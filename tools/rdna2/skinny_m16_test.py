"""Correctness of the gfx1030 skinny decode kernels at M = 1..16 against torch references.

Covers gemv_i8_rdna2, gemv_f16_rdna2 (native MT=16, and the 8 + remainder split for 9..15) and
moe_skinny_int4_decode (with an EP expert_map). Loads a freshly built library without installing it:
  RDNA_TEST_SO=build_rocm/_rocm_C.abi3.so ROCR_VISIBLE_DEVICES=1 python tools/rdna2/skinny_m16_test.py
"""
import os
import torch
import torch.nn.functional as F

so = os.environ.get("RDNA_TEST_SO")
if so:
    torch.ops.load_library(so)
else:
    import vllm._rocm_C  # noqa: F401
ops = torch.ops._rocm_C
dev = "cuda"
torch.manual_seed(0)


def rel(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-6)).item()


ok = True
K, N = 2560, 1536
W = torch.randn(N, K, device=dev, dtype=torch.float16) * 0.02
amax = W.abs().amax(dim=1).float().clamp_min(1e-8)
s8 = (amax / 127.0)
q8 = torch.round(W.float() / s8[:, None]).clamp_(-127, 127).to(torch.int8).contiguous()
s8h = s8.half().contiguous()
for M in (1, 4, 8, 9, 12, 15, 16):
    x = torch.randn(M, K, device=dev, dtype=torch.float16)
    y8 = ops.gemv_i8_rdna2(x, q8, s8h, None)
    r8 = F.linear(x, (q8.float() * s8[:, None]).half())
    y16 = ops.gemv_f16_rdna2(x, W, None)
    r16 = F.linear(x, W)
    e8, e16 = rel(y8, r8), rel(y16, r16)
    ok &= e8 < 1e-2 and e16 < 1e-2
    print(f"M={M:2d} gemv_i8 relerr {e8:.1e}   gemv_f16 relerr {e16:.1e}")

# MoE skinny: E local experts, hidden H=2560, intermediate I=640, group 128, symmetric uint4b8.
E_glob, E_loc, H, I, G, topk = 64, 16, 2560, 640, 128, 10  # small E: fits beside a running server
def pack(wq):  # wq int in [0,15], [..., rows, K] -> [..., rows, K/8] uint32, k-sequential nibbles
    sh = wq.shape
    w = wq.reshape(*sh[:-1], sh[-1] // 8, 8).to(torch.int64)
    out = torch.zeros(*sh[:-1], sh[-1] // 8, dtype=torch.int64, device=wq.device)
    for j in range(8):
        out |= w[..., j] << (4 * j)
    return out.to(torch.int32).view(torch.uint32) if hasattr(torch, "uint32") else out.to(torch.int32)
q13 = torch.randint(0, 16, (E_loc, 2 * I, H), device=dev, dtype=torch.uint8)
s13 = (torch.rand(E_loc, 2 * I, H // G, device=dev) * 0.01 + 0.002).half()
q2 = torch.randint(0, 16, (E_loc, H, I), device=dev, dtype=torch.uint8)
s2 = (torch.rand(E_loc, H, I // G, device=dev) * 0.01 + 0.002).half()
w13p, w2p = pack(q13).contiguous(), pack(q2).contiguous()
d13 = ((q13.float() - 8) * s13.float().repeat_interleave(G, dim=2))
d2 = ((q2.float() - 8) * s2.float().repeat_interleave(G, dim=2))
emap = torch.full((E_glob,), -1, dtype=torch.int32, device=dev)
local_ids = torch.randperm(E_glob, device=dev)[:E_loc]
emap[local_ids] = torch.arange(E_loc, dtype=torch.int32, device=dev)
for M in (1, 8, 16):
    x = torch.randn(M, H, device=dev, dtype=torch.float16) * 0.5
    ids = torch.stack([torch.randperm(E_glob, device=dev)[:topk] for _ in range(M)]).to(torch.int64)
    tw = torch.softmax(torch.randn(M, topk, device=dev), dim=-1).float().contiguous()
    act = torch.empty(M, topk, I, device=dev, dtype=torch.float16)
    out = torch.empty(M, H, device=dev, dtype=torch.float16)
    ops.moe_skinny_int4_decode(x, w13p, s13, w2p, s2, tw, ids, act, out, G, emap)
    ref = torch.zeros(M, H, device=dev)
    for m in range(M):
        for j in range(topk):
            e = int(emap[ids[m, j]])
            if e < 0:
                continue
            gu = d13[e] @ x[m].float()
            a = F.silu(gu[:I]) * gu[I:]
            ref[m] += tw[m, j] * (d2[e] @ a.half().float())
    e = rel(out, ref)
    ok &= e < 2e-2
    print(f"M={M:2d} moe_skinny_int4_decode relerr {e:.1e}")
print("PASS" if ok else "FAIL")
