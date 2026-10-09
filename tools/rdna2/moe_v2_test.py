"""moe_skinny_int4_decode_v2 vs a torch reference and vs v1, EP expert_map, with/without residual.
  RDNA_TEST_SO=build_rocm/_rocm_C.abi3.so ROCR_VISIBLE_DEVICES=1 python tools/rdna2/moe_v2_test.py"""
import os, torch, torch.nn.functional as F
so = os.environ.get("RDNA_TEST_SO")
torch.ops.load_library(so) if so else __import__("vllm._rocm_C")
o = torch.ops._rocm_C; dev = "cuda"; torch.manual_seed(0)
E_GLOB, E_LOC, H, I, G, TOPK = 64, 16, 2560, 640, 128, 10
def pack(q):
    sh = q.shape; w = q.reshape(*sh[:-1], sh[-1] // 8, 8).to(torch.int64)
    out = torch.zeros(*sh[:-1], sh[-1] // 8, dtype=torch.int64, device=dev)
    for j in range(8): out |= w[..., j] << (4 * j)
    return out.to(torch.int32).contiguous()
q13 = torch.randint(0, 16, (E_LOC, 2 * I, H), device=dev, dtype=torch.uint8)
q2 = torch.randint(0, 16, (E_LOC, H, I), device=dev, dtype=torch.uint8)
s13 = (torch.rand(E_LOC, 2 * I, H // G, device=dev) * 0.01 + 0.002).half()
s2 = (torch.rand(E_LOC, H, I // G, device=dev) * 0.01 + 0.002).half()
w13p, w2p = pack(q13), pack(q2)
d13 = (q13.float() - 8) * s13.float().repeat_interleave(G, dim=2)
d2 = (q2.float() - 8) * s2.float().repeat_interleave(G, dim=2)
emap = torch.full((E_GLOB,), -1, dtype=torch.int32, device=dev)
emap[torch.randperm(E_GLOB, device=dev)[:E_LOC]] = torch.arange(E_LOC, dtype=torch.int32, device=dev)
def rel(a, b): return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-6)).item()
ok = True
for M in (1, 3, 8, 16):
    for ids_dt, w_dt in ((torch.int64, torch.float32), (torch.int32, torch.float16)):
        x = torch.randn(M, H, device=dev, dtype=torch.float16) * 0.5
        ids = torch.stack([torch.randperm(E_GLOB, device=dev)[:TOPK] for _ in range(M)]).to(ids_dt).contiguous()
        tw = torch.softmax(torch.randn(M, TOPK, device=dev), -1).to(w_dt).contiguous()
        res = torch.randn(M, H, device=dev, dtype=torch.float16) * 0.1
        ref = torch.zeros(M, H, device=dev)
        for m in range(M):
            for j in range(TOPK):
                e = int(emap[int(ids[m, j])])
                if e < 0: continue
                gu = d13[e] @ x[m].float(); a = (F.silu(gu[:I]) * gu[I:]).half().float()
                ref[m] += float(tw[m, j]) * (d2[e] @ a)
        act = torch.empty(M, TOPK, I, device=dev, dtype=torch.float16)
        o1 = torch.empty(M, H, device=dev, dtype=torch.float16); o2 = torch.empty_like(o1); o3 = torch.empty_like(o1)
        o.moe_skinny_int4_decode(x, w13p, s13, w2p, s2, tw, ids, act, o1, G, emap)
        o.moe_skinny_int4_decode_v2(x, w13p, s13, w2p, s2, tw, ids, act, o2, G, emap, None)
        o.moe_skinny_int4_decode_v2(x, w13p, s13, w2p, s2, tw, ids, act, o3, G, emap, res)
        e1, e2, e3 = rel(o1, ref), rel(o2, ref), rel(o3, ref + res.float())
        ok &= e2 < 2e-2 and e3 < 2e-2 and torch.isfinite(o3).all().item()
        print(f"M={M:2d} ids {str(ids_dt)[6:]:5s} w {str(w_dt)[6:]:7s}: v1 {e1:.1e}  v2 {e2:.1e}  v2+residual {e3:.1e}")
print("PASS" if ok else "FAIL")
