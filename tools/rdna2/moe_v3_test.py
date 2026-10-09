"""moe_decode_v3 (shared expert fused into the routed pair) vs the composite it replaces:
rdna_se_gate_up_silu + rdna_se_down_gated + moe_skinny_int4_decode_v2, summed (top-k from topk_softmax for both)."""
import torch, vllm._custom_ops  # noqa: F401
R, MO = torch.ops._rocm_C, torch.ops._moe_C
dev = "cuda"; torch.manual_seed(0)
E_GLOB, E_LOC, H, I, G, TOPK, ISH = 512, 64, 2560, 640, 128, 10, 160
def i8(r, c, std):
    w = torch.randn(r, c, device=dev, dtype=torch.float16) * std
    s = (w.abs().amax(1).float() / 127).clamp_min(1e-8)
    return torch.round(w.float() / s[:, None]).clamp(-127, 127).to(torch.int8).contiguous(), s.half().contiguous()
router, se1, se2 = i8(E_GLOB, H, 0.02), i8(2 * ISH, H, 0.02), i8(H, ISH, 0.05)
seg = (torch.randn(H, device=dev, dtype=torch.float16) * 0.02).contiguous()
w13 = torch.randint(-2**31, 2**31 - 1, (E_LOC, 2 * I, H // 8), device=dev, dtype=torch.int32)
w2 = torch.randint(-2**31, 2**31 - 1, (E_LOC, H, I // 8), device=dev, dtype=torch.int32)
s13 = (torch.rand(E_LOC, 2 * I, H // G, device=dev) * 2e-3).half(); s2 = (torch.rand(E_LOC, H, I // G, device=dev) * 2e-3).half()
emap = torch.full((E_GLOB,), -1, dtype=torch.int32, device=dev)
emap[torch.randperm(E_GLOB, device=dev)[:E_LOC]] = torch.arange(E_LOC, dtype=torch.int32, device=dev)
def rel(a, b): return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-6)).item()
ok = True
for M in (1, 4, 8, 16):
    x = torch.randn(M, H, device=dev, dtype=torch.float16) * 0.5
    lg = R.gemv_i8_rdna2(x, router[0], router[1], None)
    tw = torch.empty(M, TOPK, device=dev); ti = torch.empty(M, TOPK, device=dev, dtype=torch.int32); tei = torch.empty_like(ti)
    MO.topk_softmax(tw, ti, tei, lg, True, None, None)
    ti[0, -1] = -1  # a padding slot (vLLM marks them -1): must be skipped, not looked up in expert_map
    act = torch.empty(M, TOPK, I, device=dev, dtype=torch.float16); routed = torch.empty(M, H, device=dev, dtype=torch.float16)
    R.moe_skinny_int4_decode_v2(x, w13, s13, w2, s2, tw, ti, act, routed, G, emap, None)
    sh = R.rdna_se_down_gated(R.rdna_se_gate_up_silu(x, se1[0], se1[1]), se2[0], se2[1], x, seg)
    ref = routed.float() + sh.float()
    act3 = torch.empty(M, TOPK, I, device=dev, dtype=torch.float16); ash = torch.empty(M, ISH, device=dev, dtype=torch.float16)
    sg3 = torch.empty(M, device=dev); out = torch.empty(M, H, device=dev, dtype=torch.float16); sh3 = torch.empty_like(out)
    R.moe_decode_v3(x, tw, ti, w13, s13, w2, s2, G, emap, se1[0], se1[1], se2[0], se2[1], seg, act3, ash, sg3, sh3, out)
    e = rel(out.float() + sh3.float(), ref)
    ok &= e < 1e-2 and torch.isfinite(out).all().item()
    print(f"M={M:2d}: output relerr vs composite {e:.1e}")
print("PASS" if ok else "FAIL")
