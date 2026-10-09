"""rdna_fused_glue ops at M=16 must equal two M=8 calls (rows are independent): int8 and fp16 weights.
  RDNA_TEST_SO=build_rocm/_rocm_C.abi3.so ROCR_VISIBLE_DEVICES=1 python tools/rdna2/glue_m16_test.py"""
import os, torch
so = os.environ.get("RDNA_TEST_SO")
torch.ops.load_library(so) if so else __import__("vllm._rocm_C")
o = torch.ops._rocm_C; dev = "cuda"; torch.manual_seed(0)
def i8(w):
    s = w.abs().amax(dim=1).float().clamp_min(1e-8) / 127.0
    return torch.round(w.float() / s[:, None]).clamp_(-127, 127).to(torch.int8).contiguous(), s.half().contiguous()
def same(f, *ins):  # ins: per-row tensors first (split), then shared args
    full = f(*[t for t in ins])
    return full
ok = True
H, R, hc, I = 2560, 320, 4, 160
for quant in (True, False):
    Wd = torch.randn(336, hc * H, device=dev, dtype=torch.float16) * 0.01
    Wu = torch.randn(hc * H, R, device=dev, dtype=torch.float16) * 0.05
    W1 = torch.randn(2 * I, H, device=dev, dtype=torch.float16) * 0.02
    W2 = torch.randn(H, I, device=dev, dtype=torch.float16) * 0.05
    wg = (torch.randn(H, device=dev, dtype=torch.float16) * 0.02).contiguous()
    P = {k: (i8(v) if quant else (v.contiguous(), None)) for k, v in dict(d=Wd, u=Wu, a=W1, b=W2).items()}
    xn = torch.randn(16, hc * H, device=dev, dtype=torch.float16)
    x = torch.randn(16, H, device=dev, dtype=torch.float16)
    lora = torch.randn(16, R, device=dev, dtype=torch.float16)
    act = torch.randn(16, I, device=dev, dtype=torch.float16)
    tests = {
        "gemv_act": lambda r: o.rdna_gemv_act(xn[r].contiguous(), P["d"][0], P["d"][1], R, 1.0 / hc),
        "hc_up_gate_mix": lambda r: o.rdna_hc_up_gate_mix(lora[r].contiguous(), P["u"][0], P["u"][1], xn[r].contiguous(), hc),
        "se_gate_up_silu": lambda r: o.rdna_se_gate_up_silu(x[r].contiguous(), P["a"][0], P["a"][1]),
        "se_down_gated": lambda r: o.rdna_se_down_gated(act[r].contiguous(), P["b"][0], P["b"][1], x[r].contiguous(), wg),
    }
    for name, f in tests.items():
        y16 = f(slice(0, 16)); y8 = torch.cat([f(slice(0, 8)), f(slice(8, 16))])
        d = (y16.float() - y8.float()).abs().max().item()
        ok &= d == 0.0 and torch.isfinite(y16).all().item()
        print(f"{'int8' if quant else 'fp16'} {name:16s} M=16 vs 2x M=8: max |diff| {d:.2e}")
print("PASS" if ok else "FAIL")
