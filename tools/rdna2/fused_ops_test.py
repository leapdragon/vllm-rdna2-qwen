#!/usr/bin/env python3
"""Check the T46 fused glue ops against torch references (fp16 and int8-shadow weights)."""
import torch, time
import vllm, vllm._rocm_C  # noqa
from vllm import _custom_ops as ops

torch.manual_seed(0)
dev = "cuda"
def i8(w):
    amax = w.abs().amax(dim=1).float().clamp_min(1e-8); s = amax / 127.0
    q = torch.round(w.float() / s[:, None]).clamp_(-127, 127).to(torch.int8)
    return q.contiguous(), s.half().contiguous(), (q.float() * s[:, None]).half()
def rel(a, b): return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-6)).item()
def t(fn, n=200):
    for _ in range(10): fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t0) / n * 1e6
ok = True
for M in (1, 4, 8):
    print(f"--- M={M} ---")
    # 1. gemv_act: hc down+inject [336 x 10240], silu(v/4) on first 320 cols
    xn = torch.randn(M, 10240, device=dev, dtype=torch.float16)
    W = (torch.randn(336, 10240, device=dev, dtype=torch.float16) * 0.02)
    ref = xn.float() @ W.float().T; a = ref[:, :320] / 4; ref[:, :320] = a * torch.sigmoid(a)
    y = ops.rdna_gemv_act(xn, W, None, 320, 0.25); e = rel(y, ref)
    q, s, Wd = i8(W); refq = xn.float() @ Wd.float().T; a = refq[:, :320] / 4; refq[:, :320] = a * torch.sigmoid(a)
    yq = ops.rdna_gemv_act(xn, q, s, 320, 0.25); eq = rel(yq, refq)
    print(f"  gemv_act      fp16 relerr {e:.1e} {t(lambda: ops.rdna_gemv_act(xn, W, None, 320, 0.25)):6.1f}us   int8 relerr {eq:.1e}")
    ok &= e < 2e-2 and eq < 2e-2
    # 2. hc_up_gate_mix: lora [M,320], W_up [10240,320], xn [M,10240] -> [M,2560]
    lora = torch.randn(M, 320, device=dev, dtype=torch.float16)
    Wu = (torch.randn(10240, 320, device=dev, dtype=torch.float16) * 0.05)
    g = torch.sigmoid(lora.float() @ Wu.float().T)  # [M, 10240]
    ref = (g.view(M, 4, 2560) * xn.float().view(M, 4, 2560)).mean(dim=1)
    y = ops.rdna_hc_up_gate_mix(lora, Wu, None, xn, 4); e = rel(y, ref)
    q, s, Wd = i8(Wu); g = torch.sigmoid(lora.float() @ Wd.float().T); refq = (g.view(M, 4, 2560) * xn.float().view(M, 4, 2560)).mean(dim=1)
    yq = ops.rdna_hc_up_gate_mix(lora, q, s, xn, 4); eq = rel(yq, refq)
    print(f"  hc_up_gate_mix fp16 relerr {e:.1e} {t(lambda: ops.rdna_hc_up_gate_mix(lora, Wu, None, xn, 4)):6.1f}us   int8 relerr {eq:.1e}")
    ok &= e < 2e-2 and eq < 2e-2
    # 3. shared expert gate_up+silu: x [M,2560], W [2*160, 2560] (per-rank I=160)
    x = torch.randn(M, 2560, device=dev, dtype=torch.float16)
    Wg = (torch.randn(320, 2560, device=dev, dtype=torch.float16) * 0.02)
    gu = x.float() @ Wg.float().T; ref = torch.nn.functional.silu(gu[:, :160]) * gu[:, 160:]
    y = ops.rdna_se_gate_up_silu(x, Wg, None); e = rel(y, ref)
    q, s, Wd = i8(Wg); gu = x.float() @ Wd.float().T; refq = torch.nn.functional.silu(gu[:, :160]) * gu[:, 160:]
    yq = ops.rdna_se_gate_up_silu(x, q, s); eq = rel(yq, refq)
    print(f"  se_gate_up_silu fp16 relerr {e:.1e} {t(lambda: ops.rdna_se_gate_up_silu(x, Wg, None)):6.1f}us   int8 relerr {eq:.1e}")
    ok &= e < 2e-2 and eq < 2e-2
    # 4. shared expert down gated: act [M,160], W_down [2560,160], x [M,2560], w_gate [2560]
    act = torch.randn(M, 160, device=dev, dtype=torch.float16)
    Wdn = (torch.randn(2560, 160, device=dev, dtype=torch.float16) * 0.05)
    wg = (torch.randn(2560, device=dev, dtype=torch.float16) * 0.02)
    gate = torch.sigmoid(x.float() @ wg.float()); ref = gate[:, None] * (act.float() @ Wdn.float().T)
    y = ops.rdna_se_down_gated(act, Wdn, None, x, wg); e = rel(y, ref)
    q, s, Wd = i8(Wdn); refq = gate[:, None] * (act.float() @ Wd.float().T)
    yq = ops.rdna_se_down_gated(act, q, s, x, wg); eq = rel(yq, refq)
    print(f"  se_down_gated  fp16 relerr {e:.1e} {t(lambda: ops.rdna_se_down_gated(act, Wdn, None, x, wg)):6.1f}us   int8 relerr {eq:.1e}")
    ok &= e < 2e-2 and eq < 2e-2
print("ALL PASS" if ok else "FAIL")
