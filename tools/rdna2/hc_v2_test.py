"""gfx1030 HC int8 kernels (rdna_gemv_act incl. the split-K variant, rdna_hc_up_gate_mix) vs torch references at
the Qwen3.8-Flash-Next shapes (hc_count 4, low rank 320, hidden 2560), plus graph-replayed timing. Run twice to
compare the split-K gemv_act with the generic one:
  ROCR_VISIBLE_DEVICES=1 python tools/rdna2/hc_v2_test.py ;  VLLM_RDNA_HC_V2=0 ROCR_VISIBLE_DEVICES=1 python ...
"""
import os
import torch
import torch.nn.functional as F
import vllm._rocm_C  # noqa: F401

R_ = torch.ops._rocm_C
dev = "cuda"
torch.manual_seed(0)
HC, D, R = 4, 2560, 320
N_DOWN, K_DOWN, ACT_COLS = R + HC + 12, HC * D, R


def i8(rows, cols, std):
    w = torch.randn(rows, cols, device=dev) * std
    s = w.abs().amax(dim=1).clamp_min(1e-8) / 127.0
    return torch.round(w / s[:, None]).clamp_(-127, 127).to(torch.int8).contiguous(), s.half().contiguous()


def rel(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-6)).item()


wd, sd = i8(N_DOWN, K_DOWN, 0.01)
wu, su = i8(HC * D, R, 0.05)
ok = True
for M in (1, 2, 8, 16):
    x = torch.randn(M, K_DOWN, device=dev, dtype=torch.float16)
    y = R_.rdna_gemv_act(x, wd, sd, ACT_COLS, 1.0)
    ref = x.float() @ (wd.float() * sd.float()[:, None]).t()
    ref[:, :ACT_COLS] = F.silu(ref[:, :ACT_COLS])
    e1 = rel(y, ref)
    lora = torch.randn(M, R, device=dev, dtype=torch.float16)
    xn = torch.randn(M, HC * D, device=dev, dtype=torch.float16)
    o = R_.rdna_hc_up_gate_mix(lora, wu, su, xn, HC)
    gate = torch.sigmoid(lora.float() @ (wu.float() * su.float()[:, None]).t())
    ref2 = (gate * xn.float()).view(M, HC, D).mean(dim=1)
    e2 = rel(o, ref2)
    ok &= e1 < 1e-2 and e2 < 1e-2
    print(f"M={M:2d}: gemv_act relerr {e1:.1e}   hc_up_gate_mix relerr {e2:.1e}")

v2 = os.environ.get("VLLM_RDNA_HC_V2", "1") != "0"
for M in (1, 2):
    x = torch.randn(M, K_DOWN, device=dev, dtype=torch.float16)
    lora = torch.randn(M, R, device=dev, dtype=torch.float16)
    xn = torch.randn(M, HC * D, device=dev, dtype=torch.float16)
    # 24 distinct weight sets so the 128 MB Infinity Cache does not hold them (as in the model)
    ws = [(i8(N_DOWN, K_DOWN, 0.01), i8(HC * D, R, 0.05)) for _ in range(24)]
    for name, fn in (("gemv_act", lambda W: R_.rdna_gemv_act(x, W[0][0], W[0][1], ACT_COLS, 1.0)),
                     ("hc_up_gate_mix", lambda W: R_.rdna_hc_up_gate_mix(lora, W[1][0], W[1][1], xn, HC))):
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            for W in ws:
                fn(W)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            for W in ws:
                fn(W)
        g.replay(); torch.cuda.synchronize()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(20):
            g.replay()
        b.record(); torch.cuda.synchronize()
        print(f"[{'v2' if v2 else 'generic'}] M={M} {name}: {a.elapsed_time(b) * 1e3 / (20 * len(ws)):.2f} us per call")
print("PASS" if ok else "FAIL")
