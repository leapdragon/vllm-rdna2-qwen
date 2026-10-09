"""QSA sparse paged attention over an int8_per_token_head cache vs the fp16 path over the dequantized cache.

Same inputs both ways: K/V are quantized per (token, head) to int8 + fp32 scale (absmax / 127), and the fp16
reference cache holds exactly the dequantized values, so any difference is the int8 kernel path itself (the
scale applied to scores / probabilities instead of to K / V), not quantization error.
  ROCR_VISIBLE_DEVICES=1 python tools/rdna2/qsa_int8_test.py
"""
import torch

from vllm.models.qwen4_exp.amd.ops.qsa import qsa_sparse_paged_attention

dev = "cuda"
torch.manual_seed(0)


def quant(x):  # [..., hd] -> int8 [..., hd], fp32 scale [...]
    s = x.float().abs().amax(dim=-1).clamp_min(1e-8) / 127.0
    q = torch.round(x.float() / s[..., None]).clamp_(-127, 127).to(torch.int8)
    return q, s


ok = True
HD, NKV, GROUP, PAGE, BLOCKS = 256, 1, 6, 16, 64
for rows, topk in ((1, 2048), (2, 2048), (8, 512), (64, 256)):
    nq = NKV * GROUP
    q = (torch.randn(rows, nq, HD, device=dev) * 0.5).half()
    k = torch.randn(BLOCKS, PAGE, NKV, HD, device=dev) * 2
    v = torch.randn(BLOCKS, PAGE, NKV, HD, device=dev)
    kq, ks = quant(k)
    vq, vs = quant(v)
    k_ref = (kq.float() * ks[..., None]).half()
    v_ref = (vq.float() * vs[..., None]).half()
    reqs = 2
    block_table = torch.stack([torch.randperm(BLOCKS, device=dev)[: BLOCKS // 2] for _ in range(reqs)]).int()
    token_to_req = torch.randint(0, reqs, (rows,), device=dev, dtype=torch.int32)
    ctx = (BLOCKS // 2) * PAGE
    idx = torch.stack([torch.randperm(ctx, device=dev)[:topk] for _ in range(rows)]).int()
    idx[:, -5:] = -1  # unselected slots
    ref = qsa_sparse_paged_attention(q, k_ref, v_ref, idx, block_table, token_to_req)
    out = qsa_sparse_paged_attention(q, kq, vq, idx, block_table, token_to_req, k_scale=ks, v_scale=vs)
    err = ((out.float() - ref.float()).abs().max() / ref.float().abs().max()).item()
    good = err < 5e-3 and torch.isfinite(out).all().item()
    ok &= good
    print(f"rows={rows:3d} topk={topk:4d}: int8 path vs fp16-on-dequantized relerr {err:.1e} {'ok' if good else 'FAIL'}")
print("PASS" if ok else "FAIL")
