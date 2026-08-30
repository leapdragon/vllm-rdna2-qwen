#!/usr/bin/env python3
"""Micro-benchmark of the CPU n-gram row gather on the real int4 sidecar (2026-08-30).

Compares, for a decode-sized batch (num_reqs requests x 16 heads = rows):
  torch   - the worker's _PleQuantTable.gather_into (argsort/unique/per-shard torch ops)
  numpy   - a fused path: per-row numpy indexing on the same mmaps, one vectorised dequant
and checks both produce identical rows. Usage: ple_gather_bench.py [sidecar_dir] [num_reqs]
"""
import os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.expanduser("~/repos/vllm-rdna2-qwen"))
from vllm.v1.ple_offload.worker import _PleQuantTable  # noqa: E402

quant_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/repos/vllm-rdna2-qwen/models/qwen38-flash-next-ple/ples_int4")
num_reqs = int(sys.argv[2]) if len(sys.argv) > 2 else 1
HEADS, WIDTH, ROWS = 16, 160, 320001536
tbl = _PleQuantTable(quant_dir, ROWS, WIDTH)
q_np = [t.numpy() for t in tbl._q]            # (rows_per_shard, 80) uint8, zero-copy mmap views
s_np = [t.numpy() for t in tbl._s]            # (rows_per_shard, 10) fp16
RPS = tbl.ROWS_PER_SHARD
G = WIDTH // s_np[0].shape[1]

def fused_numpy(ids: np.ndarray, out: np.ndarray) -> None:
    """ids int64 (n,), out float32 (n, WIDTH)."""
    n = ids.shape[0]
    packed = np.empty((n, WIDTH // 2), dtype=np.uint8)
    scales = np.empty((n, WIDTH // G), dtype=np.float16)
    for k in range(n):
        i = int(ids[k]); s = i // RPS; l = i - s * RPS
        packed[k] = q_np[s][l]
        scales[k] = s_np[s][l]
    lo = (packed & 0xF).astype(np.float32)
    hi = (packed >> 4).astype(np.float32)
    nib = np.empty((n, WIDTH), dtype=np.float32)
    nib[:, 0::2] = lo; nib[:, 1::2] = hi
    out[:] = (nib - 8.0) * np.repeat(scales.astype(np.float32), G, axis=1)

rng = np.random.default_rng(0)
def rand_ids(n):
    return rng.integers(0, ROWS, size=n, dtype=np.int64)

# correctness on a few batches
for trial in range(5):
    ids = rand_ids(num_reqs * HEADS)
    ref = torch.empty((ids.shape[0], WIDTH), dtype=torch.float32)
    tbl.gather_into(torch.from_numpy(ids), ref)
    out = np.empty((ids.shape[0], WIDTH), dtype=np.float32)
    fused_numpy(ids, out)
    assert np.array_equal(ref.numpy(), out), f"mismatch in trial {trial}: max diff {np.abs(ref.numpy()-out).max()}"
print("fused numpy == torch gather_into on 5 random batches: OK")

# timing: fresh random ids each iteration (cold-ish pages, like real decode), then repeated ids (warm)
def bench(fn, n_iter, same_ids):
    ids = rand_ids(num_reqs * HEADS)
    ts = []
    for _ in range(n_iter):
        if not same_ids:
            ids = rand_ids(num_reqs * HEADS)
        t0 = time.perf_counter(); fn(ids); ts.append(time.perf_counter() - t0)
    ts.sort(); return ts[len(ts)//2] * 1e3, ts[int(len(ts)*0.9)] * 1e3

ref = torch.empty((num_reqs * HEADS, WIDTH), dtype=torch.float32)
out = np.empty((num_reqs * HEADS, WIDTH), dtype=np.float32)
for name, fn in [("torch gather_into", lambda ids: tbl.gather_into(torch.from_numpy(ids), ref)),
                 ("fused numpy", lambda ids: fused_numpy(ids, out))]:
    p50c, p90c = bench(fn, 200, same_ids=False)
    p50w, p90w = bench(fn, 200, same_ids=True)
    print(f"{name:18s} {num_reqs} req x {HEADS} heads: random ids p50 {p50c:.2f} ms p90 {p90c:.2f} ms | same ids p50 {p50w:.3f} ms")
