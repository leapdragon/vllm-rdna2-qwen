# TunableOp rows (PyTorch TunableOp, rocBLAS GEMM solutions)

One directory per **rocBLAS build**: `rocblas-<sha256[:12] of librocblas.so.5>/tunableop_results{0..3}.csv`
(one file per TP rank; PyTorch appends the device ordinal to the configured filename).

The rows are keyed by the library's hash, not its version string, because solution ids come from the
Tensile library as built — two builds with the same version string can offer different solution sets, and
a row naming a solution the runtime lacks aborts the first GEMM (`Expected iter != ops_.end()`).
`tools/rdna2/serve-qwen38-flash-next.sh` computes the hash, uses the matching directory lookup-only, and
disables TunableOp with a log line when there is none. To produce rows for a new build: boot once with
`TUNEOP_TUNING=1` (cards capped low; the tuning autotunes every new GEMM shape on first sight, so drive
a set of representative prompt lengths through it), stop the server gracefully, and commit the directory.

| directory | rocBLAS build |
|---|---|
| `rocblas-3b4878bc6c37` | TheRock 7.14.0rc3 host install (rocBLAS 5.5.0.cd957402), tuned 2026-09-04 on 4× Radeon PRO V620 |
| `rocblas-9847aecc4bf8` | TheRock 7.14.1 public tarball — the `ghcr.io/leapdragon/vllm-rdna2-qwen` image (same rocBLAS version string, ~190 of 290 solution ids differ), tuned 2026-09-05 inside the image: 27 prompt lengths 74–22,883 tokens, 392 rows per rank |
