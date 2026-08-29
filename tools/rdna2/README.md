# tools/rdna2 — build, serve, measure, validate

The scripts that built, served, measured and validated the configuration in
`docs/rdna2/`. Benchmarks assume the server on `localhost:8000`.

| file | what |
|---|---|
| `build-torch-rocm714.sh` | build PyTorch (ROCm/pytorch release/2.12) + Triton wheels for gfx1030 against TheRock ROCm 7.14 |
| `serve-qwen38-flash-next.sh` | the serving configuration (host, no container); every knob explained in `docs/rdna2/CHANGES.md` §4 |
| `validate.py [base_url]` | four greedy sanity checks against the running server; exit 0 only if all pass |

| file | what | how to run |
|---|---|---|
| `bench.py [runs] [max_tokens]` | the decode yardstick: streamed completion, t/s first→last chunk | host, against the running server |
| `bench_ctx.py <ctx_tokens> [max_tokens] [runs]` | decode t/s at a context length, unique prompts (no prefix-cache hits) | host |
| `trace_agg.py <trace.json.gz> [top]` | kernel time by family / top kernels / GPU busy from a torch-profiler trace | host (`logs/traces/`) |
| `trace_attr.py <trace.json.gz> <steps>` | eager-mode attribution: kernel → CPU op → innermost model source line | host; needs an `EAGER=1 PROFILE=1` boot |
| `gemm_probe.py` | production GEMM route vs rocBLAS on the model's 12 dense shapes | on a free card |
| `ar_ops_test.py [W]` | cross-process test of the one-shot all-reduce torch ops (eager + graph replay, two instances) | all four cards, server stopped |
| `fused_ops_test.py`, `rdna_ops_test.py` | fused glue kernels and runtime-dispatch ops vs torch references | on a free card |
| `gemv_f16_harness.cu`, `gemv_i8_harness.cu`, `moe_ep_harness.cu` | standalone hipcc harnesses with CPU references and bandwidth numbers | `hipcc -O3 --offload-arch=gfx1030` |
| `ar4_test.cpp` | standalone W-rank all-reduce test (correctness, bit-identity, graph replay, skew soak, µs/op) | same; `./ar4_test W n iters dev0 dev1 ...` |

Run the kernel tests and harnesses on a card the server is not using (`ROCR_VISIBLE_DEVICES=<id>`),
from the venv that has this fork installed; the `.cu`/`.cpp` harnesses build with
`hipcc -O3 --offload-arch=gfx1030 <file>` from `/opt/rocm/bin`. `ar4_test`/`ar_ops_test.py` need
all four cards and a stopped server.

Rebuilding only the ROCm extension after a kernel edit (editable install):

    cmake -G Ninja -S . -B build_rocm -DVLLM_TARGET_DEVICE=rocm \
          -DVLLM_PYTHON_EXECUTABLE=$(which python3) -DCMAKE_BUILD_TYPE=Release
    cmake --build build_rocm --target _rocm_C -j 12 && cp build_rocm/_rocm_C.abi3.so vllm/
