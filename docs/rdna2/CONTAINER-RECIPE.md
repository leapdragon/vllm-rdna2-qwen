# Container recipe (validated environment) — Qwen3.8-Flash-Next on 4× Radeon PRO V620 under vLLM, reproducibly

**Copyright © 2026 Aron Hsiao. GPL-3.0-or-later.**

This is the end-to-end recipe for the state reached on 2026-08-29 (T46): **Qwen3.8-Flash-Next
decoding at 97–101 t/s** (single stream, 256 tokens; 95 t/s over 1024; 78–108 t/s across 3k–27k
context, acceptance-dependent) on four gfx1030 cards, with correct greedy output. It is written so another engineer or LLM can
rebuild it from a clean host without this repository's history. Where the repo has the long
form (`docs/TESTS_RESULTS.md` T42–T46, `docs/nightly-notes/20260829.md`,
`qwen4exp/README.md`), this file has the *procedure*.

Everything below was measured, not planned. Numbers are from inside the serving process.

> **Note (this fork).** This is the container-based procedure the numbers in `RESULTS.md` were
> measured with (ROCm 7.2.3 inside the image, the same source tree and kernels as this fork's
> `rdna2/qwen38-flash-next` branch). The host build against TheRock 7.14 in `README.md` is the
> distribution path; if you want the exact validated environment, this is it. Paths and patch
> numbers refer to the research repository the fork was assembled from.

---

## 0. Result and the one-paragraph story

| stage | decode t/s | what changed |
|---|---|---|
| first working run (bf16 n-gram table memmapped from disk, eager) | 5.7 | model runs at all |
| int4 PLE sidecar (table served from a 30 GB int4 copy in page cache) | 5.5 | no throughput change — only memory headroom |
| + CUDA graphs | 16.5 | 3.0× |
| + MTP=3 speculative decoding (llama.cpp parity: 29–30 t/s on the same box) | 30–35 | 1.9× |
| + T43: own fp16 skinny GEMV for every dense projection; int4 MoE GEMV under EP | 73–77 | 2.1× |
| + T44: own one-shot P2P all-reduce (33 µs vs RCCL 156 µs) | 79–89 | 1.1× |
| **+ T45/T46: int8 shadows of the dense projections; dispatch-count fusion (2,900 → 1,800 kernels/step), all behind runtime-dispatch custom ops** | **97–101** | **1.2×** |

The T43 win came from profiling inside the server: 60 % of GPU time was rocBLAS running
decode-shaped fp16 GEMMs at ~35 % of memory bandwidth, because vLLM's skinny-GEMV route is gated
to gfx9/gfx11+ and (it turns out) its RDNA kernels miscompute when built for gfx10. A
90-line kernel of our own replaced them. A second, older kernel of ours (the int4 MoE GEMV)
had silently never run on this model because its hook rejected expert parallelism.

---

## 1. Hardware and host

- **GPUs**: 5× AMD Radeon PRO V620 (Navi 21, gfx1030, 32 GB, 72 CUs, ~506 GB/s measured
  DRAM ceiling — `docs/PROFILE-NAVI21.md`). vLLM uses four of them:
  `ROCR_VISIBLE_DEVICES=1,2,3,4` = PCI `0d:00.0, 10:00.0, 43:00.0, 46:00.0`. ROCR device 0 =
  PCI `0a:00.0` is left free (the host's llama.cpp production stack shares the machine).
  **ROCR ordering ≠ `/dev/dri/cardN` ordering**: `rocm-smi --showbus` is the authority.
- **Kernel** 7.0.0-27-generic with the TP=4 stability line (T41, `docs/TESTS_RESULTS.md`):
  `amdgpu.pcie_gen_cap=0x00070007 amdgpu.aspm=0 amdgpu.runpm=0 amdgpu.gpu_recovery=1
  amdgpu.noretry=1 amdgpu.ras_enable=0 amdgpu.ppfeaturemask=0xfff77fff amdgpu.gartsize=4096
  amd_iommu=on iommu=pt`. Without this stack every earlier flat-TP attempt lost cards off the
  PCIe bus (`TROUBLESHOOTING.md` §4). Which subset is load-bearing is not isolated — use all of it.
- **Host ROCm is mixed** (apt 7.1 + TheRock 7.14). **Never mount host ROCm into a container.**
  The container brings its own ROCm 7.2.3.
- Docker 29.x. Power caps 232 W. A fan controller (`manage-gpu-fans`) must be running before
  any timing is trusted — thermal throttling has invalidated measurements before.
- Host RAM ≥ 96 GB: the int4 n-gram sidecar (30 GB) must sit in page cache; the CPU offload
  worker gathers from it every forward.

---

## 2. Software: image, source tree, patches

### 2.1 Image

`vllm-gfx1030:qwen4exp` — our "v7" gfx1030 base (ROCm 7.2.3, PyTorch 2.11.0+git d0c8b1f,
HIP 7.2.53211, Python 3.12, Triton (ROCm fork), `HSA_OVERRIDE_GFX_VERSION=10.3.0`, stock AMD
`libhsa-runtime64`). Built from `docs/Dockerfile.phase7` on top of phases 1–6
(`docs/Dockerfile.phase1..6` — the gfx1030 arch line is added by hand, patch 0008). The vLLM
tree is **not baked into the image**: it is bind-mounted at `/src` and installed editable
(`pip install -e . --no-build-isolation --no-deps` inside the image, once; 52 objects, the
`csrc/rocm/` kernels compile for `PYTORCH_ROCM_ARCH=gfx1030`). The compiled extensions live in
the tree itself: `/src/vllm/_C.abi3.so`, `/src/vllm/_rocm_C.abi3.so`, …

### 2.2 Source tree

```
git clone --depth 1 --branch release/qwen38next https://github.com/peakcrosser7/vllm.git qwen4exp/vllm
cd qwen4exp/vllm && git checkout -b rdna2-port          # base: 2a4cd64 "fix csa-linear packed tensor stride" (head of vLLM PR #53896)
git am ../patches/*.patch                                 # 22 patches, in order
```

vLLM reports itself as `0.28.0+rocm723`. The `qwen4_exp` model family in this tree has an
NVIDIA path and an AMD path (`vllm/models/qwen4_exp/amd/`); `on_gfx10x()` selects the AMD path.

### 2.3 The patch series (`qwen4exp/patches/`)

| # | what | why it is needed |
|---|---|---|
| 0001 | gfx10x platform support (`on_gfx10x()`, device-name/amdsmi index fixes, RDNA custom-AR opt-in) | vLLM does not know gfx1030 |
| 0002 | LDS tile for head_dim 256 | 64 KiB LDS hard cap on this chip |
| 0003 | softmax segments | prefill attention on gfx1030 |
| 0004 | W4 blocking 256 | int4 kernels' tile shapes |
| 0005 | moe_wna16 for gfx1030 | Triton WNA16 MoE config for the chip |
| 0006 | **int4 MoE skinny GEMV** (`moe_skinny_int4_decode`, `csrc/rocm/skinny_gemms_int4.cu`) | the wave-per-row decode MoE kernel (T38) |
| 0007 | RDNA hybrid W4A16 (context drift) | int4 linear path on gfx1030 (T35) |
| 0008 | base-image arch line | gfx1030 in the ROCm base image |
| 0009 | **FP16 alongside BF16 in the qwen4_exp AMD path** | gfx1030 has no native bf16; QSA is faster and more accurate in fp16 |
| 0010 | let ROCm run FlashAttention-derived backends | QSA guard + `reshape_and_cache_flash` import gate |
| 0011–0013 | **PLE offload** infrastructure, HIP driver shim (`hipStreamWriteValue32`/`WaitValue32`/`hipHostRegister` via ctypes), AMD-path fixes (meta-tensor materialisation, CPU-worker op bypass) | the 51 B-row n-gram table cannot live on the GPUs; the offload worker was CUDA-only |
| 0014 | disk-backed n-gram tables (vLLM PR #54070) | bf16 table memmapped from disk when RAM is short |
| 0015 | real-weights fixes (meta-buffer `copy_` no-op, offload guard in `load_weights`, meta parameter binding) | three bugs only real weights expose |
| 0016–0017 | **PLE quantized sidecar** (`VLLM_PLE_QUANT_DIR`): int4/fp8 table served by the CPU worker; completeness-check registration | replaces the 102 GB bf16 shard with a 30 GB int4 copy |
| 0018 | **T43**: `gemv_f16_rdna2` + route; EP-aware `moe_skinny_int4_decode` | the 2.1× |
| 0019 | **T44**: `rdna_ar_oneshot` one-shot P2P all-reduce + `rdna_all_reduce.py` + communicator hook | RCCL 156 µs → 33 µs per collective |
| 0020 | **T45/T46**: `gemv_i8_rdna2` + int8 shadows; MoE hook glue in-kernel; indexer norm/rope ops; fused qk-norm-rope on ROCm | bytes and launches |
| 0021–0022 | **T46**: `rdna_fused_glue.cu` (hc mix, shared expert) + runtime-dispatch custom ops (`rdna_ops.py`) | the trace-time freeze; 2,900 → 1,800 kernels/step |

### 2.4 Rebuilding only the ROCm extension after a kernel edit

The editable install has no persistent build directory, so configure one once and build the
single target (~4 min on 12 cores). Do it in a throwaway container from the image with the
tree mounted; install the `.so` by **rename** so a running server's mapping stays valid:

```bash
docker run -d --name qwen4exp-cbuild --entrypoint bash -e PYTORCH_ROCM_ARCH=gfx1030 \
  -v $REPO/qwen4exp/vllm:/src -v $REPO/logs:/logs vllm-gfx1030:qwen4exp -c '
  cd /src && cmake -G Ninja -S . -B build_rocm -DVLLM_TARGET_DEVICE=rocm \
      -DVLLM_PYTHON_EXECUTABLE=/usr/bin/python3 -DCMAKE_BUILD_TYPE=Release &&
  cmake --build build_rocm --target _rocm_C -j 12 > /logs/rocm_C-rebuild.log 2>&1 && echo BUILD_OK >> /logs/rocm_C-rebuild.log'
# then:
cp qwen4exp/vllm/build_rocm/_rocm_C.abi3.so qwen4exp/vllm/vllm/_rocm_C.abi3.so.new && mv qwen4exp/vllm/vllm/_rocm_C.abi3.so.new qwen4exp/vllm/vllm/_rocm_C.abi3.so
```

The image's entrypoint is not `bash`; always pass `--entrypoint bash` (or `python3`) or the
arguments go to `sleep`.

---

## 3. Model artifacts (`hf-cache/pleq/`)

The model is ~176 B parameters (~6 B active): 48 layers (36 Gated-DeltaNet linear-attention +
12 sparse-attention "QSA" layers, interval 4), hidden 2560, 512 experts top-10 with
`moe_intermediate_size` 640, hyper-connections (`hc_count` 4, low-rank 320), vocab 248 320, a
built-in MTP head, a vision tower, and **a 51 B-parameter n-gram embedding table in one layer**
(`ple_layer_ids=[2]`: 320 001 536 rows × 160, 102 GB in bf16). Text-only serving is all we do.

### 3.1 Backbone: `wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16`

```
hf download wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16 --local-dir hf-cache/pleq/backbone \
   --exclude "model-00001-of-00005.safetensors"      # shard 1 is the 102.4 GB unquantized n-gram table: not needed
```

73 GB: shards 2–5 + `model_mtp.safetensors`, config, tokenizer. Format `compressed-tensors`
(`pack-quantized`, int4 symmetric, group 128). **Only the 512 routed experts are quantized**
(67.3 GB). Everything else is bf16 (served fp16): `linear_attn` 4.17 GB, `self_attn` 1.24 GB,
both hyper-connections 1.27 GB, `lm_head` 1.27 GB, router + shared expert 0.6 GB, MTP 0.18 GB.
That fact drives T43.

### 3.2 n-gram table: int4 sidecar from `primitive-ai/Qwen3.8-Flash-Next-PLE-quant`

```
hf download primitive-ai/Qwen3.8-Flash-Next-PLE-quant --include "ples_int4/*" --local-dir hf-cache/pleq/pleq
```

30 GB, 128 shards `shard_N.safetensors` + `META.json`
(`layout: group16_int4_fp16scale_lownibblefirst, rows 320001536, width 160,
worst_shard_rel_err 0.071`). Patch 0016 teaches the CPU offload worker to gather from it
(`_PleQuantTable` in `vllm/v1/ple_offload/worker.py`: safetensors mmap + pure-torch dequant,
no CUDA). Rationale: the bf16 table needs ~103 GB of RAM or disk paging; this needs 30 GB of
page cache. It changed throughput by <4 % (T42) — its value is headroom, not speed.

Removing weights that were created by a container: remove them via a container too
(`docker run --rm -v …:/x … rm -rf /x/…`) — they are root-owned.

---

## 4. Serving configuration (`qwen4exp-pleq/serve-pleq.sh`)

```bash
docker run -d --name pleq-serve --entrypoint python3 \
  --device /dev/kfd --device /dev/dri --group-add 991 --group-add 44 \
  --ipc=host --shm-size=32g --network=host \
  -e HSA_OVERRIDE_GFX_VERSION=10.3.0 -e ROCR_VISIBLE_DEVICES=1,2,3,4 \
  -e HSA_NO_SCRATCH_RECLAIM=1 -e NCCL_P2P_LEVEL=PXB \
  -e VLLM_ROCM_USE_AITER=0 -e TORCH_BLAS_PREFER_HIPBLASLT=0 \
  -e FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
  -e VLLM_PLE_CPU_OFFLOAD=1 -e VLLM_PLE_QUANT_DIR=/ples_int4 -e VLLM_PLE_OFFLOAD_READY_TIMEOUT=3600 \
  -v $REPO/qwen4exp/vllm:/src -v $REPO/hf-cache/pleq/backbone:/model \
  -v $REPO/hf-cache/pleq/pleq/ples_int4:/ples_int4 -v $REPO/logs/traces:/traces \
  -w / vllm-gfx1030:qwen4exp -m vllm.entrypoints.openai.api_server \
  --model /model --served-model-name qwen38-flash-next \
  --dtype float16 --tensor-parallel-size 4 --enable-expert-parallel \
  --max-model-len 65536 --gpu-memory-utilization ${GPUUTIL:-0.90} \
  --max-num-seqs 4 --max-num-batched-tokens 2048 \
  ${EAGER:+--enforce-eager} --language-model-only --skip-mm-profiling --enable-prefix-caching \
  ${MTP:+--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP}"} \
  ${PROFILE:+--profiler-config.profiler=torch --profiler-config.torch_profiler_dir=/traces} \
  --host 0.0.0.0 --port 8000
```

The fast configuration is **`DENSE_INT8=1 MTP=3 GPUUTIL=0.86 ./qwen4exp-pleq/serve-pleq.sh`**
(add `PROFILE=1` to be able to trace; `P2P=` overrides `NCCL_P2P_LEVEL`; `COMPILE_CACHE_OFF=0`
re-enables vLLM's torch.compile cache, which is off by default because its key ignores edits to
our Python and served a stale graph for a whole boot). Each knob that is not optional:

| setting | why |
|---|---|
| `--tensor-parallel-size 4 --enable-expert-parallel` | **EP is mandatory**: `moe_intermediate_size` 640 / TP4 = 160 is not divisible by the int4 group size 128, so TP-sharding the experts would straddle scale groups (CompressedTensors refuses). EP shards whole experts (128 per rank). |
| `--dtype float16` | no native bf16 on gfx1030 (patch 0009) |
| `--language-model-only --skip-mm-profiling` | otherwise the vision tower's SDPA warm-up tries a 256 GiB allocation |
| `VLLM_PLE_CPU_OFFLOAD=1 VLLM_PLE_QUANT_DIR=/ples_int4` | n-gram layer runs in a CPU worker process, gathering from the int4 sidecar; GPUs hold only the 73 GB backbone (18.3 GiB/card) |
| `GPUUTIL=0.86` with MTP | leaves room for the drafter + graphs; KV ends up ~3.6 GiB/card (~370 k tokens without MTP at 0.90) |
| `--max-num-batched-tokens 2048`, `HSA_NO_SCRATCH_RECLAIM=1`, `NCCL_P2P_LEVEL=PXB`, kernel line | the T41 stability stack for flat TP=4 |
| CUDA graphs on (no `EAGER`) | worth 3.0× on this model: 48 layers of small kernels; eager mode is launch-bound |
| `MTP=3` | 2.4–3.1 accepted tokens per step; 1.9× |
| `VLLM_ROCM_USE_AITER=0 TORCH_BLAS_PREFER_HIPBLASLT=0` | neither exists for gfx1030 |
| `DENSE_INT8=1` (→ `VLLM_RDNA_DENSE_INT8=1`) | int8 weight-only shadows of every dense fp16 projection for decode; +1 GB/card (T45) |
| one-shot all-reduce on by default (`VLLM_RDNA_AR=0` disables) | RCCL costs 156 µs per 20 KB collective on 4 PCIe cards; ours 33 µs (T44) |
| `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` | the FA-derived backends run their Triton path |

Boot takes ~15 min (weights 2.5 min, torch.compile ~80 s, graph capture, MTP head compile);
`/health` returns 200 at "Application startup complete". The first request at a new context
length is slow (PLE pages cold: prefill 100 t/s vs 570–750 t/s warm).

### 4.1 Validate and measure

Greedy sanity (chat completions, `enable_thinking: false`, temperature 0): `17 * 23` → `391`,
capital of Australia → `Canberra`, first five primes → `2, 3, 5, 7, 11`.

The decode yardstick used for every number in this file (`bench.py`, scratch; reproduce it):
stream one completion with `stream_options.include_usage`, record wall time from the first to
the last content chunk, tokens from the final `usage.completion_tokens`; report
`(tokens − 1) / (t_last − t_first)`. Prefill is excluded by construction. Prompt: the 40-word
"explain how a hash table handles collisions" prompt, 256 tokens, three runs; a 1024-token run
as a long-generation control. For context sweeps, generate **unique** pseudo-random prose of
the target length (prefix caching is on; repeated prompts flatter prefill 10×).

```
warm-up (64 tok)          — first request after boot pays cold PLE pages
bench 3×256               — 76.9 / 73.4 / 74.4 t/s   (2.8–2.9 tok/step)
bench 1×1024              — 74.2 t/s
context 2.9k / 10.6k / 33k — 60–63 / 67–81 / 67 t/s (acceptance-dependent; no context slope)
```

Check card health after every run: `journalctl -k` for `amdgpu` faults/resets/"Runlist
oversubscribed" (host journal is local time, container logs are UTC — convert before
attributing), `rocm-smi --showtemp --showclocks`, fan controller alive.

---

## 5. Profiling inside the server (the method that found the 2×)

1. Boot with `PROFILE=1` (adds `--profiler-config.profiler=torch --profiler-config.torch_profiler_dir=/traces`).
2. `curl -X POST localhost:8000/start_profile`, run one 128-token bench, `curl -X POST localhost:8000/stop_profile`.
3. One `*.pt.trace.json.gz` per rank lands in `logs/traces/` (~25 MB each; ignored by git) plus
   `profiler_out_N.txt` summary tables.
4. Aggregate rank 0 by kernel family (`trace_agg.py`, scratch — reproduce it): load
   `traceEvents`, keep `cat == "kernel"`, sum `dur` by name, classify by name prefix
   (`ncclDevKernel*` → NCCL, `Cijk_*` → rocBLAS Tensile, `fused_moe_kernel*` → Triton MoE,
   `gemv_f16_rdna2*`/`moe_w13*`/`moe_w2*` → ours, `elementwise*`/`index_*`/`triton_poi*` → glue),
   and compute GPU busy as the union of kernel intervals over the wall span.

Kernels inside CUDA graphs carry no CPU-op linkage, so attribute by *count and duration
signature* (e.g. n = 4558 = once per hyper-connection module per step) and by standalone
timing of candidate shapes — not by guessing. `record_shapes` is off in vLLM's profiler config.

What the first trace said (42 decode steps, ~78 ms/step): rocBLAS fp16 GEMMs **60 %**, NCCL
20 %, Triton WNA16 MoE 10 %, glue 5 %, attention+GDN 1.5 %; GPU busy 82 %; **zero** skinny GEMV
kernels. That last fact is the whole diagnosis.

---

## 6. T43 change 1: `gemv_f16_rdna2` — fp16 skinny GEMM for M ≤ 8

### 6.1 Why

`vllm/model_executor/layers/utils.py::rocm_unquantized_gemm_impl` routes decode-shaped fp16
GEMMs to `wvSplitK`/`LLMM1` only when `on_gfx9() or on_gfx1x()`; `_ON_GFX1X` is defined as
gfx11|gfx12. On gfx1030 the gate is false and every dense projection at M ≤ 5 runs
`torch.nn.functional.linear` → rocBLAS Tensile tiles (`Cijk_*MT32x32x8`, `MT128x256x16` with a
GSU post-reduce for K = 10240) at ~35 % of bandwidth. Widening upstream's
`__HIP__GFX1X__` macro to `__GFX10__` builds and runs `wvSplitK_hf_sml_` but returns **wrong
results** (relerr 0.6–0.9 on every shape) — do not do that. The int4 sibling file
(`skinny_gemms_int4.cu`, T35) is gfx1030-clean, so the new kernel lives there.

### 6.2 The kernel (appended to `csrc/rocm/skinny_gemms_int4.cu`)

Design: one wave (32 lanes) per output row `n`; lanes stride K in 16-byte (`uint4`, 8 half)
loads with a 4-deep unroll so four weight loads are in flight per lane; activations for all
M tokens are read from global memory (L1/L2-resident, ≤ 8 × 20 KB); `v_dot2_f32_f16`
accumulates in fp32; `__shfl_xor` reduces across the wave; lane 0 writes M outputs. WAVES per
block is chosen from N so small-N shapes still spread over all 72 CUs.

```cpp
template <int WAVES, int MT>
__global__ void __launch_bounds__(WAVES * 32)
gemv_f16_rdna2_(const half* __restrict__ x, const half* __restrict__ w,
                const half* __restrict__ bias, half* __restrict__ y, const int N, const int K) {
  const int wave = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int n = blockIdx.x * WAVES + wave;
  if (n >= N) return;
  const int K8 = K / 8;
  const uint4* __restrict__ wrow = reinterpret_cast<const uint4*>(w + (size_t)n * K);
  const uint4* __restrict__ xr = reinterpret_cast<const uint4*>(x);
  float acc[MT];
#pragma unroll
  for (int m = 0; m < MT; m++) acc[m] = 0.f;
  for (int i = lane; i < K8; i += 32 * 4) {
    uint4 wq[4];
#pragma unroll
    for (int u = 0; u < 4; u++) { const int idx = i + 32 * u; wq[u] = (idx < K8) ? wrow[idx] : make_uint4(0, 0, 0, 0); }
#pragma unroll
    for (int u = 0; u < 4; u++) {
      const int idx = i + 32 * u;
      if (idx < K8) {
        const half2* wh = reinterpret_cast<const half2*>(&wq[u]);
#pragma unroll
        for (int m = 0; m < MT; m++) {
          const uint4 xq = xr[(size_t)m * K8 + idx];
          const half2* xh = reinterpret_cast<const half2*>(&xq);
          float a = acc[m];
          a = __builtin_amdgcn_fdot2(wh[0], xh[0], a, false);
          a = __builtin_amdgcn_fdot2(wh[1], xh[1], a, false);
          a = __builtin_amdgcn_fdot2(wh[2], xh[2], a, false);
          a = __builtin_amdgcn_fdot2(wh[3], xh[3], a, false);
          acc[m] = a;
        }
      }
    }
  }
#pragma unroll
  for (int m = 0; m < MT; m++) {
#pragma unroll
    for (int off = 16; off >= 1; off >>= 1) acc[m] += __shfl_xor(acc[m], off);
  }
  if (lane == 0) {
    const float b = bias ? __half2float(bias[n]) : 0.f;
#pragma unroll
    for (int m = 0; m < MT; m++) y[(size_t)m * N + n] = __float2half(acc[m] + b);
  }
}
// host: WAVES = 8 if N >= 1152, 4 if N >= 576, 2 if N >= 288, else 1; MT = M in 1..8 (switch);
// at::Tensor gemv_f16_rdna2(x[M,K] fp16 contiguous, w[N,K] fp16 contiguous, optional bias[N]) -> y[M,N]
// TORCH_CHECKs: 1 <= M <= 8, K % 8 == 0, fp16, contiguous.
```

Binding: `csrc/rocm/ops.h` declaration; `csrc/rocm/torch_bindings.cpp`
`rocm_ops.def("gemv_f16_rdna2(Tensor x, Tensor w, Tensor? bias) -> Tensor"); rocm_ops.impl(..., torch::kCUDA, &gemv_f16_rdna2);`;
`vllm/_custom_ops.py` wrapper `ops.gemv_f16_rdna2(x, w, bias)`.

### 6.3 The route (`rocm_unquantized_gemm_impl`, ahead of the upstream skinny gate)

```python
if (envs.VLLM_ROCM_USE_SKINNY_GEMM and on_gfx10x()
        and x.dtype == torch.float16 and weight.dtype == torch.float16
        and 0 < n <= 8 and k % 8 == 0
        and weight.is_contiguous() and (bias is None or bias.is_contiguous())):
    x_view = x.reshape(-1, k).contiguous()
    out = ops.gemv_f16_rdna2(x_view, weight, bias)
    return out.reshape(*x.shape[:-1], m)
```

(`n` = tokens, `m` = output rows, `k` = K, as in the surrounding code.) Every vLLM linear —
`ColumnParallelLinear`, `RowParallelLinear`, `ReplicatedLinear`, `ParallelLMHead` — reaches this
through `UnquantizedLinearMethod.apply → dispatch_unquantized_gemm()`, including the AMD-path
hyper-connections (which use vLLM linears, unlike the NVIDIA path's raw `F.linear`).

### 6.4 Validation and numbers

Standalone harness (hipcc, `--offload-arch=gfx1030`, CPU fp32 reference, all 12 dense shapes,
M ∈ {1, 4, 8}): ALL PASS, relerr ≈ 3e-4. Through the production route on a free card:

| shape (per rank) | rocBLAS | ours (M = 4) |
|---|---|---|
| gdn.in_proj_qkv [2560×2560] | 62 µs | 15.6 µs |
| gdn.in_proj_z [1536×2560] | 138 µs | 11.8 µs |
| qsa.q_proj [3072×2560] | 77 µs | 24.5 µs |
| router.gate [512×2560] | 88 µs | 6.2 µs |
| hc.down [320×10240] / hc.up [10240×320] / hc.inject [4×10240] | 110 / 100 / 79 µs | 24.5 / 15.7 / 9.2 µs |
| lm_head [62080×2560] | 4228 µs (75 GB/s) | 1276 µs (249 GB/s) |
| **sum of 12** | **5129 µs** | **1432 µs** |

**Warm the card before standalone timing**: on an idle GPU the first pass read 5–10× slow on
some shapes (clock ramp), which looks exactly like a bad kernel.

---

## 7. T43 change 2: the int4 MoE GEMV under expert parallelism

`TritonWNA16Experts.apply` (and `TritonExperts.apply` — **both** bodies carry the hook; a
previous campaign lost half a day to editing only one) calls `ops.moe_skinny_int4_decode` for
M ≤ 8 fp16 decode when `VLLM_ROCM_MOE_SKINNY` (default "1") and the op exists. The hook was
gated `expert_map is None`; EP always passes an `expert_map` (global → local id, −1 for
experts on other ranks), so the kernel had never run on this model.

Change (`vllm/model_executor/layers/fused_moe/experts/triton_moe.py`): drop that condition;
before the call, `if expert_map is not None: topk_ids = expert_map[topk_ids]`.
Kernel (`csrc/rocm/skinny_gemms_int4.cu`): in `moe_w13_silu_gemv_`, after reading `expert`,
`if (expert < 0) { if (lane == 0) act[...] = 0; return; }`; in `moe_w2_gemv_`,
`if (expert < 0) continue;`. Each rank thus produces its partial over local experts (zeros for
tokens with none); the layer's existing all-reduce sums the ranks. Harness: PASS at M = 1/4/8
including a token with no local expert; 98 µs per w13+w2 pair in the serving trace vs 2 × 76.5 µs
for the Triton path.

---

## 8. T44–T46: all-reduce, int8 shadows, dispatch fusion (what took 74 → 100 t/s)

### 8.1 One-shot P2P all-reduce (`csrc/rocm/rdna_allreduce.{cuh,cu}`, `rdna_all_reduce.py`)

Post-T43 the trace was 41 % RCCL: 119 collectives per step, 156 µs each, ~20 KB messages.
`NCCL_P2P_LEVEL=SYS` did nothing; vLLM's custom all-reduce is gated off on RDNA and for
>2 PCIe GPUs. The kernel: each rank pushes its contribution into every peer's **uncached**
staging buffer (`hipExtMallocWithFlags(..., hipDeviceMallocUncached)`, IPC-mapped), signals a
**host-coherent** flag (shm page + `hipHostRegister`), waits for W−1 peer flags, then reduces
in **fixed rank order** in fp32 (bit-identical across ranks). The sequence number is read from
a device counter, never passed as an argument (frozen at graph capture). 16 blocks for 20 KB.
Measured on the four serving cards: 33 µs (20 KB), 15 µs (5 KB). Wired into
`CudaCommunicator.all_reduce` ahead of every other path, one instance per process group
(handles), init made collective-safe (every rank runs every barrier; failures agreed via
all_gather). Fast path ≤ 64 KB; prefill chunks stay on RCCL.

Two traps: (1) vLLM constructs several `GroupCoordinator`s over the same ranks — a singleton
extension state deadlocked ranks 1–3 (only one card loaded weights); (2)
`hipErrorPeerAccessAlreadyEnabled` is sticky and surfaces at torch's next launch check —
clear with `hipGetLastError()`.

### 8.2 int8 shadows (`gemv_i8_rdna2`, `rdna_dense_int8.py`)

Per-output-channel symmetric int8 of every fp16 `weight` built in
`process_weights_after_loading` (fp16 kept for prefill). Same wave-per-row kernel with 16
weights per 16-byte load. Halves the streamed bytes: lm_head 1.27 → 0.64 ms.

### 8.3 The trace-time freeze (read this before adding any decode-only path)

vLLM compiles the model once for a dynamic token range. Any Python `if 0 < n <= 8:` inside the
traced region is decided on the tracing example and baked in — our int8 branch never ran
inside the graph (176 int8 vs 418 fp16 GEMVs per step; only the untraced hyper-connection
linears used it). The decode/prefill choice must live inside an **opaque custom op**
(`direct_register_custom_op` with a fake impl): `rdna_ops.py` provides `rdna_dense_gemm`,
`rdna_hc_mix`, `rdna_shared_expert`, each choosing at runtime on the real batch size.

### 8.4 Dispatch-count fusion

Method: boot with `EAGER=1 PROFILE=1`, profile, and run `trace_attr.py` (kernel → CPU op via
`External id` → innermost `/src/vllm` python frame) to rank launch sites per step. Then, in
order of launches saved: MoE hook glue into the kernel (`expert_map`, id/weight dtypes handled
in-kernel: −144/step); indexer RMSNorm and rope as single `_C.rms_norm` / `_C.rotary_embedding`
launches (−200); the fused Triton `fused_qk_rmsnorm_rope_gate` enabled on ROCm
(`num_stages=1`); `rdna_fused_glue.cu` — hyper-connection down+inject GEMV with silu epilogue,
up GEMV + sigmoid + gated mean in one kernel (5 → 2 launches per hc), shared expert
gate_up+silu·mul and down×sigmoid(gate) (6 → 2). Kernels/step 2,931 → 1,788.

hipify traps: a launch after `else` inside a macro body becomes `elsehipLaunchKernelGGL`
(keep launches in plain template functions); local classes cannot hold member templates.

### 8.5 Where the step goes now (boot 9b, decode only, 45 steps)

30.6 ms/step wall (28.8 in the un-profiled run), kernel-sum 21.7 ms, GPU busy 71 %, 1,788
kernels: all-reduce 4.9 ms (95 × 52 µs — the kernel is 33; the rest is rank skew), MoE int4
GEMV 4.3, int8 GEMV 3.5, glue 2.5 (754 kernels: 214 `elementwise<128,8>`, 108 D2D copies),
fused hc 2.4, attention 0.9, all-gather 0.8, fused shared expert 0.7, MTP-head unquantized
MoE 0.6. Idle ≈ 9 ms of ~4 µs bubbles. Untried: GDN split-output GEMV (kills ~250 copies/step),
a fatter MoE GEMV (its per-block LDS staging of x is the suspect; a 4-deep unroll did not
help), an fp16 expert GEMV for the MTP head, the rank skew.

## 10. What a reproducer actually needs (inventory)

| need | where it is | reproducible? |
|---|---|---|
| 4× gfx1030 cards on one host, kernel line from §1, fans, ≥96 GB RAM, ~110 GB disk for the model | you | hardware |
| **Foundation image** `vllm-gfx1030:latest` (ROCm 7.2.3, PyTorch 2.11.0+gitd0c8b1f built for gfx1030, Python 3.12) | built by hand in July 2026 from vLLM's `docker/Dockerfile.rocm_base` + the one-line arch patch (0005/0008); the public recipe repo's `02-VERSIONS.md` says so and warns nobody has rebuilt it end to end | **the weakest link** — hours; verify `torch.cuda.get_arch_list()` includes gfx1030 with devices attached |
| Phase images v1…v7 (`docs/Dockerfile.phase1..7`) | this repo | yes, in order; v7 = stock AMD runtime |
| `vllm-gfx1030:qwen4exp` = v7 + `pip install -e /src` (editable, ~10 min) committed as an image | `docker commit` of a v7 container after the install (one 450 MB layer) | yes: run the install in a v7 container and commit, or just install at first use |
| vLLM source: `peakcrosser7/vllm` `release/qwen38next` @ `2a4cd64` | GitHub | yes (shallow clone) |
| **22 patches** `qwen4exp/patches/0001–0022` | this repo | `git am` in order |
| Backbone `wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16` shards 2–5 + `model_mtp.safetensors` (73 GB) | Hugging Face | download; skip shard 1 |
| n-gram sidecar `primitive-ai/Qwen3.8-Flash-Next-PLE-quant` → `ples_int4/` (30 GB, 128 shards + META.json) | Hugging Face | download as-is — **no conversion script**; patch 0016 reads it directly |
| Launcher `qwen4exp-pleq/serve-pleq.sh` | this repo | edit the four absolute `/home/perfekt/repos/vllm-rdna2/...` mounts and the `--group-add 991 --group-add 44` ids (= `render`, `video` on this host) |
| Validation + measurement scripts | `qwen4exp/tools/` (copied from the campaign's scratch dir) | yes |
| The `build_rocm/` CMake tree for `_rocm_C` rebuilds | regenerated by the configure line in §2.4 | yes |
| Nothing else: no weight forking, no offline tuning tables, TunableOp off, no HF hub access at serve time | | |

`qwen4exp-pleq/ple_layer_quant.py` and `worker_image_quant.py` are reference copies of two
patched vLLM files from the sidecar bring-up; they are not part of the build.

## 9. Lessons that cost time (do not repeat)

1. **Profile in the serving process before theorizing.** T42 killed "disk paging is the
   bottleneck" (int4 sidecar: <4 %) and "GPUs at 38–99 % means compute-bound" (eager launch
   overhead; graphs alone gave 3×). T24 (an earlier model) sized a collective against an
   out-of-server microbenchmark that overstated it 4×. T43 found the real 60 % in one trace.
2. **Absence in the trace is evidence.** Zero `wvSplitK` kernels meant the skinny route never
   ran; no amount of tuning the Triton MoE would have found it.
3. **Upstream "RDNA" code paths mean gfx11/12.** `on_gfx1x()` excludes gfx1030; `__HIP__GFX1X__`
   kernels built for gfx10 can miscompute silently. Validate numerically against `F.linear`.
4. **Two `apply()` bodies** (`TritonExperts`, `TritonWNA16Experts`) — hook both.
5. **`docker rm -f` can report success with the container still up** — verify with `docker ps`.
6. **Timestamps**: host journal is local time (MDT), containers log UTC. A "no faults during the
   benchmark" claim was once wrong by exactly six hours.
7. **Side jobs go on ROCR device 0** (PCI 0A). ROCR 1 is a serving card; a harness there put
   "Runlist is getting oversubscribed" in the kernel log during graph capture.
8. **Clock ramp** on an idle card fakes slow kernels; **Infinity Cache** (128 MB) fakes fast ones
   for weights ≤ 16 MB in a timing loop — the DRAM-streaming number is lm_head's.
9. **Prefix caching on + repeated prompt** = 10× flattering prefill. Sweep with unique prompts.
10. Stop containers with `docker stop`; rebuild extensions in a separate container; install
    `.so` files by rename.
11. **torch.compile freezes Python-level decode/prefill branches.** Put the choice inside an
    opaque custom op. Profile per-kernel *counts* after every boot; a lever that "did nothing"
    usually did not run.
12. `sleep`-based wait loops in this environment: a 17-minute boot with the compile cache off
    outlasts a 10-minute tool budget — poll in two calls rather than assuming a hang.
