# What this fork changes, and why

This fork carries everything needed to serve **Qwen3.8-Flash-Next** on **four AMD Radeon PRO
V620 (Navi 21, gfx1030)** cards under vLLM at ~100 tokens/s single-stream decode, on a stock
[TheRock](https://github.com/ROCm/TheRock) ROCm 7.14 install. Upstream vLLM does neither: it
does not know gfx1030, the Flash-Next model family (vLLM PR #53896) is unmerged, and the
model's 51-billion-row n-gram embedding table does not fit on any consumer GPU.

Every change is on the `rdna2/qwen38-flash-next` branch as a separate commit; the commit
subjects carry the original patch numbers (`port: 000N …`, `T43 …`) so they can be matched to
the experiment log in `RESULTS.md`. Nothing here is a config-only tweak: the numbers came
from profiling inside the serving process and writing kernels for what the profile showed.

**2026-08-31 — container image.** `containers/` builds the fork as a Docker image in two layers:
a base (TheRock ROCm 7.14 from the public *legacy* multi-arch tarball index — 7.14.1 by default, the
host-exact 7.14.0rc3 candidate as an alternative pin, both sha256-pinned — plus torch 2.12 / Triton
3.7 / torchvision compiled for gfx1030 by `tools/rdna2/build-torch-rocm714.sh`) and a runtime image
(this tree built per README §4, dependency pins captured from the validated venv, RDNA-op and
PLE-offload verification at build time). The entrypoint is the serve script; weights mount at
`/models`. TheRock publishes no gfx103X torch wheels, and the 7.14 line has moved off AMD's current
channels — the published base image is the durable artifact. See containers/README.md.

## How to see exactly what this fork changed (on GitHub)

The branch has three layers, and GitHub's compare view can show each:

1. **Everything vs upstream vLLM** — [`main...rdna2/qwen38-flash-next`](https://github.com/leapdragon/vllm-rdna2-qwen/compare/main...rdna2/qwen38-flash-next):
   the Flash-Next model branch (vLLM PR #53896, not ours) *plus* this fork's work. Large.
2. **Only this fork's work** — [`2a46f85b43...rdna2/qwen38-flash-next`](https://github.com/leapdragon/vllm-rdna2-qwen/compare/2a46f85b43...rdna2/qwen38-flash-next):
   `2a46f85b43` is the merge commit that brought the Flash-Next branch in; every commit after it
   is ours (the 22 ported patches, the env declarations, `docs/rdna2/`, `tools/rdna2/`).
   The **Commits** tab of that compare lists them with their original subjects
   (`port: 000N …`, `rdna2: …`, `PLE offload: …`, `T43 …`, `T44 …`, `T45/T46 …`); the
   **Files changed** tab is the whole diff.
3. **One change at a time** — the commit list
   [`commits/rdna2/qwen38-flash-next`](https://github.com/leapdragon/vllm-rdna2-qwen/commits/rdna2/qwen38-flash-next);
   click a commit to see its diff. The sections above are in that order.

Locally: `git log --first-parent 2a46f85b43..rdna2/qwen38-flash-next` and
`git diff --stat 2a46f85b43 rdna2/qwen38-flash-next`.

Where our code lives (new or modified files, by area):

| area | files |
|---|---|
| `csrc/rocm/` | `rdna_allreduce.cuh/.cu` (one-shot all-reduce), `rdna_fused_glue.cu` (fused hc / shared-expert kernels), `skinny_gemms_int4.cu` (int4 MoE GEMV, `gemv_f16_rdna2`, `gemv_i8_rdna2`), `ops.h`, `torch_bindings.cpp` |
| `vllm/model_executor/layers/` | `rdna_ops.py` (runtime-dispatch custom ops), `rdna_dense_int8.py` (int8 shadows), `utils.py` (the GEMM route), `linear.py` / `vocab_parallel_embedding.py` (hooks), `fused_qk_norm_rope.py`, `fused_moe/experts/triton_moe.py` (MoE hook), `ple_offload_layer.py` |
| `vllm/distributed/device_communicators/` | `rdna_all_reduce.py`, `cuda_communicator.py` (hook) |
| `vllm/v1/ple_offload/` | the PLE offload worker/connector, `hip_driver.py` (HIP shim) |
| `vllm/models/qwen4_exp/amd/` | `ple_layer.py` (offload + sidecar), `hyperconnection.py`, `qsa.py`, `indexer_qsa.py`, fp16 enablement |
| `vllm/platforms/rocm.py`, `vllm/envs.py` | `on_gfx10x()`, amdsmi device filtering, the fork's environment variables |
| `docs/rdna2/`, `tools/rdna2/` | this documentation and the build/serve/measure tools |

## 0. The base

Validated twice: in the container environment the numbers in `RESULTS.md` were measured in
(ROCm 7.2.3), and on the host against TheRock ROCm 7.14 (same source, PyTorch built from
source; `RESULTS.md` last section) — 98 / 105 / 96 t/s.

`main` of this fork = upstream `vllm-project/vllm` main (post-0.28.0, `6cddad414`) merged with
`peakcrosser7/vllm` `release/qwen38next` (the Flash-Next PR branch, head `91a6b555d`). The
merge was clean. Our commits sit on top. The Flash-Next AMD code path (`vllm/models/qwen4_exp/amd/`)
is the one that runs on gfx1030 (`on_gfx10x()` selects it).

## 1. Platform: teaching vLLM about gfx1030 (patches 0001–0008)

| commit | change | why |
|---|---|---|
| `port: 0001-gfx10x-platform-support` | `on_gfx10x()`; amdsmi lookups filtered to compute-capable devices; an RDNA opt-in for the custom all-reduce | vLLM's ROCm platform layer only knows gfx9/gfx11/gfx12; amdsmi enumerates *physical* devices and ignores `ROCR_VISIBLE_DEVICES`, so a display card first on the bus shifts every device lookup onto the wrong card (`TROUBLESHOOTING.md` §2) |
| `port: 0002-lds-tile-headdim256` | attention tile sizes for head_dim 256 under a 64 KiB LDS | gfx1030 has 64 KiB of LDS, not 160 |
| `port: 0003-softmax-segments` | segmented softmax in the prefill attention kernel | same LDS cap |
| `port: 0004-w4-blocking-256` | int4 kernel blocking | tile shapes that fit the chip |
| `port: 0007-moe-wna16-gfx1030` | Triton WNA16 MoE config for gfx1030 | the default config targets CDNA |
| `port: 0008-moe-skinny-gemv-gfx1030` | `moe_skinny_int4_decode`: wave-per-output-row int4 expert GEMV pair (`csrc/rocm/skinny_gemms_int4.cu`) | vLLM's tiled MoE kernels read *across* K and ran 10× off the bandwidth ceiling at decode batch sizes; this kernel streams rows at ~85 % of it |
| `port: 0006-rdna-hybrid-w4a16-gfx1030` | int4 linear path for gfx1030 (`wvSplitK_int4_*`) | asymmetric-uint4 checkpoints needed a kernel path that exists on this ISA |
| `port: 0005-base-image-gfx1030-arch` | `gfx1030` in `PYTORCH_ROCM_ARCH` of `docker/Dockerfile.rocm_base` | upstream now includes gfx1030 there; kept for the record |

## 2. The Flash-Next AMD path on gfx1030 (patches 0009–0010)

- **fp16 alongside bf16** (`rdna2: allow FP16 alongside BF16 in the qwen4_exp AMD path`): the AMD
  path hard-coded bf16 in 12 places. gfx1030 has no native bf16 arithmetic; measured on a V620,
  the QSA (sparse attention) kernel is faster *and* ~7× more accurate in fp16 against an fp32
  reference at head_dim 256. Serve with `--dtype float16`.
- **FlashAttention-derived backends on ROCm** (`rdna2: let ROCm run FlashAttention-derived
  backends that compute elsewhere`): the QSA backend asserted "requires FlashAttention" although
  it computes in Triton and never calls `flash_attn`; and `reshape_and_cache_flash` was import-gated
  behind CUDA although vLLM's own C++ op provides it on ROCm.

## 3. The n-gram table: PLE offload and the int4 sidecar (patches 0011–0017)

The model's position-learning-enhancement layer holds a 320,001,536 × 160 n-gram embedding
table (102 GB in bf16) in **one** layer. It cannot live on 32 GB cards. Upstream vLLM has a
CPU-offload design for it (PRs #53899 / #54070) that was CUDA-only. This fork:

- brings that infrastructure in (`vllm/v1/ple_offload/`), wired to the AMD path;
- replaces its four `cuda-python` driver calls with a ctypes shim over `libamdhip64`
  (`hipStreamWriteValue32` / `hipStreamWaitValue32` / `hipHostRegister`), a hardware stream wait,
  not CPU polling;
- makes the AMD PLE layer materialise its meta-tensor workspace, load weights when offloaded
  (three bugs only real weights expose: a `copy_` onto a meta buffer is a silent no-op; the
  offload guard was missing from `load_weights`; a meta parameter needs binding, not `.data`
  assignment), and bypass a CUDA-only custom op in the CPU worker;
- serves the table from a **quantized sidecar** (`VLLM_PLE_QUANT_DIR`): a 30 GB int4 copy of
  the table (128 safetensors shards + `META.json`, layout `group16_int4_fp16scale_lownibblefirst`)
  memory-mapped in the CPU worker and dequantised on gather. Measured: throughput identical to
  the bf16 table (<4 %); the value is 30 GB of page cache instead of 103 GB of RAM or disk paging.

The driver selection is platform-aware, not import-aware: a full venv has `cuda-bindings`
installed as a transitive dependency, and "try cuda-python, fall back to the HIP shim" then
picks cuda-python on ROCm and dies on `dlopen libcuda.so.1` at the first host registration
(found on the first host build). On ROCm (`torch.version.hip`) the shim is always used.

Only the AMD PLE layer subclasses `PleOffloadLayer`; the NVIDIA layer stays as upstream
(their later refactor and piecewise-graph n-gram-id fix are in this tree; the offload hooks
were not re-ported to it).

## 4. Serving constraints that are not optional

- **Expert parallelism** (`--enable-expert-parallel`): `moe_intermediate_size` 640 / TP4 = 160 is
  not divisible by the checkpoint's int4 group size 128; TP-sharding the experts would straddle
  scale groups and compressed-tensors refuses. EP shards whole experts (128 per rank).
- `--language-model-only --skip-mm-profiling`: the vision tower's SDPA warm-up otherwise attempts
  a 256 GiB allocation.
- CUDA graphs on (3.0× on this model — 48 layers of small kernels are launch-bound in eager
  mode) and MTP speculative decoding (`num_speculative_tokens` 3: 2.7–3.1 accepted tokens per
  step, 1.9×).
- The TP=4 stability stack (kernel line `amdgpu.pcie_gen_cap=0x00070007 aspm=0 runpm=0
  gpu_recovery=1`, `HSA_NO_SCRATCH_RECLAIM=1`, `NCCL_P2P_LEVEL=PXB`, batched tokens 2048). Without
  it flat TP across four of these cards drops cards off the PCIe bus (`TROUBLESHOOTING.md` §4).

## 5. T43 — the dense projections were on rocBLAS (74 t/s)

The first in-process profile said 60 % of GPU time was rocBLAS Tensile tiles running decode-shaped
fp16 GEMMs at ~35 % of memory bandwidth. Only the routed experts of this checkpoint are int4;
the GDN/QSA projections, router, shared expert, both hyper-connections and lm_head are fp16 —
~2.1 GB per rank streamed every forward. vLLM's skinny-GEMV route is gated
`on_gfx9() or on_gfx1x()`, and `on_gfx1x()` means gfx11/12. Widening upstream's macro to gfx10
builds and launches `wvSplitK` but returns wrong results (relerr 0.6–0.9), so:

- **`gemv_f16_rdna2`** (`csrc/rocm/skinny_gemms_int4.cu`): wave-per-output-row fp16 GEMM for
  M ≤ 8 tokens, 16-byte loads, 4-deep unroll, `v_dot2_f32_f16`, wave32 shuffle reduce. relerr
  ~3e-4 on all 12 dense shapes; 1.43 ms vs 5.13 ms rocBLAS for the set. Routed in
  `rocm_unquantized_gemm_impl` ahead of the upstream gate.
- **EP-aware `moe_skinny_int4_decode`**: the hook rejected `expert_map`, which expert
  parallelism always sets, so the T38 kernel had never run on this model. The kernel now takes
  the ids as produced and applies `expert_map` in-kernel, skipping non-resident experts.

## 6. T44 — a one-shot P2P all-reduce (82 t/s)

Post-T43, RCCL was 41 % of GPU time: 119 latency-bound collectives per step at 156 µs for
~20 KB. vLLM's custom all-reduce is unavailable on this platform twice over (RDNA gate; the
XGMI "fully connected" check refuses four PCIe GPUs). `NCCL_P2P_LEVEL=SYS` changed nothing.

**`rdna_ar_oneshot`** (`csrc/rocm/rdna_allreduce.{cuh,cu}`,
`vllm/distributed/device_communicators/rdna_all_reduce.py`): each rank pushes its contribution
into every peer's *uncached* staging buffer (IPC-mapped; coarse-grained memory is invisible to a
peer mid-kernel on this platform), signals a *host-coherent* flag (device-memory flags cannot be
polled across PCIe), waits for the peers, and reduces in fixed rank order in fp32 so all ranks
are bit-identical. The sequence number is read from a device counter, never passed as a kernel
argument — arguments are frozen at CUDA-graph capture. Bounded spins abort instead of hanging
the GPU. 33 µs per 20 KB message on the four cards. Hooked into `CudaCommunicator.all_reduce`
ahead of every other path, one instance per process group (vLLM builds several
`GroupCoordinator`s over the same ranks — a singleton deadlocked three of four ranks), fast
path ≤ 64 KB (prefill chunks are faster on RCCL). `VLLM_RDNA_AR=0` disables.

**Fabric-friendliness knobs (2026-09-01).** The pushes now go out in a rank-staggered peer order —
at any instant each destination GPU is written by one source instead of all W−1 at once (a pure
reorder: 30.9 µs/op vs 33 before) — and two environment knobs bound the burst into the receiving
GPU's root complex, where other devices' DMA completions queue behind it: `VLLM_RDNA_AR_BLOCKS`
caps the blocks per launch (4 → 42 µs/op) and `VLLM_RDNA_AR_PACE` (0..127) idles each wave ~64
clocks per unit between strided stores (4 blocks + pace 16 → 45 µs/op). At ~95 collectives per
step that is ≈ +1.0 / +1.4 ms per ~39 ms step (2.5–3.5 %) — versus RCCL's ~156 µs/op (+11 ms).
Defaults are unchanged (auto blocks, pace 0); all variants pass `tools/rdna2/ar_ops_test.py`
(eager + graph replay, bit-identical across ranks). Motivation and what the knobs do NOT touch:
the one-shot path only carries decode-size messages (≤ 64 KB); prefill collectives are RCCL.

## 7. T45 — int8 shadows of the dense projections

`gemv_i8_rdna2` + `vllm/model_executor/layers/rdna_dense_int8.py`: per-output-channel symmetric
int8 copies of every dense fp16 weight, built in `process_weights_after_loading`, used for the
decode GEMV only (fp16 kept for prefill). Halves the streamed bytes; +1 GB per card. Enabled with
`VLLM_RDNA_DENSE_INT8=1`. Output validated with greedy checks; the MTP acceptance rate moves by
about ±0.1 tokens/step.

## 8. T46 — dispatch count (98–101 t/s)

The decode step was ~2,700 kernels with ~4 µs of bubble between each — one third of the step.
An eager-mode profile attributed to source lines (`tools/rdna2/trace_attr.py`) ranked the
launch sites; then, in order of launches saved:

- MoE hook glue moved in-kernel (id dtype, weight dtype, `expert_map`): −144/step.
- Indexer RMSNorm and rope as single `_C.rms_norm` / `_C.rotary_embedding` launches instead of
  ~7 native kernels each: −200/step.
- The fused Triton qk-norm+rope+gate kernel (`fused_qk_norm_rope.py`) enabled on ROCm — it has
  no CUDA-only pieces; `num_stages=1` on gfx1030 (the default halves occupancy on this chip).
- **`csrc/rocm/rdna_fused_glue.cu`** (fp16 or int8 weights): hyper-connection down+inject GEMV
  with the silu epilogue and the up GEMV + sigmoid + gated mean in one kernel (5 → 2 launches per
  hyper-connection); shared expert gate_up+silu·mul and down×sigmoid(gate) (6 → 2).

### The trace-time freeze (important if you add anything)

vLLM compiles the model once for a dynamic token range. Any Python `if 0 < n <= 8:` inside the
traced region is decided on the tracing example and baked into the graph — the int8 path never
ran inside the compiled graph for a whole boot. Every decode/prefill choice therefore lives inside
an **opaque custom op** with a fake impl (`vllm/model_executor/layers/rdna_ops.py`:
`rdna_dense_gemm`, `rdna_hc_mix`, `rdna_shared_expert`).

**The torch.compile cache and these ops.** The cache key (`vllm/compilation/backends.py`) hashes
every declared `VLLM_*` env var, the vLLM config, and the *contents of the Python files Dynamo
traced*; a hit is then reused with Dynamo guards disabled. It does **not** see the bodies,
schemas or fake impls of opaque ops (Dynamo never enters them), `rdna_dense_int8.py`'s
per-layer eligibility, or the `.so` — change any of those and clear
`~/.cache/vllm/torch_compile_cache/` or boot once with `COMPILE_CACHE_OFF=1`. Two facts learned
the hard way (2026-08-30): `VLLM_DISABLE_COMPILE_CACHE=1` disables *writing* as well as reading
(so a boot with it set saves nothing for the next one; the per-boot hash directory is still
created, holding only `computation_graph.py` dumps), and a cached graph is executed before any
forward pass has run the lazy `import rdna_ops` at the call sites — so the ops are now
registered eagerly at import time of `rdna_dense_int8.py` and the model's `hyperconnection.py`
(a warm boot otherwise dies with `'_OpNamespace' 'vllm' object has no attribute 'rdna_hc_mix'`).
The serve script keeps the cache off by default (`COMPILE_CACHE_OFF=1`) as the safe setting for
anyone editing the fork; with it on, a boot whose key matches skips the ~700 s of Inductor work.

## 8a. The n-gram (PLE) offload handshake and the all-reduce barrier — 2026-08-30

Two ROCm-specific bugs, found while chasing garbled characters (`">>"`, `charsetset`) in
long tool-call generations after MTP was switched off:

**HIP graphs drop `hipStreamWaitValue32`.** vLLM's PLE offload makes the GPU wait for the
CPU n-gram lookup with a stream-memory wait (`ple_offload_wait`, inside the compiled graph).
On ROCm the call is accepted during stream capture but not recorded, so every CUDA-graph
decode step read whatever was in the output buffer — usually the previous step's lookup. The
"Duplicate PLE request … skipping" warnings were the fingerprint (forwards finished without
waiting, the host ran ahead, the CPU worker drained two requests and dropped one). MTP=3 had
masked it by keeping the worker ahead of the GPU through timing luck. Moving the wait outside
the graph is *not* the fix: a pending WAIT_REG_MEM cannot be preempted when KFD evicts queues
(`svm_range_restore`, frequent here), the eviction times out, the driver resets the GPU and
on this machine the reset loses the card from the PCIe bus. The protocol now has **no
GPU-side waits at all** (`vllm/v1/ple_offload/`): the worker processes requests strictly in
order, DMAs the result to every TP worker's buffer, then bumps a shared-memory counter per
worker; each model thread blocks in `prepare_forward` until its counter reaches the launch
number and only then enqueues the forward. The chain is inherently serial (step N+1's lookup
needs step N's token), so nothing is lost by waiting on the host — but the wait is now real:
decode first went from 70–72 t/s (which was the *no-wait* speed) to ~55 t/s, with ~3.3 ms per
step in the worker. Three follow-ups the same day brought it back to **61–65 t/s over 256
tokens, 62 over 1024**: (1) a fused numpy decode path (`_fused_decode_lookup`) — for plain
decode batches the n-gram hashing and the int4 row gather run in numpy straight into the
result buffer (0.05 ms for 16 rows vs 1.6 ms of torch dispatch; bit-identical, and
`PLE_OFFLOAD_FUSED_CHECK=1` verifies every step against `forward_impl`); (2) the sidecar is
prefaulted into the page cache at worker start (32 GB in 30 s, overlapped with weight loading;
`PLE_OFFLOAD_PREFAULT=0` disables) so random rows are minor faults, not 0.4 ms disk reads;
(3) the result no longer crosses processes on the GPU: each TP worker registers a shared
pinned result buffer, the offload worker writes rows + a plain-store sequence number, and each
model thread DMAs the rows to its own device buffer on its model stream. Worker time per
request: 3.33 → 0.88 ms. What remains on the critical path (~2.3 ms of a ~15.4 ms step) is the
D2H of the sampled token, the zmq hop and the lookup itself; the forward is ~13 ms.
Tests: `tools/rdna2/ple_consistency_test.py`
(two identical greedy runs must match, garble scan, no skips, idle GPUs; `--trace` with
`PLE_OFFLOAD_DEBUG_TRACE` on the worker compares the lookups themselves),
`tools/rdna2/ple_coherence_test.py` (cross-process DMA visibility, standalone).

**The one-shot all-reduce's barrier flags were in a host-coherent page.** Every rank polled
system memory over PCIe for the whole barrier (~10 k barriers/s at MTP=0), and — the subtle
part — its payload went to a peer's VRAM while its flag went to host memory: posted PCIe
writes are only ordered per destination, so a rank could see the flag before the last payload
bytes landed and reduce a few stale elements. That was the run-to-run logprob noise
(1e-5…1e-2) that made two greedy runs diverge. Flags now live in a 4 KB page appended to each
rank's *uncached* staging buffer (same IPC handle): a rank announces with one posted P2P
store per peer and polls its own memory with `s_sleep(8)` between polls. Two greedy runs are
byte-identical, waiting generates no PCIe traffic, and a 5-minute soak (`soak_fabric_watch.sh`)
logged no `mpt2sas`/amdgpu events — this box audibly resets its tape drive under fabric
stress, and twice that day cards dropped off the bus during generation.

## 8b. Vision (2026-08-30)

The checkpoint ships a full Qwen3-VL-style vision tower (27 blocks, hidden 1152, patch 16,
2×2 merge; 333 `model.visual.*` tensors, 0.90 GB bf16 in shard 2) which `--language-model-only`
had been skipping. `VISION=1` in the serve script enables it. Facts that matter on gfx1030:

- The ViT attention backend resolves to **Torch SDPA** (no `flash_attn` package, AITER is
  CDNA-only, the Triton-AMD FA subpackage is absent). SDPA's math path materialises the
  N² attention matrix, and the image processor's default ceiling is `longest_edge: 16777216`
  — the startup memory profiler builds a 16 MP dummy image and dies asking for **64 GiB**.
  The serve script therefore caps images at `--mm-processor-kwargs '{"max_pixels": 1638400}'`
  (≈1280×1280 → 6,400 patches → a ~1.3 GB attention matrix). Raise it only with a
  linear-memory ViT backend.
- The tower is **replicated on every TP rank** (not sharded): ~0.9 GB/card plus encoder
  activations; at 196k context the KV pool drops from 541k to 405k tokens (2.06×).
- Verified with `tools/rdna2/vision_test.py` (synthetic images, deterministic answers):
  colour+shape, OCR ("SUNRISE 42" read exactly), counting (5 squares), quadrant colours,
  two-images-which-has-the-triangle, an 1800×1400 star, a 5th image rejected with HTTP 400,
  and text-only decode afterwards. TTFT 1.3–5 s per image prompt (encoder is eager);
  text decode and the PLE consistency test are unaffected.
- **Over-limit conversations are elided, not rejected** (`MM_ELIDE=1`, the default with
  vision on). Stock vLLM 400s a prompt whose accumulated images exceed
  `--limit-mm-per-prompt`; in an agent loop the history only grows, so after the Nth
  screenshot every subsequent request fails and platforms that cannot rewrite past turns
  (litellm-fronted chat UIs, Kilocode-style agents) are wedged. The renderer
  (`_elide_over_limit_images`, `vllm/renderers/online_renderer.py`) now keeps the newest
  `limit` images and replaces older image parts with a short text marker before templating,
  preserving turn structure. Verified: 6 images at limit 4 → HTTP 200, log line
  "Elided 2 over-limit image(s)", and the model demonstrably no longer sees the elided
  (oldest) image while a 4-image control still does. `MM_ELIDE=0` restores the strict 400.

## 9. What was measured but not adopted

- YTILE=2 (two rows per wave) and LDS-staged activations for `gemv_f16_rdna2`: no gain.
- A 4-deep unroll of the MoE int4 GEMV: no gain (not memory-level-parallelism bound).
- `NCCL_P2P_LEVEL=SYS`: no gain; cards read 99 % busy while idle.
- `num_speculative_tokens` 4: +0.4 accepted tokens/step for +1 draft forward — a wash at 256
  tokens, +2 % on 1024-token generations.

## 10. What is left (per ~29 ms step)

All-reduce 4.9 ms (the kernel is 33 µs; the rest is waiting for the slowest rank — EP
imbalance), MoE int4 GEMV 4.3 ms (per-block LDS staging of activations suspected), ~9 ms of
launch bubbles over 1,788 kernels (the GDN core op's internal copies are the next ~250), the MTP
head's unquantised MoE (0.6 ms), lm_head int8 at 49 % of the bandwidth ceiling, prefill
(500–750 t/s warm; 100–180 t/s on first touch of a context length while sidecar pages are cold).
