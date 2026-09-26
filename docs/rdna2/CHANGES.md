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
| `vllm/model_executor/layers/` | `rdna_ops.py` (runtime-dispatch custom ops), `rdna_dense_int8.py` (int8 shadows; shadows-only mode), `utils.py` (the GEMM route), `linear.py` / `vocab_parallel_embedding.py` (hooks), `fused_qk_norm_rope.py`, `fused_moe/experts/triton_moe.py` (MoE hook), `ple_offload_layer.py` |
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
(eager + graph replay, bit-identical across ranks), and the staggered kernel serves at the recorded
rate (60–62 t/s at MTP=0 in-server, 2026-09-01). **Boot self-test hardened the same day:** it used to
time a single collective per size against a 50 ms bound, which on a freshly rebooted machine failed on
trial 1 (59 ms — rank skew at boot, not P2P speed) and silently fell back to RCCL at −25 % decode. It
now warms each size up untimed and judges the minimum of three timed repeats, which a genuinely slow
P2P path cannot pass. If you see `rdna_ar: disabled -- boot self-test failed` on an otherwise healthy
board, update to a tree with this change before concluding anything about your fabric. Motivation and what the knobs do NOT touch:
the one-shot path only carries decode-size messages (≤ 64 KB); prefill collectives are RCCL.

**Wedge handling (T44b, 2026-09-07).** Two boards reported the same picture: after a while one
GPU, or three, sits at 99 % and generation stops until the engine's 300 s execute timeout. That
is a collective whose peer flag never arrived: the kernel's spin was bounded (~2 s) and set a
sticky flag, but nothing read the flag after the boot self-test, so every later collective also
spun to its cap and returned without writing its output. Now the first abort records phase, peer
and sequence number in a host-mapped word (`rdna_ar_timeout_info`), the model runner reads it
once per step with a plain load, and on the first hit the process logs `rdna_ar: WEDGED -- rank
r timed out after ~N ms at collective #s: peer rank j's flag never arrived`, writes
`$VLLM_CACHE_ROOT/rdna_ar_wedged`, and fails the step. Graph-captured collectives cannot be
re-routed in a live process, so the fallback is the next boot, which sees the marker and starts
on RCCL with a warning naming it; delete the marker to retry P2P, `VLLM_RDNA_AR=0` forces RCCL.
`VLLM_RDNA_AR_SPIN_CAP` (polls per wait, default 2,000,000 ≈ 2 s) tunes the bound.

## 7. T45 — int8 shadows of the dense projections

`gemv_i8_rdna2` + `vllm/model_executor/layers/rdna_dense_int8.py`: per-output-channel symmetric
int8 copies of every dense fp16 weight, built in `process_weights_after_loading`, used for the
decode GEMV only (fp16 kept for prefill). Halves the streamed bytes; +1 GB per card. Enabled with
`VLLM_RDNA_DENSE_INT8=1`. Output validated with greedy checks; the MTP acceptance rate moves by
about ±0.1 tokens/step.

**Shadows only (`VLLM_RDNA_DENSE_INT8_ONLY=1`, 2026-09-06).** Releases the fp16 copy of every
shadowed projection once its shadow exists (the parameter becomes a 0-element placeholder), so the
int8 copy is the only resident one: weights per card 20.29 → 17.05 GiB, the KV pool roughly
doubles. Prefill-shaped calls (M > 8) dequantise the shadow on the fly inside the runtime-dispatch
ops (`rdna_dense_gemm`, `rdna_hc_mix`, `rdna_shared_expert`) into persistent per-shape fp16
scratch buffers (`rdna_dense_int8.dequant` / `linear_released`, 8192-row blocks for the lm_head);
the fakes take their shapes from the int8 tensor. Layers whose shadow is skipped keep their fp16
weight. Measured against the fp16-prefill layout on 9,210 teacher-forced tokens: NLL/token +0.1 %
(inside the sampling error), top-1 agreement −0.2 points (inside its error), mean per-token
logprob shift 0.24 nats; prefill and single-stream decode unchanged within noise, 12 concurrent
streams −2 % (the M > 8 path). Two operational notes: the compiled graph changes (0-element
weights), so give this mode its own `VLLM_CACHE_ROOT` and expect one recompile; and re-derive
`--gpu-memory-utilization` — at 0.96 the freed memory all became KV cache and an unprofiled
`prompt_logprobs` transient (full-vocabulary logits, ~1.5 GB) ran the card out of memory, while
0.93 keeps ~2 GiB of idle headroom. Serve script knob: `DENSE_INT8_ONLY=1`.

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
  (proxy-fronted chat UIs, Kilocode-style agents) are wedged. The renderer
  (`_elide_over_limit_images`, `vllm/renderers/online_renderer.py`) now keeps the newest
  `limit` images and replaces older image parts with a short text marker before templating,
  preserving turn structure. Verified: 6 images at limit 4 → HTTP 200, log line
  "Elided 2 over-limit image(s)", and the model demonstrably no longer sees the elided
  (oldest) image while a 4-image control still does. `MM_ELIDE=0` restores the strict 400.

## 8c. PLE round-trip: doorbell, page-table populate, faster lookup — 2026-09-05

Sized first (the "PLE offload host wait" log line is a host-thread wait overlapped with the
previous forward, not a stall — see RESULTS.md, 2026-09-05): a
`PLE_OFFLOAD_DEBUG_DELAY_MS` sweep gave a slope of 1.0 ms/step per ms of lookup latency (fully
serial) and a decode profile put the exposure at ~1.0–1.4 ms of a 16.2 ms step, the gap that
sits right before `__amd_rocclr_copyBuffer` (the result copy). Per-hop stamps
(`PLE_OFFLOAD_DEBUG_HOPS=1`: both processes write `perf_counter_ns()` into spare int64 slots of
the shared done page; medians/means/max logged every 500 decode launches) split the 1.42 ms:
d2h→sent 303 µs, sent→recv 172, recv→lookup 623, lookup→publish 41, publish→seen 105,
seen→enqueued 172. Changes, all bit-exact against the kept `_fused_decode_lookup_ref`
(thousands of random batches offline; `PLE_OFFLOAD_FUSED_CHECK=1` in-server: 995/1000 fused,
0 mismatches, max abs diff 0):

1. **Doorbell instead of ZMQ on the hot path** (`PLE_OFFLOAD_DOORBELL=1`, default; `=0` restores
   the ZMQ request). TP rank 0's *model thread* stages the D2H copies, sleep-polls the event until
   just before the forward is expected to end (running mean − 1 ms), then writes
   num_tokens/num_reqs/seq into int32 slots 5/6/4 of its done page. The sidecar spins on that
   page for 50 ms after each request, then sleep-polls (200 µs) the page *and* the ZMQ socket, so
   idle CPU stays low and ZMQ clients still work. The old request thread is bypassed: once the
   model thread spun, the two threads fought for the GIL exactly when the D2H completed
   (d2h→sent 0.3 → 1.5 ms, decode 55 t/s in the intermediate build). Spin loops call
   `time.sleep(0)` every 64 iterations to yield the GIL.
2. **Two-phase wait** in `_wait_lookup_done`: sleep-poll (100 µs) until the running mean of the
   decode wait minus 1.5 ms, then spin (rank 0 on the doorbell path spins immediately). The old
   20 µs sleep was ~70–100 µs real (the publish→seen hop).
3. **`madvise(MADV_POPULATE_READ)` over every shard mapping** at worker start (1.6–2 s warm)
   instead of the sequential read: the table was 100 % page-cache resident but each decode step
   took ~30 first-touch minor faults (3.8 µs each → 0.5 µs). Falls back to the read pass if the
   kernel refuses. Log line: `PLE sidecar populated (page cache + page tables): 32.0 GB in 2 s`.
4. **Lookup**: plain-ndarray views (an `np.memmap` subclass index costs µs per row), hashing
   constants cached on the layer, pure-Python hashing for the single-request case (74 → 4 µs),
   the float32→bf16 store done in numpy (round-to-nearest-even, same as torch) straight into a
   cached uint16 view of the pinned buffer. Offline one-token lookup 264 → 124 µs; in the sidecar
   0.62 → 0.25 ms (the rest of the old figure was the core waking cold from the ZMQ poll).

Result (170 W, MTP=0): hops median d2h→sent 34 µs, sent→recv 5, recv→lookup 181, lookup→publish
23, publish→seen 73, seen→enqueued 117, **total 485 µs** (was 1,416 mean). **Decode 61.9 →
64.0–64.1 t/s** over 256 tokens; prefill unchanged (1107/1070 @3.3k, 1178 @30k); validate PASS.
Rejected: a RAM-contiguous copy of the table (gather 27 → 9 µs for an unevictable 32 GB).
Left on the table: seen→enqueued (a zero-copy read of the pinned buffer would drop the per-step
H2D copy) and the ~180 µs lookup — together ≤0.3 ms/step.

**Trap:** a numpy view of a shared CPU tensor taken *before* the registration pickle dangles —
pickling under torch's `file_system` strategy copies the storage into a new mapping and swaps
the data pointer (torch views follow, numpy views do not) — all four workers SIGSEGV'd at the
first step after graph capture. Take numpy views of shared pages only after registration.

## 8d. Prefill on gfx1030 — 2026-09-04/05

Prefill had been left where the decode work put it (767–778 tok/s at a 3.3k prompt, 835 at 30k,
measured inside the server at 170 W power caps). Two days of prefill-only work, every lever
sized from in-server measurements, and every closed lever closed with data:

- **`NCCL_P2P_LEVEL=SYS` is now the serve script's default** (`P2P=` overrides): +8 % prefill,
  decode unchanged. On this 2+2 PCIe layout RCCL's ring is already die-local
  (`NCCL_GRAPH_DUMP_FILE` shows two cross-die hops, the minimum) with one channel over PHB,
  so the remaining knobs do nothing: `NCCL_PROTO=LL` −12 %, 16/32 channels neutral,
  `NCCL_NTHREADS` neutral, tree −3 %, `NCCL_BUFFSIZE` neutral. The collective log
  (`NCCL_DEBUG_SUBSYS=COLL`) shows every prefill collective is a per-layer hidden-state
  AllReduce (two per layer per chunk, ~10.5 MB), so EP-versus-TP cannot change the volume.
- **TunableOp rows for the dense GEMM shapes ship in `tunableop/rocblas-<build>/`** and the serve
  script enables TunableOp **lookup-only** (`PYTORCH_TUNABLEOP_TUNING=0`; `TUNEOP_TUNING=1` for a
  deliberate tuning boot). The rows are specific to the **rocBLAS build**, not its version string:
  solution ids come from the Tensile library as built, and the 7.14.1 container tarball's rocBLAS
  (same `5.5.0.cd957402` string as the 7.14.0rc3 host install) offers a different solution set
  for ~190 of the 290 rows — a row naming a solution the runtime lacks aborts the first GEMM with
  `Expected iter != ops_.end()`. The serve script therefore keys the directory by
  `sha256(librocblas.so)[:12]`, runs lookup-only when a directory matches, and otherwise disables
  TunableOp with a log line (`tunableop/README.md`). The top dense prefill kernel (`Cijk_… MT32x32x8`) went 734 → 259 ms
  per 3.3k prefill on its own. Tuning mode must never run in production: it autotunes every
  never-seen GEMM shape mid-request (prefill M is prompt-length dependent), which makes prefill
  bimodal (771 vs 37 tok/s measured) and perturbs greedy output while it runs.
- **`ROCR_VISIBLE_DEVICES` is honoured in the logical→physical device map** used for the amdsmi
  device-name lookup (`vllm/platforms/rocm.py`). Tuned-kernel configs are keyed by device
  name; with a different card at physical index 0 the lookup named the wrong device.
- **A tuned fused-MoE config for the int4 wna16 Triton prefill kernel**
  (`vllm/model_executor/layers/fused_moe/configs/E=128,N=640,device_name=AMD_Radeon_Pro_V620,dtype=int4_w4a16.json`):
  the default 64×64×32 tiles, 4 warps, **`num_stages=1`** — prefill **767/835 → 1038/1127 tok/s
  (+31 % / +35 %)**, decode unchanged (decode takes the CUDA `moe_wna16_gemm` path below
  `M·topk/E_local ≤ 6`). Every larger K tile and the 128×128 tiles lost, some below untuned.
  The file is keyed **per rank and packed**: `E` = experts ÷ EP, `N` = the packed `w2` N × 2 —
  not the model-level 512/1280 — and `device_name` comes from amdsmi. The serve log says
  `Using configuration from …` when it loads and `Using default MoE config … Config file not found
  at <exact paths>` when it does not; the latter names the filename it wants. Re-tune if EP/TP or
  the quantisation packing changes (`benchmarks/kernels/benchmark_moe.py` recognises Qwen4Exp).
- **QSA launch tiles** (`vllm/models/qwen4_exp/amd/ops/qsa.py`): sparse prefill attention
  BLOCK_N 64 → 32 with 4 warps, indexer (MQA scoring) BLOCK_N 32 → 128 with 8 warps: +3 % at
  3.3k, +5 % at 30k. Sparse BLOCK_N = 128 is catastrophic; MQA `num_stages=1` neutral. Env
  overrides remain for re-tuning: `VLLM_RDNA_QSA_MQA_BLOCK_N`, `VLLM_RDNA_QSA_MQA_WARPS`,
  `VLLM_RDNA_QSA_MQA_STAGES`, `VLLM_RDNA_QSA_BLOCK_N`, `VLLM_RDNA_QSA_SPLITS`,
  `VLLM_RDNA_QSA_WARPS`, `VLLM_RDNA_QSA_STAGES`; the boot log line `QSA launch params: …`
  shows what took effect.
- **The gfx1030 `num_stages` rule.** Triton's AMD backend defaults to `num_stages=2` (CUDA's is 3);
  on this chip that halves occupancy and `num_stages=1` wins. Three instances so far: the prefill
  attention kernels, the QSA partial kernel, and the fused-MoE int4 kernel above — the last one
  hides differently, because its stages come from the config JSON, so *no JSON* means the
  default. Never trust a tuning result whose load you did not verify in the serve log.

Result at 170 W caps: prefill **767/778 → 1074–1107 at 3.3k (+40 %), 835 → 1178–1189 at 30k
(+42 %)**; decode unchanged. Profile of a 3.3k prefill afterwards (3,278 ms of kernels): MoE
27.6 %, RCCL 22.1 %, QSA 21.5 %, dense GEMM 21.1 %.

Measured and closed: hipBLASLt (unsupported on gfx1030 — PyTorch logs "Attempting to use
hipBLASLt on an unsupported architecture" and falls back to hipBLAS; rocBLAS already runs the
dense GEMMs at 34.4 TFLOP/s ≈ 76–93 % of fp16 peak); `--max-num-batched-tokens` (2048 is the
optimum: 1024 −15 % / −27 %, 4096 −5 % / −10 %); `compile_sizes: [2048]` (crashes: the runner
slices a `[3, 2049]` MRoPE positions buffer and Inductor's static-stride guard wants `[3, 2048]`
contiguous — root-caused in `gpu_model_runner.py`, not fixed). Short-prompt time-to-first-token
has a ~0.3 s floor that is **host-dispatch-bound**: a 40-word prompt has ~0.17 s of GPU work
per rank and the rest is the CPU issuing ~3,500 eager launches through Inductor wrappers and
custom-op dispatch across 48 layers while the ranks wait on each other; long prefill is 91 %
GPU-busy. Graph capture (or launch fusion) for small prefill batches is the lever there.

## 8e. Time-to-first-token: CUDA-graph capture sizes for prefill batches — 2026-09-06

The serve configuration already ran `FULL_AND_PIECEWISE`, but vLLM caps the default piecewise
capture list at `min(max_num_seqs × decode_query_len × 2, 512)` tokens — with `--max-num-seqs 4`
that is **8**, so the list was `[1, 2, 4, 8]` and every real prefill batch ran its compiled pieces
eagerly (~3,500 launches for a 40-word prompt, CPU-bound, ranks waiting on each other every layer,
§8d). All attention/GDN/QSA/PLE-conv ops are splitting ops, so capturing larger sizes covers exactly
the dense/MoE/norm launches. The serve script now passes `--cudagraph-capture-sizes 1 2 4 8 16 32 64
128 256` (`CG_SIZES=` overrides; empty restores vLLM's default).

Measured at 160 W caps, MTP=0, streaming time to first content token:

| case | before | after |
|---|---|---|
| short prompt (40 words) | 0.38–0.42 s | **0.25–0.26 s** (−35 %) |
| 3.3k prompt, cold | 2.59 s | 2.48–2.50 s |
| 3.3k prompt, fully cached | 0.745 s | 0.74 s |
| cached prefix + ~55-token tail (agent turn) | 0.83–0.89 s | 0.83–0.88 s |
| decode / prefill / validate | 63.4–63.8 / 1043–1068 @3.3k | 64.0–64.6 / unchanged / pass |

Cost: graph memory 0.64 → 1.31 GiB per card, KV pool 353k → 309k tokens (−12 %). Larger lists
buy nothing more here: 14 sizes ≤512 → 262k tokens, 18 sizes ≤2048 → 221k (+2–3 % prefill at 3.3k
from replaying the 2048 chunks, −4 % cold 3.3k TTFT) — the memory is per captured size and grows
with size. Boot adds ~20–40 s of capture, and the capture list is part of the compile-cache key
(a changed list is a cold compile).

**The cached-turn floor is a different mechanism, and this vLLM's knobs do not reach it.** The
prefix-cache counters (`/metrics` deltas: `prefix_cache_queries_total` / `hits_total`) show a
2642-token prompt re-sent hits 2352 = 3 × 784 tokens and recomputes ~300, at small-batch efficiency
(~600 tok/s) — that is the ~0.5 s above the floor. With `--block-size 256 --mamba-block-size 256`
the hit *fell* to 2048 (recompute 594): in `align` mode the Mamba state is checkpointed only where
a prefill scheduler step ends on a block boundary, and a 2048-token chunk plus a 594-token remainder
leaves nothing cached past 2048 (784 happened to give 1568 + 784 = 2352). `--mamba-cache-mode all`
did not change the counters for this model. The lever is a scheduler change: end the final prefill
step of a prompt on a block boundary so the largest aligned prefix is checkpointed, leaving a
recompute of < block_size tokens on re-send (~0.5 s → ~0.45 s with 256-token blocks). Not done.

## 9. What was measured but not adopted

- YTILE=2 (two rows per wave) and LDS-staged activations for `gemv_f16_rdna2`: no gain.
- A 4-deep unroll of the MoE int4 GEMV: no gain (not memory-level-parallelism bound).
- `NCCL_P2P_LEVEL=SYS`: no gain at decode (2026-08-29); re-measured for prefill 2026-09-04 at +8 %
  and made the default (§8d).
- A RAM-contiguous copy of the n-gram table in the sidecar: gather 27 → 9 µs per step for an
  unevictable 32 GB of host memory (§8c).
- Hand-written GDN decode kernels: GDN is 0.36 ms of a 15.6 ms decode step (§10), so even an
  infinite speed-up buys ≤ 0.4 ms.
- An MoE tile sweep beyond `num_stages=1`: every K ≥ 64 and 128×128 tile lost (§8d).
- `num_speculative_tokens` 4: +0.4 accepted tokens/step for +1 draft forward — a wash at 256
  tokens, +2 % on 1024-token generations.

## 10. What is left

**Decode (per 15.6 ms step at MTP=0, 2026-09-05 kernel budget, rank 0):** kernels 14.6 ms —
MoE decode kernels ~4.8 (int4 GEMV family + `topkGating`), dense int8 GEMV 2.6 (158 launches),
the one-shot all-reduce 2.5 (97 launches, includes peer spin), norm/elementwise ~1.1, QSA 0.4,
GDN recurrent 0.4, the lm_head logits AllGather 0.08 (one RCCL call per step, 62,080 fp16 per
rank × 4 = the 248,320 vocabulary). Host-side idle ≈ 0.5 ms, nearly all the PLE round-trip
(§8c; a zero-copy read of the pinned result would drop the per-step H2D copy, ≤ 0.3 ms more).
With MTP=3 the same fixed per-step costs amortise over ~2.4 accepted tokens per step but the
drafter's forwards and the larger verify batch (each extra token pulls its own experts) absorb
the gain at typical acceptance — choose by measured acceptance, not by default.

**Prefill (per 3.3k prompt):** MoE, RCCL, QSA and the dense GEMMs at roughly a quarter each
(§8d); the all-reduces are per-layer hidden-state exchanges over PCIe that no RCCL knob improves
at this topology; `compile_sizes` static shapes are blocked by the positions-buffer stride
(§8d); short-prompt TTFT is host-dispatch-bound (§8d).

## 10. W4A8 MoE prefill kernel (2026-09-24)

`vllm/model_executor/layers/fused_moe/rdna_w4a8_moe.py`, opt-in with `VLLM_RDNA_MOE_W4A8=1`, hooked at the
top of `invoke_fused_moe_wna16_triton_kernel` (the modular `TritonExperts` path calls it directly). A prefill
profile put the int4 MoE Triton kernel at 28 % of prefill and ~4 TFLOPS, about 11 % of the V620's measured
fp16 dot rate: the kernel converts every int4 weight to fp16 in the inner loop (shift, mask, int→fp32,
subtract, scale, fp32→fp16). The W4A8 kernel quantises activations per token to int8 once per GEMM,
unpacks weights to int8 with integer ops only, accumulates `tl.dot(int8, int8) → int32` (lowered to
`v_dot4_i32_i8`) over each 128-wide quantisation group, applies the group scale once per output element
and the token scale at the store. Symmetric int4 only; decode-shaped configs (BLOCK_SIZE_M < 32) keep the
stock path, so decode is unchanged.

Tiles matter more than usual on RDNA2: any K block ≥ 64 hits the 256-VGPR cap and spills 190–390 registers
(up to 5× slower). Swept best at Flash-Next per-rank shapes: 64×128×16, 8 warps, 2 stages — 2.0–2.45× over
W4A16 from M = 256 to 2048, exact against a torch reference of the same maths (2e-4). BLOCK_SIZE_M is taken
from the caller because the token sort was aligned to it.

In-server (4× V620, TP=4/EP=4, 120 W caps): prefill 908 → 1,160 t/s @3.3k, 957 → 1,217 @14k, 962 → 1,232
@26k (+27–28 %, more than the MoE share alone predicts — halving MoE time also shrinks the all-reduce wait
behind the slowest rank); decode 62.8–62.9 t/s unchanged; validate PASS; teacher-forced NLL/token vs W4A16
−0.1 % general / −0.2 % identifier-heavy (within noise), deterministic run to run.

Side finding: the MoE kernel's time is nearly flat in chunk size (per-rank ~40 tokens per expert per 2048
chunk, so every expert fills one 64-row block regardless) — the 2048-token prefill chunk was tuned before
this kernel and is worth re-sweeping.

## 11. QSA indexer: score only the visible columns (2026-09-24)

The sparse-attention indexer scored the page table's full capacity for every query row: the caller sized the
logits as `page_table.shape[1] * k_cache.shape[1]` (≈ max_model_len / 4 = 65,536 columns at 262k) and never
passed `num_columns`; the kernel has no early exit past the causal edge, so it ran the head loop on masked
loads and wrote −inf for every capacity column (~0.5 GB of fp32 logits per layer per 2048-token chunk), and
the logits workspace cap split each chunk into many launches. Its cost was therefore flat in context and
proportional to `--max-model-len`, and it was the largest prefill kernel at long context (24 %).

`QSAForwardMetadata` now carries the batch's host-side `max_seq_len` (from vLLM's common metadata, no device
sync), and `qsa_select_paged_tokens` scores `ceil(max_seq_len / compress_ratio)` columns rounded up to the
tile. Only eager batches of at least `VLLM_RDNA_QSA_BOUND_MIN_TOKENS` rows (default 512) outside graph
capture are bounded: graph-replayed batches (decode, piecewise prefill ≤ 256 tokens) keep full capacity,
because a column count captured into a graph would truncate candidates for later, longer sequences.

Exact by construction — the dropped columns were always −inf and never selected. Harness: selected indices
identical on every row at 7k / 30k / 65k context; per layer per chunk 33.5 → 1.2 ms @7k, 35.6 → 5.5 ms @30k,
37.8 → 12.7 ms @65k. In-server (W4A8 on, 120 W): prefill 1217 → 1566 t/s @14k, 1232 → 1582 @26k, 1213 → 1528
@58k, 1207 → 1469 @88k (+22–29 %); 3.3k 1160 → 1200–1425 (noisy); decode unchanged (62.5–63.0); validate PASS;
teacher-forced logprobs bit-identical to the unbounded server.

A blocked `tl.dot` variant of the scoring kernel (`_qsa_mqa_paged_dot_kernel`, one program per 16 rows x 32
keys, key tile reused across heads and rows, fp16 dot2) ships OPT-IN (`VLLM_RDNA_QSA_DOT=1`): once columns
are bounded the kernel is gather- and mask-bound rather than compute-bound, and the best swept tile is
0.84–1.08× the per-row kernel from 7k to 88k context. Single-request batches only.

## 12. Cached-turn TTFT: keep the linear-attention state at the prompt's last full block (2026-09-25)

In `align` mode the GDN (linear-attention) state is cacheable per 784-token Mamba block, and a block holds
a usable state only if some prefill step *ended exactly* at that block's end. Upstream enforces this by
block-aligning chunk ends, but in `scheduler.block_size` units, which on this model is the 4-token
attention granularity. Steps are therefore sized by the 2048-token budget and land on a 784-token boundary
only by chance. Two blocks matter for reuse:

* **the prompt's last full block**, which is what the next conversational turn resumes from;
* with `--prefix-cache-retention-interval`, **one block per segment**.

Both were mostly lost. After an 81k-token cold prompt the linear-attention groups held states only up to
37,632 tokens. The full-attention groups hit 80,752 tokens, but a hybrid hit is the minimum over groups,
so the first follow-up turn recomputed 43.5k tokens (31 s to the first token). Later turns were fast
because the follow-up's own prefill happened to cache the boundary.

`_mamba_block_aligned_split` now adds a stop at the end of each retained block: the last full block of
the prompt, and every retained segment block inside the chunk. This costs one extra, shorter step per
retained block, with no measurable change in cold prefill time. Knob: `VLLM_RDNA_MAMBA_RETENTION_STOPS`
(default 1). It is a no-op when the scheduler block size already equals the Mamba block size.

A stop placed *inside* the block is a trap. It gets the block allocated and marked cached, but the block
then holds the state a few tokens short of its end, and a resumed turn silently drifts.

Measured with `turn_probe.py` (cold prompt, then three follow-up turns, greedy):

| history | first follow-up before | first follow-up after |
|---|---|---|
| 3k | 2,352 hit, 0.96 s | 2,352 hit, 0.77–0.87 s |
| 27k | 25,088–26,656 hit, 0.46–1.9 s | 26,656 hit, 0.59–0.66 s |
| 81k | 37,632 hit, 43.5k recomputed, 31 s | 80,752 hit, 434 recomputed, 0.98–1.09 s |

Every retained block of a 344-block run was cached; none were lost. Correctness was checked by comparing a
cache-resumed follow-up against the same request with a `cache_salt`, which forces a full recompute. Two
recomputes already diverge after 2–4 greedy tokens on this server, so that is the noise floor. Resumed and
recomputed answers match for all 48 tokens at 34k and 84k, and first-token log-probabilities agree within
0.02–0.2 nats.

## 13. W8A8 prefill GEMMs for the dense projections (2026-09-25)

With `DENSE_INT8_ONLY=1` the dense projections exist only as per-channel int8 shadows. Prefill used to
expand each shadow to fp16 and run rocBLAS. `rdna_w8a8_dense.py` (`VLLM_RDNA_DENSE_W8A8=1`, default off)
instead quantises the activations per token to int8 and runs `tl.dot(int8, int8) -> int32` (`v_dot4`),
applying the token and channel scales in the epilogue. It is hooked into the prefill branch of
`rdna_dense_gemm` and, opt-in, `rdna_hc_mix`. Any M works, so the prompt's tail chunk no longer falls
back to an untuned rocBLAS tile for these shapes.

**Measure under sustained load.** At the 120 W cap a V620 throttles after a few hundred milliseconds:
rocBLAS ran a 2048×4096×2560 GEMM in 2.05 ms on its first burst and 3.0 ms once settled. Short
benchmarks mix the two regimes; the harness warms for 0.6 s and measures for 1 s.

Per-rank GEMM at M = 2048, sustained (`bench/w8a8-dense/harness.py`), including activation quantisation:

| shape (N×K) | rocBLAS | W8A8 | |
|---|---|---|---|
| GDN in_proj_qkvz 4096×2560 | 3.27 ms | 1.81 ms | 1.81× |
| QSA qkv 3584×2560 | 3.02 ms | 1.68 ms | 1.79× |
| GDN out / QSA o 2560×1536 | 1.38 ms | 0.76 ms | 1.83× |
| hyper-connection up 10240×320 | 1.13 ms | 0.89 ms | 1.27× |
| hyper-connection down 336×10240 | 1.34 ms | 1.11 ms | 1.21× |
| router, shared expert, indexer | | | 0.89–0.96× (not used) |

Tiles: 128×256×16, 8 warps, 2 stages for the three projections. Quality was gated with teacher-forced
logprobs against the fp16-GEMM path; that baseline is bit-identical across boots, so every change seen
is real.

| variant (vs baseline, general / identifier-heavy texts) | mean \|Δlogprob\| | NLL | top-1 |
|---|---|---|---|
| projections (default set) | 0.21 / 0.16 | +0.14 % / +0.23 % | ±0 |
| hc up projection alone | 0.36 / 0.28 | −1.1 % / −1.1 % | −0.4 pt |
| hc down projection alone (per-branch activation scales) | 0.20 / 0.15 | ±0 | −0.3 pt |

The hyper-connection up projection produces the branch gates. Quantising its input moves the outputs
about twice as much as any change adopted so far, so it is opt-in (`VLLM_RDNA_DENSE_W8A8_SHAPES`). The
down projection reads four residual branches side by side. It gets one activation scale per branch
(`VLLM_RDNA_DENSE_W8A8_HC_GROUP`, default 2560), which fixed the precision loss on imbalanced branches in
the harness, but in-server it gained nothing, so it is opt-in as well.

In-server, 120 W, projections only: prefill +3.7–3.9 % from 14k to 88k and +5–7 % at 3.3k; decode
unchanged; validate PASS.

## 14. int8-compressed prefill all-reduce (2026-09-25)

A 2048-token chunk all-reduces 10.5 MB of fp16 twice per layer, and those calls are PCIe-bound.
`rdna_q8_all_reduce.py` (`VLLM_RDNA_AR_Q8=1`, default off) halves the bytes with a two-shot algorithm
built from RCCL collectives on the existing pynccl communicator, so no peer-memory kernel is needed:

1. Quantise each row in 64-element blocks to int8, with an fp16 scale per block.
2. All-to-all the row shards, using grouped send/recv.
3. Dequantise, sum in fp32, and requantise the reduced shard.
4. All-gather, then dequantise.

It takes eager, non-captured fp16 calls of at least 1 MB (`VLLM_RDNA_AR_Q8_MIN_KB`) whose last dim is
in `VLLM_RDNA_AR_Q8_DIMS` (default 2560, the language model). The vision tower's 1152-wide all-reduces
stay on RCCL because they are not quality-tested. vLLM's own quantised all-reduce, QuickReduce, is
built for MI300: it uses wave64 and CDNA3 buffer-descriptor and cache-scope bits, so it does not run
on gfx1030 without a port.

4-GPU microbenchmark (`bench/int8-allreduce/ar_bench.py`):

| M × 2560 | RCCL fp16 | int8 two-shot | |
|---|---|---|---|
| 2048 | 2.30 ms | 1.41–1.47 ms | 1.56–1.63× |
| 1024 | 1.17 ms | 0.76 ms | 1.53× |
| 512 | 0.60 ms | 0.44 ms | 1.38× |

The relative error is 0.76–0.92 % on Gaussian data (block 32–128), against 3e-4 for RCCL. The size is
set by rounding twice, so smaller blocks barely help. In-server the gain is smaller than the
microbenchmark suggests, because part of the all-reduce time is waiting for the slowest rank: +1.8–4.6 %
prefill alone. Teacher-forced drift: 0.20 / 0.15 nats, NLL +0.05 % / +0.26 %.

**Together with §13** (projections), in-server at 120 W:

| | 3.3k | 14k | 26k | 58k | 88k |
|---|---|---|---|---|---|
| baseline | 1,272–1,299 | 1,473 | 1,509 | 1,459 | 1,424 |
| W8A8 projections + int8 all-reduce | 1,356–1,371 | 1,590 | 1,639 | 1,572 | 1,523 |
| gain | +6 % | +7.9 % | +8.6 % | +7.7 % | +7.0 % |

Decode is unchanged (62.4 t/s) and validate passes. Drift is 0.23 / 0.16 nats, NLL −0.06 % / +0.13 %,
top-1 −0.15 / +0.01 pt, which is in the range of earlier adopted changes.

## 15. Host-staged one-shot all-reduce, `VLLM_RDNA_AR_MODE=host` (2026-09-25)

The one-shot all-reduce (T44, §6) pushes each rank's contribution straight into its peers'
uncached VRAM through their PCIe BARs. At ~95 collectives per decode step that is thousands of
small posted writes per second into every card, and on a 2-die X399 board about half of them
cross between the dies. Several users reported cards wedging or dropping with it until they set
`VLLM_RDNA_AR=0`. On 2026-09-25 our own board lost two V620s off the bus twice under the same
heavy multi-stream load; with `VLLM_RDNA_AR=0` the load ran clean, at −26 % single-stream decode
(62 → 46 t/s).

`VLLM_RDNA_AR_MODE=host` keeps the one-shot protocol but moves the payload to host memory:

- **One shared pinned host buffer.** Rank 0 creates POSIX shared memory during the ordered
  init, every rank maps it and registers it with its GPU (`hipHostRegister`, fine-grained). There
  is no peer access, no IPC mapping and no GPU-to-GPU write of any kind.
- **Tagged words instead of flags.** Each rank writes its slot as 8-byte words: 4 bytes of data
  and the collective's sequence number (RCCL's "LL" idea). A reader accepts a word only when its
  tag matches, so correctness does not depend on write ordering. Posted writes to different
  destinations, or relaxed-ordered ones, can overtake each other; that ordering gap caused the
  2026-08-30 stale-element noise.
- **Cheap waiting.** One thread per block polls one word per peer, with backoff, before the bulk
  reads. The bulk reads re-poll individually only if a word is still in flight.
- **Unchanged guarantees.** There are two parities, a fixed-order fp32 reduction so results are
  bit-identical across ranks, a sequence counter on the device for graph replay, and the T44b
  bounded spins with abort record, wedge marker and RCCL on the next boot.

The selection is made inside the extension (`rdna_ar_init` reads the variable), so the op
signatures and the Python side are unchanged. Default is still `p2p`.

`tools/rdna2/ar_ops_test.py` on 4× V620, fp16 4×2560 (20 KB), host mode: eager and graph replay
correct on every rank, bit-identical across ranks, **32.1 µs per all-reduce eager, 33.3 µs in a
CUDA graph**. The p2p kernel measures 31–33 µs and RCCL ~156 µs. The op test can load a freshly
built library without installing it (`RDNA_AR_TEST_SO=build_rocm/_rocm_C.abi3.so`).

Not yet done: a serving soak under the load that dropped the cards.

Build note: `cmake/hipify.py` now resolves symlinks for the project, output and source paths.
With the checkout reached through a symlink, the sources were hipified under the resolved path
and the headers under the symlink path. Dependents then kept `#include "cuda_compat.h"`
un-hipified, and `attention.hip` and `skinny_gemms.hip` failed with `unknown type name
'cudaDeviceProp'`.

## 16. Cheap prefill items, 2026-09-26: M-bucketed rocBLAS GEMMs, de-spilled QSA kernel, no own-shard copy

- **M buckets (`VLLM_RDNA_GEMM_MBUCKET=128`, default off).** Since the cached-turn fix (§12), prefill steps end
  at 784-token block boundaries, so many steps have odd M. TunableOp rows are exact-M, so those GEMMs fell back to
  rocBLAS's MT32x32x8 tile: 3.3 % of a 14k prefill. The knob pads M > 256 up to a multiple of 128 for the GEMMs still
  on rocBLAS (hyper-connection mixers, router, shared expert, indexer, and one 2560×2560). The new TunableOp rows for
  M = 384 … 2048 are in `tunableop/rocblas-3b4878bc6c37/`.
  In-server at 120 W vs off: prefill +4–6 % at 3.3k, +2.5 % at 14k, ±0 from 26k to 88k; decode unchanged.
  Teacher-forced NLL −0.06 % / +0.16 %, within noise.
- **De-spilled sparse attention (`VLLM_RDNA_QSA_DESPILL=1`, default off).** 1.17× on the kernel in the harness,
  but **no end-to-end gain in-server** (14k 1,588 vs 1,591 t/s). Left off.
- **int8 all-reduce reads its own shard in place,** so there is no copy per collective. Exact.
- `tools/rdna2/trace_agg.py` classifies the fork's kernels: dense-int8-GEMM, rdna-allreduce, int8-quant, GDN pieces.

**Stability note.** A V620 dropped off the bus during the de-spilled-only measurement boot (host-staged decode
all-reduce, int8 prefill all-reduce on). All three drops since 2026-09-25 had `VLLM_RDNA_AR_Q8=1`; none occurred
in the weeks before it existed. Treat the int8 prefill all-reduce (§14) as suspect on multi-root-complex boards.
Its all-to-all sends to every peer at once over P2P.

## 17. Staggered exchange for the int8 all-reduce, `VLLM_RDNA_AR_Q8_STAGGER` (default on), 2026-09-26

The int8 all-reduce (§14) exchanged its shards with one grouped all-to-all, so every card sent to and received
from all three peers at once: twelve flows, three inbound per card. Every V620 bus drop on our board since
2026-09-25 happened with that enabled (§16, stability note). The exchange now runs in W−1 rounds. In round k
each card sends only to rank+k and receives only from rank−k, so each card has one inbound and one outbound
transfer at a time, four flows per round. The bytes and the reduction order are unchanged.

`tools/rdna2/q8_ar_test.py` drives the real module over vLLM's PyNcclCommunicator on 4× V620. The staggered
result is **bit-identical** to the grouped one and identical across ranks at M = 2048, 1300 and 700 (×2560
fp16), with rel. error 8.4e-3 vs the exact sum, as before. It is also slightly **faster**:

| M × 2560 | RCCL fp16 | int8 grouped | int8 staggered |
|---|---|---|---|
| 2048 | 2.29 ms | 1.43 ms | 1.40 ms |
| 1300 | 1.60 ms | 1.09 ms | 1.04 ms |
| 700 | 0.95 ms | 1.00 ms | 0.87 ms |

`VLLM_RDNA_AR_Q8_STAGGER=0` restores the grouped all-to-all. Whether staggering removes the bus drops is not yet
known; it needs a heavy-load soak with `VLLM_RDNA_AR_Q8=1`.
