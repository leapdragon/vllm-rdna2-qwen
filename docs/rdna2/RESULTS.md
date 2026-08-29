# Measured results — Qwen3.8-Flash-Next on 4× Radeon PRO V620 (T42–T46)

These are the experiment-log entries, verbatim, from the campaign that produced this fork
(`docs/TESTS_RESULTS.md` in the research repository). "T-numbers" are experiment ids; "boot N"
is one server restart. Every number was measured inside the serving process on this hardware.
Read `CHANGES.md` for what each change is and why, and `README.md` for how to reproduce.

## T42 — Qwen3.8-Flash-Next (qwen4_exp) on 4x V620: llama.cpp parity (2026-08-29)

First working deployment of Qwen3.8-Flash-Next under vLLM on RDNA2, and the
performance campaign that followed. Config: `qwen4exp-pleq/serve-pleq.sh` --
AWQ-W4A16 backbone (native W4A16 kernels), TP=4 + **expert parallel** (mandatory:
moe_intermediate 640/TP4 = 160 is not divisible by group_size 128), FP16,
int4 PLE sidecar, `language_model_only`, `skip_mm_profiling`, BATCHTOK 2048.

| step | 3.5k | 13k | 41k | note |
|---|---|---|---|---|
| bf16 table on disk, eager | 5.66 | 5.50 | 5.31 | first working run |
| int4 PLE sidecar, eager | 5.49 | 5.32 | 5.46 | **no throughput change** |
| + CUDA graphs | 16.45 | 16.55 | 16.67 | **3.0x** |
| + MTP=3 | **30.96** | **29.19** | **28.76** | **1.9x more; 5.6x total** |

**llama.cpp on this same box and model: 29.05 t/s at -c 4096, 30.10 t/s with the
full server flag set** (`~/repos/llama.cpp-qwen4exp-official/LOCAL-CHANGES.md`).
So vLLM now matches it at short context and, unlike that measurement (taken at
-c 4096), holds 28.76 t/s at 41k.

MTP acceptance 1.40 tokens/draft-step at K=3 = 2.40 tokens/step. 0 gpu fault
events across the whole campaign. Cards 41-42 C, 88% VRAM.

### Two hypotheses this campaign killed

1. **"Disk-paging the 102 GB table is the bottleneck."** Wrong. Moving it to a
   32 GB page-cache-resident int4 sidecar changed throughput by <4%. Its value is
   memory headroom: 103 GiB downloaded instead of 169 GiB, host RAM 34 GB used /
   90 GB free where the bf16 table did not fit at all.
2. **"GPUs at 38-99% utilization means compute-bound, so it must be the Triton
   GDN kernels."** Wrong, or at least premature. With 48 layers of small Triton
   kernels, eager mode keeps the GPU nominally busy while mostly paying launch
   overhead -- graphs alone were worth 3x. High utilization is not evidence of
   useful work.

### Still open

GDN decode runs Triton/FLA for 36 of 48 layers: the fused kernel is CUDA-only
(`torch.ops._C.fused_gdn_decode_post_conv_mtp` is absent from a ROCm build; its
source is `csrc/libtorch_stable/gdn/fused_gdn_decode_kernel.cu`), so the bf16
requirement in `_fused_gdn_decode_unsupported_reason` never even applies to us.
The `rdna2_extras` fork's hand-written HIP GDN chain claims ~9.3x over Triton at
B=1 -- the remaining upside, and a real porting project.
KV cache is down to 3.64 GiB with the MTP drafter resident; worth tuning.

## T43 — Flash-Next 35 → 74 t/s: the dense GEMMs were on rocBLAS (2026-08-29)

**Goal**: 50 t/s decode on Qwen3.8-Flash-Next (T42 config: 4×V620 TP=4/EP, int4 PLE
sidecar, CUDA graphs, MTP=3). **Result: 76.9 / 73.4 / 74.4 t/s** (256-token runs),
**74.2 t/s over 1024 tokens**, from 36.7 / 34.8 / 34.7 on the same boot config an hour
earlier. Greedy validation: 17×23 → 391, capital of Australia → Canberra, first five
primes → 2, 3, 5, 7, 11. MTP acceptance unchanged (2.78–2.88 tok/step). Zero fault
events, cards 47–52 °C, fans on.

Yardstick: `bench.py` streams one completion and measures decode t/s first→last chunk
(prefill excluded; tokens from the usage block). T42's 30 t/s was a different prompt
(2.40 tok/step) measured wall-inclusive; the T43 baseline on the new yardstick was 35.

**Method: profile inside the serving process first** (torch profiler via
`--profiler-config`, `PROFILE=1` launcher knob). Rank-0 trace, 42 decode steps:

| family | before | after |
|---|---|---|
| rocBLAS Tensile fp16 GEMM | **2039 ms, 60 %** | 55 ms (prefill + M>8 only) |
| NCCL | 680 ms, 20 % | 815 ms, **41 %** |
| MoE (Triton WNA16 → skinny int4 GEMV) | 346 ms, 10 % | 207 ms (skinny) + 83 (prefill/MTP) |
| own fp16 GEMV (`gemv_f16_rdna2`) | — | 485 ms, 24 % |
| glue (60k elementwise kernels) | 168 ms | 162 ms |
| per MTP step | ~78 ms | ~43 ms |

**Finding 1 (the 60 %).** Only the routed experts are int4; GDN in/out proj, QSA q/o,
router, shared expert, both hyper-connections and lm_head are fp16 — ~2.1 GB/rank
streamed every forward. vLLM's skinny-GEMV route is gated `on_gfx9() or on_gfx1x()`, and
`_ON_GFX1X` means gfx11|gfx12, so on gfx1030 every one of them ran on Tensile tiles at
~35 % of bandwidth (the K=10240 hyper-connection GEMMs at 114+74 µs per module for 13 MB).
The trace contained zero wvSplitK/LLMM1 kernels. Widening the upstream macro to gfx10
compiles and launches wvSplitK but returns **wrong results** (relerr 0.6–0.9 on every
shape), so `gemv_f16_rdna2` is our own: wave-per-output-row, 16-B loads, 4-deep unroll,
`v_dot2_f32_f16`, wave32 shuffle reduce, M ≤ 8 as a template parameter, WAVES/block from
N. relerr ~3e-4 on all 12 dense shapes; sum of the 12 at M=4 **1.43 ms vs 5.13 ms**;
lm_head/rank 1.28 vs 4.23 ms. Standalone timing lesson: warm the card first — the first
pass on the idle GPU read 5–10× slow on some shapes (clock ramp), not the kernel.

**Finding 2.** The T38 int4 MoE GEMV (patch 0006) had never run on this model: its hook is
gated `expert_map is None`, and EP always sets `expert_map`. Made EP-aware (remap ids,
skip `expert < 0`, rank partials summed by the existing all-reduce); harness PASS incl.
an all-non-local token; 98 µs per w13+w2 pair in the serving trace vs 2×76.5 µs Triton.
Both `apply()` bodies carry the hook (the T38 trap again).

**Also learned.** Custom all-reduce is unavailable here twice over (`use_custom_allreduce()`
false on RDNA unless `VLLM_ROCM_FORCE_CUSTOM_ALLREDUCE=1`, and `world_size > 2 and not
fully_connected` — the XGMI check — disables it anyway). GPU idle is 17–23 % of wall but it
is 128k gaps of ~4 µs between kernels, not CPU stalls: dispatch count, as the llama.cpp
field guide said. Side jobs on ROCR device 1 land on a serving card (PCI 0D) — use
ROCR 0 (PCI 0A) — a harness there put "Runlist is getting oversubscribed" in the kernel
log during graph capture.

Patches: `qwen4exp/patches/0017` (PLE worker, previously uncommitted) and `0018` (this).
`_rocm_C` rebuilt alone with `cmake --build build_rocm --target _rocm_C` in ~4 min.

**Next levers, by expected value** (per ~43 ms step): NCCL 19 ms — 119 latency-bound
collectives (156 µs each for ~20 KB): a one-shot P2P/host-memory all-reduce for small
messages (P2P + native atomics verified on all pairs, docs/old/OPTIMIZATION-PHASE.md §1a)
or fewer collectives; dispatch — ~3,000 kernels/step, 60k glue kernels per trace, fusion
targets: hc mix/combine chains, router+topk, MTP-head unquantized MoE (`fused_moe_kernel`,
117 µs × 6/step — an fp16 expert GEMV would do); `gemv_f16_rdna2` itself at 250 GB/s on
lm_head (49 % of ceiling; YTILE=2 rows/wave to halve x re-reads and raise MLP); the
Triton GDN decode kernel still at `num_stages=3`.

## T44 — one-shot P2P all-reduce for 4 ranks: 74 → 82 t/s (2026-08-29)

Post-T43 profile: RCCL 41 % of GPU time, 119 latency-bound collectives per step at 156 µs
for ~20 KB. `NCCL_P2P_LEVEL=SYS` (boot 3): no better (73.8 / 68.6 / 72.1) and cards 99 %
"busy" at idle — reverted. vLLM's custom all-reduce is gated off twice on this platform.

`rdna_ar_oneshot` (`csrc/rocm/rdna_allreduce.{cuh,cu}`, W = 2..8): the WS2 TP=2 design
(push into every peer's uncached staging, host-coherent flags, device-side sequence numbers
so graph replay is safe, bounded spins) plus W-slot staging and a fixed-order fp32 reduction
(bit-identical ranks). Cross-PCIe on the four serving cards: **33 µs** per 20 KB message
(RCCL in-server 156), 15 µs / 5 KB, 85 µs / 80 KB; block sweep 1/4/16 = 76/36/33 µs.
Hooked into `CudaCommunicator.all_reduce` ahead of every other path; per-process-group
instances (vLLM builds several GroupCoordinators over the same ranks), init collective-safe
(boot 4 deadlocked ranks 1–3 when a singleton's second init raised on rank 0 inside an ordered
barrier loop — the user spotted one card loaded, three at 1 %); `hipGetLastError()` after
`hipErrorPeerAccessAlreadyEnabled` (sticky). Fast path capped at 64 KB (`VLLM_RDNA_AR_MAX_KB`;
prefill chunks are faster on RCCL).

Boot 5: **89.4 / 79.2 / 82.4 t/s**, 1024-tok 75.1, greedy checks pass. In decode the kernel
runs 25 µs median but ~60 µs mean in later boots: a barrier shows rank skew as its own time.

## T45 — int8 weight-only shadows of the dense projections (2026-08-29)

`gemv_i8_rdna2` + `rdna_dense_int8.py` (`VLLM_RDNA_DENSE_INT8=1`): per-channel symmetric
int8 built at load in `process_weights_after_loading`, fp16 kept for prefill; +1 GB/card.
Harness: lm_head 640 µs vs 1270 fp16 (bytes halve); cache-resident small shapes slower at
M=4 (unpack ALU). Boot 6 was flat (81.5 / 78.7 / 84.1) because **torch.compile froze the
`0 < n <= 8` decode/prefill branch at trace time** — only the untraced hyper-connection
linears took the shadow (176 int8 vs 418 fp16 GEMVs per step). Every Python-level decode
switch has this problem; the fix is an opaque custom op with the choice inside
(`rdna_ops.py: rdna_dense_gemm`). With that (boot 9b) fp16 GEMVs are 32/step (the <64-row
layers) and int8 3.5 ms/step vs 10.9 fp16 before.

## T46 — dispatch-count fusion: 82 → 98–101 t/s (2026-08-29)

Decode-only budget (boot 5): 41 ms/step, kernel-sum 26 ms, **2,711 kernels/step**, GPU busy
64 % — 12.7 ms/step of ≤100 µs inter-kernel bubbles (median 3.9 µs). An eager-mode boot with
`trace_attr.py` (kernel → CPU op → innermost model frame) ranked launch sites per step:
GemmaRMSNorm native 683 (inductor fuses these except inside opaque ops), rotary
`forward_static` 239 (indexer), MoE hook glue 192 (my own `expert_map[topk_ids]`/`.to()`),
GDN copies/cat 395, hc glue 3 × 109, shared-expert sigmoid·mul 105, PLE 48.

Done: MoE hook glue moved in-kernel (int64 ids, fp16/fp32 weights, `expert_map` applied in
the kernel); indexer RMSNorm and rope through `_C.rms_norm` / `_C.rotary_embedding` (one
launch each); the fused Triton qk-norm+rope+gate kernel enabled on ROCm (`num_stages=1`);
`rdna_fused_glue.cu` (fp16 or int8 weights): hc down+inject GEMV with silu(v/hc) epilogue,
hc up GEMV + sigmoid + gated mean in one kernel (5 → 2 launches per hyper-connection),
shared expert gate_up+silu·mul and down×sigmoid(gate) (6 → 2); all behind runtime-dispatch
custom ops (`rdna_hc_mix`, `rdna_shared_expert`). hipify traps: a launch after `else` inside
a macro body becomes `elsehipLaunchKernelGGL`; local classes cannot hold member templates.

| boot | change | 256-tok t/s | ms/step | kernels/step |
|---|---|---|---|---|
| 5 | one-shot all-reduce | 89.4 / 79.2 / 82.4 | 35 | 2,711 |
| 8 | + MoE glue in-kernel, indexer ops, fused qk-norm-rope | 83.5 / 85.1 / 82.8 | 32.4 | 2,205 |
| **9b** | + runtime-dispatch int8, fused hc, fused shared expert | **98.4 / 101.1 / 97.3** | **28.8** | **1,788** |

Boot 9b: 1024-tok 95.2; context 2.9k / 10.6k / 27k = 78 / 108 / 88 t/s (2.35 / 3.12 / 2.53
tok/step — acceptance, not context, moves it); greedy checks pass; 41–47 °C; journal clean.
Remaining per step: all-reduce 4.9 ms (95 × 52 µs, mostly rank skew), MoE int4 4.3, int8
GEMV 3.5, glue 2.5 (754 kernels), fused hc 2.4, attention 0.9, all-gather 0.8; idle ~9 ms
(1,788 bubbles). A 4-deep unroll of the MoE GEMV did not help (not MLP-bound; its per-block
LDS staging of x is the next suspect).
