# Environment variables

Every environment variable this fork adds or depends on, what it does, and what its default is.
Section numbers (§N) point at [`CHANGES.md`](CHANGES.md), which has the measurements and the history
behind each one. The configuration we actually run is in [`PRODUCTION.md`](../../PRODUCTION.md).

**How they are read.** All of these are read once, when the process (or the extension library) starts.
Changing one needs a server restart. Boolean switches are on when set to `1` and off for anything else,
unless noted. A variable that is unset takes the default shown.

**Two kinds of setting.** [`tools/rdna2/serve-qwen38-flash-next.sh`](../../tools/rdna2/serve-qwen38-flash-next.sh)
takes short *launcher knobs* (`GPUUTIL`, `MTP`, …) and turns them into vLLM arguments and variables.
Everything named `VLLM_RDNA_*`, `VLLM_PLE_*` or `PLE_OFFLOAD_*` is read by the fork's code directly and
passes through the launcher unchanged, so it can be set in the environment of the serve script.

**A note on the torch.compile cache.** Its key does not cover this fork's custom operators or most of the
switches below. After changing a switch that alters which kernels run (the int8 shadows, W8A8, W4A8),
either keep `COMPILE_CACHE_OFF=1` (the serve script's default) or clear the compile cache.

---

## 1. Launcher knobs (serve script)

| Variable | Default | Meaning |
|---|---|---|
| `MODEL` | required | Directory of the AWQ-W4A16 backbone (checkpoint shards 2–5 plus `model_mtp.safetensors`). |
| `PLE_INT4` | required unless `PLE_DISK_DIR` | Directory of the quantised n-gram table sidecar (int4 or fp8, 128 shards + `META.json`). Exported as `VLLM_PLE_QUANT_DIR`. |
| `PLE_DISK_DIR` | unset | Alternative to `PLE_INT4`: a writable directory for the unquantised bf16 n-gram table (~95 GiB). The first boot copies the table out of checkpoint shard 1; later boots memory-map the file. Exported as `VLLM_PLE_DISK_OFFLOAD_DIR`. |
| `GPUS` | `1,2,3,4` | ROCr device IDs to serve on (`ROCR_VISIBLE_DEVICES`). Excludes device 0, typically a display card. |
| `PORT` | `8000` | HTTP port of the OpenAI-compatible API. |
| `MTP` | `3` | Speculative tokens from the model's multi-token-prediction head. `0` disables speculation. |
| `GPUUTIL` | `0.90` | `--gpu-memory-utilization`: the fraction of each card's VRAM vLLM may use. KV cache fills whatever the model and activations leave. With `DENSE_INT8_ONLY=1`, stay at 0.93 or below. |
| `MAXLEN` | `131072` | `--max-model-len`. The model supports 262,144. |
| `DENSE_INT8` | `1` | Sets `VLLM_RDNA_DENSE_INT8` (§3 below). |
| `DENSE_INT8_ONLY` | `0` | Sets `VLLM_RDNA_DENSE_INT8_ONLY` (§3). Changes the compiled graph: use a separate `VLLM_CACHE_ROOT`. |
| `COMPILE_CACHE_OFF` | `1` | Sets `VLLM_DISABLE_COMPILE_CACHE`. `0` saves ~700 s per boot, but see the note above. |
| `P2P` | `SYS` | Sets `NCCL_P2P_LEVEL`. `SYS` allows direct card-to-card transfers between all cards, including across CPU dies (+8 % prefill vs `PXB`). |
| `CG_SIZES` | `1 2 4 8 16 32 64 128 256` | Batch sizes captured as CUDA graphs. Covers decode and short prefills; empty = vLLM's default list. |
| `EAGER` | unset | Any value: `--enforce-eager` (no CUDA graphs). For debugging only; much slower decode. |
| `PROFILE` | unset | Any value: enable the torch profiler, writing traces to `TRACES`. |
| `TRACES` | `./logs/traces` | Output directory for profiler traces. |
| `TOOLS` | unset | `1`: OpenAI tool calling with `tool_choice: auto`, for agentic clients. |
| `TOOL_PARSER` | `qwen3_coder` | Tool-call parser: the model emits Qwen3-Coder-style `<function=…>` XML. |
| `REASONING_PARSER` | `qwen3` | Moves `<think>` blocks into `reasoning_content`. |
| `CHAT_KWARGS` | unset | JSON merged into every request's `chat_template_kwargs`; request values win. The template understands `enable_thinking`, `preserve_thinking` and `reasoning_effort` (`xhigh`, `medium`, `low`). |
| `VISION` | `0` | `1`: load the vision tower and accept images on the chat API. The tower is replicated on every rank, ~0.9 GB less KV per card. |
| `MM_LIMIT` | `{"image": 4, "video": 0}` | `--limit-mm-per-prompt`. |
| `MM_PROCESSOR_KWARGS` | `{"max_pixels": 1638400}` | Images are resized to at most ~1280×1280 before the encoder. The vision attention on gfx1030 materialises N² memory, so large images must be bounded. |
| `MM_ELIDE` | `1` | When a conversation holds more images than `MM_LIMIT`, replace the oldest with a text marker instead of failing the request with HTTP 400. |
| `EXTRA_ARGS` | unset | Extra `vllm serve` arguments, appended verbatim (e.g. `--prefix-cache-retention-interval 6272`). |
| `TUNEOP_TUNING` | `0` | `1`: a TunableOp **tuning** boot that autotunes every new GEMM shape and writes rows. Offline only, with the cards capped low: tuning runs all cards flat out and perturbs outputs. Production is lookup-only. |
| `TUNEOP_FILE` | unset | Use this TunableOp results file instead of the per-rocBLAS-build directory under `tunableop/`. |

---

## 2. All-reduce (tensor-parallel communication)

Every transformer layer ends in all-reduces across the four cards. The fork routes each one by size: small
decode messages to a custom one-shot kernel, large prefill messages optionally to an int8-compressed
path, and everything else to RCCL.

### One-shot all-reduce for small messages (§6, §15, §18)

| Variable | Default | Meaning |
|---|---|---|
| `VLLM_RDNA_AR` | `1` | Use the fork's one-shot all-reduce for messages up to `VLLM_RDNA_AR_MAX_KB`. `0` sends everything to RCCL: always works, about 26 % slower single-stream decode. |
| `VLLM_RDNA_AR_MODE` | `p2p` (serve script: `wide`) | How the one-shot kernel moves data. See below. The serve script defaults to `wide` (since 2026-09-27). |
| `VLLM_RDNA_AR_MAX_KB` | `64` | Largest message (KB) taken by the one-shot kernel. Decode messages (batch × 2560 fp16) fit; larger ones go to RCCL, which is faster there. |
| `VLLM_RDNA_AR_BLOCKS` | `0` (auto) | Cap on thread blocks per launch. Fewer blocks mean fewer concurrent PCIe write streams, at some latency cost (20 KB: 16/4/1 blocks = 33/36/76 µs). |
| `VLLM_RDNA_AR_PACE` | `0` | Idle time inserted between strided stores, in units of ~64 GPU clocks (0–127). Spaces out bursts on the bus. `0` = off. |
| `VLLM_RDNA_AR_SPIN_CAP` | `2000000` | How many polls a card waits for a peer before giving up (~2 s). On timeout the step fails with a diagnosis naming the missing peer, and a marker file (`$VLLM_CACHE_ROOT/rdna_ar_wedged`) makes later boots use RCCL until it is deleted. |

**`VLLM_RDNA_AR_MODE` values:**

- **`p2p`** — each card writes its contribution straight into the other cards' memory, one fp16 element
  (2 bytes) per thread, then raises a flag in each peer's memory; each card waits on its own flags.
  Fastest in isolation, but it produces many tiny PCIe writes from all four cards at once.
- **`host`** — contributions go through one shared pinned buffer in host memory, as 8-byte words carrying
  4 bytes of data plus a sequence tag. No card writes into another card. Waiting means reading host
  memory across the bus.
- **`wide`** — the `p2p` protocol with controlled write size: each thread stores 16 bytes, so each group
  of 8 threads fills one aligned 128-byte line. Writes only: nothing is read across the bus, one flag per
  peer, waiting on local memory. Takes only messages that are a multiple of 16 bytes from 16-byte-aligned
  tensors (decode messages always are); others go to RCCL. Per call it is 7–20 % slower than `p2p`,
  with the same decode speed in the server.

### int8-compressed prefill all-reduce (§14, §17)

| Variable | Default | Meaning |
|---|---|---|
| `VLLM_RDNA_AR_Q8` | `0` | `1`: large prefill all-reduces are sent as int8 with an fp16 scale per group, roughly halving the bytes on the bus. Two-shot: exchange shards, sum in fp32 and requantise, all-gather. Relative error ~0.8 % vs fp16. Prefill +2–5 % alone, +7–9 % together with `VLLM_RDNA_DENSE_W8A8`. |
| `VLLM_RDNA_AR_Q8_STAGGER` | `1` | Exchange shards in rounds: in round k each card sends to rank+k and receives from rank−k, so each card has one inbound and one outbound transfer at a time. Bit-identical to the grouped exchange (`0`, all cards to all cards at once) and slightly faster. |
| `VLLM_RDNA_AR_Q8_MIN_KB` | `1024` | Smallest message (KB) that takes the int8 path. |
| `VLLM_RDNA_AR_Q8_GROUP` | `64` | Elements per quantisation group; each group gets its own fp16 scale. |
| `VLLM_RDNA_AR_Q8_DIMS` | `2560` | Comma-separated last-dimension sizes allowed on the int8 path. The default is the language model's hidden size; other tensors, e.g. the vision tower's, keep full precision. |

Only eager (not graph-captured), contiguous fp16 tensors qualify. Captured prefill graphs and small
batches keep RCCL.

---

## 3. Dense projections (§7, §13, §16)

The routed experts are int4. The remaining large layers (linear-attention and sparse-attention
projections, router, shared expert, hyper-connection mixers, output head) ship in fp16.

| Variable | Default | Meaning |
|---|---|---|
| `VLLM_RDNA_DENSE_INT8` | `0` (serve script: `1`) | Keep an int8 weight-only copy of each fp16 projection, with one symmetric scale per output channel, and use it for decode-sized calls (≤ 8 tokens). This halves the bytes streamed per decode step. Prefill keeps using fp16. |
| `VLLM_RDNA_DENSE_INT8_MIN_ROWS` | `64` | Skip layers with fewer output rows than this. |
| `VLLM_RDNA_DENSE_INT8_ONLY` | `0` | Also **free the fp16 weight** once its int8 copy exists, so int8 is the only resident copy: about 2–3 GB per card back to the KV cache. Prefill then dequantises the int8 weight on the fly for each GEMM (~0.3 % of a 2048-token chunk). Quality stays within noise of fp16. |
| `VLLM_RDNA_DENSE_W8A8` | `0` | Prefill GEMMs for selected projections run as int8 × int8, quantising activations per token on the fly and using the int8 weight copies. Needs `VLLM_RDNA_DENSE_INT8=1`. Prefill +4–7 % alone. |
| `VLLM_RDNA_DENSE_W8A8_MIN_TOKENS` | `64` | Smallest batch (tokens) that takes the W8A8 path. |
| `VLLM_RDNA_DENSE_W8A8_SHAPES` | `4096x2560,3584x2560,2560x1536` | Weight shapes (N×K) that use W8A8. The hyper-connection mixers (`336x10240`, `320x10240`) are opt-in because they cost more accuracy. |
| `VLLM_RDNA_DENSE_W8A8_HC_GROUP` | `2560` | Activation-scale group for the hyper-connection shapes: one scale per 2560-wide residual branch instead of per token. The four branches differ in magnitude. |
| `VLLM_RDNA_DENSE_W8A8_TILE` | unset | Override the W8A8 kernel tile as `BM,BN,BK,GM,warps,stages`. For tuning sweeps. |
| `VLLM_RDNA_GEMM_MBUCKET` | `0` (off) | Pad the token count M of rocBLAS prefill GEMMs above 256 up to a multiple of this value (e.g. `128`). TunableOp rows are exact-M, and prefill steps often end at odd sizes; bucketing makes every M hit a tuned row. +2.5–6 % on short and mid prompts. |

---

## 4. MoE experts

| Variable | Default | Meaning |
|---|---|---|
| `VLLM_RDNA_MOE_W4A8` | `0` | Prefill MoE kernel with int8 activations × int4 experts, using the chip's packed int8 dot-product instructions (`v_dot4`). About 2× on the MoE GEMMs, +27 % prefill. |
| `VLLM_RDNA_MOE_W4A8_MIN_TOKENS` | `64` | Smallest batch (tokens) that takes the W4A8 kernel. Decode keeps the int4 GEMV below. |
| `VLLM_RDNA_MOE_W4A8_BN` / `_BK` / `_WARPS` / `_STAGES` | `128` / `16` / `8` / `2` | Kernel tile. Larger K blocks spill registers on this chip and lose up to 5×. |
| `VLLM_ROCM_MOE_SKINNY` | `1` | Decode MoE uses a wave-per-row int4 GEMV written for gfx1030 instead of the generic Triton kernel. |

---

## 5. Fused decode kernels (§8)

| Variable | Default | Meaning |
|---|---|---|
| `VLLM_RDNA_FUSED_HC` | `1` | The hyper-connection mix (the model's four residual branches) runs as one fused kernel in decode instead of several small ones. |
| `VLLM_RDNA_FUSED_SE` | `1` | The shared-expert MLP runs as one fused decode kernel. |
| `VLLM_RDNA_MOE_V2` | `1` | The routed-expert decode kernels (batches up to 16) use the v2 pair: 16-byte weight loads, no shared-memory staging, only the experts held on this rank do work. `0` restores v1. |
| `VLLM_RDNA_MOE_V3` | `1` | At decode batches of 1–2, the shared expert is computed inside the routed-expert kernel pair instead of as its own launches. Larger batches use v2 + the separate shared expert, which is as fast there. |

All of these cut kernel launches or wasted work per decode step; launch overhead was a major cost on this chip.

**int8 KV cache (`--kv-cache-dtype int8_per_token_head`).** The 12 full-attention (QSA) layers can keep their main
K/V cache as int8 with one fp32 scale per token and head (absmax/127). This about doubles the KV-cache token
capacity at unchanged decode and prefill speed: on 4 × V620 at `GPUUTIL=0.93`, 482k → 939k tokens. Needle retrieval
at 32k/64k was unchanged (10/10), and greedy output stays within near-tie noise of the fp16 cache. The QSA indexer
caches stay fp16; they are about 3 % of the per-token bytes. gfx1030 has no fp8 hardware, so use int8 rather than
the fp8 cache types. Changing the flag changes the compile cache key, so expect one cold boot.

---

## 6. Sparse attention (QSA indexer) (§8d, §11, §16)

| Variable | Default | Meaning |
|---|---|---|
| `VLLM_RDNA_QSA_BOUND_MIN_TOKENS` | `512` | For eager prefill batches of at least this many tokens, the indexer scores only the compressed keys that can actually be visible, not the page table's full capacity. Smaller and graph-captured batches keep full capacity, so a captured graph never bakes in a column count. |
| `VLLM_RDNA_QSA_DOT` | `0` | `1`: use the blocked `tl.dot` scoring kernel (roughly parity). Tile: `VLLM_RDNA_QSA_DOT_BLOCK_M` / `_BLOCK_N` / `_WARPS` / `_STAGES` (16 / 32 / 8 / 1). |
| `VLLM_RDNA_QSA_MQA_BLOCK_N` / `_MQA_WARPS` / `_MQA_STAGES` | `128` / `8` / Triton default | Tile of the indexer's scoring kernel. |
| `VLLM_RDNA_QSA_BLOCK_N` / `_SPLITS` / `_WARPS` / `_STAGES` | `32` / `1` / `4` / `1` on ROCm | Tile of the sparse-attention prefill kernel. `num_stages=1` matters on this chip: Triton's AMD default of 2 halves occupancy. |
| `VLLM_RDNA_QSA_DESPILL` | `0` | Use a register-lean variant of the sparse prefill kernel: 1.17× in isolation, no end-to-end gain. |
| `VLLM_RDNA_QSA_DUMP` | unset | Debug: a directory to dump the indexer's inputs and outputs on a few large calls. Synchronises the GPU; never set it in production. |

Any `VLLM_RDNA_QSA_*` override is logged as a warning at startup.

---

## 7. Scheduler (§12)

| Variable | Default | Meaning |
|---|---|---|
| `VLLM_RDNA_MAMBA_RETENTION_STOPS` | `1` | End prefill steps exactly at the ends of the linear-attention state blocks (784 tokens) that the prefix cache retains. The saved state is only exact when a step ends on a block boundary. Without this, a follow-up turn on a long conversation could re-prefill tens of thousands of tokens (e.g. 31 s instead of seconds to first token). |

---

## 7b. CPU KV-cache offload (§19)

| Setting | Default | Meaning |
|---|---|---|
| `--kv-offloading-size N` (vLLM argument, e.g. via `EXTRA_ARGS`) | off | Keep N GiB (total across all TP ranks) of pinned CPU RAM as a second KV-cache tier. Prefixes evicted from the GPU reload from RAM instead of being recomputed. |
| `VLLM_USE_SIMPLE_KV_OFFLOAD` | `0` | **Set to `1` with this model.** Selects `SimpleCPUOffloadConnector`, which reuses vLLM's prefix-cache logic for every cache group. The default `OffloadingConnector` refuses to start on the QSA compressor ring (`tokens_per_block=4 not divisible by tokens_per_hash=784`). |
| `VLLM_RDNA_OFFLOAD_TIMING` | `0` | Diagnostic: log each offload copy's duration, and every worker step slower than `VLLM_RDNA_OFFLOAD_TIMING_SLOW_MS` (default 500), with whether a copy overlapped it. |
| `--kv-transfer-config '{"kv_connector_extra_config":{"lazy_offload":true}}'` | eager | Lazy mode: copy a block to RAM only when it is about to leave the GPU, so the two tiers hold different prefixes (resumable ≈ GPU + RAM instead of RAM alone). Works on this model only with the fork's fixes (§19b). |
| `VLLM_RDNA_OFFLOAD_LAZY_TARGET` | computed (24 here) | Lazy mode: GPU blocks at the eviction end kept copied to RAM. 96 covers the allocation burst of reloading a ~48k-token prompt; larger reloads may lose some old tails. |
| `VLLM_RDNA_OFFLOAD_LAZY_CURSOR` | `0` | `1` restores upstream's lazy walk (§19b fixes off). For comparison only. |
| `VLLM_RDNA_OFFLOAD_LAZY_RESCUE` | `1` | Lazy mode: copy a cached block the GPU allocator evicts before it was offloaded (at the start of the step), and copy a block's prefix ancestors and Mamba states with it so RAM hits are contiguous (§19b items 6–7). |
| `VLLM_RDNA_OFFLOAD_PIN_CHUNK_MB` | `128` | Register the RAM tier with the GPU driver in block-aligned chunks of about this size, so a page migration revalidates one chunk instead of the whole tier. `0` = one registration (upstream). |
| `VLLM_RDNA_OFFLOAD_THP` | `1` | Back the RAM tier with transparent huge pages (`MADV_HUGEPAGE`), which memory compaction does not move. |
| `VLLM_RDNA_OFFLOAD_MLOCK` | `1` | `mlock` the RAM tier and `mlockall(ONFAULT)` the GPU worker so compaction cannot migrate pages the GPU has mapped (each migration stalls all of that rank's GPU queues). Needs `RLIMIT_MEMLOCK` ≥ tier per rank + ~3 GB and `vm.compact_unevictable_allowed=0`; logs a warning and continues otherwise. |
| `VLLM_RDNA_OFFLOAD_DEBUG` | `0` | Diagnostic: per cache group, lazy-walk counts every 30 s, and for a long request with no RAM hit, which group's blocks are missing. |

---

## 8. N-gram table offload (PLE) (§3, §8a, §8c)

The model's per-layer n-gram embedding table has about 51 billion rows, far too large for the cards. A
CPU worker process serves lookups from host memory, and the GPUs wait for its results each step.

| Variable | Default | Meaning |
|---|---|---|
| `VLLM_PLE_CPU_OFFLOAD` | `0` (serve script: `1`) | Serve the n-gram table from the CPU offload worker. |
| `VLLM_PLE_QUANT_DIR` | unset | Directory of the quantised table sidecar (int4 or fp8). |
| `VLLM_PLE_DISK_OFFLOAD_DIR` | unset | Directory for the memory-mapped bf16 table (alternative to the sidecar). The first boot streams it out of the checkpoint; later boots map it instantly. |
| `VLLM_PLE_OFFLOAD_READY_TIMEOUT` | `600` (serve script: `3600`) | Seconds to wait for the worker to load the table and for all ranks to register. |
| `PLE_OFFLOAD_DOORBELL` | `1` | Rank 0 posts each step's request into a shared memory page that the worker polls, instead of sending a ZMQ message. Saves ~170 µs per step. `0` restores ZMQ. |
| `PLE_OFFLOAD_PREFAULT` | `1` | At startup, touch the table's pages in the background so the first requests don't stall on page faults. |
| `PLE_OFFLOAD_ANON` | `0` | `1`: copy the quantized table into the worker's own (anonymous) memory at start and drop the file's page-cache copy. The kernel can then no longer evict table pages under memory pressure, which otherwise causes multi-second lookups and uneven TP ranks. **Required alongside `--kv-offloading-size`** (§19). Adds ~1–2 min to boot. Needs enough RAM for the whole table. |
| `PLE_OFFLOAD_DEBUG_HOPS` | `0` | Test hook: timestamp each hop of the per-step round trip and log per-hop averages. |
| `PLE_OFFLOAD_DEBUG_TRACE` | unset | Test hook: append one line per request (sequence, sizes, input and result hashes) to this file, so two runs can be compared. |
| `PLE_OFFLOAD_DEBUG_DELAY_MS` | `0` | Test hook: add an artificial delay to every worker request. |
| `PLE_OFFLOAD_FUSED_CHECK` | `0` | Test hook: re-run every fused lookup with the reference path and compare. Slow. |

---

## 9. Test-tool variables

| Variable | Used by | Meaning |
|---|---|---|
| `RDNA_AR_TEST_SO` | `tools/rdna2/ar_ops_test.py` | Path to a freshly built `_rocm_C` library to test without installing it (e.g. `build_rocm/_rocm_C.abi3.so`). |
| `RDNA_AR_TEST_MAXB` | `tools/rdna2/ar_ops_test.py` | Staging size (bytes) for the all-reduce under test. |

---

## 10. Standard variables the serve script sets, and why

Not added by this fork, but part of the working configuration.

| Variable | Value | Why |
|---|---|---|
| `HSA_NO_SCRATCH_RECLAIM` | `1` | Keeps the ROCm runtime from reclaiming kernel scratch memory mid-run. Part of the multi-card stability setup. |
| `NCCL_P2P_LEVEL` | `SYS` (from `P2P`) | RCCL may use direct card-to-card transfers between all cards. |
| `NCCL_GRAPH_MIXING_SUPPORT` | `1` | Declares that one RCCL communicator is used both in CUDA graphs and eagerly (graph usage mode 2). Without it, RCCL warns on every captured collective that the mix "can lead to hangs". No measurable cost. |
| `NCCL_PROTO` | `Simple` (since 2026-09-27; set it empty to let RCCL choose) | Forces RCCL onto its large-chunk protocol instead of "LL", which moves data as 8-byte stores each carrying a flag. Fewer, larger bus transactions for mid-sized messages. |
| `VLLM_ROCM_USE_AITER` | `0` | AITER is CDNA-only; it has nothing for gfx1030. |
| `TORCH_BLAS_PREFER_HIPBLASLT` | `0` | Use rocBLAS: hipBLASLt has no tuned kernels for gfx1030. |
| `FLASH_ATTENTION_TRITON_AMD_ENABLE` | `TRUE` | Use the Triton flash-attention backend on AMD. |
| `PYTORCH_TUNABLEOP_*` | lookup-only | Tuned GEMM rows are loaded from `tunableop/rocblas-<build id>/` and never tuned inside a serving process. The rows are specific to the rocBLAS build, so a different build runs untuned until it is tuned once. |
| `VLLM_DISABLE_COMPILE_CACHE` | `1` (from `COMPILE_CACHE_OFF`) | See the compile-cache note at the top. |
