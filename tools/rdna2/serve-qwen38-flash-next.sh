#!/usr/bin/env bash
# Serve Qwen3.8-Flash-Next on 4x Radeon PRO V620 (gfx1030) with this fork, on the host
# (no container), against TheRock ROCm 7.14. This file *is* the configuration that produced
# ~100 t/s single-stream decode (docs/rdna2/RESULTS.md, T46). Every non-obvious knob is
# explained in docs/rdna2/CHANGES.md section 4.
#
#   MODEL=models/qwen38-flash-next PLE_INT4=models/qwen38-flash-next-ple/ples_int4 \
#     tools/rdna2/serve-qwen38-flash-next.sh
#
# Knobs (env): GPUS (ROCR device ids, default 1,2,3,4), PORT (8000), MTP (3), GPUUTIL (0.90),
#   DENSE_INT8 (1), EAGER (unset), PROFILE (unset), TRACES (dir for torch-profiler traces),
#   COMPILE_CACHE_OFF (1), P2P (SYS; NCCL_P2P_LEVEL, +8% prefill), CG_SIZES (piecewise CUDA-graph
#   capture sizes, default '1 2 4 8 16 32 64 128 256'; empty = vLLM default), MAXLEN (131072; the model allows 262144 and the KV pool
#   at GPUUTIL 0.86 held ~177-197k tokens, i.e. 1.35-1.5 concurrent 128k requests), EXTRA_ARGS,
#   TOOLS (1: OpenAI tool calling with tool_choice "auto" for agentic clients such as Kilocode /
#   Cline / Roo; the model's chat template emits Qwen3-Coder-style <function=…><parameter=…> XML,
#   parsed by vLLM's "qwen3_coder" parser; <think> blocks go to reasoning_content via the "qwen3"
#   reasoning parser), TOOL_PARSER (qwen3_coder), REASONING_PARSER (qwen3),
#   CHAT_KWARGS (JSON merged into every request's chat_template_kwargs, request values win;
#   this template understands enable_thinking, preserve_thinking and reasoning_effort =
#   xhigh|medium|low, e.g. CHAT_KWARGS='{"preserve_thinking": true, "reasoning_effort": "medium"}'),
#   VISION (0: text only -- the 0.9 GB Qwen3-VL-style vision tower is not loaded, --language-model-only;
#   1: images accepted on the chat API, tower replicated on every rank, ~0.9 GB/card less KV;
#   the ViT runs Torch SDPA attention on gfx1030), MM_LIMIT (JSON for --limit-mm-per-prompt,
#   default '{"image": 4, "video": 0}'), MM_PROCESSOR_KWARGS (JSON for --mm-processor-kwargs,
#   default '{"max_pixels": 1638400}' = images resized to <= 1280x1280 before the encoder: the
#   processor's own ceiling is 16 megapixels, and on gfx1030 the ViT's attention is SDPA's math
#   path, which materialises N^2 -- the memory profiler's 16 MP dummy image asked for 64 GiB),
#   MM_ELIDE (1: when a conversation has accumulated more than the image limit, the OLDEST
#   images are replaced with a text marker and the newest MM_LIMIT are kept -- agent platforms
#   cannot rewrite past turns, so the strict HTTP 400 wedges them; 0: stock vLLM 400).
set -euo pipefail

: "${MODEL:?set MODEL to the AWQ-W4A16 backbone directory (shards 2-5 + model_mtp.safetensors)}"
: "${PLE_INT4:?set PLE_INT4 to the ples_int4 sidecar directory (128 shards + META.json)}"
GPUS="${GPUS:-1,2,3,4}"
PORT="${PORT:-8000}"
MTP="${MTP:-3}"
GPUUTIL="${GPUUTIL:-0.90}"
MAXLEN="${MAXLEN:-131072}"
TRACES="${TRACES:-$PWD/logs/traces}"
mkdir -p "$TRACES"

# --- platform -----------------------------------------------------------------------------
export ROCR_VISIBLE_DEVICES="$GPUS"
export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
# T41 stability stack for flat TP=4 on PCIe (with the kernel line in docs/rdna2/README.md #1)
export HSA_NO_SCRATCH_RECLAIM=1
# RCCL P2P level. SYS (host-staged collectives) measured FASTER than PXB here: prefill
# 738/735 t/s at 3.3k and 760 at 30k vs 682/679/703 on PXB (+8%), decode unchanged at
# 61.5-61.9 t/s, validate PASS (2026-09-04, two independent boots, cap 220 W). Cross-die
# P2P on this 2-die X399 traverses the weak inter-die fabric; host staging avoids it.
# P2P=PXB restores the old behaviour.
export NCCL_P2P_LEVEL="${P2P:-SYS}"
# nothing from the CDNA world exists on gfx1030
export VLLM_ROCM_USE_AITER=0
export TORCH_BLAS_PREFER_HIPBLASLT=0
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
# TunableOp: LOOKUP-ONLY in production; never autotune inside a serving process.
# Tuned rows are worth +6% prefill at 3.3k and +13% at 30k vs untuned on gfx1030 (2026-09-04),
# decode unchanged. Offline retune only: TUNEOP_TUNING=1 with the cards capped LOW and the
# workload paced -- tuning drives all four cards to full power with no idle between GEMM
# candidates, makes prefill bimodal and perturbs greedy output, so its numbers mean nothing.
#
# Rows are specific to the rocBLAS BUILD, not its version string: solution ids come from the
# Tensile library as built, and two builds with the same version string can offer different
# solution sets (TheRock 7.14.0rc3 vs the 7.14.1 tarball: ~190 of 290 rows differ). A row naming
# a solution the runtime lacks aborts the first GEMM with "Expected iter != ops_.end()". So rows
# live under tunableop/rocblas-<sha256[:12] of librocblas.so>/; no directory for the running
# build means lookup stays OFF (logged), and a TUNEOP_TUNING=1 boot writes into that build's
# directory. TUNEOP_FILE= overrides the location outright.
_SERVE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_REPO_ROOT="$(cd "$_SERVE_DIR/../.." && pwd)"
_ROCBLAS_LIB="$(readlink -f "${ROCM_PATH:-/opt/rocm}/lib/librocblas.so.5" 2>/dev/null || true)"
_ROCBLAS_ID="unknown"; [ -f "$_ROCBLAS_LIB" ] && _ROCBLAS_ID="$(sha256sum "$_ROCBLAS_LIB" | cut -c1-12)"
_TUNEOP_DIR="$_REPO_ROOT/tunableop/rocblas-$_ROCBLAS_ID"
export PYTORCH_TUNABLEOP_TUNING="${TUNEOP_TUNING:-0}"
export PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED=0
if [ -n "${TUNEOP_FILE:-}" ]; then
  export PYTORCH_TUNABLEOP_ENABLED="${PYTORCH_TUNABLEOP_ENABLED:-1}"
  export PYTORCH_TUNABLEOP_FILENAME="$TUNEOP_FILE"
elif [ "$PYTORCH_TUNABLEOP_TUNING" = 1 ]; then
  mkdir -p "$_TUNEOP_DIR"
  export PYTORCH_TUNABLEOP_ENABLED=1
  export PYTORCH_TUNABLEOP_FILENAME="$_TUNEOP_DIR/tunableop_results.csv"
  echo "TunableOp: TUNING boot -- rows for rocBLAS build $_ROCBLAS_ID will be written under $_TUNEOP_DIR"
elif [ -f "$_TUNEOP_DIR/tunableop_results0.csv" ]; then
  export PYTORCH_TUNABLEOP_ENABLED="${PYTORCH_TUNABLEOP_ENABLED:-1}"
  export PYTORCH_TUNABLEOP_FILENAME="$_TUNEOP_DIR/tunableop_results.csv"
else
  export PYTORCH_TUNABLEOP_ENABLED=0
  echo "TunableOp: no tuned rows for this rocBLAS build ($_ROCBLAS_ID, $_ROCBLAS_LIB) under $_REPO_ROOT/tunableop/ -- lookup disabled (dense GEMMs run untuned, ~6-13% slower prefill). Tune once with TUNEOP_TUNING=1; see docs/rdna2/CHANGES.md section 8d."
fi

# --- this fork's features ---------------------------------------------------------------
# n-gram table served from the int4 sidecar by a CPU worker process (CHANGES.md #3)
export VLLM_PLE_CPU_OFFLOAD=1
export VLLM_PLE_QUANT_DIR="$PLE_INT4"
export VLLM_PLE_OFFLOAD_READY_TIMEOUT=3600
# int8 shadows of the dense fp16 projections for decode (CHANGES.md #7)
export VLLM_RDNA_DENSE_INT8="${DENSE_INT8:-1}"
# one-shot P2P all-reduce (CHANGES.md #6); VLLM_RDNA_AR=0 falls back to RCCL
export VLLM_RDNA_AR="${VLLM_RDNA_AR:-1}"
# torch.compile cache key does not cover this fork's Python; keep it off (CHANGES.md #8)
export VLLM_DISABLE_COMPILE_CACHE="${COMPILE_CACHE_OFF:-1}"

SPEC=()
if [ -n "$MTP" ] && [ "$MTP" != "0" ]; then
  SPEC=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP}")
fi
CHATARGS=()
if [ -n "${CHAT_KWARGS:-}" ]; then
  CHATARGS=(--default-chat-template-kwargs "$CHAT_KWARGS")
fi
TOOLARGS=()
if [ "${TOOLS:-1}" = "1" ]; then
  TOOLARGS=(--enable-auto-tool-choice --tool-call-parser "${TOOL_PARSER:-qwen3_coder}")
  # REASONING_PARSER= (explicitly empty) turns OFF vLLM-side reasoning splitting: thinking
  # text then arrives inline in `content` instead of `reasoning_content`. Note that vLLM's
  # chat endpoint still strips the template's </think> terminator from the content stream
  # (measured 2026-09-04), so a client cannot split on it either way.
  # ${VAR-default}, NOT ${VAR:-default}: the ':-' form replaces an *empty* value with the
  # default, making disable impossible.
  if [ -n "${REASONING_PARSER-qwen3}" ]; then
    TOOLARGS+=(--reasoning-parser "${REASONING_PARSER-qwen3}")
  fi
fi
VISIONARGS=(--language-model-only --skip-mm-profiling)
if [ "${VISION:-0}" = "1" ]; then
  # Profiling stays on so the encoder's activations are accounted for before the KV pool
  # is sized; the limit keeps a prompt from carrying more than MM_LIMIT images.
  _MM_LIMIT_DEFAULT='{"image": 4, "video": 0}'   # never inline JSON in ${VAR:-...}: bash closes at the first '}'
  _MM_PROC_DEFAULT='{"max_pixels": 1638400}'
  VISIONARGS=(--limit-mm-per-prompt "${MM_LIMIT:-$_MM_LIMIT_DEFAULT}"
              --mm-processor-kwargs "${MM_PROCESSOR_KWARGS:-$_MM_PROC_DEFAULT}")
  export MM_ELIDE_OVER_LIMIT="${MM_ELIDE:-1}"
fi
PROF=()
# CUDA-graph capture sizes (2026-09-06). vLLM caps the default piecewise list at
# min(max_num_seqs * 2, 512) tokens -- with --max-num-seqs 4 that is 8, so every real prefill batch
# ran its compiled pieces eagerly (~3,500 launches for a 40-word prompt). Capturing up to 256 tokens
# takes short-prompt TTFT from 0.38-0.42 s to 0.25 s (-35%); decode and long prefill unchanged.
# Cost: graph memory 0.64 -> 1.31 GiB per card (KV pool 353k -> 309k tokens). Larger lists cost
# more (<=512: 262k tokens; <=2048: 221k) for no further TTFT gain on this workload; the
# cached-prefix turn is bounded by the partial-block recompute, not by launches.
# CG_SIZES= (empty) restores vLLM's default list.
CG_SIZES="${CG_SIZES-1 2 4 8 16 32 64 128 256}"
CGARGS=(); [ -n "$CG_SIZES" ] && CGARGS=(--cudagraph-capture-sizes $CG_SIZES)
if [ -n "${PROFILE:-}" ]; then
  PROF=(--profiler-config.profiler=torch --profiler-config.torch_profiler_dir="$TRACES")
fi

CMD=(python3 -m vllm.entrypoints.openai.api_server
  --model "$MODEL" --served-model-name qwen38-flash-next
  --dtype float16
  --tensor-parallel-size 4 --enable-expert-parallel
  --max-model-len "$MAXLEN" --gpu-memory-utilization "$GPUUTIL"
  --max-num-seqs 4 --max-num-batched-tokens 2048
  ${EAGER:+--enforce-eager}
  "${VISIONARGS[@]}"
  --enable-prefix-caching
  "${SPEC[@]}" "${TOOLARGS[@]}" "${CHATARGS[@]}" "${PROF[@]}" "${CGARGS[@]}" ${EXTRA_ARGS:-}
  --host 0.0.0.0 --port "$PORT")

if [ "${DRYRUN:-0}" = "1" ]; then
  # tools/rdna2/system-report.sh uses this: show the resolved environment and command, launch nothing
  echo "# environment this script exports:"
  env | grep -E '^(ROCR_|ROCM_|HSA_|NCCL_|VLLM_|TORCH_BLAS|FLASH_ATTENTION|PYTORCH_|MM_ELIDE)' | sort | sed 's/^/#   /'
  echo "# command:"
  printf '%q ' "${CMD[@]}"; echo
  exit 0
fi
exec "${CMD[@]}"
