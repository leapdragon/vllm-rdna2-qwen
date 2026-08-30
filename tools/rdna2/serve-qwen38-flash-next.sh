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
#   COMPILE_CACHE_OFF (1), P2P (PXB), MAXLEN (131072; the model allows 262144 and the KV pool
#   at GPUUTIL 0.86 held ~177-197k tokens, i.e. 1.35-1.5 concurrent 128k requests), EXTRA_ARGS,
#   TOOLS (1: OpenAI tool calling with tool_choice "auto" for agentic clients such as Kilocode /
#   Cline / Roo; the model's chat template emits Qwen3-Coder-style <function=…><parameter=…> XML,
#   parsed by vLLM's "qwen3_coder" parser; <think> blocks go to reasoning_content via the "qwen3"
#   reasoning parser), TOOL_PARSER (qwen3_coder), REASONING_PARSER (qwen3),
#   CHAT_KWARGS (JSON merged into every request's chat_template_kwargs, request values win;
#   this template understands enable_thinking, preserve_thinking and reasoning_effort =
#   xhigh|medium|low, e.g. CHAT_KWARGS='{"preserve_thinking": true, "reasoning_effort": "medium"}').
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
export NCCL_P2P_LEVEL="${P2P:-PXB}"
# nothing from the CDNA world exists on gfx1030
export VLLM_ROCM_USE_AITER=0
export TORCH_BLAS_PREFER_HIPBLASLT=0
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
# TunableOp: lookup-only; never autotune inside a serving process on this chip
export PYTORCH_TUNABLEOP_ENABLED="${PYTORCH_TUNABLEOP_ENABLED:-0}"

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
  TOOLARGS=(--enable-auto-tool-choice --tool-call-parser "${TOOL_PARSER:-qwen3_coder}"
            --reasoning-parser "${REASONING_PARSER:-qwen3}")
fi
PROF=()
if [ -n "${PROFILE:-}" ]; then
  PROF=(--profiler-config.profiler=torch --profiler-config.torch_profiler_dir="$TRACES")
fi

exec python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name qwen38-flash-next \
  --dtype float16 \
  --tensor-parallel-size 4 --enable-expert-parallel \
  --max-model-len "$MAXLEN" --gpu-memory-utilization "$GPUUTIL" \
  --max-num-seqs 4 --max-num-batched-tokens 2048 \
  ${EAGER:+--enforce-eager} \
  --language-model-only --skip-mm-profiling \
  --enable-prefix-caching \
  "${SPEC[@]}" "${TOOLARGS[@]}" "${CHATARGS[@]}" "${PROF[@]}" ${EXTRA_ARGS:-} \
  --host 0.0.0.0 --port "$PORT"
