#!/bin/bash
# entrypoint of the vllm-rdna2-qwen image.
#
#   docker run … IMAGE                      serve Qwen3.8-Flash-Next with the fork's tuned configuration
#                                           (tools/rdna2/serve-qwen38-flash-next.sh — every knob it documents
#                                           works here as -e: MTP, MAXLEN, GPUUTIL, VISION, CHAT_KWARGS, TOOLS,
#                                           GPUS, PORT, EXTRA_ARGS, DRYRUN=1 …)
#   docker run … IMAGE <any command>        passthrough (e.g. bash, python3, tools/rdna2/system-report.sh)
#
# Weights are mounted, not shipped: -v <models-dir>:/models where <models-dir> holds
#   qwen38-flash-next/           the AWQ-W4A16 backbone (shards 2-5 + model_mtp.safetensors; README §5)
#   qwen38-flash-next-ple/ples_int4/   the int4 n-gram sidecar (128 shards + META.json)
# Override with -e MODEL=… -e PLE_INT4=…. Persist /compile-cache (torch.compile: cold ~13 min, warm ~3).
set -euo pipefail
cd /app/vllm
if [ $# -gt 0 ] && [ "$1" != "serve" ]; then
  exec "$@"
fi
[ "${1:-}" = "serve" ] && shift
export MODEL="${MODEL:-/models/qwen38-flash-next}"
export PLE_INT4="${PLE_INT4:-/models/qwen38-flash-next-ple/ples_int4}"
for d in "$MODEL" "$PLE_INT4"; do
  [ -d "$d" ] || { echo "entrypoint: $d is not a directory — mount your models dir at /models (see containers/README.md)" >&2; exit 2; }
done
[ -f "$PLE_INT4/META.json" ] || { echo "entrypoint: $PLE_INT4/META.json missing — not the ples_int4 sidecar" >&2; exit 2; }
# Container defaults that differ from the host script: the compile cache is ON when /compile-cache is
# mounted (the host default is off for development), and the device list is whatever is visible.
if [ -d /compile-cache ]; then
  export VLLM_CACHE_ROOT=/compile-cache
  export COMPILE_CACHE_OFF="${COMPILE_CACHE_OFF:-0}"
fi
[ -d /triton-cache ] && export TRITON_CACHE_DIR=/triton-cache
export GPUS="${GPUS:-${ROCR_VISIBLE_DEVICES:-0,1,2,3}}"
export TRACES="${TRACES:-/tmp/traces}"
[ $# -gt 0 ] && export EXTRA_ARGS="${EXTRA_ARGS:-} $*"
exec tools/rdna2/serve-qwen38-flash-next.sh
