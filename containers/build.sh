#!/usr/bin/env bash
# containers/build.sh — build, tag and (optionally) push the vllm-rdna2-qwen images.
#
#   ./containers/build.sh                 runtime image FROM the published base
#   ./containers/build.sh --base          first build the base: TheRock 7.14 (pinned public tarball) +
#                                         torch/triton/vision from source (2–3 h on 32 cores)
#   ROCM_DIR=/opt/rocm ./containers/build.sh --base
#                                         …but from a LOCAL TheRock install instead of the download
#                                         (symlinks resolved) — e.g. the exact 7.14.0rc3 we validated on
#   ./containers/build.sh --push          push what this run built (docker login ghcr.io first)
#
# Env: REGISTRY_REPO (ghcr.io/leapdragon/vllm-rdna2-qwen)  VERSION (<date>-g<sha>)  BASE_TAG  BASE_IMAGE
#      ROCM_DIR (/opt/rocm)  MAX_JOBS (24)
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; root="$(cd "$here/.." && pwd)"
REGISTRY_REPO="${REGISTRY_REPO:-ghcr.io/leapdragon/vllm-rdna2-qwen}"
BASE_TAG="${BASE_TAG:-therock7.14.1-torch2.12-gfx1030}"
BASE_IMAGE="${BASE_IMAGE:-${REGISTRY_REPO}-base:${BASE_TAG}}"
ROCM_DIR="${ROCM_DIR:-}"; [ -n "$ROCM_DIR" ] && ROCM_DIR="$(readlink -f "$ROCM_DIR")"
MAX_JOBS="${MAX_JOBS:-24}"
sha="$(git -C "$root" rev-parse --short HEAD 2>/dev/null || echo unknown)"
VERSION="${VERSION:-$(date -u +%Y%m%d)-g${sha}}"
VLLM_VERSION_OVERRIDE="${VLLM_VERSION_OVERRIDE:-0.28.1rc1+rdna2.g${sha}}"
do_base=0; do_push=0
for a in "$@"; do case "$a" in --base) do_base=1;; --push) do_push=1;; -h|--help) sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 0;; *) echo "unknown arg $a" >&2; exit 2;; esac; done
export DOCKER_BUILDKIT=1
built=()
log() { printf '\n==> %s\n' "$*"; }
if [ "$do_base" = 1 ]; then
  ROCM_CTX=()
  if [ -n "$ROCM_DIR" ]; then
    [ -x "$ROCM_DIR/bin/hipcc" ] || { echo "ROCM_DIR=$ROCM_DIR has no bin/hipcc — point it at a TheRock 7.14 install" >&2; exit 2; }
    ROCM_CTX=(--build-context "rocm=$ROCM_DIR"); log "base $BASE_IMAGE from LOCAL TheRock at $ROCM_DIR (MAX_JOBS=$MAX_JOBS)"
  else
    log "base $BASE_IMAGE from the pinned public TheRock tarball (MAX_JOBS=$MAX_JOBS)"
  fi
  docker buildx build --load --progress=plain -f "$here/Dockerfile.base" --target final \
    "${ROCM_CTX[@]}" --build-arg "MAX_JOBS=$MAX_JOBS" \
    --label "org.opencontainers.image.revision=$sha" --label "org.opencontainers.image.created=$(date -u +%FT%TZ)" \
    -t "$BASE_IMAGE" "$here"
  built+=("$BASE_IMAGE")
fi
log "runtime $REGISTRY_REPO:$VERSION FROM $BASE_IMAGE (vllm version $VLLM_VERSION_OVERRIDE)"
docker buildx build --load --progress=plain -f "$here/Dockerfile" \
  --build-arg "BASE_IMAGE=$BASE_IMAGE" --build-arg "MAX_JOBS=$MAX_JOBS" \
  --build-arg "VLLM_VERSION_OVERRIDE=$VLLM_VERSION_OVERRIDE" \
  --build-arg "BUILD_DATE=$(date -u +%FT%TZ)" --build-arg "VCS_REF=$sha" --build-arg "IMAGE_VERSION=$VERSION" \
  -t "$REGISTRY_REPO:$VERSION" -t "$REGISTRY_REPO:latest" "$root"
built+=("$REGISTRY_REPO:$VERSION" "$REGISTRY_REPO:latest")
if [ "$do_push" = 1 ]; then for t in "${built[@]}"; do log "push $t"; docker push "$t"; done; fi
printf '\nbuilt:\n'; printf '  %s\n' "${built[@]}"
