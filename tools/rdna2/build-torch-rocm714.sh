#!/usr/bin/env bash
# Build PyTorch (ROCm/pytorch release/2.12) and Triton (ROCm/triton) from source for gfx1030
# against a TheRock ROCm 7.14 install, producing wheels. TheRock publishes no torch wheels for
# the gfx103X family, so this is the only path. Pins match this fork's docker/Dockerfile.rocm_base.
#
#   source ~/venvs/vllm-rdna2-qwen/bin/activate      # a Python 3.12 venv
#   tools/rdna2/build-torch-rocm714.sh                 # ~2-3 h on 32 cores
#
# Env: ROCM_PATH (/opt/rocm), SRC (~/src/rdna2), OUT (~/wheels/rdna2), MAX_JOBS (nproc),
#      PYTORCH_REF (6bbd260), TRITON_REF (f0b55c0), SKIP_TRITON (unset)
set -euo pipefail

ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
SRC="${SRC:-$HOME/src/rdna2}"
OUT="${OUT:-$HOME/wheels/rdna2}"
MAX_JOBS="${MAX_JOBS:-$(nproc)}"
PYTORCH_REF="${PYTORCH_REF:-6bbd260}"   # ROCm/pytorch release/2.12 as of 2026-08-02
TRITON_REF="${TRITON_REF:-f0b55c0}"     # ROCm/triton release/internal/3.7.x as of 2026-08-18
mkdir -p "$SRC" "$OUT"

[ -x "$ROCM_PATH/bin/hipcc" ] || { echo "no hipcc under $ROCM_PATH"; exit 1; }
python3 -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version' || { echo "use a Python 3.12 venv"; exit 1; }
python3 -m pip install -q --upgrade pip setuptools wheel ninja cmake pyyaml numpy typing_extensions requests packaging

export ROCM_PATH ROCM_HOME="$ROCM_PATH" HIP_PATH="$ROCM_PATH" MAX_JOBS
export PATH="$ROCM_PATH/bin:$PATH"
export PYTORCH_ROCM_ARCH=gfx1030
export USE_ROCM=1 USE_CUDA=0
# No aotriton in TheRock 7.14 and vLLM brings its own attention: skip torch's SDPA flash paths.
export USE_FLASH_ATTENTION=0 USE_MEM_EFF_ATTENTION=0
export BUILD_TEST=0 USE_DISTRIBUTED=1 USE_RCCL=1 USE_MPI=0 USE_KINETO=1 USE_ROCM_KERNEL_ASSERT=0
export CMAKE_PREFIX_PATH="$(python3 -c 'import sys; print(sys.prefix)'):$ROCM_PATH"

# ---- PyTorch ------------------------------------------------------------------------------
if [ ! -d "$SRC/pytorch" ]; then
  git clone https://github.com/ROCm/pytorch.git "$SRC/pytorch"
fi
cd "$SRC/pytorch"
git fetch -q origin
git checkout -q "$PYTORCH_REF"
git submodule sync -q && git submodule update -q --init --recursive
python3 -m pip install -q -r requirements.txt
# hipify the CUDA sources in-tree (idempotent)
python3 tools/amd_build/build_amd.py
python3 setup.py bdist_wheel --dist-dir "$OUT" 2>&1 | tee "$OUT/torch-build.log" | grep -E "error|Error|^-- |Building wheel|Finished" | tail -20
ls -la "$OUT"/torch-*.whl

# ---- Triton -------------------------------------------------------------------------------
if [ -z "${SKIP_TRITON:-}" ]; then
  if [ ! -d "$SRC/triton" ]; then
    git clone https://github.com/ROCm/triton.git "$SRC/triton"
  fi
  cd "$SRC/triton"
  git fetch -q origin
  git checkout -q "$TRITON_REF"
  export TRITON_BUILD_WITH_CCACHE=false TRITON_BUILD_WITH_CLANG_LLD=false
  python3 -m pip install -q -r python/requirements.txt 2>/dev/null || true
  python3 -m pip wheel --no-build-isolation --no-deps -w "$OUT" . 2>&1 | tee "$OUT/triton-build.log" | tail -5
  ls -la "$OUT"/triton-*.whl
fi

echo "wheels in $OUT:"; ls "$OUT"/*.whl
echo "install:  pip install $OUT/torch-*.whl $OUT/triton-*.whl"
