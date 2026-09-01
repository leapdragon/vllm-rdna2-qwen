# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------------------------
# vllm-rdna2-qwen — runtime image: this fork built on the TheRock-7.14 / torch-2.12 base
# (containers/Dockerfile.base), as docs/rdna2/README.md §4 prescribes, dependency pins captured
# from the validated host venv (containers/constraints-runtime.txt).
#   Build context: repository root.   ./containers/build.sh
#   Run:           containers/README.md — weights are mounted at /models, never shipped.
# ---------------------------------------------------------------------------------------------
ARG BASE_IMAGE=ghcr.io/leapdragon/vllm-rdna2-qwen-base:therock7.14.0rc3-torch2.12-gfx1030
FROM ${BASE_IMAGE}
ARG BASE_IMAGE
ARG MAX_JOBS=24
ARG VLLM_VERSION_OVERRIDE=0.28.1rc1+rdna2
ENV DEBIAN_FRONTEND=noninteractive PIP_ROOT_USER_ACTION=ignore PIP_NO_CACHE_DIR=1

COPY . /app/vllm
WORKDIR /app/vllm

# README §4 verbatim: build deps WITHOUT torch/triton (requirements/build/rocm.txt would drag in a
# PyPI torch with its own bundled ROCm — the mixed-ROCm trap), then the extensions, then runtime deps.
# Versions come from the constraints file (the validated venv), not from README §4's ranges — the
# venv itself drifted past some of them (cmake 4.x) and is what the numbers were measured with.
RUN pip install -c containers/constraints-runtime.txt \
      cmake packaging setuptools setuptools-scm setuptools-rust wheel jinja2 ninja \
      -r requirements/common.txt
RUN VLLM_TARGET_DEVICE=rocm PYTORCH_ROCM_ARCH=gfx1030 ROCM_PATH=/opt/rocm MAX_JOBS=${MAX_JOBS} \
    VLLM_VERSION_OVERRIDE=${VLLM_VERSION_OVERRIDE} \
    pip install -e . --no-build-isolation --no-deps
RUN pip install -c containers/constraints-runtime.txt -r requirements/rocm.txt --no-deps \
 && pip check || true

# Verification — no GPU needed: torch's compiled arch flags, HIP 7.14, this fork's RDNA ops bound.
RUN python - <<'PY'
import torch, vllm, vllm._rocm_C  # noqa: F401
flags = (torch._C._cuda_getArchFlags() or "").split(); assert "gfx1030" in flags, flags
assert torch.version.hip.startswith("7.14"), torch.version.hip
for op in ("rdna_ar_all_reduce", "rdna_ar_init", "rdna_hc_up_gate_mix", "rdna_gemv_act", "rdna_se_gate_up_silu"):
    assert hasattr(torch.ops._rocm_C, op), f"fork op {op} not bound"
import vllm.v1.ple_offload.connector, vllm.v1.ple_offload.worker  # noqa: F401  (PLE offload present)
print(f"runtime ok: vllm {vllm.__version__} on torch {torch.__version__} hip {torch.version.hip}; RDNA ops + PLE offload present")
open("/opt/versions.txt", "a").write(f"vllm={vllm.__version__} (leapdragon/vllm-rdna2-qwen)\n")
PY

COPY containers/entrypoint.sh /usr/local/bin/vllm-rdna2-qwen
# Build-time dry run of the entrypoint → serve script (needs no GPU, no weights): the command resolves.
RUN chmod +x /usr/local/bin/vllm-rdna2-qwen \
 && mkdir -p /tmp/m /tmp/p && echo '{}' > /tmp/p/META.json \
 && MODEL=/tmp/m PLE_INT4=/tmp/p DRYRUN=1 /usr/local/bin/vllm-rdna2-qwen serve | grep -q -- "--enable-expert-parallel" \
 && rm -rf /tmp/m /tmp/p && echo "entrypoint dry run ok"

# Runtime defaults every serve needs (the serve script exports the rest per knob).
ENV ROCM_PATH=/opt/rocm \
    HEALTHCHECK_PORT=8000 \
    HF_HUB_OFFLINE=1
ARG BUILD_DATE=unknown
ARG VCS_REF=unknown
ARG IMAGE_VERSION=unknown
LABEL maintainer="Aron Hsiao (github.com/leapdragon)" \
      org.opencontainers.image.title="vllm-rdna2-qwen — Qwen3.8-Flash-Next on 4× gfx1030 (TheRock ROCm 7.14)" \
      org.opencontainers.image.description="vLLM fork with the Flash-Next n-gram (PLE) CPU offload, RDNA2 decode kernels and one-shot all-reduce; torch 2.12 built for gfx1030 on TheRock ROCm 7.14." \
      org.opencontainers.image.source="https://github.com/leapdragon/vllm-rdna2-qwen" \
      org.opencontainers.image.documentation="https://github.com/leapdragon/vllm-rdna2-qwen/blob/rdna2/qwen38-flash-next/containers/README.md" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.version="${IMAGE_VERSION}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.base.name="${BASE_IMAGE}"
# Cold boot: ~13 min torch.compile + the 30 GB sidecar prefault; healthcheck start period sized for it.
HEALTHCHECK --interval=30s --timeout=10s --start-period=40m --retries=3 \
  CMD python3 -c 'import os, urllib.request as u; u.urlopen("http://127.0.0.1:%s/health" % os.environ.get("HEALTHCHECK_PORT", "8000"), timeout=5)' || exit 1
ENTRYPOINT ["/usr/local/bin/vllm-rdna2-qwen"]
CMD ["serve"]
