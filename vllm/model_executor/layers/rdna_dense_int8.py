# SPDX-License-Identifier: Apache-2.0
"""T45: int8 weight-only shadow copies of the dense fp16 projections for gfx1030 decode.

On Qwen3.8-Flash-Next only the routed experts are quantised; the GDN/QSA
projections, router, shared expert, hyper-connections and lm_head are fp16 and
stream ~2.1 GB per rank per forward -- the largest remaining per-step cost after
T43/T44. Weight-only int8 with a per-output-channel symmetric scale halves the
bytes at negligible accuracy cost. The fp16 weight is kept for prefill (rocBLAS,
M > 8); the shadow is used only where the decode GEMV would run.

Enabled with VLLM_RDNA_DENSE_INT8=1 on gfx10x (off by default until validated).
VLLM_RDNA_DENSE_INT8_MIN_ROWS (default 64) skips tiny layers.
"""

import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_ENABLED: bool | None = None


def enabled() -> bool:
    global _ENABLED
    if _ENABLED is None:
        from vllm.platforms import current_platform

        on = os.getenv("VLLM_RDNA_DENSE_INT8", "0") == "1" and current_platform.is_rocm()
        if on:
            from vllm.platforms.rocm import on_gfx10x

            on = on_gfx10x()
        _ENABLED = on
    return _ENABLED


@torch.no_grad()
def make_shadow(layer: torch.nn.Module) -> None:
    """Attach `weight_i8` / `weight_i8_scale` to a layer whose `weight` is fp16 [N, K]."""
    if not enabled():
        return
    w = getattr(layer, "weight", None)
    if w is None or w.dtype != torch.float16 or w.dim() != 2 or not w.is_cuda:
        return
    n, k = w.shape
    if k % 16 != 0 or n < int(os.getenv("VLLM_RDNA_DENSE_INT8_MIN_ROWS", "64")):
        return
    amax = w.abs().amax(dim=1).float().clamp_min(1e-8)
    scale = (amax / 127.0)
    q = torch.round(w.float() / scale[:, None]).clamp_(-127, 127).to(torch.int8)
    layer.weight_i8 = q.contiguous()
    layer.weight_i8_scale = scale.to(torch.float16).contiguous()


def apply(layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None):
    """Return the int8-GEMV result for decode-shaped x, or None to fall through."""
    w8 = getattr(layer, "weight_i8", None)
    if w8 is None or x.dtype != torch.float16:
        return None
    n = x.numel() // x.size(-1)
    if not (0 < n <= 8):
        return None
    if bias is not None and (bias.dtype != torch.float16 or not bias.is_contiguous()):
        return None
    from vllm import _custom_ops as ops

    x_view = x.reshape(-1, x.size(-1)).contiguous()
    out = ops.gemv_i8_rdna2(x_view, w8, layer.weight_i8_scale, bias)
    return out.reshape(*x.shape[:-1], w8.shape[0])
