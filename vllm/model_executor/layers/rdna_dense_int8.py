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

VLLM_RDNA_DENSE_INT8_ONLY=1 additionally RELEASES the fp16 weight once its shadow exists
(the parameter becomes a 0-element placeholder), so the int8 copy is the only resident
copy: ~2 GB/rank less than shadow+fp16. Prefill-shaped calls (M > 8) then dequantise the
shadow on the fly (`dequant`) into a transient fp16 [N, K] for the rocBLAS GEMM -- a
per-call cost of one N*K*3-byte pass, ~0.3 % of a 2048-token chunk. Layers whose shadow
is skipped (tiny / K % 16 != 0) keep their fp16 weight and are unaffected.
"""

import os

import torch

from vllm.logger import init_logger

# Register torch.ops.vllm.rdna_* eagerly. With the torch.compile cache enabled a cached
# graph that calls these ops is loaded and run before any forward pass has executed the
# lazy imports at the call sites -> "'_OpNamespace' 'vllm' object has no attribute
# 'rdna_hc_mix'" (2026-08-30). Importing the module is what registers the ops.
from vllm.model_executor.layers import rdna_ops  # noqa: F401

logger = init_logger(__name__)

_ENABLED: bool | None = None
_ONLY: bool | None = None


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


def only_mode() -> bool:
    """True when the fp16 weight is released after shadowing (VLLM_RDNA_DENSE_INT8_ONLY=1)."""
    global _ONLY
    if _ONLY is None:
        _ONLY = enabled() and os.getenv("VLLM_RDNA_DENSE_INT8_ONLY", "0") == "1"
    return _ONLY


def is_released(weight: torch.Tensor) -> bool:
    """A released fp16 weight is the 0-element placeholder left by `make_shadow`."""
    return weight.numel() == 0


_SCRATCH: dict[tuple, torch.Tensor] = {}


def dequant(weight_i8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """fp16 [N, K] view of the per-channel int8 shadow, written into a persistent scratch buffer.

    One buffer per (rows, K, device) is allocated on first use and reused for every later call
    of that shape, so prefill never churns short-lived N*K*2-byte temporaries through the caching
    allocator (that churn fragmented it and OOMed a full-vocab logits allocation at 96 %). The
    handful of shapes (GDN in/out, QSA qkv/o, shared expert, hc, one lm_head block) costs
    ~100 MB/rank, allocated during the profile run and therefore inside vLLM's budget. Callers
    must consume the result before the next dequant of the same shape.
    """
    n, k = weight_i8.shape
    key = (n, k, weight_i8.device)
    buf = _SCRATCH.get(key)
    if buf is None:
        buf = torch.empty((n, k), dtype=torch.float16, device=weight_i8.device)
        _SCRATCH[key] = buf
    torch.mul(weight_i8, scale[:, None], out=buf)
    return buf


def linear_released(
    x: torch.Tensor,
    weight_i8: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None,
    block_rows: int = 8192,
) -> torch.Tensor:
    """x @ dequant(weight_i8)^T + bias, dequantising `block_rows` output rows at a time.

    A whole-weight temporary for the lm_head is 318 MB/rank; freed at once it stays reserved
    in the caching allocator in a size the following full-vocab logits cannot reuse, which
    OOMed a prompt_logprobs request at 96 % utilisation. 42 MB blocks fragment nothing.
    """
    n = weight_i8.shape[0]
    if n <= block_rows:
        return torch.nn.functional.linear(x, dequant(weight_i8, scale), bias)
    out = x.new_empty((*x.shape[:-1], n))
    for i in range(0, n, block_rows):
        j = min(i + block_rows, n)
        out[..., i:j] = torch.nn.functional.linear(
            x, dequant(weight_i8[i:j], scale[i:j]), None if bias is None else bias[i:j]
        )
    return out


def weight_for_gemm(
    weight: torch.Tensor, weight_i8: torch.Tensor | None, scale: torch.Tensor | None
) -> torch.Tensor:
    """The fp16 weight for a prefill-shaped GEMM: the real one, or a dequantised shadow."""
    if weight_i8 is not None and is_released(weight):
        return dequant(weight_i8, scale)
    return weight


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
    if only_mode():
        # Release the fp16 copy: keep the Parameter object (loader attrs, registration)
        # but drop its storage. Every consumer of `weight` on the fp16 path goes through
        # rdna_ops.*, which dequantises the shadow when it sees the placeholder.
        layer.weight.data = w.new_empty((0,))


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
