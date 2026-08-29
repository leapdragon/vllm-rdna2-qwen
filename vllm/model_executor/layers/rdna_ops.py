# SPDX-License-Identifier: Apache-2.0
"""T46: opaque custom ops with *runtime* decode/prefill dispatch for gfx1030.

torch.compile traces the model once for a dynamic token range, so a Python
branch like `if 0 < n <= 8` is decided at trace time and the decode kernels
never run inside the compiled graph (boot 6/8 of T45: only the hyper-connection
linears, which sit outside the traced region, took the int8 path). Wrapping the
decision in a custom op makes it a runtime choice on the real batch size.

  rdna_dense_gemm   int8-shadow GEMV for decode, fp16 rocBLAS for prefill
  rdna_hc_mix       hyper-connection mix: 2 fused kernels for decode, torch for prefill
  rdna_shared_expert shared expert (gate_up+silu*mul, down*sigmoid(gate)): 2 kernels / torch
"""

import torch
import torch.nn.functional as F

from vllm.utils.torch_utils import direct_register_custom_op

_DECODE_MAX = 8


def _ntok(x: torch.Tensor) -> int:
    return x.numel() // x.size(-1)


# ---------------------------------------------------------------- dense int8 shadow
def _rdna_dense_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_i8: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    n = _ntok(x)
    if 0 < n <= _DECODE_MAX and x.dtype == torch.float16:
        from vllm import _custom_ops as ops

        x2 = x.reshape(-1, x.size(-1)).contiguous()
        out = ops.gemv_i8_rdna2(x2, weight_i8, scale, bias)
        return out.reshape(*x.shape[:-1], weight_i8.shape[0])
    return F.linear(x, weight, bias)


def _rdna_dense_gemm_fake(x, weight, weight_i8, scale, bias):
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


# ---------------------------------------------------------------- hyper-connection mix
def _rdna_hc_mix(
    xn: torch.Tensor,
    w_down: torch.Tensor,
    w_down_i8: torch.Tensor | None,
    s_down: torch.Tensor | None,
    w_up: torch.Tensor,
    w_up_i8: torch.Tensor | None,
    s_up: torch.Tensor | None,
    lora_rank: int,
    hc_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (block_input [M, H], down_and_injection [M, N_down])."""
    n = _ntok(xn)
    if 0 < n <= _DECODE_MAX and xn.dtype == torch.float16 and xn.is_contiguous():
        from vllm import _custom_ops as ops

        wd, sd = (w_down_i8, s_down) if w_down_i8 is not None else (w_down, None)
        wu, su = (w_up_i8, s_up) if w_up_i8 is not None else (w_up, None)
        dai = ops.rdna_gemv_act(xn, wd, sd, lora_rank, 1.0 / hc_count)
        lora = dai[:, :lora_rank].contiguous()
        block_input = ops.rdna_hc_up_gate_mix(lora, wu, su, xn, hc_count)
        return block_input, dai
    # prefill / fallback: the original op sequence
    from vllm.models.qwen4_exp.amd.ops.hc import hc_gate_mix, hc_silu

    dai = F.linear(xn, w_down)
    lora = hc_silu(dai[:, :lora_rank].contiguous(), hc_count)
    gate = F.linear(lora, w_up)
    block_input = hc_gate_mix(xn, gate, hc_count)
    return block_input, dai


def _rdna_hc_mix_fake(xn, w_down, w_down_i8, s_down, w_up, w_up_i8, s_up, lora_rank, hc_count):
    m = xn.shape[0]
    return (
        xn.new_empty((m, xn.shape[1] // hc_count)),
        xn.new_empty((m, w_down.shape[0])),
    )


# ---------------------------------------------------------------- shared expert
def _rdna_shared_expert(
    x: torch.Tensor,
    w1: torch.Tensor,
    w1_i8: torch.Tensor | None,
    s1: torch.Tensor | None,
    w2: torch.Tensor,
    w2_i8: torch.Tensor | None,
    s2: torch.Tensor | None,
    w_gate: torch.Tensor,
) -> torch.Tensor:
    """Per-rank partial of sigmoid(w_gate.x) * down(silu(gate)*up); caller reduces."""
    n = _ntok(x)
    if 0 < n <= _DECODE_MAX and x.dtype == torch.float16 and x.dim() == 2 and x.is_contiguous():
        from vllm import _custom_ops as ops

        a, sa = (w1_i8, s1) if w1_i8 is not None else (w1, None)
        b, sb = (w2_i8, s2) if w2_i8 is not None else (w2, None)
        act = ops.rdna_se_gate_up_silu(x, a, sa)
        return ops.rdna_se_down_gated(act, b, sb, x, w_gate)
    gu = F.linear(x, w1)
    half = gu.shape[-1] // 2
    act = F.silu(gu[..., :half]) * gu[..., half:]
    out = F.linear(act, w2)
    return torch.sigmoid(F.linear(x, w_gate.reshape(1, -1))) * out


def _rdna_shared_expert_fake(x, w1, w1_i8, s1, w2, w2_i8, s2, w_gate):
    return x.new_empty((*x.shape[:-1], w2.shape[0]))


direct_register_custom_op(
    op_name="rdna_dense_gemm",
    op_func=_rdna_dense_gemm,
    mutates_args=[],
    fake_impl=_rdna_dense_gemm_fake,
)
direct_register_custom_op(
    op_name="rdna_hc_mix",
    op_func=_rdna_hc_mix,
    mutates_args=[],
    fake_impl=_rdna_hc_mix_fake,
)
direct_register_custom_op(
    op_name="rdna_shared_expert",
    op_func=_rdna_shared_expert,
    mutates_args=[],
    fake_impl=_rdna_shared_expert_fake,
)
