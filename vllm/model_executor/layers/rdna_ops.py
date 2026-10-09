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

import os

import torch
import torch.nn.functional as F

from vllm.utils.torch_utils import direct_register_custom_op

_DECODE_MAX = 16  # gemv_i8/f16_rdna2 handle M <= 16 (2026-10-09)
# The fused glue kernels (rdna_fused_glue.cu: rdna_gemv_act, rdna_hc_up_gate_mix, rdna_se_*) take M <= 8 and
# M == 16 (graph-padded decode batches 9..16); other M take the unfused paths.
_FUSED_DECODE_MAX = 8


def _fused_ok(n: int) -> bool:
    return 0 < n <= _FUSED_DECODE_MAX or n == 16


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
    from vllm.model_executor.layers import rdna_w8a8_dense as w8a8
    from vllm.model_executor.layers.rdna_dense_int8 import is_released, linear_bucketed, linear_released

    if w8a8.can_use(x, weight_i8):
        return w8a8.linear(x, weight_i8, scale, bias)
    if is_released(weight):
        return linear_released(x, weight_i8, scale, bias)
    return linear_bucketed(x, weight, bias)


def _rdna_dense_gemm_fake(x, weight, weight_i8, scale, bias):
    return x.new_empty((*x.shape[:-1], weight_i8.shape[0]))


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
    if _fused_ok(n) and xn.dtype == torch.float16 and xn.is_contiguous():
        from vllm import _custom_ops as ops

        wd, sd = (w_down_i8, s_down) if w_down_i8 is not None else (w_down, None)
        wu, su = (w_up_i8, s_up) if w_up_i8 is not None else (w_up, None)
        dai = ops.rdna_gemv_act(xn, wd, sd, lora_rank, 1.0 / hc_count)
        lora = dai[:, :lora_rank].contiguous()
        block_input = ops.rdna_hc_up_gate_mix(lora, wu, su, xn, hc_count)
        return block_input, dai
    # prefill / fallback: the original op sequence
    from vllm.models.qwen4_exp.amd.ops.hc import hc_gate_mix, hc_silu
    from vllm.model_executor.layers import rdna_w8a8_dense as w8a8
    from vllm.model_executor.layers.rdna_dense_int8 import linear_bucketed, weight_for_gemm

    if w8a8.can_use(xn, w_down_i8):
        dai = w8a8.linear(xn, w_down_i8, s_down)
    else:
        dai = linear_bucketed(xn, weight_for_gemm(w_down, w_down_i8, s_down))
    lora = hc_silu(dai[:, :lora_rank].contiguous(), hc_count)
    if w8a8.can_use(lora, w_up_i8):
        gate = w8a8.linear(lora, w_up_i8, s_up)
    else:
        gate = linear_bucketed(lora, weight_for_gemm(w_up, w_up_i8, s_up))
    block_input = hc_gate_mix(xn, gate, hc_count)
    return block_input, dai


def _rdna_hc_mix_fake(xn, w_down, w_down_i8, s_down, w_up, w_up_i8, s_up, lora_rank, hc_count):
    m = xn.shape[0]
    n_down = w_down_i8.shape[0] if w_down_i8 is not None else w_down.shape[0]
    return (
        xn.new_empty((m, xn.shape[1] // hc_count)),
        xn.new_empty((m, n_down)),
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
    if _fused_ok(n) and x.dtype == torch.float16 and x.dim() == 2 and x.is_contiguous():
        from vllm import _custom_ops as ops

        a, sa = (w1_i8, s1) if w1_i8 is not None else (w1, None)
        b, sb = (w2_i8, s2) if w2_i8 is not None else (w2, None)
        act = ops.rdna_se_gate_up_silu(x, a, sa)
        return ops.rdna_se_down_gated(act, b, sb, x, w_gate)
    from vllm.model_executor.layers.rdna_dense_int8 import linear_bucketed, weight_for_gemm

    w1 = weight_for_gemm(w1, w1_i8, s1)
    w2 = weight_for_gemm(w2, w2_i8, s2)
    gu = linear_bucketed(x, w1)
    half = gu.shape[-1] // 2
    act = F.silu(gu[..., :half]) * gu[..., half:]
    out = linear_bucketed(act, w2)
    return torch.sigmoid(F.linear(x, w_gate.reshape(1, -1))) * out


def _rdna_shared_expert_fake(x, w1, w1_i8, s1, w2, w2_i8, s2, w_gate):
    n_out = w2_i8.shape[0] if w2_i8 is not None else w2.shape[0]
    return x.new_empty((*x.shape[:-1], n_out))


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


# ---------------------------------------------------------------- MoE v3: shared expert inside the routed pair
# gfx1030 fork (2026-10-09): at M <= 2 the shared expert (gate_up+silu, down * sigmoid(gate)) runs inside the
# routed int4 kernel pair (moe_decode_v3): 4 launches per MoE layer instead of 6, 71 -> 61.5 us/layer at M=1 in
# bench/fused-moe. At M >= 3 the extra grid slice no longer hides and v2 + the separate shared expert is as fast
# or faster, so the window stops at 2. VLLM_RDNA_MOE_V3=0 disables it.
#
# The MoE runner owns the shared expert and the routed kernel never sees it, so the hand-off is a one-slot
# mailbox: the runner posts the shared expert's weights before the routed call (moe_v3_offer), the int4 skinny
# hook takes them if every kernel precondition holds (try_moe_v3), and the runner falls back to the normal
# shared-expert call if nobody did (moe_v3_taken). Both sides run inside the opaque moe_forward op, so this is
# plain eager Python at capture time. The shared output is a fresh tensor written by the w2 kernel, never a
# persistent buffer: the compiled graph may reuse the op's outputs as scratch once the add has consumed them.
_MOE_V3 = os.environ.get("VLLM_RDNA_MOE_V3", "1") == "1"
_MOE_V3_MAX = 2
_moe_v3_slot: list = [None]


def _v3_shared_weights(mlp):
    """(se1, se1_s, se2, se2_s, gate_w) for a Qwen2MoeMLP with int8 shadows, else None (cached on the module)."""
    cached = getattr(mlp, "_rdna_v3_weights", False)
    if cached is not False:
        return cached
    out = None
    g, d, e = (getattr(mlp, n, None) for n in ("gate_up_proj", "down_proj", "expert_gate"))
    if g is not None and d is not None and e is not None:
        ts = (getattr(g, "weight_i8", None), getattr(g, "weight_i8_scale", None),
              getattr(d, "weight_i8", None), getattr(d, "weight_i8_scale", None), e.weight)
        if (all(t is not None for t in ts) and getattr(e, "bias", None) is None
                and all(getattr(m, "bias", None) is None for m in (g, d))
                and ts[0].dim() == 2 and ts[2].dim() == 2 and ts[4].dtype == torch.float16
                and ts[0].size(0) == 2 * ts[2].size(1)):
            out = (ts[0].contiguous(), ts[1].reshape(-1).contiguous(), ts[2].contiguous(),
                   ts[3].reshape(-1).contiguous(), ts[4].reshape(-1).contiguous())
    mlp._rdna_v3_weights = out
    return out


def moe_v3_offer(mlp, x: torch.Tensor) -> bool:
    """Runner side: post the shared expert for the routed kernel to fuse. False = run it normally."""
    if not _MOE_V3 or x.dtype != torch.float16 or x.dim() != 2 or not 0 < x.size(0) <= _MOE_V3_MAX:
        return False
    if not x.is_contiguous() or x.data_ptr() % 16:
        return False
    w = _v3_shared_weights(mlp)
    if w is None:
        return False
    _moe_v3_slot[0] = (x, w)
    return True


def moe_v3_taken() -> torch.Tensor | None:
    """Runner side, after the routed call: the shared expert's output if the routed kernel computed it, None if
    the offer was not taken (the caller runs the shared expert)."""
    slot, _moe_v3_slot[0] = _moe_v3_slot[0], None
    return slot if isinstance(slot, torch.Tensor) else None


def try_moe_v3(x, w13, s13, w2, s2, topk_weights, topk_ids, output, group_size, expert_map) -> bool:
    """Routed-kernel side: run moe_decode_v3 (routed into `output`, shared into a tensor left in the slot for the
    runner) if a shared expert was offered for this input and the kernel takes these tensors."""
    offer = _moe_v3_slot[0]
    if not isinstance(offer, tuple) or offer[0].data_ptr() != x.data_ptr() or offer[0].shape != x.shape:
        return False
    if (topk_ids.dtype != torch.int32 or topk_weights.dtype != torch.float32 or group_size % 32
            or not output.is_contiguous() or output.dtype != torch.float16
            or (expert_map is not None and expert_map.dtype != torch.int32)
            or not hasattr(torch.ops._rocm_C, "moe_decode_v3")):
        return False
    se1, se1_s, se2, se2_s, seg = offer[1]
    M, topk, inter = x.size(0), topk_ids.size(1), w13.size(1) // 2
    act = torch.empty((M, topk, inter), dtype=torch.float16, device=x.device)
    act_sh = torch.empty((M, se2.size(1)), dtype=torch.float16, device=x.device)
    sgate = torch.empty((M,), dtype=torch.float32, device=x.device)
    shared = torch.empty_like(output)
    torch.ops._rocm_C.moe_decode_v3(
        x, topk_weights.contiguous(), topk_ids.contiguous(), w13, s13, w2, s2, group_size, expert_map,
        se1, se1_s, se2, se2_s, seg, act, act_sh, sgate, shared, output,
    )
    _moe_v3_slot[0] = shared
    return True
