# SPDX-License-Identifier: Apache-2.0
"""W4A8 fused-MoE prefill kernel for gfx1030 (RDNA2), opt-in via VLLM_RDNA_MOE_W4A8=1.

The stock W4A16 Triton kernel (`fused_moe_kernel_gptq_awq`) dequantises every int4 weight to
fp16 inside the K loop (shift, mask, int->fp32, subtract zero point, multiply by the group
scale, fp32->fp16) before an fp16 `tl.dot`. On a V620 that inner loop, not the dot rate,
bounds the kernel: ~4 TFLOPS against 38 TFLOPS measured for `v_dot2_f32_f16` (2026-09-24
profile). This kernel instead

  * quantises the activations per token to int8 (symmetric, absmax / 127), once per GEMM;
  * unpacks the int4 weights to int8 with integer ops only (shift, mask, subtract 8);
  * accumulates `tl.dot(int8, int8) -> int32` across one quantisation group (Triton lowers it
    to `v_dot4_i32_i8`, measured 64.2 TOPS, PROFILE-NAVI21 T-C1/T-I2);
  * folds the group's weight scale in once per output element per group, and the token's
    activation scale once at the store.

Symmetric int4 weights only (uint4b8: stored value - 8), K a multiple of the group size and
group size a multiple of BLOCK_SIZE_K. Anything else falls back to the stock kernel.
"""

import os

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

_ENABLED: bool | None = None
_MIN_TOKENS = int(os.getenv("VLLM_RDNA_MOE_W4A8_MIN_TOKENS", "64"))
_LOGGED = False
# Tiles from the 2026-09-24 sweep on a V620 (Flash-Next per-rank shapes, M=256..2048): 64x128x16,
# 8 warps, 2 stages = 2.0-2.45x over the W4A16 kernel. BLOCK_SIZE_M is NOT set here: the token sort
# (moe_align_block_size) was aligned to the caller's BLOCK_SIZE_M, so it must be kept. Larger K
# blocks spill (256 VGPR cap: 190-390 spilled at BK>=64) and lose up to 5x.
_TILES = {
    "BLOCK_SIZE_N": int(os.getenv("VLLM_RDNA_MOE_W4A8_BN", "128")),
    "BLOCK_SIZE_K": int(os.getenv("VLLM_RDNA_MOE_W4A8_BK", "16")),
    "num_warps": int(os.getenv("VLLM_RDNA_MOE_W4A8_WARPS", "8")),
    "num_stages": int(os.getenv("VLLM_RDNA_MOE_W4A8_STAGES", "2")),
}


def enabled() -> bool:
    global _ENABLED
    if _ENABLED is None:
        _ENABLED = os.getenv("VLLM_RDNA_MOE_W4A8", "0") == "1"
    return _ENABLED


@triton.jit
def _quant_rows_int8_kernel(x_ptr, q_ptr, s_ptr, K, stride_xm, stride_qm,
                            BLOCK_K: tl.constexpr):
    """One program per row: absmax/127 symmetric int8, scale in fp32."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_K)
    mask = offs < K
    x = tl.load(x_ptr + row * stride_xm + offs, mask=mask, other=0.0).to(tl.float32)
    amax = tl.max(tl.abs(x), axis=0)
    scale = tl.maximum(amax, 1e-10) / 127.0
    v = x / scale
    q = tl.where(v >= 0, tl.floor(v + 0.5), tl.ceil(v - 0.5))   # round half away from zero
    q = tl.minimum(tl.maximum(q, -127.0), 127.0).to(tl.int8)
    tl.store(q_ptr + row * stride_qm + offs, q, mask=mask)
    tl.store(s_ptr + row, scale)


def quant_rows_int8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row symmetric int8 quantisation of a 2-D fp16/bf16/fp32 tensor."""
    assert x.dim() == 2
    m, k = x.shape
    q = torch.empty((m, k), dtype=torch.int8, device=x.device)
    s = torch.empty((m,), dtype=torch.float32, device=x.device)
    if m == 0:
        return q, s
    _quant_rows_int8_kernel[(m,)](x, q, s, k, x.stride(0), q.stride(0),
                                  BLOCK_K=triton.next_power_of_2(k), num_warps=4)
    return q, s


@triton.jit
def fused_moe_kernel_w4a8(
    a_ptr, a_scale_ptr, b_ptr, c_ptr, b_scale_ptr,
    topk_weights_ptr, sorted_token_ids_ptr, expert_ids_ptr, num_tokens_post_padded_ptr,
    N: tl.constexpr, K: tl.constexpr, EM, num_valid_tokens,
    stride_am, stride_ak,
    stride_be, stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_bse, stride_bsk, stride_bsn,
    group_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr, top_k: tl.constexpr, compute_type: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts == -1:
        tl.store(c_ptrs, tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type), mask=c_mask)
        return

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_rows = offs_token // top_k
    a_ptrs = a_ptr + (a_rows[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = (b_ptr + off_experts * stride_be + (offs_k[:, None] // 2) * stride_bk
              + offs_bn[None, :] * stride_bn)
    b_shifter = (offs_k[:, None] % 2) * 4
    bs_ptrs = b_scale_ptr + off_experts * stride_bse + offs_bn * stride_bsn

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for g in range(0, K // group_size):
        acc_i = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
        for kk in range(0, group_size // BLOCK_SIZE_K):
            a = tl.load(a_ptrs, mask=token_mask[:, None], other=0)
            b = tl.load(b_ptrs)
            b = (((b >> b_shifter) & 0xF).to(tl.int8) - 8).to(tl.int8)
            acc_i = tl.dot(a, b, acc=acc_i, out_dtype=tl.int32)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += (BLOCK_SIZE_K // 2) * stride_bk
        scale = tl.load(bs_ptrs + g * stride_bsk).to(tl.float32)
        acc += acc_i.to(tl.float32) * scale[None, :]

    a_scale = tl.load(a_scale_ptr + a_rows, mask=token_mask, other=0.0)
    acc = acc * a_scale[:, None]
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        acc = acc * moe_weight[:, None]
    tl.store(c_ptrs, acc.to(compute_type), mask=c_mask)


def can_use(A: torch.Tensor, B_zp, use_int4_w4a16: bool, group_size: int,
            config: dict) -> bool:
    if not enabled() or not use_int4_w4a16 or B_zp is not None:
        return False
    if A.dim() != 2 or A.size(0) < _MIN_TOKENS or A.dtype not in (torch.float16, torch.bfloat16):
        return False
    if config.get("BLOCK_SIZE_M", 0) < 32:     # decode-shaped configs keep the stock path
        return False
    bk = _TILES["BLOCK_SIZE_K"]
    return group_size > 0 and A.size(1) % group_size == 0 and group_size % bk == 0 and bk >= 16


def invoke(A, B, C, B_scale, topk_weights, sorted_token_ids, expert_ids,
           num_tokens_post_padded, mul_routed_weight: bool, top_k: int,
           config: dict, compute_type, group_size: int) -> None:
    """Drop-in for invoke_fused_moe_wna16_triton_kernel on symmetric int4 weights."""
    global _LOGGED
    if not _LOGGED:
        logger.info("rdna W4A8 MoE: int8 activations x int4 weights (v_dot4) for prefill "
                    "(M >= %d), tiles BM=%d %s", _MIN_TOKENS, config["BLOCK_SIZE_M"], _TILES)
        _LOGGED = True
    A_q, A_s = quant_rows_int8(A)
    M = A.size(0)
    EM = sorted_token_ids.size(0)
    if M < config["BLOCK_SIZE_M"]:
        EM = min(EM, M * top_k * config["BLOCK_SIZE_M"])
    cfg = {"BLOCK_SIZE_M": config["BLOCK_SIZE_M"], "GROUP_SIZE_M": config.get("GROUP_SIZE_M", 1), **_TILES}
    grid = lambda META: (triton.cdiv(EM, META["BLOCK_SIZE_M"])
                         * triton.cdiv(B.size(1), META["BLOCK_SIZE_N"]),)
    fused_moe_kernel_w4a8[grid](
        A_q, A_s, B, C, B_scale,
        topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded,
        B.size(1), A.size(1), EM, M * top_k,
        A_q.stride(0), A_q.stride(1),
        B.stride(0), B.stride(2), B.stride(1),
        C.stride(1), C.stride(2),
        B_scale.stride(0), B_scale.stride(2), B_scale.stride(1),
        group_size=group_size,
        MUL_ROUTED_WEIGHT=mul_routed_weight, top_k=top_k, compute_type=compute_type,
        **cfg,
    )
