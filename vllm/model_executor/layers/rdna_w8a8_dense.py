# SPDX-License-Identifier: Apache-2.0
"""W8A8 prefill GEMM for the dense int8 shadows on gfx1030 (RDNA2), opt-in via VLLM_RDNA_DENSE_W8A8=1.

With VLLM_RDNA_DENSE_INT8_ONLY=1 the dense projections live only as per-output-channel int8
shadows (`rdna_dense_int8`), and prefill-shaped calls dequantise the shadow to fp16 for a rocBLAS
GEMM. This path instead quantises the activations per token to int8 (symmetric, absmax / 127) and
runs `tl.dot(int8, int8) -> int32` (Triton lowers it to `v_dot4_i32_i8`), applying the token scale
and the channel scale once per output element. No fp16 copy of the weight is materialised.

K up to 10240 cannot overflow the int32 accumulator (127 * 127 * 10240 < 2**31).
"""

import os

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

_ENABLED: bool | None = None
_MIN_TOKENS = int(os.getenv("VLLM_RDNA_DENSE_W8A8_MIN_TOKENS", "64"))
_LOGGED = False


def enabled() -> bool:
    global _ENABLED
    if _ENABLED is None:
        _ENABLED = os.getenv("VLLM_RDNA_DENSE_W8A8", "0") == "1"
    return _ENABLED


# Tile per (N, K) per rank, from the harness sweeps (bench/w8a8-dense/); other shapes use _DEFAULT_TILE.
_DEFAULT_TILE = dict(BM=64, BN=128, BK=32, GROUP_M=8, num_warps=8, num_stages=1)
_TILES: dict[tuple[int, int], dict] = {
    # (N, K) per rank -> tile; 2026-09-25 sweep on a V620 at 120 W, M = 2048 (bench/w8a8-dense/)
    (4096, 2560): dict(BM=128, BN=256, BK=16, GROUP_M=8, num_warps=8, num_stages=2),
    (3584, 2560): dict(BM=128, BN=256, BK=16, GROUP_M=8, num_warps=8, num_stages=2),
    (2560, 1536): dict(BM=128, BN=256, BK=16, GROUP_M=8, num_warps=8, num_stages=2),
    (10240, 320): dict(BM=64, BN=128, BK=16, GROUP_M=8, num_warps=4, num_stages=1),
    # per-branch activation scales (_A_GROUP): grouped K loop, re-swept (hc_group_sweep.py)
    (336, 10240): dict(BM=64, BN=128, BK=64, GROUP_M=8, num_warps=4, num_stages=2),
    (320, 10240): dict(BM=64, BN=128, BK=64, GROUP_M=8, num_warps=4, num_stages=2),
}
# Shapes with a tile are the ones W8A8 can run; every smaller one (router 512x2560, shared expert,
# indexer 640x2560) was 0.89-0.96x of rocBLAS under sustained load, and keeping the router on the fp16
# GEMM also keeps expert selection exactly as before.
#
# Default set: the GDN/QSA projections only. In-server (2026-09-25, teacher-forced logprobs vs the
# fp16-GEMM baseline) they drift 0.21 nats with NLL +0.1-0.2 %, like earlier adopted changes. The
# hyper-connection up projection (10240x320, whose output is the branch gate) drifted 0.36 nats on its
# own -- NLL -1.1 % but top-1 -0.4 pt, a larger behavioural change than anything shipped so far -- and
# the down projection (336x10240) is clean but gained nothing in-server. Both stay opt-in through
# VLLM_RDNA_DENSE_W8A8_SHAPES.
_DEFAULT_SHAPES = {(4096, 2560), (3584, 2560), (2560, 1536)}


def _tile(n: int, k: int) -> dict:
    env = os.getenv("VLLM_RDNA_DENSE_W8A8_TILE")        # "BM,BN,BK,GM,warps,stages" (sweeps)
    if env:
        bm, bn, bk, gm, nw, ns = map(int, env.split(","))
        return dict(BM=bm, BN=bn, BK=bk, GROUP_M=gm, num_warps=nw, num_stages=ns)
    return _TILES.get((n, k), _DEFAULT_TILE)


@triton.jit
def _quant_rows_kernel(x_ptr, q_ptr, s_ptr, K, stride_xm, stride_qm, BLOCK_K: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_K)
    mask = offs < K
    x = tl.load(x_ptr + row * stride_xm + offs, mask=mask, other=0.0).to(tl.float32)
    amax = tl.max(tl.abs(x), axis=0)
    scale = tl.maximum(amax, 1e-10) / 127.0
    v = x / scale
    q = tl.where(v >= 0, tl.floor(v + 0.5), tl.ceil(v - 0.5))
    q = tl.minimum(tl.maximum(q, -127.0), 127.0).to(tl.int8)
    tl.store(q_ptr + row * stride_qm + offs, q, mask=mask)
    tl.store(s_ptr + row, scale)


def quant_rows(x: torch.Tensor, group: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """int8 [M, K] and fp32 scales [M, K // group]; group = K (default) is one scale per token."""
    m, k = x.shape
    group = group or k
    ng = k // group
    q = torch.empty((m, k), dtype=torch.int8, device=x.device)
    s = torch.empty((m, ng), dtype=torch.float32, device=x.device)
    if m:
        if ng > 1 and x.stride(0) != k:
            x = x.contiguous()
        # one program per (row, group) on the [M * ng, group] view (ng == 1: the rows themselves)
        stride = group if ng > 1 else x.stride(0)
        bk = triton.next_power_of_2(group)
        _quant_rows_kernel[(m * ng,)](x, q, s, group, stride, group,
                                      BLOCK_K=bk, num_warps=8 if bk >= 4096 else 4)
    return q, s


@triton.jit
def _w8a8_kernel(
    a_ptr, as_ptr, b_ptr, bs_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_bn, stride_cm, stride_as,
    HAS_BIAS: tl.constexpr, EVEN_K: tl.constexpr, A_GROUPS: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr,
):
    """A_GROUPS == 1: one activation scale per token (applied in the epilogue). A_GROUPS > 1: K is
    split into A_GROUPS equal groups (EVEN_K, multiple of BK), each with its own per-token scale,
    folded into an fp32 accumulator after the group's int32 dot products."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    rm = tl.where(offs_m < M, offs_m, 0).to(tl.int64)
    rn = tl.where(offs_n < N, offs_n, 0).to(tl.int64)
    a_ptrs = a_ptr + rm[:, None] * stride_am + offs_k[None, :]
    b_ptrs = b_ptr + rn[None, :] * stride_bn + offs_k[:, None]        # [BK, BN] view of row-major [N, K]

    if A_GROUPS == 1:
        acc = tl.zeros((BM, BN), dtype=tl.int32)
        for kk in range(0, tl.cdiv(K, BK)):
            if EVEN_K:
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs)
            else:
                kmask = offs_k + kk * BK < K
                a = tl.load(a_ptrs, mask=kmask[None, :], other=0)
                b = tl.load(b_ptrs, mask=kmask[:, None], other=0)
            acc = tl.dot(a, b, acc=acc, out_dtype=tl.int32)
            a_ptrs += BK
            b_ptrs += BK
        sa = tl.load(as_ptr + rm * stride_as)
        out = acc.to(tl.float32) * sa[:, None]
    else:
        out = tl.zeros((BM, BN), dtype=tl.float32)
        steps = K // A_GROUPS // BK
        for g in range(0, A_GROUPS):
            acc = tl.zeros((BM, BN), dtype=tl.int32)
            for kk in range(0, steps):
                acc = tl.dot(tl.load(a_ptrs), tl.load(b_ptrs), acc=acc, out_dtype=tl.int32)
                a_ptrs += BK
                b_ptrs += BK
            sa = tl.load(as_ptr + rm * stride_as + g)
            out += acc.to(tl.float32) * sa[:, None]

    sb = tl.load(bs_ptr + rn).to(tl.float32)
    out = out * sb[None, :]
    if HAS_BIAS:
        out += tl.load(bias_ptr + rn).to(tl.float32)[None, :]
    c_ptrs = c_ptr + offs_m[:, None].to(tl.int64) * stride_cm + offs_n[None, :]
    tl.store(c_ptrs, out.to(tl.float16), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def gemm_q(q: torch.Tensor, qs: torch.Tensor, w_i8: torch.Tensor, w_scale: torch.Tensor,
           bias: torch.Tensor | None = None, out: torch.Tensor | None = None) -> torch.Tensor:
    """fp16 [M, N] = (q * qs) @ (w_i8 * w_scale)^T (+ bias), q int8 [M, K] row-major."""
    m, k = q.shape
    n = w_i8.shape[0]
    t = _tile(n, k)
    if out is None:
        out = torch.empty((m, n), dtype=torch.float16, device=q.device)
    grid = (triton.cdiv(m, t["BM"]) * triton.cdiv(n, t["BN"]),)
    a_groups = qs.shape[1] if qs.dim() == 2 else 1
    _w8a8_kernel[grid](
        q, qs, w_i8, w_scale, bias if bias is not None else w_scale, out,
        m, n, k, q.stride(0), w_i8.stride(0), out.stride(0), a_groups,
        HAS_BIAS=bias is not None, EVEN_K=(k % t["BK"] == 0), A_GROUPS=a_groups,
        BM=t["BM"], BN=t["BN"], BK=t["BK"], GROUP_M=t["GROUP_M"],
        num_warps=t["num_warps"], num_stages=t["num_stages"],
    )
    return out


_SHAPES_ENV = os.getenv("VLLM_RDNA_DENSE_W8A8_SHAPES")   # e.g. "4096x2560,3584x2560" (default: all tiled)
_USE = ({tuple(map(int, t.split("x"))) for t in _SHAPES_ENV.split(",")} if _SHAPES_ENV else _DEFAULT_SHAPES)


# Activation-scale group per shape (default: one scale per token). The hyper-connection down
# projection reads 4 residual branches side by side (K = 4 x 2560) whose magnitudes differ; one
# scale per token cost the small branches their precision (teacher-forced drift 0.37 nats vs 0.21
# for the projections alone), so it gets one scale per branch.
_A_GROUP = {(336, 10240): int(os.getenv("VLLM_RDNA_DENSE_W8A8_HC_GROUP", "2560")),
            (320, 10240): int(os.getenv("VLLM_RDNA_DENSE_W8A8_HC_GROUP", "2560"))}


def can_use(x: torch.Tensor, w_i8: torch.Tensor | None) -> bool:
    return (w_i8 is not None and enabled() and x.dtype == torch.float16
            and tuple(w_i8.shape) in _USE and x.size(-1) == w_i8.shape[1]
            and x.numel() // x.size(-1) >= _MIN_TOKENS)


def linear(x: torch.Tensor, w_i8: torch.Tensor, w_scale: torch.Tensor,
           bias: torch.Tensor | None = None) -> torch.Tensor:
    """Drop-in for F.linear(x, dequant(w_i8, w_scale), bias) on prefill-shaped fp16 x."""
    global _LOGGED
    if not _LOGGED:
        logger.info("rdna W8A8 dense prefill GEMM active (M >= %d tokens)", _MIN_TOKENS)
        _LOGGED = True
    x2 = x.reshape(-1, x.size(-1))
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    q, qs = quant_rows(x2, _A_GROUP.get(tuple(w_i8.shape)))
    out = gemm_q(q, qs, w_i8, w_scale, bias)
    return out.reshape(*x.shape[:-1], w_i8.shape[0])
