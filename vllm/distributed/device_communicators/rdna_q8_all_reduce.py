# SPDX-License-Identifier: Apache-2.0
"""int8-compressed prefill all-reduce for gfx1030 (RDNA2), opt-in via VLLM_RDNA_AR_Q8=1.

Prefill-sized all-reduces (a 2048-token chunk is 10.5 MB of fp16 per call, two per layer) are
PCIe-bound on 4x V620. This path halves the bytes on the wire with a two-shot algorithm built from
RCCL collectives on the existing pynccl communicator -- no peer-memory kernel:

  1. quantise each row in blocks of G elements to int8 with an fp16 scale per block, packed per row;
  2. all-to-all: rank r receives every rank's copy of row shard r (grouped send/recv);
  3. dequantise the W copies, sum in fp32, requantise the reduced shard;
  4. all-gather the reduced shards; dequantise to fp16.

Wire bytes per rank ~(1 + 2/G) / 2 of an fp16 ring all-reduce. The result is quantised twice
(rel. error ~0.8 % with G = 64 on Gaussian data, vs 3e-4 for RCCL fp16). QuickReduce, vLLM's
quantised all-reduce, is MI300-only (wave64 and CDNA3 buffer/cache bits).

Only eager (non-captured) fp16 calls at or above VLLM_RDNA_AR_Q8_MIN_KB (default 1024) whose last
dim is a multiple of G take this path; graph-captured decode and small prefills keep their route.
"""

import os

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

_G = int(os.getenv("VLLM_RDNA_AR_Q8_GROUP", "64"))


def enabled() -> bool:
    return os.getenv("VLLM_RDNA_AR_Q8", "0") == "1"


@triton.jit
def _quant_pack(x_ptr, out_ptr, rows, stride_x, NG: tl.constexpr, NGP: tl.constexpr,
                G: tl.constexpr, H: tl.constexpr, ROWB: tl.constexpr):
    r = tl.program_id(0)
    g = tl.arange(0, NGP)[:, None]
    c = tl.arange(0, G)[None, :]
    gm = (g < NG) & (r < rows)                     # rows past `rows` are zero padding
    x = tl.load(x_ptr + r.to(tl.int64) * stride_x + g * G + c, mask=gm, other=0.0).to(tl.float32)
    s = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-12) / 127.0
    q = x / s[:, None]
    q = tl.where(q >= 0, tl.floor(q + 0.5), tl.ceil(q - 0.5)).to(tl.int8)
    base = out_ptr + r.to(tl.int64) * ROWB
    tl.store(base + g * G + c, q.to(tl.uint8, bitcast=True), mask=g < NG)
    sg = tl.arange(0, NGP)
    tl.store((base + H).to(tl.pointer_type(tl.float16)) + sg, s.to(tl.float16), mask=sg < NG)


@triton.jit
def _sum_requant(recv_ptr, own_ptr, out_ptr, rows_per, rank, W: tl.constexpr, NG: tl.constexpr, NGP: tl.constexpr,
                 G: tl.constexpr, H: tl.constexpr, ROWB: tl.constexpr):
    # own_ptr: our own shard, read straight from the send buffer (no copy into recv)
    r = tl.program_id(0)
    g = tl.arange(0, NGP)[:, None]
    c = tl.arange(0, G)[None, :]
    gm = g < NG
    sg = tl.arange(0, NGP)
    acc = tl.zeros((NGP, G), dtype=tl.float32)
    for w in tl.static_range(W):
        if w == rank:
            base = own_ptr + r.to(tl.int64) * ROWB
        else:
            base = recv_ptr + (w * rows_per + r).to(tl.int64) * ROWB
        q = tl.load(base + g * G + c, mask=gm, other=0).to(tl.int8, bitcast=True).to(tl.float32)
        s = tl.load((base + H).to(tl.pointer_type(tl.float16)) + sg, mask=sg < NG, other=0.0)
        acc += q * s.to(tl.float32)[:, None]
    s2 = tl.maximum(tl.max(tl.abs(acc), axis=1), 1e-12) / 127.0
    q2 = acc / s2[:, None]
    q2 = tl.where(q2 >= 0, tl.floor(q2 + 0.5), tl.ceil(q2 - 0.5)).to(tl.int8)
    ob = out_ptr + r.to(tl.int64) * ROWB
    tl.store(ob + g * G + c, q2.to(tl.uint8, bitcast=True), mask=gm)
    tl.store((ob + H).to(tl.pointer_type(tl.float16)) + sg, s2.to(tl.float16), mask=sg < NG)


@triton.jit
def _dequant(in_ptr, y_ptr, NG: tl.constexpr, NGP: tl.constexpr, G: tl.constexpr,
             H: tl.constexpr, ROWB: tl.constexpr):
    r = tl.program_id(0)
    g = tl.arange(0, NGP)[:, None]
    c = tl.arange(0, G)[None, :]
    gm = g < NG
    sg = tl.arange(0, NGP)
    base = in_ptr + r.to(tl.int64) * ROWB
    q = tl.load(base + g * G + c, mask=gm, other=0).to(tl.int8, bitcast=True).to(tl.float32)
    s = tl.load((base + H).to(tl.pointer_type(tl.float16)) + sg, mask=sg < NG, other=0.0).to(tl.float32)
    tl.store(y_ptr + r.to(tl.int64) * H + g * G + c, (q * s[:, None]).to(tl.float16), mask=gm)


class RdnaQ8AllReduce:
    def __init__(self, pynccl_comm, rank: int, world_size: int):
        self.comm = pynccl_comm
        self.rank = rank
        self.world_size = world_size
        self.min_bytes = int(os.getenv("VLLM_RDNA_AR_Q8_MIN_KB", "1024")) * 1024
        # last dims allowed (default: the language model's hidden size; the vision tower's 1152-wide
        # all-reduces are not quality-tested with int8 transport)
        self.dims = {int(d) for d in os.getenv("VLLM_RDNA_AR_Q8_DIMS", "2560").split(",") if d}
        # VLLM_RDNA_AR_Q8_STAGGER (default 1 since 2026-09-26; 0 = old grouped all-to-all): exchange the shards in W-1 rounds instead of one grouped
        # all-to-all. Round k: send to rank+k, receive from rank-k -- every card has ONE outgoing and ONE
        # incoming transfer at a time (W flows per round instead of W*(W-1) at once). Same bytes, same order
        # of reduction: bit-identical results. Motivation: every V620 bus drop since 2026-09-25 had the grouped
        # all-to-all enabled, which has each card receive from all three peers at once.
        self.stagger = os.getenv("VLLM_RDNA_AR_Q8_STAGGER", "1") == "1"
        self._logged = False
        logger.info("rdna_ar_q8: int8 two-shot prefill all-reduce enabled (group %d, >= %d KB, %s exchange)",
                    _G, self.min_bytes // 1024, "staggered" if self.stagger else "grouped all-to-all")

    def should_use(self, inp: torch.Tensor) -> bool:
        return (inp.dtype == torch.float16 and inp.is_cuda and inp.dim() >= 2
                and inp.numel() * 2 >= self.min_bytes and inp.shape[-1] in self.dims and inp.shape[-1] % _G == 0
                and inp.is_contiguous() and not torch.cuda.is_current_stream_capturing())

    def all_reduce(self, inp: torch.Tensor) -> torch.Tensor:
        W, H = self.world_size, inp.shape[-1]
        x = inp.view(-1, H)
        m = x.shape[0]
        rp = triton.cdiv(m, W)
        mp = rp * W
        ng = H // _G
        ngp = triton.next_power_of_2(ng)
        rowb = H + 2 * ng
        dev = inp.device
        send = torch.empty(mp * rowb, dtype=torch.uint8, device=dev)
        recv = torch.empty(mp * rowb, dtype=torch.uint8, device=dev)
        red = torch.empty(rp * rowb, dtype=torch.uint8, device=dev)
        gath = torch.empty(mp * rowb, dtype=torch.uint8, device=dev)
        y = torch.empty((mp, H), dtype=torch.float16, device=dev)
        kw = dict(NG=ng, NGP=ngp, G=_G, H=H, ROWB=rowb, num_warps=4)
        _quant_pack[(mp,)](x, send, m, x.stride(0), **kw)
        shard = rp * rowb
        if self.stagger:
            for k in range(1, W):
                dst = (self.rank + k) % W
                src = (self.rank - k) % W
                self.comm.group_start()
                self.comm.send(send[dst * shard:(dst + 1) * shard], dst)
                self.comm.recv(recv[src * shard:(src + 1) * shard], src)
                self.comm.group_end()
        else:
            self.comm.group_start()
            for p in range(W):
                if p != self.rank:
                    self.comm.send(send[p * shard:(p + 1) * shard], p)
                    self.comm.recv(recv[p * shard:(p + 1) * shard], p)
            self.comm.group_end()
        own = send[self.rank * shard:(self.rank + 1) * shard]
        _sum_requant[(rp,)](recv, own, red, rp, self.rank, W=W, **kw)
        self.comm.all_gather(gath, red)
        _dequant[(mp,)](gath, y, **kw)
        if not self._logged:
            logger.info("rdna_ar_q8: first prefill all-reduce %s via int8 two-shot", tuple(inp.shape))
            self._logged = True
        return y[:m].view(inp.shape)
