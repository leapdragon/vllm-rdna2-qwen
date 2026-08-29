# SPDX-License-Identifier: Apache-2.0
"""T44: push-based one-shot all-reduce for small TP messages on gfx1030 (2..8 ranks).

vLLM's custom all-reduce is unavailable on RDNA (platform gate, and the XGMI
"fully connected" check refuses 4 PCIe GPUs), and RCCL costs ~156 us per 20 KB
all-reduce on this 4-card PCIe topology -- 119 of them per decode step, 41% of
GPU time after T43. This kernel (csrc/rocm/rdna_allreduce.cuh) pushes each
rank's contribution straight into every peer's uncached staging buffer, signals
through host-coherent flags, and reduces in fixed rank order so all ranks
produce bit-identical results. Graph-capture safe (sequence numbers live on the
device, not in kernel arguments).

One instance per process group (vLLM builds several GroupCoordinators over the
same ranks); the extension hands out a handle per instance. Initialisation is
collective-safe: every rank runs every barrier, and a failure on any rank
disables the instance on all ranks (a rank that bailed out of an ordered
barrier loop deadlocked its peers in boot 4 of T44).

Enabled by default on gfx10x for world sizes 2..8; VLLM_RDNA_AR=0 disables,
VLLM_RDNA_AR_MAX_KB (default 512) bounds the fast path; larger tensors and
other dtypes take the stock path.
"""

import os

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from vllm.logger import init_logger

logger = init_logger(__name__)

_instances = 0


class RdnaOneShotAllReduce:
    def __init__(self, group: ProcessGroup, device: torch.device) -> None:
        global _instances
        from vllm import _custom_ops as ops

        self.disabled = True
        self.handle = -1
        self._ops = ops
        self.rank = dist.get_rank(group=group)
        self.world_size = dist.get_world_size(group=group)
        max_kb = int(os.getenv("VLLM_RDNA_AR_MAX_KB", "512"))
        self.max_bytes = max_kb * 1024
        if not (2 <= self.world_size <= 8):
            return
        dev_idx = device.index if device.index is not None else torch.cuda.current_device()
        gathered: list = [None] * self.world_size
        dist.all_gather_object(gathered, int(dev_idx), group=group)
        device_ids = torch.tensor(gathered, dtype=torch.int64)
        # rank 0 names the flag page; one per instance
        my_name = f"/vllm_rdna_ar_{os.getpid()}_{_instances}"
        names: list = [None] * self.world_size
        dist.all_gather_object(names, my_name, group=group)
        shm_name = names[0]
        _instances += 1

        # ordered init: rank 0 (re)creates the flag page before anyone opens it.
        # Every rank executes every barrier no matter what happens locally.
        packed = None
        err: str | None = None
        for r in range(self.world_size):
            if r == self.rank and err is None:
                try:
                    with torch.cuda.device(device):
                        packed = ops.rdna_ar_init(
                            self.rank, self.world_size, device_ids, self.max_bytes, shm_name
                        )
                except Exception as e:  # noqa: BLE001
                    err = str(e)
            dist.barrier(group=group)
        status: list = [None] * self.world_size
        dist.all_gather_object(status, err, group=group)
        if any(s is not None for s in status):
            logger.warning(
                "rdna_ar: disabled for this group -- init failed on some rank: %s",
                [s for s in status if s is not None][:1],
            )
            return
        raw = packed.numpy().tobytes()
        self.handle = int.from_bytes(raw[:8], "little", signed=True)
        handles: list = [None] * self.world_size
        dist.all_gather_object(handles, raw[8:], group=group)
        buf = torch.frombuffer(bytearray(b"".join(handles)), dtype=torch.uint8).view(
            self.world_size, -1
        )
        err = None
        try:
            with torch.cuda.device(device):
                ops.rdna_ar_connect(self.handle, buf.contiguous())
        except Exception as e:  # noqa: BLE001
            err = str(e)
        dist.all_gather_object(status, err, group=group)
        if any(s is not None for s in status):
            logger.warning("rdna_ar: disabled -- connect failed: %s", status)
            return
        dist.barrier(group=group)
        self.disabled = False
        logger.info(
            "rdna_ar: one-shot all-reduce active (handle %d, rank %d/%d, devices %s, max %d KB)",
            self.handle, self.rank, self.world_size, gathered, max_kb,
        )

    def should_use(self, inp: torch.Tensor) -> bool:
        return (not self.disabled) and self._ops.rdna_ar_can(self.handle, inp)

    def all_reduce(self, inp: torch.Tensor) -> torch.Tensor:
        return self._ops.rdna_ar_all_reduce(self.handle, inp)

    def timed_out(self) -> bool:
        return (not self.disabled) and self._ops.rdna_ar_timed_out(self.handle)
