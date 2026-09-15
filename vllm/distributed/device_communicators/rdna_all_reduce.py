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
VLLM_RDNA_AR_BLOCKS caps the blocks per launch and VLLM_RDNA_AR_PACE (0..127)
idles each wave between strided pushes -- fabric-friendliness knobs (2026-09-01):
fewer, paced push streams into the receiving GPU's root complex, at a few us per
collective (T44: 20 KB at 16/4 blocks = 33/36 us). Peer order is always rank-staggered.
VLLM_RDNA_AR_MAX_KB (default 512) bounds the fast path; larger tensors and
other dtypes take the stock path.

T44b (2026-09-07) -- wedge handling. A collective that hits its spin cap aborts without
writing its output; before, nothing read the sticky flag after boot, so a fabric that drops
or stalls a peer's posted write showed as GPUs pinned at 99 % and generation stopped until
the 300 s engine timeout (two boards reported it). Now the kernel records phase/peer/sequence
in a host-mapped word, `rdna_ar_check()` reads it once per engine step (plain load, no sync),
and on the first abort the process logs the diagnosis, writes a marker under VLLM_CACHE_ROOT
and fails the step. Graph-captured collectives cannot be re-routed in a live process, so the
honest fallback is the NEXT boot, which sees the marker and starts on RCCL (delete the marker
to retry P2P; VLLM_RDNA_AR=0 forces RCCL regardless). VLLM_RDNA_AR_SPIN_CAP sets the polls
per wait before abort (default 2,000,000, ~2 s).
"""

import os
import time

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from vllm.logger import init_logger

logger = init_logger(__name__)

_instances = 0
_MARKER_NAME = "rdna_ar_wedged"


def marker_path() -> str:
    from vllm import envs

    return os.path.join(envs.VLLM_CACHE_ROOT, _MARKER_NAME)


def describe_abort(code: int, rank: int) -> str:
    """Decode the kernel's abort record (layout in rdna_allreduce.cuh) into one sentence."""
    phase = (code >> 8) & 0xF
    peer = (code >> 12) & 0xF
    ms = (code >> 16) & 0xFFFF
    seq = (code >> 32) & 0xFFFFFFFF
    if phase == 1:
        what = "its own blocks never reached the grid barrier (a launch on this GPU stalled)"
    else:
        what = (f"peer rank {peer}'s flag never arrived (the posted P2P write from GPU {peer} "
                "was lost or stalled on this fabric)")
    return f"rank {rank} timed out after ~{ms} ms of spinning at collective #{seq}: {what}"


_active: "RdnaOneShotAllReduce | None | bool" = False  # False = not looked up yet


def rdna_ar_check() -> None:
    """Per-step wedge check for the TP group's instance; a no-op unless the fast path is active."""
    global _active
    if _active is False:
        try:
            from vllm.distributed.parallel_state import get_tp_group

            _active = getattr(get_tp_group().device_communicator, "rdna_ar_comm", None)
        except Exception:  # noqa: BLE001 -- no TP group (single rank / not initialised)
            _active = None
    if _active is not None and not _active.disabled:
        _active.check()


class RdnaOneShotAllReduce:
    def __init__(self, group: ProcessGroup, device: torch.device) -> None:
        global _instances
        from vllm import _custom_ops as ops

        self.disabled = True
        self.handle = -1
        self._ops = ops
        self.rank = dist.get_rank(group=group)
        self.world_size = dist.get_world_size(group=group)
        max_kb = int(os.getenv("VLLM_RDNA_AR_MAX_KB", "64"))  # decode messages; prefill chunks are faster on RCCL
        self.max_bytes = max_kb * 1024
        if not (2 <= self.world_size <= 8):
            return
        # T44b: a previous run on this machine wedged -- stay on RCCL until the marker is removed.
        marker = marker_path()
        if os.path.exists(marker):
            try:
                why = open(marker).read().strip().replace("\n", " ")[:400]
            except OSError:
                why = "unreadable marker"
            logger.warning(
                "rdna_ar: disabled -- a previous run wedged on this machine (%s). Using RCCL for "
                "the small collectives. Delete %s to try the one-shot path again (a slow or "
                "ACS-redirected GPU P2P path is the usual cause), or set VLLM_RDNA_AR=0 to keep "
                "RCCL without this warning.",
                why, marker,
            )
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

        # Boot self-test (2026-08-31): on boards where GPU P2P is slow or broken
        # (ACS redirect, chipset-routed slots, cross-socket paths) init can succeed
        # while every collective then spins to its ~2 s cap and aborts WITHOUT
        # writing the output -- silent corruption plus stalls that get reported by
        # whatever waits next (usually the PLE handshake). Verify the path with
        # known patterns before trusting it; all ranks agree on the verdict.
        err = self._self_test(device)
        dist.all_gather_object(status, err, group=group)
        if any(s is not None for s in status):
            logger.warning(
                "rdna_ar: disabled -- boot self-test failed on some rank "
                "(weak GPU peer-to-peer on this board? falling back to RCCL): %s",
                [s for s in status if s is not None][:1],
            )
            return
        dist.barrier(group=group)
        self.disabled = False
        logger.info(
            "rdna_ar: one-shot all-reduce active (handle %d, rank %d/%d, devices %s, max %d KB; blocks cap %s, pace %s)",
            self.handle, self.rank, self.world_size, gathered, max_kb, os.getenv("VLLM_RDNA_AR_BLOCKS", "auto"), os.getenv("VLLM_RDNA_AR_PACE", "0"))

    def _self_test(self, device: torch.device) -> str | None:
        """Verified all-reduces on the fast path at three sizes; returns an error string or None.

        Per size: one untimed warm-up collective (first launch loads the code object and
        first-touches the IPC mappings), then REPEATS timed collectives judged on their MINIMUM.
        The minimum is what a slow P2P path cannot hide; the maximum only measures how far the
        ranks were out of step at boot -- which once failed a healthy machine at 59 ms (2026-09-01).
        Every collective is checked for the spin-cap timeout and for the exact fp32 result.
        """
        import time

        REPEATS = 3
        try:
            with torch.cuda.device(device):
                for trial, numel in enumerate((1024, 4096, self.max_bytes // 2)):
                    inp = torch.full(
                        (numel,), float(self.rank + 1) * (trial + 1), dtype=torch.float16, device=device
                    )
                    expect = float((trial + 1) * self.world_size * (self.world_size + 1) // 2)
                    times: list[float] = []
                    for rep in range(REPEATS + 1):  # rep 0 = warm-up, untimed
                        t0 = time.perf_counter()
                        out = self._ops.rdna_ar_all_reduce(self.handle, inp)
                        torch.cuda.synchronize(device)
                        dt = time.perf_counter() - t0
                        code = int(self._ops.rdna_ar_timeout_info(self.handle))
                        if code:
                            return (f"spin-cap timeout in self-test trial {trial} rep {rep} ({dt * 1e3:.0f} ms): "
                                    f"{describe_abort(code, self.rank)}")
                        if not bool((out == expect).all()):
                            got = out.float().mean().item()
                            return f"wrong result in self-test trial {trial} rep {rep}: mean {got:.2f}, expected {expect:.1f}"
                        if rep > 0:
                            times.append(dt)
                    best = min(times)
                    if best > 0.05:
                        return (f"self-test trial {trial} best {best * 1e3:.1f} ms of {REPEATS} for {numel * 2} bytes "
                                f"(all: {', '.join(f'{t * 1e3:.1f}' for t in times)} ms; P2P too slow, RCCL will be faster)")
        except Exception as e:  # noqa: BLE001
            return str(e)
        return None

    def should_use(self, inp: torch.Tensor) -> bool:
        return (not self.disabled) and self._ops.rdna_ar_can(self.handle, inp)

    def all_reduce(self, inp: torch.Tensor) -> torch.Tensor:
        return self._ops.rdna_ar_all_reduce(self.handle, inp)

    def timed_out(self) -> bool:
        return (not self.disabled) and self._ops.rdna_ar_timed_out(self.handle)

    def _write_marker(self, msg: str) -> str | None:
        path = marker_path()
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} world={self.world_size} {msg}\n")
            return path
        except OSError as e:
            logger.warning("rdna_ar: could not write the wedge marker %s: %s", path, e)
            return None

    def check(self) -> None:
        """Once per engine step (T44b). A collective that hit its spin cap returned WITHOUT
        writing its output, so the current step is already wrong and every later collective
        would spin to its cap too. Diagnose, leave the marker for the next boot, fail now."""
        if self.disabled:
            return
        code = int(self._ops.rdna_ar_timeout_info(self.handle))
        if code == 0:
            return
        self.disabled = True
        msg = describe_abort(code, self.rank)
        path = self._write_marker(msg)
        logger.error(
            "rdna_ar: WEDGED -- %s. The one-shot all-reduce is disabled for this process; "
            "graph-captured steps cannot be re-routed live, so the engine stops here instead "
            "of grinding to the execute timeout. The next boot starts on RCCL automatically "
            "(marker: %s); VLLM_RDNA_AR=0 forces RCCL; delete the marker to retry P2P after "
            "checking ACS / IOMMU / slot topology (docs/rdna2/TROUBLESHOOTING.md).",
            msg, path or "not written",
        )
        raise RuntimeError(f"rdna_ar wedged: {msg} (see the log line above)")
