# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dedicated CPU-offload process for PLE embedding layers.

This module implements a standalone process that:
1. Loads only the :class:`PleOffloadLayer` weights into CPU memory.
2. Accepts per-step computation requests from GPU worker processes.
3. Runs ``forward_impl()`` on CPU, copies results to every TP worker's GPU
   output buffer for the requesting DP rank, and signals the corresponding
   IPC semaphore.

The TP workers within one DP rank receive identical inputs, so the CPU result
is computed once per DP rank and fanned out to all of its TP ranks.

Class structure mirrors the GPU worker pattern in multiproc_executor.py:

  PleOffloadWorkerHandle -- handle held by the spawning GPU worker
  PleOffloadWorker       -- process lifecycle and READY handshake
  PleOffloadRunner       -- owns weights and serves inference requests
"""

import contextlib
import json
import mmap as _mmap
import multiprocessing.process
import os
import pickle
import signal
import tempfile
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from multiprocessing.connection import Connection
from typing import Any, cast

import time
import msgspec
import torch
import torch.distributed as dist
import zmq

import vllm.envs as envs
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed.parallel_state import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.ple_offload_layer import (
    CpuGpuSemaphore,
    PleOffloadLayer,
    mark_as_offload_worker,
)
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader
from vllm.model_executor.model_loader.utils import (
    initialize_model,
    process_weights_after_loading,
)
from vllm.model_executor.model_loader.weight_utils import initialize_dummy_weights
from vllm.utils.system_utils import decorate_logs, get_mp_context
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.ple_offload.protocol import (
    _PLE_OFFLOAD_REQUEST_DECODER,
    PleOffloadRegistration,
    PleOffloadRequest,
)

logger = init_logger(__name__)


@dataclass
class PleOffloadOutputTarget:
    """GPU output destination and semaphore for one TP worker."""

    tp_rank: int
    gpu_output_buffer: torch.Tensor  # IPC-mapped GPU buffer for this TP worker
    sem: CpuGpuSemaphore  # semaphore paired with gpu_output_buffer
    copy_stream: torch.cuda.Stream
    done_seq_buf: torch.Tensor | None = None  # shared CPU counter of completed requests


@dataclass
class PleOffloadInputBuffers:
    """Shared-memory input buffers registered for one DP rank."""

    input_ids_buf: torch.Tensor  # int32 (max_num_tokens,)
    query_start_loc_buf: torch.Tensor  # int32 (max_num_reqs + 1,)
    ngram_context_buf: torch.Tensor | None  # int32 (max_num_reqs, ngram_context_len)


@dataclass
class PleOffloadWorkerHandle:
    """Resources owned by the GPU worker that spawned the offload process."""

    proc: Any
    death_writer: Connection | None
    ready_pipe_reader: Connection | None

    def close(self) -> None:
        """Release all process resources. Safe to call more than once."""
        if self.ready_pipe_reader is not None:
            self.ready_pipe_reader.close()
            self.ready_pipe_reader = None
        if self.death_writer is not None:
            self.death_writer.close()
            self.death_writer = None
        # First allow the child to exit after observing the closed death pipe.
        if self.proc.is_alive():
            self.proc.join(timeout=5)
        # Fall back to SIGTERM if graceful shutdown times out.
        if self.proc.is_alive():
            self.proc.terminate()
            self.proc.join(timeout=5)
        # Use SIGKILL as the final fallback for a stuck child.
        if self.proc.is_alive():
            self.proc.kill()
            self.proc.join(timeout=5)


def _init_offload_distributed() -> None:
    """Initialize the single-rank Gloo world required by TP-aware layers."""
    if dist.is_initialized():
        return

    # VocabParallelEmbedding reads the TP process group during construction.
    # The offload process owns the full embedding table, so it uses an isolated
    # TP1/PP1 Gloo world and never joins the GPU workers' NCCL groups.
    store_dir = tempfile.mkdtemp(prefix="vllm_ple_offload_")
    init_distributed_environment(
        world_size=1,
        rank=0,
        distributed_init_method=f"file://{store_dir}/store",
        local_rank=0,
        backend="gloo",
    )
    # initialize_model_parallel reads the active VllmConfig in the current
    # vLLM version. Explicitly configure DP1/TP1/PP1 to match the isolated
    # world, regardless of any DP environment variables inherited from the GPU
    # worker. The real DP/TP configuration is used later for model construction,
    # registration, and request routing.
    offload_config = VllmConfig()
    offload_parallel_config = offload_config.parallel_config
    offload_parallel_config.data_parallel_size = 1
    offload_parallel_config.data_parallel_size_local = 1
    offload_parallel_config.data_parallel_rank = 0
    offload_parallel_config.data_parallel_rank_local = 0
    offload_parallel_config.data_parallel_index = 0
    offload_parallel_config.tensor_parallel_size = 1
    offload_parallel_config.pipeline_parallel_size = 1
    offload_parallel_config.prefill_context_parallel_size = 1
    offload_parallel_config.decode_context_parallel_size = 1
    offload_parallel_config.world_size = 1
    offload_parallel_config.nnodes = 1
    offload_parallel_config.node_rank = 0
    with set_current_vllm_config(offload_config):
        ensure_model_parallel_initialized(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            backend="gloo",
        )
    logger.info(
        "Distributed environment initialized (backend=gloo, rank=0, world_size=1)."
    )


class PleOffloadWorker:
    """Manage process creation, READY handshake, and the child entry point."""

    READY_STR = "READY"

    @staticmethod
    def make_process(
        vllm_config: VllmConfig,
        num_workers: int,
        ipc_addr: str,
    ) -> PleOffloadWorkerHandle:
        """Spawn one CPU offload process for all local DP and TP workers."""
        context = get_mp_context()
        ready_reader, ready_writer = context.Pipe(duplex=False)
        death_reader, death_writer = context.Pipe(duplex=False)
        proc = context.Process(
            target=PleOffloadWorker.proc_main,
            kwargs={
                "vllm_config": vllm_config,
                "num_workers": num_workers,
                "ipc_addr": ipc_addr,
                "ready_pipe": (ready_reader, ready_writer),
                "death_pipe": death_reader,
            },
            name="PleOffloadWorker",
            daemon=True,
        )

        # Python normally forbids a daemon WorkerProc from spawning children.
        # vLLM owns this process through death_pipe and explicit shutdown, so
        # temporarily clear the daemon flag while the child is created.
        parent = multiprocessing.process._current_process  # type: ignore[attr-defined]
        saved_daemon = parent._config.get("daemon")
        parent._config["daemon"] = False
        try:
            proc.start()
        finally:
            parent._config["daemon"] = saved_daemon
        ready_writer.close()
        return PleOffloadWorkerHandle(
            proc=proc,
            death_writer=death_writer,
            ready_pipe_reader=ready_reader,
        )

    @staticmethod
    def wait_for_ready(handle: PleOffloadWorkerHandle) -> None:
        """Wait until weights and all GPU registrations are ready to serve."""
        reader = handle.ready_pipe_reader
        if reader is None:
            return
        if not reader.poll(envs.VLLM_PLE_OFFLOAD_READY_TIMEOUT):
            raise TimeoutError(
                "PLE offload worker did not become ready within "
                f"{envs.VLLM_PLE_OFFLOAD_READY_TIMEOUT}s."
            )
        try:
            message = reader.recv()
        except EOFError as error:
            raise RuntimeError("PLE offload worker exited during startup") from error
        finally:
            reader.close()
            handle.ready_pipe_reader = None
        if message.get("status") != PleOffloadWorker.READY_STR:
            raise RuntimeError(
                "PLE offload worker failed during startup: "
                f"{message.get('error', 'unknown error')}"
            )
        layer_names = message["layer_names"]
        logger.info(
            "Worker ready - %d PleOffloadLayer(s): %s",
            len(layer_names),
            layer_names,
        )

    @staticmethod
    def proc_main(
        vllm_config: VllmConfig,
        num_workers: int,
        ipc_addr: str,
        ready_pipe: tuple[Connection, Connection],
        death_pipe: Connection,
    ) -> None:
        """Load PLE weights, accept registrations, and run the request loop."""
        decorate_logs("PleOffloadWorker")
        ready_reader, ready_writer = ready_pipe
        ready_reader.close()
        shutdown_event = threading.Event()

        def monitor_parent() -> None:
            try:
                death_pipe.recv()
            except EOFError:
                logger.info("Parent exited, shutting down.")
                shutdown_event.set()

        def handle_signal(_signum: int, _frame: object) -> None:
            shutdown_event.set()

        signal.signal(signal.SIGTERM, handle_signal)
        signal.signal(signal.SIGINT, handle_signal)
        threading.Thread(
            target=monitor_parent,
            daemon=True,
            name="PleOffloadDeathMonitor",
        ).start()

        zmq_context: zmq.Context | None = None
        pull_socket: zmq.Socket | None = None
        try:
            # The flag lets PleOffloadLayer subclasses execute their complete
            # constructors instead of becoming empty GPU-worker placeholders.
            mark_as_offload_worker()

            # Initialize Gloo before installing the real VllmConfig. This keeps
            # the CPU process in an isolated rank-zero, world-size-one group.
            _init_offload_distributed()

            # Model components read the active VllmConfig while the meta model
            # is constructed, so keep the context around runner initialization.
            with set_current_vllm_config(vllm_config):
                runner = PleOffloadRunner(vllm_config)

            zmq_context = zmq.Context()
            pull_socket = zmq_context.socket(zmq.PULL)
            pull_socket.bind(ipc_addr)
            logger.info(
                "Bound IPC address %s; waiting for %d GPU worker registration(s).",
                ipc_addr,
                num_workers,
            )

            # READY means that the process can immediately serve requests. Wait
            # for every DP/TP worker to register before notifying the parent.
            runner.accept_registrations(pull_socket, num_workers)
            ready_writer.send(
                {
                    "status": PleOffloadWorker.READY_STR,
                    "layer_names": sorted(runner.layer_names),
                }
            )
            ready_writer.close()
            ready_writer = None  # type: ignore[assignment]

            runner.busy_loop(pull_socket, shutdown_event)
        except Exception as error:
            logger.exception("Unexpected failure in PLE offload worker.")
            if ready_writer is not None:
                with contextlib.suppress(Exception):
                    ready_writer.send({"status": "FAILURE", "error": repr(error)})
            raise
        finally:
            if pull_socket is not None:
                pull_socket.close(linger=0)
            if zmq_context is not None:
                zmq_context.term()
            if ready_writer is not None:
                ready_writer.close()
            death_pipe.close()


def _ple_disk_shard_of(mapped_name: str) -> str | None:
    """"<layer>.a.b.shard_3.weight" -> "<layer>.a.b" (the parameter the shard fills)."""
    import re

    m = re.match(r"^(.*)\.shard_\d+\.weight$", mapped_name)
    return m.group(1) if m else None


def _ple_disk_dir() -> str | None:
    return envs.VLLM_PLE_DISK_OFFLOAD_DIR or None


_PLE_DISK_MAPS: dict[str, object] = {}


def _disk_backed_tensor(path: str, shape: tuple[int, ...], dtype: torch.dtype,
                        writable: bool) -> torch.Tensor:
    """Map ``path`` as a tensor of ``shape``/``dtype``.

    numpy has no bfloat16, so the file is mapped with a same-width integer dtype
    and reinterpreted. ``writable`` selects a shared read-write mapping (first
    boot, shard writes must reach the file) versus copy-on-write (steady state).
    MADV_RANDOM is applied either way: gathers are random-access and readahead
    only evicts useful pages.
    """
    import numpy as np

    _NP = {torch.bfloat16: (np.uint16, torch.uint16), torch.float16: (np.uint16, torch.uint16),
           torch.float32: (np.uint32, torch.uint32), torch.float8_e4m3fn: (np.uint8, torch.uint8)}
    np_dtype, torch_int = _NP[dtype]
    arr = np.memmap(path, dtype=np_dtype, mode="r+" if writable else "c", shape=shape)
    with contextlib.suppress(Exception):
        arr._mmap.madvise(_mmap.MADV_RANDOM)  # noqa: SLF001 - numpy has no public madvise
    _PLE_DISK_MAPS[path] = arr
    return torch.from_numpy(arr).view(dtype)


# ---------------------------------------------------------------------------
# Quantized PLE sidecar (primitive-ai/Qwen3.8-Flash-Next-PLE-quant).
#
# The bf16 n-gram table is 95-102 GB, which does not fit this host's RAM, so the
# disk path below pages it and every decoded token pays random read latency. The
# sidecar ships the same table at int4 g16 (32 GB) / fp8 per-row (49 GB), small
# enough to sit in page cache. Lifted from that repo's worker overlay: pure torch
# plus safetensors mmap, no CUDA, so it runs unchanged on ROCm.
# ---------------------------------------------------------------------------

class _PleQuantTable:
    """Shard-mmapped quantized n-gram table; gathers dequantize to BF16."""

    ROWS_PER_SHARD = 2_500_012

    def __init__(self, quant_dir: str, total_rows: int, width: int) -> None:
        import json
        import os

        from safetensors import safe_open

        meta = json.load(open(os.path.join(quant_dir, "META.json")))
        self.layout = meta["layout"]
        assert meta["rows"] == total_rows and meta["width"] == width, (
            f"sidecar built for {meta['rows']}x{meta['width']}, "
            f"table is {total_rows}x{width}"
        )
        n_shards = meta["shards"]
        assert n_shards * self.ROWS_PER_SHARD == total_rows, "non-uniform shards"
        # Order matters: the e2m1 layout string contains "e4m3" (its scale dtype).
        if "e2m1" in self.layout:
            key = "weight_e2m1"
        elif "e4m3" in self.layout:
            key = "weight_fp8"
        else:
            key = "weight_i4"
        self._q, self._s, self._s2 = [], [], []
        for n in range(n_shards):
            f = safe_open(os.path.join(quant_dir, f"shard_{n}.safetensors"),
                          framework="pt")
            self._q.append(f.get_tensor(key))
            self._s.append(f.get_tensor("weight_scale"))
            self._s2.append(
                f.get_tensor("weight_scale_2").item()
                if "weight_scale_2" in f.keys() else 1.0
            )
        self.width = width
        self._lut = None
        logger.info("PLE quant table: %s, %d shards mmapped from %s",
                    self.layout, n_shards, quant_dir)

    def gather_into(self, ids: torch.Tensor, out: torch.Tensor) -> None:
        ids = ids.long()
        shard = ids // self.ROWS_PER_SHARD
        local = ids - shard * self.ROWS_PER_SHARD
        order = torch.argsort(shard)
        s_sorted, l_sorted = shard[order], local[order]
        uniq, counts = torch.unique_consecutive(s_sorted, return_counts=True)
        pos = 0
        for s, c in zip(uniq.tolist(), counts.tolist()):
            sel = l_sorted[pos:pos + c]
            rows = self._dequant(s, sel)
            out[order[pos:pos + c]] = rows.to(out.dtype)
            pos += c

    def _dequant(self, s: int, sel: torch.Tensor) -> torch.Tensor:
        if "e2m1" in self.layout:
            if self._lut is None:
                mags = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
                self._lut = torch.tensor(mags + [-m for m in mags],
                                         dtype=torch.float32)
            packed = self._q[s].index_select(0, sel)
            lo = (packed & 0xF).long()
            hi = (packed >> 4).long()
            nib = torch.stack((lo, hi), dim=-1).view(packed.shape[0], self.width)
            scale = self._s[s].index_select(0, sel).to(torch.float32)
            g = self.width // scale.shape[1]
            return (self._lut[nib]
                    * scale.repeat_interleave(g, dim=1)
                    * self._s2[s])
        if "e4m3" in self.layout:
            q = self._q[s].index_select(0, sel).to(torch.float32)
            return q * self._s[s].index_select(0, sel)[:, None]
        packed = self._q[s].index_select(0, sel)
        lo = (packed & 0xF).to(torch.int16)
        hi = (packed >> 4).to(torch.int16)
        nib = torch.stack((lo, hi), dim=-1).view(packed.shape[0], self.width)
        scale = self._s[s].index_select(0, sel).to(torch.float32)
        g = self.width // scale.shape[1]
        return (nib.to(torch.float32) - 8) * scale.repeat_interleave(g, dim=1)


def _ple_quant_dir() -> str | None:
    import os

    return os.environ.get("VLLM_PLE_QUANT_DIR") or None


def _ple_quant_attach(layer_name: str, layer: torch.nn.Module,
                      quant_dir: str) -> str | None:
    """Swap the layer's table for a sidecar-backed quant store.

    Returns the stubbed parameter's name, or None when the layer has no
    parameter large enough to be a table (>= 1 GiB).
    """
    named = sorted(layer.named_parameters(), key=lambda kv: kv[1].numel(), reverse=True)
    if not named or named[0][1].numel() * named[0][1].element_size() < (1 << 30):
        return None
    pname, param = named[0]
    rows, width = param.shape
    owner = layer
    parts = pname.split(".")
    for p in parts[:-1]:
        owner = getattr(owner, p)
    owner._ple_quant = _PleQuantTable(quant_dir, rows, width)
    # Stub before anything writes the parameter: the original 95 GB allocation
    # is lazy virtual memory and stays unmaterialized.
    stub = torch.empty(0, width, dtype=param.dtype)
    target = getattr(owner, parts[-1])
    if target.is_meta:
        # Same constraint as _ple_disk_attach: this runs before weights are
        # materialized, so the parameter is still on meta and `.data = stub`
        # trips set_data's type check (meta is not cpu). Swap the Parameter and
        # carry over vLLM's weight-loading metadata.
        new_param = torch.nn.Parameter(stub, requires_grad=False)
        for attr, value in vars(target).items():
            setattr(new_param, attr, value)
        setattr(owner, parts[-1], new_param)
    else:
        target.data = stub
    logger.info("PLE quant: %s.%s stubbed, gathers served from sidecar.",
                layer_name, pname)
    return pname


def _ple_disk_attach(layer_name: str, layer: torch.nn.Module,
                     disk_dir: str) -> tuple[str, bool] | None:
    """Swap the layer's largest parameter (the n-gram table) for a disk-backed map.

    Returns ``(param_name, file_complete)`` or ``None`` when the layer has no
    parameter large enough to be worth spilling (>= 1 GiB).
    """
    import os

    named = sorted(layer.named_parameters(), key=lambda kv: kv[1].numel(), reverse=True)
    if not named or named[0][1].numel() * named[0][1].element_size() < (1 << 30):
        return None
    pname, param = named[0]
    shape, dtype = tuple(param.shape), param.dtype
    nbytes = param.numel() * param.element_size()
    os.makedirs(disk_dir, exist_ok=True)
    base = os.path.join(disk_dir, layer_name.replace("/", "_") + "." + pname)
    bin_path, done_path = base + ".bin", base + ".done.json"

    complete = False
    if os.path.exists(done_path) and os.path.exists(bin_path)             and os.path.getsize(bin_path) == nbytes:
        meta = json.load(open(done_path))
        complete = meta.get("shape") == list(shape) and meta.get("dtype") == str(dtype)
    if not complete:
        with contextlib.suppress(FileNotFoundError):
            os.remove(done_path)
        with open(bin_path, "ab") as f:
            f.truncate(nbytes)

    mapped = _disk_backed_tensor(bin_path, shape, dtype, writable=not complete)
    # Replace the parameter data in place; module structure and names are unchanged,
    # so load_weights and the gather path are untouched.
    owner = layer
    parts = pname.split(".")
    for p in parts[:-1]:
        owner = getattr(owner, p)
    target = getattr(owner, parts[-1])
    if target.is_meta:
        # The offload worker builds the model under torch.device("meta") and this
        # attach runs before any weights are materialized, so `.data = mapped`
        # hits set_data's type check (meta is not cpu). Swap the Parameter itself
        # and carry over vLLM's weight-loading metadata (weight_loader, output_dim,
        # ...), which set_weight_attrs stores in the instance __dict__.
        new_param = torch.nn.Parameter(mapped, requires_grad=False)
        for attr, value in vars(target).items():
            setattr(new_param, attr, value)
        setattr(owner, parts[-1], new_param)
    else:
        target.data = mapped
    logger.info(
        "PLE disk offload: %s.%s -> %s (%.1f GiB, %s)",
        layer_name, pname, bin_path, nbytes / (1 << 30),
        "reusing finished file" if complete else "first boot, writing through",
    )
    return pname, complete


def _ple_disk_finalize(layer_name: str, layer: torch.nn.Module, pname: str,
                       disk_dir: str) -> None:
    """Flush the written mapping, record completion, and remap copy-on-write."""
    import os

    owner = layer
    parts = pname.split(".")
    for p in parts[:-1]:
        owner = getattr(owner, p)
    param = getattr(owner, parts[-1])
    base = os.path.join(disk_dir, layer_name.replace("/", "_") + "." + pname)
    arr = _PLE_DISK_MAPS.get(base + ".bin")
    if arr is not None:
        with contextlib.suppress(Exception):
            arr.flush()
    json.dump({"shape": list(param.shape), "dtype": str(param.dtype)},
              open(base + ".done.json", "w"))
    param.data = _disk_backed_tensor(base + ".bin", tuple(param.shape), param.dtype,
                                     writable=False)
    logger.info("PLE disk offload: %s.%s finalized and remapped copy-on-write.",
                layer_name, pname)


class PleOffloadRunner:
    """Own all discovered PLE tables and serve every local DP rank."""

    def __init__(self, vllm_config: VllmConfig) -> None:
        self.vllm_config = vllm_config
        self._clamp_input_ids = (
            getattr(vllm_config, "speculative_config", None) is not None
        )
        # name -> PleOffloadLayer (CPU)
        self._layers: dict[str, PleOffloadLayer] = {}
        # dp_rank -> layer_name -> one destination per TP rank
        self._worker_targets: dict[int, dict[str, list[PleOffloadOutputTarget]]] = {}
        # Each (dp_rank, layer_name) pair owns a separate pinned scratch buffer.
        # Sharing one buffer is unsafe because an asynchronous H2D copy may still
        # be reading it when another layer or DP rank starts writing.
        self._pinned_bufs: dict[int, dict[str, torch.Tensor]] = {}
        # Shared-memory inputs are registered once per DP rank by TP rank zero.
        self._input_bufs: dict[int, PleOffloadInputBuffers] = {}
        self._load_weights()

    @property
    def layer_names(self) -> list[str]:
        """Return PleOffloadLayer names in model traversal order."""
        return list(self._layers)

    def _load_weights(self) -> None:
        """Load only :class:`PleOffloadLayer` subtrees into CPU memory.

        Strategy:
        1. Build the entire model on ``meta`` so non-offloaded parameters use no
           physical memory. PleOffloadLayer constructors explicitly target CPU.
        2. Discover all PleOffloadLayer modules from the complete model.
        3. Stream the checkpoint through a prefix filter so only matching PLE
           tensors are materialized and passed to ``model.load_weights``.
        4. Run post-load processing only on the CPU-owned PLE subtrees.
        """
        model_config = self.vllm_config.model_config
        load_config = self.vllm_config.load_config

        # Step 1: build complete structure, while only PLE subtrees allocate CPU
        # memory. All transformer, MoE, and vision parameters remain on meta.
        logger.info("Initializing model structure for PLE weight discovery ...")
        model_dtype = cast(torch.dtype, model_config.dtype)
        with set_default_torch_dtype(model_dtype), torch.device("meta"):
            model = initialize_model(
                vllm_config=self.vllm_config,
                model_config=model_config,
            )

        # Step 2: preserve named_modules DFS order so CPU execution follows the
        # same layer order as the GPU model forward.
        offload_layers = {
            name: module
            for name, module in model.named_modules()
            if isinstance(module, PleOffloadLayer)
        }
        if not offload_layers:
            raise RuntimeError(
                "VLLM_PLE_CPU_OFFLOAD is enabled, but no PleOffloadLayer "
                "was found in the initialized model"
            )
        logger.info(
            "Found %d PleOffloadLayer(s): %s",
            len(offload_layers),
            sorted(offload_layers),
        )
        offload_prefixes = tuple(f"{name}." for name in offload_layers)

        quant_dir = _ple_quant_dir()
        # A quantized sidecar replaces the checkpoint table outright, so the
        # bf16 disk-paging path is not used when one is configured.
        disk_dir = _ple_disk_dir() if quant_dir is None else None
        disk_attached: dict[str, str] = {}
        disk_complete_params: set[str] = set()
        disk_complete_tables: tuple[str, ...] = ()
        if quant_dir is not None:
            # The sidecar supplies the table, so its parameter must be marked
            # already-complete: the checkpoint no longer carries it (we do not
            # even download that shard) and the post-load completeness check
            # would otherwise fail on
            # '...ple_embedding.ngram_embedding.weight'. Same bookkeeping the
            # disk path does for a finished file.
            table_prefixes = []
            for layer_name, layer in offload_layers.items():
                pname = _ple_quant_attach(layer_name, layer, quant_dir)
                if pname is None:
                    continue
                full = f"{layer_name}.{pname}"
                disk_complete_params.add(full)
                table_prefixes.append(full.rsplit(".", 1)[0])
            disk_complete_tables = tuple(table_prefixes)
        if disk_dir is not None:
            table_prefixes = []
            for layer_name, layer in offload_layers.items():
                attached = _ple_disk_attach(layer_name, layer, disk_dir)
                if attached is None:
                    continue
                pname, complete = attached
                disk_attached[layer_name] = pname
                if complete:
                    full = f"{layer_name}.{pname}"
                    disk_complete_params.add(full)
                    # "...ngram_embedding.weight" -> "...ngram_embedding": the module
                    # whose shard_N.weight checkpoint tensors fill this table.
                    table_prefixes.append(full.rsplit(".", 1)[0])
            # Shard tensors that land inside an already-finished table are not
            # re-read from the checkpoint: mapping the file replaces them.
            disk_complete_tables = tuple(table_prefixes)

        # Step 3: filter checkpoint tensors before model.load_weights(). The
        # conditional-generation checkpoint uses HF names such as
        # ``model.language_model.*`` while named_modules exposes mapped vLLM
        # names such as ``language_model.model.*``. Apply the model mapper only
        # for matching, then yield the original pair so load_weights performs
        # its normal single mapping pass.
        mapper = getattr(model, "hf_to_vllm_mapper", None)
        matched_checkpoint_tensors = 0

        def offload_only_iter(
            weights: Iterable[tuple[str, torch.Tensor]],
        ) -> Iterable[tuple[str, torch.Tensor]]:
            nonlocal matched_checkpoint_tensors
            for weight_name, tensor in weights:
                mapped_name: str | None = weight_name
                if mapper is not None:
                    mapped_names = mapper.apply_list([weight_name])
                    mapped_name = mapped_names[0] if mapped_names else None
                if mapped_name is not None and mapped_name.startswith(offload_prefixes):
                    matched_checkpoint_tensors += 1
                    if disk_complete_tables:
                        table = _ple_disk_shard_of(mapped_name)
                        if table is not None and table.startswith(disk_complete_tables):
                            continue
                    yield weight_name, tensor

        loader = get_model_loader(load_config)
        if isinstance(loader, DummyModelLoader):
            logger.info(
                "Initializing dummy weights for %d PleOffloadLayer(s) ...",
                len(offload_layers),
            )
            for layer in offload_layers.values():
                # The model was built under torch.device("meta"), and
                # initialize_dummy_weights fills existing storage -- it cannot
                # materialize meta tensors. The DefaultModelLoader path gets
                # storage implicitly by copying checkpoint data in; the dummy
                # path has to ask for it, or the layer's parameters and its
                # persistent buffers (layer_multipliers, ngram_heads_*) stay on
                # meta and the first forward dies with
                # "Tensor on device meta is not on the expected device cpu".
                layer.to_empty(device=torch.device("cpu"))
                initialize_dummy_weights(layer, model_config)
        elif isinstance(loader, DefaultModelLoader):
            all_weights = loader.get_all_weights(model_config, model)
            loaded_params = model.load_weights(offload_only_iter(all_weights))
            if matched_checkpoint_tensors == 0:
                raise RuntimeError(
                    "PLE offload checkpoint filter matched no weights for "
                    f"layers: {sorted(offload_layers)}"
                )

            expected_offload_params = {
                f"{layer_name}.{param_name}"
                for layer_name, layer in offload_layers.items()
                for param_name, _ in layer.named_parameters()
            }
            loaded_offload_entries = {
                name for name in loaded_params if name.startswith(offload_prefixes)
            }
            loaded_expected_params = expected_offload_params.intersection(loaded_params)
            missing_offload_params = sorted(
                expected_offload_params.difference(loaded_expected_params)
                - disk_complete_params
            )
            if missing_offload_params:
                raise RuntimeError(
                    "PLE offload checkpoint did not load all materialized "
                    f"parameters: {missing_offload_params}"
                )
            logger.info(
                "PLE offload matched %d checkpoint tensor(s), loaded %d "
                "offload entries, and verified %d/%d materialized "
                "parameter(s) for layers: %s",
                matched_checkpoint_tensors,
                len(loaded_offload_entries),
                len(loaded_expected_params),
                len(expected_offload_params),
                sorted(offload_layers),
            )
        else:
            raise NotImplementedError(
                "PLE offload requires the default or dummy model loader, got "
                f"{type(loader).__name__}"
            )

        # Step 4: post-load processing is restricted to CPU-owned PLE modules;
        # the remainder of the model is still on meta and must not be visited.
        for layer in offload_layers.values():
            process_weights_after_loading(layer, model_config, torch.device("cpu"))

        if disk_dir is not None:
            for layer_name, pname in disk_attached.items():
                if f"{layer_name}.{pname}" not in disk_complete_params:
                    _ple_disk_finalize(layer_name, offload_layers[layer_name],
                                       pname, disk_dir)

        self._layers.update(offload_layers)
        del model
        logger.info("PLE weight loading complete.")

    def accept_registrations(
        self,
        pull_socket: zmq.Socket,
        num_workers: int,
    ) -> None:
        """Receive every local DP/TP worker's IPC and shared-memory buffers."""
        logger.info("Waiting for %d GPU worker registration(s) ...", num_workers)
        registrations: list[PleOffloadRegistration] = []
        for index in range(num_workers):
            item = pickle.loads(pull_socket.recv())
            if not isinstance(item, PleOffloadRegistration):
                raise RuntimeError(
                    "Expected PleOffloadRegistration during setup, got "
                    f"{type(item).__name__} ({index + 1}/{num_workers})"
                )
            registrations.append(item)
            logger.info(
                "GPU worker %d registered (dp_rank=%d, tp_rank=%d, layers=%s).",
                item.worker_id,
                item.dp_rank,
                item.tp_rank,
                sorted(item.gpu_output_buffers),
            )

        dp_size = self.vllm_config.parallel_config.data_parallel_size
        tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        if num_workers != dp_size * tp_size:
            raise RuntimeError(
                f"Expected {dp_size * tp_size} registrations for DP={dp_size}, "
                f"TP={tp_size}, got {num_workers}"
            )

        registrations_by_dp: dict[int, list[PleOffloadRegistration]] = {}
        for registration in registrations:
            registrations_by_dp.setdefault(registration.dp_rank, []).append(
                registration
            )
        if set(registrations_by_dp) != set(range(dp_size)):
            raise RuntimeError(
                f"Expected DP ranks {set(range(dp_size))}, "
                f"got {set(registrations_by_dp)}"
            )
        for dp_rank, dp_registrations in registrations_by_dp.items():
            tp_ranks = {registration.tp_rank for registration in dp_registrations}
            if tp_ranks != set(range(tp_size)):
                raise RuntimeError(
                    f"DP rank {dp_rank} expected TP ranks {set(range(tp_size))}, "
                    f"got {tp_ranks}"
                )

        for registration in registrations:
            if set(registration.gpu_output_buffers) != set(self.layer_names):
                raise RuntimeError(
                    "Registered PLE layers do not match CPU layers: "
                    f"registered={sorted(registration.gpu_output_buffers)}, "
                    f"cpu={sorted(self.layer_names)}"
                )
            targets_for_dp = self._worker_targets.setdefault(registration.dp_rank, {})
            for layer_name, gpu_buffer in registration.gpu_output_buffers.items():
                target = PleOffloadOutputTarget(
                    tp_rank=registration.tp_rank,
                    gpu_output_buffer=gpu_buffer,
                    sem=CpuGpuSemaphore.from_ipc_tensor(
                        registration.sem_flag_tensors[layer_name]
                    ),
                    copy_stream=torch.cuda.Stream(device=gpu_buffer.device),
                    done_seq_buf=registration.done_seq_buf,
                )
                targets_for_dp.setdefault(layer_name, []).append(target)
            # All TP ranks in one DP group receive the same input, so buffers
            # registered by TP rank zero are sufficient for that DP rank.
            if registration.tp_rank == 0:
                self._input_bufs[registration.dp_rank] = PleOffloadInputBuffers(
                    input_ids_buf=registration.input_ids_buf,
                    query_start_loc_buf=registration.query_start_loc_buf,
                    ngram_context_buf=registration.ngram_context_buf,
                )

        if set(self._input_bufs) != set(range(dp_size)):
            raise RuntimeError(
                "TP rank zero did not register PLE input buffers for every DP "
                f"rank: expected={set(range(dp_size))}, got={set(self._input_bufs)}"
            )

        config = self.vllm_config.model_config.hf_text_config
        max_tokens = self.vllm_config.scheduler_config.max_num_batched_tokens
        embedding_dim = int(config.ple_embed_dim)
        for dp_rank, layer_targets in self._worker_targets.items():
            self._pinned_bufs[dp_rank] = {}
            for layer_name, targets in layer_targets.items():
                if len(targets) != tp_size:
                    raise RuntimeError(
                        f"PLE layer {layer_name} for DP rank {dp_rank} received "
                        f"{len(targets)} targets, expected {tp_size}"
                    )
                targets.sort(key=lambda target: target.tp_rank)
                self._pinned_bufs[dp_rank][layer_name] = torch.empty(
                    max_tokens,
                    embedding_dim,
                    dtype=self._layers[layer_name].get_offload_output_dtype(
                        self.vllm_config.model_config.dtype
                    ),
                    pin_memory=True,
                )
        logger.info(
            "Registrations complete (dp_size=%d, tp_size=%d, layers=%s).",
            dp_size,
            tp_size,
            sorted(self.layer_names),
        )

    @torch.inference_mode()
    def busy_loop(
        self,
        pull_socket: zmq.Socket,
        shutdown_event: threading.Event,
    ) -> None:
        """Decode and batch available requests by DP rank until shutdown."""
        self._debug_delay_s = float(os.getenv("PLE_OFFLOAD_DEBUG_DELAY_MS", "0")) / 1e3
        # Test hook: PLE_OFFLOAD_DEBUG_TRACE=<file> appends one line per request
        # "seq num_tokens num_reqs ids_hash result_hash" so two runs can be compared.
        self._debug_trace = None
        trace_path = os.getenv("PLE_OFFLOAD_DEBUG_TRACE", "")
        if trace_path:
            self._debug_trace = open(trace_path, "a", buffering=1)
            logger.warning("PLE_OFFLOAD_DEBUG_TRACE=%s: tracing every request (test hook).", trace_path)
        self._done_seq: dict[int, int] = {}
        if self._debug_delay_s:
            logger.warning("PLE_OFFLOAD_DEBUG_DELAY_MS=%.0f: every lookup is delayed (test hook).", self._debug_delay_s * 1e3)
        logger.info("Busy-loop started.")
        poller = zmq.Poller()
        poller.register(pull_socket, zmq.POLLIN)
        while not shutdown_event.is_set():
            if pull_socket not in dict(poller.poll(timeout=100)):
                continue

            requests = []
            try:
                requests.append(_PLE_OFFLOAD_REQUEST_DECODER.decode(pull_socket.recv()))
                while True:
                    requests.append(
                        _PLE_OFFLOAD_REQUEST_DECODER.decode(
                            pull_socket.recv(zmq.NOBLOCK)
                        )
                    )
            except zmq.Again:
                pass
            except msgspec.DecodeError as error:
                raise RuntimeError("Unexpected PLE offload request") from error

            self._handle_requests(requests)

    def _handle_requests(self, requests: list[PleOffloadRequest]) -> None:
        """Run every drained request, strictly in arrival order.

        Protocol (2026-08-30, replaces the GPU stream-wait handshake): the GPU
        worker's model thread blocks in ``prepare_forward`` until this process
        has written request N's result into its output buffer and bumped the
        shared ``done_seq_buf`` counter, and only then enqueues the forward. No
        stream-wait packet is ever left pending on a GPU queue (on ROCm,
        hipStreamWaitValue32 is dropped from HIP graphs, and a pending
        WAIT_REG_MEM made KFD queue eviction fail and reset the GPU). Because the
        GPU worker cannot launch request N+1 before N completed, at most one
        request per DP rank is ever outstanding; several in one drain would be a
        protocol violation, so they are processed in order and logged.
        """
        if len(requests) > 1:
            logger.warning(
                "%d PLE requests drained at once (expected at most one per DP rank); "
                "processing in order.",
                len(requests),
            )
        for request in requests:
            if request.dp_rank not in self._worker_targets:
                logger.warning(
                    "No PLE output targets for dp_rank=%d; skipping request.",
                    request.dp_rank,
                )
                continue
            if self._debug_delay_s:
                # Test hook: PLE_OFFLOAD_DEBUG_DELAY_MS slows every lookup so a test
                # can prove the forward waits for the result (output unchanged,
                # decode slower) instead of reading a stale buffer.
                time.sleep(self._debug_delay_s)
            self._handle_one(request)

    def _handle_one(self, request: PleOffloadRequest) -> None:
        requests_by_dp = {request.dp_rank: request}

        # Speculative placeholders are not vocabulary IDs. Normalize each DP
        # input once before all PLE layers consume the shared buffer.
        if self._clamp_input_ids:
            for dp_rank, request in requests_by_dp.items():
                self._input_bufs[dp_rank].input_ids_buf[
                    : request.num_tokens
                ].clamp_min_(0)

        for layer_name, layer in self._layers.items():
            for dp_rank, request in requests_by_dp.items():
                targets = self._worker_targets[dp_rank][layer_name]

                # The previous DMA out of the pinned result buffer must be done
                # before forward_impl overwrites it. (The GPU side is ordered by
                # the host protocol: request N+1 is only launched after forward N
                # finished consuming buffer N, so no GPU-side wait is needed.)
                for target in targets:
                    target.copy_stream.synchronize()

                input_bufs = self._input_bufs[dp_rank]
                ngram_context = (
                    input_bufs.ngram_context_buf[: request.num_reqs]
                    if input_bufs.ngram_context_buf is not None
                    else None
                )
                result = layer.forward_impl(
                    input_bufs.input_ids_buf[: request.num_tokens],
                    input_bufs.input_ids_buf[: request.num_tokens],
                    input_bufs.query_start_loc_buf[: request.num_reqs + 1],
                    ngram_context,
                    output_buffer=self._pinned_bufs[dp_rank][layer_name],
                )
                if self._debug_trace is not None:
                    import hashlib
                    ids = input_bufs.input_ids_buf[: request.num_tokens]
                    h_ids = hashlib.md5(ids.contiguous().numpy().tobytes()).hexdigest()[:12]
                    h_res = hashlib.md5(result.contiguous().view(torch.uint8).numpy().tobytes() if result.dtype != torch.bfloat16 else result.float().contiguous().numpy().tobytes()).hexdigest()[:12]
                    self._debug_trace.write(
                        f"{self._done_seq.get(dp_rank, 0) + 1} {request.num_tokens} {request.num_reqs} {h_ids} {h_res}\n"
                    )

                # The result is identical on every TP rank in this DP group.
                # Each copy stream signals only after its DMA completes.
                slices = tuple(slice(0, size) for size in result.shape)
                for target in targets:
                    with torch.cuda.stream(target.copy_stream):
                        target.gpu_output_buffer[slices].copy_(
                            result[slices], non_blocking=True
                        )
                        target.sem.signal(target.copy_stream)

        # Publish completion: every TP worker's output buffers now hold this
        # request's result (DMA finished), so its model thread may enqueue the
        # forward. One counter per TP worker; the same tensor is shared by all
        # of that worker's layer targets, so write it once per worker.
        for dp_rank in requests_by_dp:
            seq = self._done_seq.get(dp_rank, 0) + 1
            self._done_seq[dp_rank] = seq
            written: set[int] = set()
            for layer_name in self._layers:
                for target in self._worker_targets[dp_rank][layer_name]:
                    target.copy_stream.synchronize()
                    buf = target.done_seq_buf
                    if buf is not None and id(buf) not in written:
                        buf[0] = seq
                        written.add(id(buf))
