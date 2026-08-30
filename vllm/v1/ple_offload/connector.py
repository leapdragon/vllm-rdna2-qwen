# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exchange PLE data between a GPU worker and the CPU-offload process."""

import os
import queue
import threading
import time
from multiprocessing.reduction import ForkingPickler
from typing import Any

import msgspec
import torch
import torch.nn as nn
import zmq
# See ple_offload_layer: cuda-python is NVIDIA-only, so importing it must not
# break module import on ROCm. Only the offload data path dereferences it.
# On ROCm the HIP shim is always the right driver: cuda-bindings may well be
# importable (it is a transitive dependency) but there is no libcuda to dlopen.
if torch.version.hip is not None:
    from vllm.v1.ple_offload import hip_driver as cuda_driver
else:
    try:
        from cuda.bindings import driver as cuda_driver
    except ImportError:  # pragma: no cover - platform dependent
        from vllm.v1.ple_offload import hip_driver as cuda_driver  # type: ignore[no-redef]

from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_dp_group, get_tp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.ple_offload_layer import (
    CpuGpuSemaphore,
    PleOffloadLayer,
)
from vllm.v1.ple_offload.protocol import (
    PleOffloadRegistration,
    PleOffloadRequest,
)

logger = init_logger(__name__)


def _cuda_check(result: Any, operation: str) -> Any:
    """Check the ``(CUresult, ...)`` tuple returned by cuda-python calls."""
    error = result[0] if isinstance(result, tuple) else result
    if error.value != 0:
        raise RuntimeError(f"{operation} failed: {error}")
    return result


class PleOffloadConnector:
    """Connect a GPU runner to the shared PLE CPU worker.

    MRV1 and MRV2 share the same CPU-input and CUDA-output IPC protocol.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        device: torch.device,
        ipc_addr: str,
        *,
        input_ids_source: torch.Tensor,
        query_start_loc_source: torch.Tensor,
        ngram_context_source: torch.Tensor | None,
    ) -> None:
        self.device = device
        self.dp_rank = get_dp_group().rank_in_group
        self.tp_rank = get_tp_group().rank_in_group
        self._layers = self._setup_layers(vllm_config, model)

        # Both runner paths stage into the same shared buffers. TP0 registers
        # them with CUDA so MRV2 can use asynchronous D2H copies.
        scheduler_config = vllm_config.scheduler_config
        self._input_ids_buf = torch.empty(
            scheduler_config.max_num_batched_tokens,
            dtype=torch.int32,
            device="cpu",
        ).share_memory_()
        self._query_start_loc_buf = torch.empty(
            scheduler_config.max_num_seqs + 1,
            dtype=torch.int32,
            device="cpu",
        ).share_memory_()
        self._ngram_context_buf = None
        config = vllm_config.model_config.hf_text_config
        ngram_context_len = int(config.ngram_size) - 1
        if ngram_context_len > 0:
            self._ngram_context_buf = torch.empty(
                scheduler_config.max_num_seqs,
                ngram_context_len,
                dtype=torch.int32,
                device="cpu",
            ).share_memory_()

        # Runner input allocations are address-stable, so bind them once and
        # pass only batch sizes through the per-forward request queue.
        self._input_ids_source = input_ids_source
        self._query_start_loc_source = query_start_loc_source
        self._ngram_context_source = ngram_context_source
        self._uses_cuda_inputs = self._input_ids_source.is_cuda
        self._validate_input_sources()

        self._pinned_input_buffers: list[torch.Tensor] = []
        # PLE rejects DBO, and each forward consumes its output before the
        # next launch, so one pending request is sufficient.
        # Each item carries its own input-ready event: a single re-recorded event let
        # a request thread that fell one step behind wait on the *next* step's recording,
        # which depends on this step's forward, which waits for this very request
        # (deadlock on warm boots, 2026-08-30).
        self._request_queue: queue.Queue[
            tuple[PleOffloadRequest, torch.cuda.Event | None] | None
        ] = queue.Queue(maxsize=1)
        self._request_thread: threading.Thread | None = None
        # Host-side completion protocol (see prepare_forward): the offload worker
        # bumps this shared counter once this worker's output buffers hold the
        # result of launch N; the model thread waits for it before enqueueing
        # the forward. Allocated before registration so it can be shared.
        self._done_seq_buf = torch.zeros(1, dtype=torch.int64).share_memory_()
        self._launch_seq = 0
        self._t_launch = 0.0
        self._t_wait = 0.0
        self._request_thread_ready = threading.Event()
        self._zmq_ctx: zmq.Context | None = None
        self._registration_socket: zmq.Socket | None = None
        self._d2h_stream: torch.cuda.Stream | None = None
        self._input_ready_event: torch.cuda.Event | None = None
        self._d2h_done_event: torch.cuda.Event | None = None

        try:
            self._zmq_ctx = zmq.Context()
            self._registration_socket = self._zmq_ctx.socket(zmq.PUSH)
            self._registration_socket.connect(ipc_addr)
            self._register_with_offload_worker(vllm_config, ipc_addr)

            if self.tp_rank == 0:
                # ForkingPickler may replace CPU storage while converting its
                # sharing strategy, so register only the final addresses.
                with torch.accelerator.device_index(self.device.index):
                    self._pin_input_buffers()
                    if self._uses_cuda_inputs:
                        self._d2h_stream = torch.cuda.Stream(device=self.device)
                        self._input_ready_event = torch.cuda.Event()
                        self._d2h_done_event = torch.cuda.Event()
                self._start_request_thread(ipc_addr)
        except Exception:
            self.close()
            raise

    def _setup_layers(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
    ) -> dict[str, PleOffloadLayer]:
        """Attach output buffers and semaphores to GPU PLE placeholders."""
        layers = {
            name: module
            for name, module in model.named_modules()
            if isinstance(module, PleOffloadLayer)
        }
        if not layers:
            raise RuntimeError(
                "VLLM_PLE_CPU_OFFLOAD is enabled, but the model has no PleOffloadLayer"
            )

        config = vllm_config.model_config.hf_text_config
        max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        for layer in layers.values():
            # The CPU worker writes results here through CUDA IPC. The GPU
            # placeholder waits on the paired cross-process semaphore.
            output_buffer = torch.empty(
                max_num_tokens,
                int(config.ple_embed_dim),
                dtype=layer.get_offload_output_dtype(vllm_config.model_config.dtype),
                device=self.device,
            )
            layer.setup_cross_process_offload(
                output_buffer,
                CpuGpuSemaphore(self.device),
            )
        return layers

    def _pin_input_buffers(self) -> None:
        """Page-lock shared input allocations without replacing their storage."""
        buffers = [self._input_ids_buf, self._query_start_loc_buf]
        if self._ngram_context_buf is not None:
            buffers.append(self._ngram_context_buf)
        for buffer in buffers:
            if buffer.device.type != "cpu" or not buffer.is_shared():
                raise RuntimeError("PLE input buffers must be shared CPU tensors")
            if not buffer.is_contiguous():
                raise RuntimeError("PLE input buffers must be contiguous")
            _cuda_check(
                cuda_driver.cuMemHostRegister(
                    buffer.data_ptr(),
                    buffer.numel() * buffer.element_size(),
                    cuda_driver.CU_MEMHOSTREGISTER_PORTABLE,
                ),
                "cuMemHostRegister(PLE input buffer)",
            )
            self._pinned_input_buffers.append(buffer)
            if not buffer.is_pinned():
                raise RuntimeError("CUDA did not page-lock a PLE input buffer")

    def _unpin_input_buffers(self) -> None:
        """Release CUDA registrations after the request thread has stopped."""
        for buffer in reversed(self._pinned_input_buffers):
            try:
                _cuda_check(
                    cuda_driver.cuMemHostUnregister(buffer.data_ptr()),
                    "cuMemHostUnregister(PLE input buffer)",
                )
            except RuntimeError:
                logger.exception("Failed to unregister a PLE input buffer")
        self._pinned_input_buffers.clear()

    def _register_with_offload_worker(
        self, vllm_config: VllmConfig, ipc_addr: str
    ) -> None:
        """Register CUDA IPC outputs and shared CPU inputs with the worker."""
        # Each GPU worker owns distinct output buffers, while TP0's shared
        # inputs become the request source for its DP rank.
        registration = PleOffloadRegistration(
            worker_id=(
                self.dp_rank * vllm_config.parallel_config.world_size
                + vllm_config.parallel_config.rank
            ),
            tp_rank=self.tp_rank,
            dp_rank=self.dp_rank,
            gpu_output_buffers={
                name: layer._gpu_output_buffer for name, layer in self._layers.items()
            },
            sem_flag_tensors={
                name: layer._sem.flag_tensor for name, layer in self._layers.items()
            },
            input_ids_buf=self._input_ids_buf,
            query_start_loc_buf=self._query_start_loc_buf,
            ngram_context_buf=self._ngram_context_buf,
            done_seq_buf=self._done_seq_buf,
        )

        # ForkingPickler transmits tensors through shared-memory and CUDA IPC.
        import torch.multiprocessing as torch_mp

        original_strategy = torch_mp.get_sharing_strategy()
        torch_mp.set_sharing_strategy("file_system")
        try:
            payload = ForkingPickler.dumps(registration)
        finally:
            torch_mp.set_sharing_strategy(original_strategy)
        assert self._registration_socket is not None
        self._registration_socket.send(payload)

        logger.info(
            "PleOffload: registered %d PleOffloadLayer(s) "
            "(dp_rank=%d, tp_rank=%d, ipc_addr=%s): %s",
            len(self._layers),
            self.dp_rank,
            self.tp_rank,
            ipc_addr,
            sorted(self._layers),
        )

    def _start_request_thread(self, ipc_addr: str) -> None:
        """Start the thread that publishes batches after inputs are ready."""
        self._request_thread = threading.Thread(
            target=self._request_loop,
            args=(ipc_addr,),
            name=f"ple-offload-dp{self.dp_rank}",
            daemon=True,
        )
        self._request_thread.start()
        if not self._request_thread_ready.wait(timeout=10):
            raise RuntimeError("Timed out starting the PLE request thread")

    def _request_loop(self, ipc_addr: str) -> None:
        """Stage fixed runner inputs, then notify the CPU worker."""
        socket: zmq.Socket | None = None
        try:
            if self._zmq_ctx is None:
                raise RuntimeError("PLE ZMQ context closed before thread startup")
            socket = self._zmq_ctx.socket(zmq.PUSH)
            socket.connect(ipc_addr)
            self._request_thread_ready.set()
            while True:
                item = self._request_queue.get()
                if item is None:
                    return
                request, input_ready = item
                self._process_request(request, input_ready, socket)
        except Exception:
            logger.exception("PLE request thread failed")
            os._exit(1)
        finally:
            self._request_thread_ready.set()
            if socket is not None:
                socket.close(linger=0)

    def _process_request(
        self,
        request: PleOffloadRequest,
        input_ready: torch.cuda.Event | None,
        socket: zmq.Socket,
    ) -> None:
        """Stage one batch from fixed sources and publish its request."""
        if self._uses_cuda_inputs:
            assert input_ready is not None
            self._copy_cuda_inputs(request, input_ready)
        else:
            self._copy_cpu_inputs(request)

        with torch.cuda.nvtx.range("ple_offload.send_request"):
            socket.send(msgspec.msgpack.encode(request))

    def _copy_cpu_inputs(self, request: PleOffloadRequest) -> None:
        """Stage MRV1's existing CPU mirrors in the notifier thread."""
        num_tokens = request.num_tokens
        num_reqs = request.num_reqs
        with torch.cuda.nvtx.range("ple_offload.copy_input_ids"):
            self._input_ids_buf[:num_tokens].copy_(self._input_ids_source[:num_tokens])
        with torch.cuda.nvtx.range("ple_offload.copy_query_start_loc"):
            self._query_start_loc_buf[: num_reqs + 1].copy_(
                self._query_start_loc_source[: num_reqs + 1]
            )
        if self._ngram_context_buf is not None:
            assert self._ngram_context_source is not None
            with torch.cuda.nvtx.range("ple_offload.copy_ngram_context"):
                self._ngram_context_buf[:num_reqs].copy_(
                    self._ngram_context_source[:num_reqs]
                )

    def _validate_input_sources(self) -> None:
        """Validate fixed runner sources against shared input buffers."""
        sources = [
            ("input_ids", self._input_ids_source, self._input_ids_buf),
            (
                "query_start_loc",
                self._query_start_loc_source,
                self._query_start_loc_buf,
            ),
        ]
        if (self._ngram_context_source is None) != (self._ngram_context_buf is None):
            raise ValueError("PLE ngram_context source and buffer must match")
        if self._ngram_context_source is not None:
            assert self._ngram_context_buf is not None
            sources.append(
                (
                    "ngram_context",
                    self._ngram_context_source,
                    self._ngram_context_buf,
                )
            )

        expected_device = self.device if self._uses_cuda_inputs else torch.device("cpu")
        for name, source, buffer in sources:
            if (
                source.device != expected_device
                or source.dtype != buffer.dtype
                or source.ndim != buffer.ndim
                or source.shape[0] < buffer.shape[0]
                or source.shape[1:] != buffer.shape[1:]
            ):
                raise ValueError(f"PLE {name} source is incompatible")

    def _copy_cuda_inputs(
        self, request: PleOffloadRequest, input_ready: torch.cuda.Event
    ) -> None:
        """Stage MRV2 inputs on the background D2H stream."""
        if self._d2h_stream is None or self._d2h_done_event is None:
            raise RuntimeError("PLE D2H resources are not initialized")

        with torch.accelerator.device_index(self.device.index):
            with torch.cuda.stream(self._d2h_stream):
                self._d2h_stream.wait_event(input_ready)
                with torch.cuda.nvtx.range("ple_offload.copy_input_ids"):
                    self._input_ids_buf[: request.num_tokens].copy_(
                        self._input_ids_source[: request.num_tokens],
                        non_blocking=True,
                    )
                with torch.cuda.nvtx.range("ple_offload.copy_query_start_loc"):
                    self._query_start_loc_buf[: request.num_reqs + 1].copy_(
                        self._query_start_loc_source[: request.num_reqs + 1],
                        non_blocking=True,
                    )
                if self._ngram_context_buf is not None:
                    assert self._ngram_context_source is not None
                    with torch.cuda.nvtx.range("ple_offload.copy_ngram_context"):
                        self._ngram_context_buf[: request.num_reqs].copy_(
                            self._ngram_context_source[: request.num_reqs],
                            non_blocking=True,
                        )
                self._d2h_done_event.record(self._d2h_stream)
            with torch.cuda.nvtx.range("ple_offload.wait_d2h"):
                self._d2h_done_event.synchronize()

    def _launch(
        self,
        num_reqs: int,
        num_tokens: int,
    ) -> None:
        """Queue one batch while keeping staging off the model thread."""
        # Inputs are replicated across TP ranks. One request per DP rank drives
        # the CPU result fan-out to every registered TP output buffer.
        if self.tp_rank != 0:
            return

        input_ready: torch.cuda.Event | None = None
        if self._uses_cuda_inputs:
            # The background copy stream waits for runner input production
            # without making the model stream wait for D2H completion. One event
            # per request (see the queue comment in __init__).
            input_ready = torch.cuda.Event()
            input_ready.record(torch.cuda.current_stream(self.device))
        request = PleOffloadRequest(
            dp_rank=self.dp_rank,
            num_tokens=num_tokens,
            num_reqs=num_reqs,
        )
        # Block instead of put_nowait: the model thread only *enqueues* GPU work, so during
        # a fast host loop (kernel warmup after a compile-cache hit, async scheduling) it can
        # run several steps ahead of the request thread, which drains one request per
        # completed GPU input stage. The bounded queue (max_num_seqs + 1) then raised
        # queue.Full and killed the worker. Back-pressure is safe: the GPU keeps executing
        # already-enqueued steps while we wait, so the request thread always makes progress.
        try:
            self._request_queue.put((request, input_ready), timeout=300)
        except queue.Full as exc:
            raise RuntimeError(
                "PLE offload request queue stayed full for 300 s; the request thread is stuck"
            ) from exc

    def prepare_forward(
        self,
        num_reqs: int,
        num_tokens: int,
        dummy_run: bool,
    ) -> None:
        """Submit real inputs or satisfy the PLE wait for a dummy forward."""
        if dummy_run:
            self.signal_dummy_outputs(num_tokens)
            return
        self._launch_seq += 1
        t0 = time.perf_counter()
        self._launch(num_reqs, num_tokens)
        t1 = time.perf_counter()
        # Host-side wait for the CPU lookup (2026-08-30). Why not the GPU-side
        # stream wait the layer's ``ple_offload_wait`` op enqueues: on ROCm,
        # hipStreamWaitValue32 is accepted during stream capture but not recorded
        # into the HIP graph, so on every CUDA-graph decode replay it was a no-op
        # and the model read a stale buffer (garbled long generations, "Duplicate
        # PLE request" skips); and a *pending* WAIT_REG_MEM packet cannot be
        # preempted when KFD evicts queues (svm_range_restore), which reset and
        # lost a GPU on this machine. Waiting here on the host costs nothing:
        # step N+1's lookup needs step N's sampled token, so the chain is serial
        # anyway, and async scheduling still overlaps this wait with forward N.
        self._wait_lookup_done(self._launch_seq)
        t2 = time.perf_counter()
        self._t_launch += t1 - t0
        self._t_wait += t2 - t1
        if self._launch_seq % 500 == 0 and self.tp_rank == 0:
            n = self._launch_seq
            logger.info(
                "PLE offload host wait over %d launches: launch %.2f ms + wait %.2f ms per step",
                n, self._t_launch / n * 1e3, self._t_wait / n * 1e3,
            )

    def _wait_lookup_done(self, seq: int) -> None:
        """Block until the offload worker reports launch ``seq`` complete."""
        buf = self._done_seq_buf
        deadline = time.monotonic() + 600.0
        warned = False
        spins = 0
        while int(buf[0]) < seq:
            spins += 1
            if spins < 2000:
                continue                      # ~sub-millisecond: spin
            time.sleep(20e-6)
            now = time.monotonic()
            if not warned and now > deadline - 595.0:
                logger.warning(
                    "PLE lookup for launch %d has taken >5 s (worker slow or stuck?)",
                    seq,
                )
                warned = True
            if now > deadline:
                raise RuntimeError(
                    f"PLE offload worker did not complete launch {seq} within 600 s "
                    f"(done={int(buf[0])})"
                )

    def signal_dummy_outputs(self, num_tokens: int) -> None:
        """Locally satisfy PLE waits for dummy and capture forwards."""
        # Dummy and capture forwards do not send CPU requests, but every PLE
        # placeholder still waits for a completed output semaphore.
        stream = torch.cuda.current_stream(self.device)
        for layer in self._layers.values():
            layer._gpu_output_buffer[:num_tokens].zero_()
            layer._sem.signal(stream)

    def release_outputs(self) -> None:
        """Mark GPU output buffers reusable after the model consumes them."""
        # Reset only after the consumer forward so the CPU worker cannot
        # overwrite an output that a GPU PLE placeholder may still read.
        stream = torch.cuda.current_stream(self.device)
        for layer in self._layers.values():
            layer.release_offloaded_output(stream)

    def close(self) -> None:
        """Stop request transport and release host registrations."""
        request_thread = self._request_thread
        if request_thread is not None and request_thread.is_alive():
            try:
                self._request_queue.put(None, timeout=5)
            except queue.Full:
                logger.error("Timed out stopping the PLE request thread")
            request_thread.join(timeout=5)
        if request_thread is not None and request_thread.is_alive():
            # The thread may still access the registered buffers or ZMQ context.
            logger.error("PLE request thread did not stop; deferring resource cleanup")
            return
        self._request_thread = None

        if self._pinned_input_buffers:
            with torch.accelerator.device_index(self.device.index):
                self._unpin_input_buffers()
        self._d2h_done_event = None
        self._input_ready_event = None
        self._d2h_stream = None
        if self._registration_socket is not None:
            self._registration_socket.close(linger=0)
            self._registration_socket = None
        if self._zmq_ctx is not None:
            self._zmq_ctx.term()
            self._zmq_ctx = None
