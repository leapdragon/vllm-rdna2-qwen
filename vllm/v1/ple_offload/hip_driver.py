# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A ``cuda.bindings.driver`` work-alike for the four calls PLE offload needs.

PLE offload signals between the GPU worker and the offload subprocess with
stream-ordered write/wait-value operations on a flag, and page-locks the shared
CPU staging buffers. On NVIDIA those come from ``cuda-python``, which is not
available on ROCm -- but every one of them has a direct HIP equivalent, so this
module binds them from ``libamdhip64`` with ctypes and presents the same surface
(same call names, same argument order, same ``.value``-bearing result) that
``_cuda_check`` and the call sites already expect.

Mapping:

    cuStreamWriteValue32  -> hipStreamWriteValue32
    cuStreamWaitValue32   -> hipStreamWaitValue32   (HIP takes an extra mask)
    cuMemHostRegister     -> hipHostRegister
    cuMemHostUnregister   -> hipHostUnregister

The flag constants coincide: hipStreamWaitValueEq == CU_STREAM_WAIT_VALUE_EQ == 0x1
and hipHostRegisterPortable == CU_MEMHOSTREGISTER_PORTABLE == 0x1.
"""

import ctypes
from enum import Enum

_LIB_CANDIDATES = ("libamdhip64.so", "libamdhip64.so.7", "libamdhip64.so.6")

_hip: ctypes.CDLL | None = None
for _name in _LIB_CANDIDATES:
    try:
        _hip = ctypes.CDLL(_name)
        break
    except OSError:
        continue

HIP_DRIVER_AVAILABLE = _hip is not None

# hipStreamWaitValue32 masks all bits by default; ctypes must pass it explicitly.
_WAIT_MASK_ALL = 0xFFFFFFFF

CU_MEMHOSTREGISTER_PORTABLE = 0x1  # == hipHostRegisterPortable


class CUstreamWaitValue_flags(Enum):
    """Mirrors the CUDA enum; values match HIP's hipStreamWaitValue* flags."""

    CU_STREAM_WAIT_VALUE_GEQ = 0x0
    CU_STREAM_WAIT_VALUE_EQ = 0x1
    CU_STREAM_WAIT_VALUE_AND = 0x2
    CU_STREAM_WAIT_VALUE_NOR = 0x3


class _HipResult:
    """Minimal stand-in for ``CUresult``: ``_cuda_check`` only reads ``.value``."""

    __slots__ = ("value",)

    def __init__(self, value: int) -> None:
        self.value = int(value)

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"hipError_t({self.value})"


def CUstream(handle: int) -> ctypes.c_void_p:  # noqa: N802 - mirrors cuda-python
    """Wrap a torch ``stream.cuda_stream`` handle (a hipStream_t on ROCm)."""
    return ctypes.c_void_p(int(handle))


def CUdeviceptr(address: int) -> ctypes.c_void_p:  # noqa: N802
    """Wrap a device address; HIP's value APIs take a plain ``void *``."""
    return ctypes.c_void_p(int(address))


def _require_hip() -> ctypes.CDLL:
    if _hip is None:
        raise RuntimeError(
            "PLE offload on ROCm requires libamdhip64, which could not be loaded"
        )
    return _hip


def cuStreamWriteValue32(  # noqa: N802
    stream: ctypes.c_void_p,
    ptr: ctypes.c_void_p,
    value: int,
    flags: int,
) -> _HipResult:
    fn = _require_hip().hipStreamWriteValue32
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32]
    return _HipResult(fn(stream, ptr, ctypes.c_uint32(value), ctypes.c_uint32(flags)))


def cuStreamWaitValue32(  # noqa: N802
    stream: ctypes.c_void_p,
    ptr: ctypes.c_void_p,
    value: int,
    flags: int,
) -> _HipResult:
    fn = _require_hip().hipStreamWaitValue32
    fn.restype = ctypes.c_int
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
    ]
    return _HipResult(
        fn(
            stream,
            ptr,
            ctypes.c_uint32(value),
            ctypes.c_uint32(flags),
            ctypes.c_uint32(_WAIT_MASK_ALL),
        )
    )


def cuMemHostRegister(address: int, size: int, flags: int) -> _HipResult:  # noqa: N802
    fn = _require_hip().hipHostRegister
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32]
    return _HipResult(
        fn(ctypes.c_void_p(int(address)), ctypes.c_size_t(int(size)),
           ctypes.c_uint32(flags))
    )


def cuMemHostUnregister(address: int) -> _HipResult:  # noqa: N802
    fn = _require_hip().hipHostUnregister
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_void_p]
    return _HipResult(fn(ctypes.c_void_p(int(address))))
