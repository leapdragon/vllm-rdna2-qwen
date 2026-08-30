#!/usr/bin/env python3
"""Can the copy stream announce its own completion into host memory? (2026-08-30)

Mirrors the planned PLE handshake: the offload worker (child) DMAs a result into an
IPC-shared device buffer and then enqueues hipStreamWriteValue32(seq) on the same stream,
targeting a shared-memory page it has hipHostRegister'ed (Mapped|Portable). The GPU worker
(parent) polls that page from the CPU and reads the buffer with a kernel.
Checks: (a) the write lands and is visible to the CPU, (b) ordering — every observed seq is
accompanied by the matching buffer contents, (c) the per-round latency.
Run on a free GPU: ROCR_VISIBLE_DEVICES=<idx> [HSA_OVERRIDE_GFX_VERSION=10.3.0] python3 ...
"""
import ctypes, sys, time
import torch
import torch.multiprocessing as mp

N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
NUMEL = 2560 * 4


def producer(dev_buf, page, ack, go, n):
    from vllm.v1.ple_offload import hip_driver as drv
    torch.cuda.set_device(dev_buf.device)
    stream = torch.cuda.Stream()
    pinned = torch.empty(NUMEL, dtype=torch.float16).pin_memory()
    # register the shared page in THIS process and get its device pointer
    addr = page.data_ptr()
    r = drv.cuMemHostRegister(addr, 4096, 0x1 | 0x2)     # Portable | Mapped
    assert r.value == 0, f"hipHostRegister failed: {r}"
    lib = drv._hip                                       # the shim's already-loaded libamdhip64
    devptr = ctypes.c_void_p()
    lib.hipHostGetDevicePointer.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_uint]
    assert lib.hipHostGetDevicePointer(ctypes.byref(devptr), ctypes.c_void_p(addr), 0) == 0
    go.wait()
    for i in range(1, n + 1):
        pinned.fill_(float(i % 2000))
        with torch.cuda.stream(stream):
            dev_buf.copy_(pinned, non_blocking=True)
            r = drv.cuStreamWriteValue32(drv.CUstream(stream.cuda_stream), devptr, i, 0)
            assert r.value == 0, f"WriteValue32 failed: {r}"
        # no synchronize: the stream itself publishes completion
        while ack[0] < i:
            time.sleep(1e-5)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    dev_buf = torch.zeros(NUMEL, dtype=torch.float16, device=dev)
    page = torch.zeros(1024, dtype=torch.int32).share_memory_()   # 4 KB
    ack = torch.zeros(1, dtype=torch.int64).share_memory_()
    go = mp.Event()
    p = mp.Process(target=producer, args=(dev_buf, page, ack, go, N))
    p.start(); time.sleep(3); go.set()
    bad = 0; first = None; lat = []
    for i in range(1, N + 1):
        t0 = time.perf_counter()
        while int(page[0]) < i:
            pass
        lat.append(time.perf_counter() - t0)
        expect = float(i % 2000)
        mn, mx = dev_buf.min().item(), dev_buf.max().item()
        if mn != expect or mx != expect:
            bad += 1
            if first is None:
                first = (i, mn, mx, expect)
        ack[0] = i
    p.join(timeout=30)
    lat.sort()
    print(f"rounds={N} ordering_violations={bad} first={first} "
          f"host-observed latency p50={lat[len(lat)//2]*1e6:.0f}us p99={lat[int(len(lat)*0.99)]*1e6:.0f}us")
    print("DMA-ANNOUNCE OK" if bad == 0 else "DMA-ANNOUNCE BROKEN")
    sys.exit(0 if bad == 0 else 1)
