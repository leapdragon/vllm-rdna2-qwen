#!/usr/bin/env python3
"""4-process torch test of the _rocm_C rdna_ar_* ops in the exact shape vLLM uses them:
one process per device, handles exchanged out-of-band, fp16 [4,2560] all-reduce, checked
against the exact sum, timed, and replayed under CUDA-graph capture."""
import os, sys, time
import torch
import torch.multiprocessing as mp

W = int(sys.argv[1]) if len(sys.argv) > 1 else 4
N = int(sys.argv[2]) if len(sys.argv) > 2 else 4 * 2560

def worker(rank, q_handles, q_out, barrier):
    torch.cuda.set_device(rank)
    import vllm  # noqa: F401
    import vllm._rocm_C  # noqa: F401  (registers the ops)
    from vllm import _custom_ops as ops
    dev_ids = torch.arange(W, dtype=torch.int64)
    shm = "/rdna_ar_optest"
    for r in range(W):           # ordered init, like the communicator
        if r == rank:
            packed = ops.rdna_ar_init(rank, W, dev_ids, 512 * 1024, shm).numpy().tobytes()
            hdl = int.from_bytes(packed[:8], "little"); h = packed[8:]
        barrier.wait()
    q_handles.put((rank, h))
    barrier.wait()
    allh = {}
    while len(allh) < W:
        r, b = q_handles.get(); allh[r] = b
        q_handles.put((r, b))    # re-post for the others
        if len(allh) == W: break
    barrier.wait()
    buf = torch.frombuffer(bytearray(b"".join(allh[r] for r in range(W))), dtype=torch.uint8).view(W, -1)
    ops.rdna_ar_connect(hdl, buf.contiguous())
    barrier.wait()
    x = (torch.arange(N, device="cuda", dtype=torch.float32) % 97 * 0.125 + (rank + 1) * 7).half()
    want = sum(((torch.arange(N, device="cuda", dtype=torch.float32) % 97 * 0.125 + (r + 1) * 7).half().float()) for r in range(W)).half()
    assert ops.rdna_ar_can(hdl, x)
    y = ops.rdna_ar_all_reduce(hdl, x); torch.cuda.synchronize()
    ok = torch.equal(y, want)
    for _ in range(50): y = ops.rdna_ar_all_reduce(hdl, x)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(2000): y = ops.rdna_ar_all_reduce(hdl, x)
    torch.cuda.synchronize(); us = (time.perf_counter() - t0) / 2000 * 1e6
    # RCCL-free reference for the same size: nothing; just graph replay next
    g = torch.cuda.CUDAGraph(); s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(3): ops.rdna_ar_all_reduce(hdl, x)
        torch.cuda.synchronize()
        with torch.cuda.graph(g, stream=s):
            y2 = ops.rdna_ar_all_reduce(hdl, x); y3 = ops.rdna_ar_all_reduce(hdl, y2)
    torch.cuda.synchronize()
    for _ in range(300): g.replay()
    torch.cuda.synchronize()
    ok2 = torch.equal(y2, want) and torch.equal(y3, (want.float() * W).half())
    t0 = time.perf_counter()
    for _ in range(1000): g.replay()
    torch.cuda.synchronize(); us_g = (time.perf_counter() - t0) / 1000 * 1e6 / 2
    # a second instance (second process group) in the same process must also work
    for r in range(W):
        if r == rank:
            packed2 = ops.rdna_ar_init(rank, W, dev_ids, 64 * 1024, shm + "_b").numpy().tobytes()
            hdl2 = int.from_bytes(packed2[:8], "little"); h2 = packed2[8:]
        barrier.wait()
    q_handles.put((rank + 100, h2)); barrier.wait()
    allh2 = {}
    while len(allh2) < W:
        r, b = q_handles.get()
        if r >= 100: allh2[r - 100] = b
        q_handles.put((r, b))
        if len(allh2) == W: break
    barrier.wait()
    buf2 = torch.frombuffer(bytearray(b"".join(allh2[r] for r in range(W))), dtype=torch.uint8).view(W, -1)
    ops.rdna_ar_connect(hdl2, buf2.contiguous()); barrier.wait()
    yb = ops.rdna_ar_all_reduce(hdl2, x); torch.cuda.synchronize()
    ok3 = torch.equal(yb, want) and hdl2 != hdl
    q_out.put((rank, ok and ok3, ok2, us, us_g, ops.rdna_ar_timed_out(hdl), y.float().sum().item()))

if __name__ == "__main__":
    mp.set_start_method("spawn")
    qh, qo = mp.Queue(), mp.Queue(); bar = mp.Barrier(W)
    ps = [mp.Process(target=worker, args=(r, qh, qo, bar)) for r in range(W)]
    for p in ps: p.start()
    res = sorted(qo.get() for _ in range(W))
    for p in ps: p.join(60)
    sums = {r[6] for r in res}
    for r in res: print(f"rank {r[0]}: eager {'OK' if r[1] else 'WRONG'} graph {'OK' if r[2] else 'WRONG'}  {r[3]:.1f} us/op eager, {r[4]:.1f} us/op in-graph  timed_out={r[5]}")
    print("PASS" if all(r[1] and r[2] and not r[5] for r in res) and len(sums) == 1 else "FAIL", "(identical across ranks)" if len(sums) == 1 else "(ranks DIFFER)")
