#!/usr/bin/env python3
"""4-process test of the int8 prefill all-reduce (vllm/distributed/device_communicators/rdna_q8_all_reduce.py)
through vLLM's own PyNcclCommunicator, as the server uses it.

Per size: the grouped all-to-all and the staggered exchange (VLLM_RDNA_AR_Q8_STAGGER) must give BIT-IDENTICAL
results (same bytes, same reduction order), both within int8 error of the exact fp32 sum, and identical across
ranks. Then times both against RCCL's fp16 all-reduce. Keeps the run short: it exercises the traffic pattern
suspected of knocking cards off the bus on 2026-09-25/26.

usage: ROCR_VISIBLE_DEVICES=1,2,3,4 q8_ar_test.py [iters]"""
import os, sys, time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

H = 2560
SIZES = (2048, 1300, 700)


def worker(rank, W, iters, port, q):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", rank=rank, world_size=W)
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.device_communicators.rdna_q8_all_reduce import RdnaQ8AllReduce
    dev = torch.device("cuda", rank)
    comm = PyNcclCommunicator(dist.group.WORLD, dev)
    ar = RdnaQ8AllReduce(comm, rank, W)
    res = []
    for M in SIZES:
        g = torch.Generator(device="cpu").manual_seed(1000 + rank * 7 + M)
        x = (torch.randn(M, H, generator=g) * (0.5 + 0.25 * rank)).half().to(dev)
        assert ar.should_use(x), "tensor not eligible for the int8 path"
        full = torch.empty(W, M, H, dtype=torch.float16, device=dev)
        comm.all_gather(full.view(-1), x.view(-1))
        exact = full.float().sum(0)
        ar.stagger = False; y_grp = ar.all_reduce(x).clone()
        ar.stagger = True;  y_stg = ar.all_reduce(x).clone()
        torch.cuda.synchronize()
        same = torch.equal(y_grp, y_stg)
        err = ((y_stg.float() - exact).norm() / exact.norm()).item()
        chk = y_stg.float().sum().item()
        def timeit(fn):
            for _ in range(3): fn()
            torch.cuda.synchronize(); dist.barrier(); t = time.perf_counter()
            for _ in range(iters): fn()
            torch.cuda.synchronize(); return (time.perf_counter() - t) / iters * 1e3
        def grp(): ar.stagger = False; ar.all_reduce(x)
        def stg(): ar.stagger = True; ar.all_reduce(x)
        t_rccl = timeit(lambda: comm.all_reduce(x))
        t_grp = timeit(grp)
        t_stg = timeit(stg)
        res.append((M, same, err, chk, t_rccl, t_grp, t_stg))
    q.put((rank, res))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    W = torch.cuda.device_count()
    mp.set_start_method("spawn")
    q = mp.Queue()
    ps = [mp.Process(target=worker, args=(r, W, iters, 29541, q)) for r in range(W)]
    for p in ps: p.start()
    out = dict(q.get() for _ in range(W))
    for p in ps: p.join(60)
    ok = True
    for i, M in enumerate(SIZES):
        rows = [out[r][i] for r in range(W)]
        same = all(r[1] for r in rows)
        err = max(r[2] for r in rows)
        ident = len({round(r[3], 3) for r in rows}) == 1
        t = [sum(r[k] for r in rows) / W for k in (4, 5, 6)]
        ok &= same and ident and err < 2e-2
        print(f"M={M:5d}: staggered == grouped {'yes' if same else 'NO'}, identical across ranks {'yes' if ident else 'NO'}, "
              f"rel err vs exact {err:.2e} | RCCL fp16 {t[0]:.3f} ms, int8 grouped {t[1]:.3f} ms, "
              f"int8 staggered {t[2]:.3f} ms ({t[1] / t[2]:.2f}x vs grouped, {t[0] / t[2]:.2f}x vs RCCL)")
    print("PASS" if ok else "FAIL")
