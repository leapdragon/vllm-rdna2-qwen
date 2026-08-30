#!/usr/bin/env python3
"""Cross-process H2D coherence test, mirroring the PLE offload data path (2026-08-30).

  producer (child, like the offload worker): pinned host buffer -> hipMemcpyAsync H2D into a
      device buffer it received through CUDA IPC, on its own stream; synchronize; publish the
      round number in a shared-memory counter.
  consumer (parent, like the GPU worker): read the buffer with a kernel (so its lines sit in
      L2), wait for the counter on the host, read again with a kernel and check every element
      equals the round number.

Any stale read means the consumer's L2 is not coherent with another process's DMA, i.e. the
model could consume an old n-gram lookup even when the handshake itself is correct.
Run on a free GPU: ROCR_VISIBLE_DEVICES=<idx> python3 ple_coherence_test.py [rounds]
"""
import sys, time
import torch
import torch.multiprocessing as mp

N_ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
NUMEL = 2048 * 8          # like an 8-token decode batch of hidden 2048 (fp16)
BUF = 32 * 1024 * 1024    # a big buffer touched between rounds to churn L2 (32 MB)


def producer(dev_buf, counter, go, n_rounds):
    torch.cuda.set_device(dev_buf.device)
    stream = torch.cuda.Stream()
    pinned = torch.empty(NUMEL, dtype=torch.float16).pin_memory()
    go.wait()
    for i in range(1, n_rounds + 1):
        pinned.fill_(float(i % 2000))
        with torch.cuda.stream(stream):
            dev_buf.copy_(pinned, non_blocking=True)
        stream.synchronize()
        counter[0] = i
        while counter[1] < i:        # wait for the consumer before overwriting (serial protocol)
            time.sleep(1e-5)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    dev_buf = torch.zeros(NUMEL, dtype=torch.float16, device=dev)
    churn = torch.zeros(BUF // 2, dtype=torch.float16, device=dev)
    counter = torch.zeros(2, dtype=torch.int64).share_memory_()
    go = mp.Event()
    p = mp.Process(target=producer, args=(dev_buf, counter, go, N_ROUNDS))
    p.start()
    time.sleep(2)
    go.set()
    stale = 0
    first = None
    t0 = time.time()
    for i in range(1, N_ROUNDS + 1):
        _ = dev_buf.sum()            # keep the buffer's lines warm in L2
        if i % 7 == 0:
            churn.add_(1)
        while counter[0] < i:
            pass
        expect = float(i % 2000)
        got_min, got_max = dev_buf.min().item(), dev_buf.max().item()
        if got_min != expect or got_max != expect:
            stale += 1
            if first is None:
                first = (i, got_min, got_max, expect)
        counter[1] = i
    p.join(timeout=30)
    dt = time.time() - t0
    print(f"rounds={N_ROUNDS} stale_reads={stale} first={first} ({dt / N_ROUNDS * 1e6:.0f} us/round)")
    print("COHERENT" if stale == 0 else "STALE READS DETECTED")
    sys.exit(0 if stale == 0 else 1)
