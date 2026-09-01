# Fixing the idle CPU spin (host source build)

TheRock ROCm 7.14's bundled `libhsa-runtime64.so` (ROCR 1.21 line) busy-spins CPU cores even
when the GPU is idle: roughly **one full core per HIP process**, plus one more per tensor-parallel
rank once RCCL initializes. On a 4-way TP serve that is several cores of standing load and heat.
Root causes (see [ROCm/TheRock#7051](https://github.com/ROCm/TheRock/issues/7051) and
[ROCm/ROCm#6522](https://github.com/ROCm/ROCm/issues/6522)): the `AsyncEventsLoop` polling mode and
several signal-wait paths lack any backoff, and an `InterruptSignal` wait can livelock returning
instantly. No stock environment knob disables it with more than one GPU visible
(`HSA_ENABLE_INTERRUPT`, `ROC_ACTIVE_WAIT_TIMEOUT`, `hipSetDeviceFlags(BlockingSync)` all measured
ineffective).

The fix is a small patch to the ROCR runtime that adds a bounded poll cadence at each spin site.
It lives in this repo at [`containers/patches/rocr-async-events-poll-backoff.patch`](../../containers/patches/rocr-async-events-poll-backoff.patch).
Container images built from this repo already bake it in (`Dockerfile.base`, stage `rocr-patch`).
This document is the **host source-build** procedure.

## What this does

Rebuilds **only** `libhsa-runtime64.so` (~5 MB). Your torch, triton, torchvision, the vLLM
extension, and your venv are **not** touched — none of them are involved in the spin. Deployment is
by `LD_PRELOAD`, so your ROCm install is left pristine.

## Placeholders

| Placeholder | Meaning |
|---|---|
| `<clone>`  | root of your `vllm-rdna2-qwen` checkout (the one with `containers/`) |
| `<work>`   | any empty scratch directory you can write to |
| `<ROCM>`   | your ROCm 7.14 install prefix (commonly `/opt/rocm`) |
| `<COMMIT>` | the `rocm-systems` commit your ROCm was built from (see step 2) |

## 1. Build prerequisites

```bash
sudo apt-get install -y cmake ninja-build build-essential xxd \
     libdrm-dev libelf-dev libnuma-dev
```

`xxd` is easy to miss — the trap-handler header generator needs it.

## 2. Get the ROCR source at the right commit

```bash
cd <work>
git clone --filter=blob:none --no-checkout --sparse \
    https://github.com/ROCm/rocm-systems.git rocm-systems
cd rocm-systems
git sparse-checkout set projects/rocr-runtime
git fetch origin <COMMIT>
git checkout <COMMIT>
```

For `<COMMIT>`: on the **public TheRock 7.14.1 tarball**, use
`ca887ee80abfb82671fe1d6d8da708a713438e05`. If you built ROCm yourself, use the `rocm-systems`
commit from your own tree. The patched runtime is ABI-compatible across 7.14.x, so the `ca887ee`
build usually preloads fine over a slightly different 7.14 install if you'd rather not hunt your
exact commit.

## 3. Apply the patch

From your existing clone:

```bash
git apply <clone>/containers/patches/rocr-async-events-poll-backoff.patch
```

Or fetch it straight from GitHub:

```bash
curl -fL https://github.com/leapdragon/vllm-rdna2-qwen/raw/rdna2/qwen38-flash-next/containers/patches/rocr-async-events-poll-backoff.patch \
  | git apply
```

## 4. Build just the runtime (~2 min)

```bash
cmake -G Ninja -S projects/rocr-runtime -B build \
      -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
      -DCMAKE_PREFIX_PATH="<ROCM>;<ROCM>/lib/llvm"
cmake --build build -j "$(nproc)"
```

Result: `<work>/rocm-systems/build/rocr/lib/libhsa-runtime64.so.1.21.0`

If CMake can't find `clang` for the blit kernels, prepend ROCm's compiler dir and re-run the
configure: `export PATH=<ROCM>/lib/llvm/bin:$PATH`.

## 5. Deploy by preloading

Set this in whatever environment your server starts in:

```bash
export LD_PRELOAD=<work>/rocm-systems/build/rocr/lib/libhsa-runtime64.so.1.21.0
# then start your server as usual
```

If you use this fork's launcher, add that one `LD_PRELOAD=` line to the launcher's site-config env
file so it applies on every start.

## 6. Verify (~20 s)

```bash
LD_PRELOAD=<work>/rocm-systems/build/rocr/lib/libhsa-runtime64.so.1.21.0 \
  python -c "import torch,time; torch.zeros(1,device='cuda'); torch.cuda.synchronize(); print('ready'); time.sleep(20)" &
# watch it in:  top -H -p $!
# patched:   hottest thread near idle.
# unpatched: one thread pinned at 100%.
```

## Tuning knobs

Baked into the patch; all have sane defaults:

| Variable | Default | Effect |
|---|---|---|
| `HSA_BUSY_WAIT_POLL_US`    | `20`   | poll cadence between empty scans; set `0` to restore the stock spin (for an A/B) |
| `HSA_BUSY_WAIT_GRACE_US`   | `1000` | full-speed spin window before backoff engages |
| `HSA_ASYNC_EVENTS_POLL_US` | `100`  | async-events loop cadence |
