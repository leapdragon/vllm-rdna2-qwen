# Container image

The fork, prebuilt: `ghcr.io/leapdragon/vllm-rdna2-qwen` — this tree built exactly as
[docs/rdna2/README.md](../docs/rdna2/README.md) §3–4 prescribe, on **TheRock ROCm 7.14** with
PyTorch 2.12 / Triton 3.7 / torchvision compiled from source for gfx1030, dependency pins captured
from the machine that produced every number in [RESULTS.md](../docs/rdna2/RESULTS.md).

| Image | Tag | What |
|---|---|---|
| `ghcr.io/leapdragon/vllm-rdna2-qwen` | `<date>-g<sha>`, `latest` | Runtime image — [`Dockerfile`](Dockerfile). |
| `ghcr.io/leapdragon/vllm-rdna2-qwen-base` | `therock7.14.0rc3-torch2.12-gfx1030` | Base: TheRock 7.14 + torch/triton/vision from source — [`Dockerfile.base`](Dockerfile.base). Only needed to rebuild the runtime image. |

## Getting started

What the host needs: **4× gfx103x cards with 32 GB** (TP=4 with expert parallelism is assumed
throughout), the `amdgpu` kernel driver, Docker, **≥ 96 GB RAM** (the 30 GB n-gram sidecar lives in
the page cache) and the platform-stability kernel line from README §1. **No ROCm on the host** —
the image carries TheRock 7.14; never mount a host ROCm into it.

**1. Weights** (README §5 — two downloads, no conversion), into one directory:

```bash
mkdir -p models && cd models
hf download wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16 --local-dir qwen38-flash-next \
   --exclude "model-00001-of-00005.safetensors"                        # 73 GB; shard 1 is NOT needed
hf download primitive-ai/Qwen3.8-Flash-Next-PLE-quant --include "ples_int4/*" \
   --local-dir qwen38-flash-next-ple                                    # 30 GB int4 sidecar
```

**2. Serve** — the image's entrypoint is the fork's own tuned launcher
(`tools/rdna2/serve-qwen38-flash-next.sh`); every knob it documents works as `-e`:

```bash
docker run -d --name qwen38 --network=host \
  --device /dev/kfd --device /dev/dri \
  --group-add "$(getent group render | cut -d: -f3)" --group-add "$(getent group video | cut -d: -f3)" \
  --ipc=host --ulimit memlock=-1 --security-opt seccomp=unconfined \
  -e ROCR_VISIBLE_DEVICES=0,1,2,3 \
  -v "$PWD/models:/models" \
  -v qwen38-compile-cache:/compile-cache \
  ghcr.io/leapdragon/vllm-rdna2-qwen:latest
```

- Flags, and why: `--device /dev/kfd --device /dev/dri` + the `render`/`video` group adds give the
  container the GPUs (group ids differ per distro, hence `getent`); `--ipc=host` and
  `--ulimit memlock=-1` are required — the n-gram worker and the four GPU workers share pinned
  host buffers; `--security-opt seccomp=unconfined` lets ROCm's memory mapping work;
  `--network=host` exposes port 8000 directly (the only mode we test).
- `ROCR_VISIBLE_DEVICES` — your four serving cards, in `rocm-smi` order.
- `/models` must hold `qwen38-flash-next/` and `qwen38-flash-next-ple/ples_int4/` (override with
  `-e MODEL=… -e PLE_INT4=…`). Weights are never shipped in the image.
- `/compile-cache` (a named volume is fine) persists torch.compile: **first boot ≈ 15–20 min**
  (measured 17–19: compile + the 30 GB sidecar prefault), later boots ≈ 4–5 min. The healthcheck
  allows 40 min; `docker inspect --format '{{.State.Health.Status}}' qwen38` says `healthy` when
  the server answers. Keep `/models` on an SSD — a spinning disk or network share under the sidecar
  produces the slow-lookup symptoms in the PLE diagnostic tree.
- Knobs: `MTP` (3; `0` disables speculative decoding and frees ~150k tokens of KV),
  `MAXLEN` (131072), `GPUUTIL` (0.90), `VISION` (0/1), `CHAT_KWARGS`
  (`'{"preserve_thinking": true, "reasoning_effort": "medium"}'`), `TOOLS` (1), `PORT` (8000 —
  add `-e HEALTHCHECK_PORT=`), `EXTRA_ARGS`, `DRYRUN=1` (print the resolved command and exit).
  Semantics and the measured worth of each: README §6 and [CHANGES.md](../docs/rdna2/CHANGES.md).

**3. Query** — a standard OpenAI-compatible server (`--served-model-name qwen38-flash-next`):

```bash
docker logs -f qwen38                      # wait for "Application startup complete"
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen38-flash-next", "messages": [{"role":"user","content":"Say hello in five words."}], "max_tokens": 64}'
```

Expect the RESULTS.md numbers once warm (measured from this image: 71–74 t/s at MTP=3, 1024-token
runs). **The first few minutes after a boot decode slower** (we saw 50–59 t/s): the 32 GB sidecar is
prefaulted, but loading the 73 GB backbone afterwards evicts part of it on a ≤128 GB host, and the
n-gram rows real requests need fault back in from disk until they are resident — the PLE worker's
`lookup` time in the log falls from ~45 ms to ~15–20 ms per request as that happens. Nothing to fix
in the container; more RAM or a warm second boot makes it disappear.

**4. Stop, restart, update**

```bash
docker stop -t 60 qwen38            # graceful: the GPU workers take ~20 s to unwind ROCm; SIGKILL leaves VRAM held until they exit
docker start qwen38                 # same configuration, warm compile cache: ~4–5 min to healthy
docker rm qwen38                    # to change knobs (e.g. -e MTP=0), remove and run again with the new -e
docker pull ghcr.io/leapdragon/vllm-rdna2-qwen:latest   # a new image version: rm + run again; a code change
                                                         # invalidates the compile cache, so expect one cold boot
```

**5. If something is wrong** — get the support report from inside the container and send it with
your issue ([TROUBLESHOOTING.md](../docs/rdna2/TROUBLESHOOTING.md) §5.0):

```bash
docker logs qwen38 > qwen38.log 2>&1
docker cp qwen38.log qwen38:/tmp/serve.log
docker exec qwen38 tools/rdna2/system-report.sh --log /tmp/serve.log --out /tmp/system-report.log
docker cp qwen38:/tmp/system-report.log .            # review it (it is redacted), then attach it
```

PLE timeouts have their own decision tree: [PLE-DIAGNOSTIC-TREE.md](../docs/rdna2/PLE-DIAGNOSTIC-TREE.md).

Any other command after the image name runs inside the environment (`bash`, `python3`,
`tools/rdna2/validate.py`, `tools/rdna2/bench.py`, …) — `serve` is the default.

## Why the base is built the way it is

TheRock publishes no PyTorch wheels for the gfx103X family, so torch, Triton and torchvision are
compiled from source (`tools/rdna2/build-torch-rocm714.sh`, 2–3 h on 32 cores). ROCm itself comes
from TheRock's public **multi-arch tarball, pinned by URL and sha256** in `Dockerfile.base`:
`repo.amd.com/rocm/tarball-multi-arch/therock-dist-linux-gfx103X-all-7.14.1.tar.gz` (2.3 GB). The
7.14 line lives on TheRock's *legacy* indexes — `repo.amd.com/rocm/tarball-multi-arch/` for
releases, `rocm.prereleases.amd.com/tarball-multi-arch/` for the `7.14.0rcN` candidates — not on the
current `*.repo.amd.com/rocm/core/` channels, which carry 10.x.

```bash
./containers/build.sh --base                          # base from the pinned public tarball; hours
ROCM_DIR=/opt/rocm ./containers/build.sh --base        # …or from a local TheRock 7.14 install
./containers/build.sh                                 # runtime image only, FROM the published base (~20 min)
```

The numbers in RESULTS.md were measured against the maintainer's own from-source TheRock
7.14.0rc3 tree. Two public pins therefore exist: the default **7.14.1** (stable index) and the
host-exact **7.14.0rc3** candidate — `THEROCK_URL=https://rocm.prereleases.amd.com/tarball-multi-arch/therock-dist-linux-gfx103X-all-7.14.0rc3.tar.gz`
`THEROCK_SHA256=ce9a5be2b43ee1bdd85de3fa9ea3c3d5dcb6875445acf7281c3763a4ee783f19` — both validated separately
(Build status below). Moving the fork to ROCm 10.x is future work.

Provenance inside the image: `/opt/versions.txt` (TheRock commit and package version, torch /
triton / torchvision versions and source refs, vLLM version), and the tree itself at `/app/vllm`.

## Build status

| Date | What | Result |
|---|---|---|
| 2026-08-31 | Base from the public TheRock **7.14.1** tarball (sha256-pinned), torch/triton/vision from source | **PASS.** Torch stage 58 min at MAX_JOBS=24 on 32 cores; base 12.5 GB; in-build checks: HIP 7.14.60850, `gfx1030` in torch's arch flags. (The same TheRock tarball layout was probed on the host's from-source 7.14.0rc3 tree first; both TheRock commits are pinned in `build.sh`'s header.) |
| 2026-08-31 | Runtime image (this tree, README §4) | **PASS after three fixes**, all in the container recipe rather than the fork: README §4's version ranges clash with the validated venv (cmake 4.x, setuptools 81); the venv snapshot is not resolver-consistent (assembled with `--no-deps`) → installed verbatim with `--no-deps`; host build residue (`.deps/` CMake FetchContent cache) must be excluded from the build context. Image 15.1 GB; RDNA ops + PLE offload verified at build. |
| 2026-08-31 | Serve test on 4× V620 (weights mounted, MTP=3 default) | **PASS.** Cold boot 1129 s; all four workers registered, sidecar prefaulted (32 GB / 65 s), one-shot all-reduce self-test passed, KV pool 248,682 tokens, zero PLE warnings; `validate.py` PASS, `toolcall_test.py` PASS. Decode: 50–59 t/s in the first minutes, **71–74.5 t/s once the sidecar rows are resident** — vs 66–73 t/s from the host-native launcher in a same-day A/B (identical config, same bench). Acceptance 2.6–2.9 tokens/step. The transient shortfall is page-cache eviction of the sidecar by the weight load, identical on the host; the container runtime itself measured equal on HIP latency, NumPy gathers and CPU limits. |
| 2026-09-01 | Rebuilt with the host's interpreter build (python-build-standalone 3.12.13 via uv; Ubuntu's 3.12.3 measured 1.4× slower on pure Python) | **PASS.** Torch stage served from cache; runtime 15.3 GB; `validate.py` PASS, zero PLE warnings; hot decode 69–73 t/s, PLE worker `lookup` 15 ms per request — host parity. This is the published build. |

## Publishing (maintainer)

```bash
gh auth refresh -h github.com -s write:packages && gh auth token | docker login ghcr.io -u leapdragon --password-stdin
./containers/build.sh --push            # runtime + latest;  add --base to publish the base too
```

First push creates the package private — make it public in Package settings.

## Known issue: idle CPU spin in the bundled ROCm runtime

TheRock 7.14's `libhsa-runtime64.so` (ROCR 1.21 line) busy-spins one full CPU core per
HIP process even when the GPU is completely idle — in a 4-way TP serve that is several
cores of standing load and heat ([ROCm/TheRock#7051](https://github.com/ROCm/TheRock/issues/7051)).
Root cause: `AsyncEventsLoop` falls into a polling mode (any async signal without a kernel
event forces it) and its rescan loop has no backoff. No environment knob of the stock
runtime disables it (`HSA_ENABLE_INTERRUPT`, `ROC_ACTIVE_WAIT_TIMEOUT`,
`hipSetDeviceFlags(BlockingSync)` all measured ineffective with >1 visible GPU).

A one-hunk patch adds a bounded poll cadence (default 100 µs, tunable via
`HSA_ASYNC_EVENTS_POLL_US`, `0` restores stock behavior); measured: idle spin eliminated,
GEMM/serve latency unchanged. Patch (applies to rocm-systems @ `ca887ee`,
`projects/rocr-runtime/.../core/runtime/runtime.cpp`):
`https://github.com/leapdragon/vllm-rdna2/blob/main/patches/rocr-async-events-poll-backoff.patch`
(also usable outside containers: build ROCR, then `LD_PRELOAD` the patched
`libhsa-runtime64.so.1.21.0` into the serve).
