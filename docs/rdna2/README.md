# Qwen3.8-Flash-Next on 4× Radeon PRO V620 (gfx1030) with vLLM — the happy path

This fork of vLLM serves **Qwen3.8-Flash-Next** (176 B parameters, ~6 B active, a 51 B-row
n-gram embedding table) on **four AMD Radeon PRO V620** cards at **~100 tokens/s** single-stream
decode, built from source against a stock **TheRock ROCm 7.14** install — no Docker required.
Upstream vLLM cannot do this: it does not target gfx1030, the model family is unmerged, and the
n-gram table does not fit on the cards. See [`CHANGES.md`](CHANGES.md) for what was changed and
why, [`RESULTS.md`](RESULTS.md) for the measurements, [`PROFILE-NAVI21.md`](PROFILE-NAVI21.md)
for the silicon profile every kernel here was designed against, and
[`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) for the failure modes that lie.

If you have the same class of machine, follow the steps in order. Budget: ~3 hours of
unattended build time (PyTorch is the long pole), ~1 hour of downloads, 15 minutes per server
boot.

---

## 1. What you need

| item | ours | notes |
|---|---|---|
| GPUs | 4× Radeon PRO V620 (Navi 21, gfx1030, 32 GB each) | Any 4 gfx103x cards with 32 GB should behave the same; 24 GB cards will not fit the 18.3 GiB/card backbone plus KV |
| Host | Ubuntu (kernel 7.0), 32 cores, 128 GB RAM, ~250 GB free disk | RAM: the 30 GB n-gram sidecar must sit in page cache. Disk: 73 GB backbone + 30 GB sidecar + ~40 GB of build trees |
| ROCm | **TheRock 7.14** runfile install at `/opt/rocm` (→ `/opt/TheRock/rocm/core-7.14`) | `rocminfo` must list your cards as `gfx1030`. Nothing else in ROCm needs to be installed; do **not** mix in apt ROCm packages |
| Kernel command line | `amdgpu.pcie_gen_cap=0x00070007 amdgpu.aspm=0 amdgpu.runpm=0 amdgpu.gpu_recovery=1 amdgpu.noretry=1 amd_iommu=on iommu=pt` | The flat-TP=4 stability stack. Without it, four of these cards under tensor parallelism drop off the PCIe bus (§4 of TROUBLESHOOTING) |
| Python | 3.12 via `uv` (`/snap/bin/uv` or `pip install uv`) | 3.13/3.14 have no working torch build path here |
| Tools | git, cmake ≥ 3.26, ninja, gcc 13+, `hf` (Hugging Face CLI) | |
| Fans / power | a fan controller that actually runs; power caps ≤ 232 W | Thermal throttling silently caps results |

Do not put a pre-Vega (gfx8) card in the machine, even display-only (TROUBLESHOOTING §1–3).

## 2. Get the code

```bash
git clone https://github.com/leapdragon/vllm-rdna2-qwen.git
cd vllm-rdna2-qwen
git checkout rdna2/qwen38-flash-next
```

The branch is upstream vLLM main (post-0.28.0) + the Flash-Next model branch (vLLM PR #53896)
+ this fork's commits. `main` tracks upstream.

## 3. Build PyTorch and Triton against TheRock 7.14

TheRock publishes PyTorch wheels for gfx110X/gfx120X/gfx94X/gfx950 — **not for gfx103X** — so
PyTorch is built from source. Use the same pins as this fork's `docker/Dockerfile.rocm_base`
(ROCm/pytorch `release/2.12` @ `6bbd260`, ROCm/triton @ `f0b55c0`). One script does it:

```bash
uv venv --python 3.12 ~/venvs/vllm-rdna2-qwen
source ~/venvs/vllm-rdna2-qwen/bin/activate
tools/rdna2/build-torch-rocm714.sh          # ~2–3 h on 32 cores; writes wheels to ~/wheels/rdna2/
pip install ~/wheels/rdna2/torch-*.whl ~/wheels/rdna2/triton-*.whl
python -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.get_arch_list())"
```

The last line must print `gfx1030` in the arch list **with GPU devices visible**; an empty
list is not an error message, it is a wrong build.

## 4. Build vLLM (this fork)

```bash
source ~/venvs/vllm-rdna2-qwen/bin/activate
# build deps WITHOUT torch/triton (requirements/build/rocm.txt would pull torch 2.11+rocm7.1
# with its own bundled ROCm 7.1 runtime — exactly the mixed-ROCm trap; you built torch above)
pip install "cmake>=3.26.1,<4" "packaging>=24.2" "setuptools>=77.0.3,<80" "setuptools-scm>=8" \
            "setuptools-rust>=1.9.0" wheel "jinja2>=3.1.6" ninja -r requirements/common.txt
export VLLM_TARGET_DEVICE=rocm PYTORCH_ROCM_ARCH=gfx1030 ROCM_PATH=/opt/rocm MAX_JOBS=24
pip install -e . --no-build-isolation --no-deps      # ~15 min: _C, _rocm_C (our kernels), _moe_C
pip install -r requirements/rocm.txt --no-deps       # runtime deps (then `pip check` and add what it names)
```

The Rust frontend (`vllm-rs`, the Rust tool parser) is optional and is skipped when `cargo` is
absent; nothing on this path needs it.

Rebuilding only the ROCm extension after a kernel edit: see `tools/rdna2/README.md`.

## 5. Get the model (two downloads, no conversion)

```bash
# backbone: AWQ-W4A16 (only the routed experts are int4; everything else is bf16, served fp16)
hf download wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16 --local-dir models/qwen38-flash-next \
   --exclude "model-00001-of-00005.safetensors"      # shard 1 is the 102 GB bf16 n-gram table: not needed
# n-gram table as an int4 sidecar (30 GB) — replaces shard 1
hf download primitive-ai/Qwen3.8-Flash-Next-PLE-quant --include "ples_int4/*" --local-dir models/qwen38-flash-next-ple
```

`models/qwen38-flash-next-ple/ples_int4/META.json` must read
`"layout": "group16_int4_fp16scale_lownibblefirst", "rows": 320001536, "width": 160`.

## 6. Serve

```bash
MODEL=models/qwen38-flash-next PLE_INT4=models/qwen38-flash-next-ple/ples_int4 \
  tools/rdna2/serve-qwen38-flash-next.sh
```

That script is the whole configuration; read it. The knobs that are not optional are explained
in `CHANGES.md` §4 (expert parallelism, fp16, language-model-only, the stability env). Boot
takes ~15 minutes (weights, torch.compile, CUDA-graph capture, the MTP head). `curl
localhost:8000/health` returns 200 at "Application startup complete". The first request at a
new context length is slow while sidecar pages are cold.

Optional knobs: `MTP=4` (+2 % on long generations), `DENSE_INT8=0` (fp16 dense projections;
−15 %), `GPUS=1,2,3,4` (ROCR device ids), `PORT=`, `PROFILE=1` (enables `/start_profile`).

## 7. Validate and measure

```bash
python tools/rdna2/bench.py 1 64          # warm-up (cold sidecar pages)
python tools/rdna2/bench.py 3 256         # the yardstick: decode t/s, prefill excluded
python tools/rdna2/bench.py 1 1024        # long-generation control
python tools/rdna2/bench_ctx.py 13000 256 # decode at ~13k context, unique prompt
```

Greedy sanity (chat completions, `enable_thinking: false`, temperature 0): `17 * 23` → `391`,
capital of Australia → `Canberra`, first five primes → `2, 3, 5, 7, 11`.

What you should see (our numbers, `RESULTS.md` T46):

| | decode t/s |
|---|---|
| 256 tokens, ×3 | 98 / 101 / 97 |
| 1024 tokens | 95 |
| 2.9k / 10.6k / 27k context | 78 / 108 / 88 (acceptance-driven; no context slope) |
| per MTP step | ~29 ms, ~1,800 kernels |

After every run: `journalctl -k | grep amdgpu` should show nothing new, and `rocm-smi --showtemp`
should be under 60 °C. Container logs are UTC and the host journal is local time — convert
before attributing a fault to a run.

## 8. Profiling — how the optimisations were found

Boot with `PROFILE=1`, then `curl -X POST localhost:8000/start_profile`, run one bench,
`curl -X POST localhost:8000/stop_profile`; traces land in `logs/traces/`.
`tools/rdna2/trace_agg.py <trace>` gives kernel time by family and GPU busy;
`tools/rdna2/trace_attr.py <trace> <steps>` (on an `EAGER=1 PROFILE=1` boot) attributes every
launch to a model source line. That pair found every lever in `CHANGES.md` §5–8.

## 9. If it does not work

`TROUBLESHOOTING.md` first. Then, in the order they bit us: cards missing from `rocminfo`
(another GPU in the machine), one card loading weights and three idle (a collective deadlocked
during init — check the log for `rdna_ar`), `hipErrorOutOfMemory` with gigabytes free (not a
memory problem), a lever that "did nothing" (profile the kernel *counts* — torch.compile may have
frozen your branch at trace time, `CHANGES.md` §8), and 10× flattering prefill numbers (prefix
caching + a repeated prompt).
