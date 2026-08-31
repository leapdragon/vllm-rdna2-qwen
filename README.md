# GETTING STARTED WITH THIS FORK

This is a fork of [vLLM](https://github.com/vllm-project/vllm) that serves the
**Qwen3.8-Flash-Next** model (176 B parameters) on **four AMD Radeon PRO V620** cards
(Navi 21 / gfx1030) at about **100 tokens per second**, built against a stock
**TheRock ROCm 7.14** install. Stock vLLM cannot do this: it does not support gfx1030, the
model is not merged upstream yet, and the model's 51-billion-row n-gram table does not fit on
the cards. Everything that makes it work is in this repository. You do not need Docker.

**What you get**: the patched vLLM source (kernels written for this chip, a P2P all-reduce,
int8 shadows, fused decode kernels, the CPU offload for the n-gram table), the build and serve
scripts, the benchmark/validation tools, and the research write-ups explaining every change.

**What you need** (details in [`docs/rdna2/README.md`](docs/rdna2/README.md) §1):

- A Linux box (ours: Ubuntu, kernel 7.0) with **4× gfx103x cards with 32 GB each**, ≥96 GB of
  RAM, ~250 GB of free disk, and **no other GPU generation in the machine**.
- The kernel command line `amdgpu.pcie_gen_cap=0x00070007 amdgpu.aspm=0 amdgpu.runpm=0
  amdgpu.gpu_recovery=1 amdgpu.noretry=1 amd_iommu=on iommu=pt` (without it, four of these
  cards under tensor parallelism fall off the PCIe bus).
- **TheRock ROCm 7.14** installed with the runfile so that `/opt/rocm` points at it and
  `/opt/rocm/bin/rocminfo` lists your cards as `gfx1030`. No apt ROCm packages mixed in.
- `git`, `cmake`, `ninja`, `gcc`, [`uv`](https://docs.astral.sh/uv/) (for a Python 3.12
  environment), and the Hugging Face CLI (`pip install -U huggingface_hub` gives you `hf`).
- About 3 hours of unattended build time and ~105 GB of downloads.

**The steps** (each one is spelled out in [`docs/rdna2/README.md`](docs/rdna2/README.md)):

1. **Get the code.** Clone this repository — the default branch is the working one:

   ```bash
   git clone https://github.com/leapdragon/vllm-rdna2-qwen.git
   cd vllm-rdna2-qwen        # you are on branch rdna2/qwen38-flash-next
   ```

2. **Build PyTorch, Triton and torchvision for gfx1030** (TheRock does not publish PyTorch
   wheels for this GPU family, so they are built from source — this is the long step):

   ```bash
   uv venv --python 3.12 ~/venvs/vllm-rdna2-qwen
   source ~/venvs/vllm-rdna2-qwen/bin/activate
   tools/rdna2/build-torch-rocm714.sh                      # ~2-3 hours; wheels land in ~/wheels/rdna2/
   uv pip install ~/wheels/rdna2/torch-*.whl ~/wheels/rdna2/triton-*.whl ~/wheels/rdna2/torchvision-*.whl
   ```

3. **Build this vLLM** (`docs/rdna2/README.md` §4 has the exact commands, including the two
   small traps: install the wheels above *before* any other dependency, and copy
   `/opt/rocm/share/amd_smi` somewhere writable before installing it):

   ```bash
   uv pip install -e . --no-build-isolation --no-deps       # ~25 minutes
   ```

4. **Download the model** — two parts, no conversion (§5):

   ```bash
   hf download wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16 --local-dir models/qwen38-flash-next \
      --exclude "model-00001-of-00005.safetensors"           # 73 GB; shard 1 is not needed
   hf download primitive-ai/Qwen3.8-Flash-Next-PLE-quant --include "ples_int4/*" \
      --local-dir models/qwen38-flash-next-ple                # 30 GB n-gram table, int4
   ```

5. **Serve it** (§6). The script *is* the whole configuration:

   ```bash
   MODEL=models/qwen38-flash-next PLE_INT4=models/qwen38-flash-next-ple/ples_int4 \
     tools/rdna2/serve-qwen38-flash-next.sh
   ```

   Boot takes ~15 minutes; `curl localhost:8000/health` returns 200 when it is up. It speaks
   the OpenAI API on port 8000 (model name `qwen38-flash-next`).

6. **Check it** (§7):

   ```bash
   python tools/rdna2/validate.py        # must print PASS
   python tools/rdna2/bench.py 3 256     # expect ~60-65 tokens/s decode (MTP=0; ~60-72 with MTP=3)
   ```

## Troubleshooting

1. **Read [`docs/rdna2/TROUBLESHOOTING.md`](docs/rdna2/TROUBLESHOOTING.md) first.** This
   platform's failure modes point away from their causes (a PLE timeout is usually storage,
   RAM or a missing env var; a "hang with MTP" is usually the drafter; a wrong GPU identity is
   a kernel-line issue). For anything involving the n-gram table, PLE timeouts or slow
   lookups, the step-by-step is [`docs/rdna2/PLE-DIAGNOSTIC-TREE.md`](docs/rdna2/PLE-DIAGNOSTIC-TREE.md).
2. **Run the report script and send it to me.**

   ```bash
   tools/rdna2/system-report.sh --probe      # add --tests if no server is running
   ```

   It writes `system-report.log` (host, PCIe links, GPUs, ROCm, venv, torch/vLLM build,
   model files and their storage, the resolved serve command, running processes, a digest of
   your newest serve log, the kernel log of this and the previous boot; `--probe` adds a live
   health/metrics/tiny-completion check). It modifies nothing and redacts home paths, user,
   hostname, IPs and anything that looks like a secret — skim it, then **DM it to me on
   Discord (`<DISCORD-HANDLE>`)** together with the exact error text and one paragraph on
   what you were doing. That file answers most questions before I have to ask them.
3. [`docs/rdna2/README.md`](docs/rdna2/README.md) §9 has the remaining known issues.

## Housekeeping

**To see what was changed**: the GitHub compare view
[`2a46f85b43...rdna2/qwen38-flash-next`](https://github.com/leapdragon/vllm-rdna2-qwen/compare/2a46f85b43...rdna2/qwen38-flash-next)
shows only this fork's commits and diff (everything after the merge of the Flash-Next model
branch); [`main...rdna2/qwen38-flash-next`](https://github.com/leapdragon/vllm-rdna2-qwen/compare/main...rdna2/qwen38-flash-next)
shows everything vs upstream vLLM. `docs/rdna2/CHANGES.md` opens with a map of where the code lives.

**To understand or reuse the work**: [`docs/rdna2/CHANGES.md`](docs/rdna2/CHANGES.md) — every
change and the reason for it; [`docs/rdna2/RESULTS.md`](docs/rdna2/RESULTS.md) — the measured
numbers; [`docs/rdna2/PROFILE-NAVI21.md`](docs/rdna2/PROFILE-NAVI21.md) — the silicon profile the
kernels were designed against; [`tools/rdna2/`](tools/rdna2/) — build, serve, benchmark, profile
and test tools. `main` tracks upstream vLLM; this fork's work is on `rdna2/qwen38-flash-next`.

---

<!-- markdownlint-disable MD001 MD041 -->
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
Easy, fast, and cheap LLM serving for everyone
</h3>

<p align="center">
| <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> |
</p>

🔥 We have built a vLLM website to help you get started with vLLM. Please visit [vllm.ai](https://vllm.ai) to learn more.
For events, please visit [vllm.ai/events](https://vllm.ai/events) to join us.

---

## About

vLLM is a fast and easy-to-use library for LLM inference and serving.

Originally developed in the [Sky Computing Lab](https://sky.cs.berkeley.edu) at UC Berkeley, vLLM has grown into one of the most active open-source AI projects built and maintained by a diverse community of many dozens of academic institutions and companies from over 2000 contributors.

vLLM is fast with:

- State-of-the-art serving throughput
- Efficient management of attention key and value memory with [**PagedAttention**](https://blog.vllm.ai/2023/06/20/vllm.html)
- Continuous batching of incoming requests, chunked prefill, prefix caching
- Fast and flexible model execution with piecewise and full CUDA/HIP graphs
- Quantization: FP8, MXFP8/MXFP4, NVFP4, INT8, INT4, GPTQ/AWQ, GGUF, compressed-tensors, ModelOpt, TorchAO, and [more](https://docs.vllm.ai/en/latest/features/quantization/index.html)
- Optimized attention kernels including FlashAttention, FlashInfer, TRTLLM-GEN, FlashMLA, and Triton
- Optimized GEMM/MoE kernels for various precisions using CUTLASS, TRTLLM-GEN, CuTeDSL
- Speculative decoding including n-gram, suffix, EAGLE, DFlash
- Automatic kernel generation and graph-level transformations using torch.compile
- Disaggregated prefill, decode, and encode

vLLM is flexible and easy to use with:

- Seamless integration with popular Hugging Face models
- High-throughput serving with various decoding algorithms, including *parallel sampling*, *beam search*, and more
- Tensor, pipeline, data, expert, and context parallelism for distributed inference
- Streaming outputs
- Generation of structured outputs using xgrammar or guidance
- Tool calling and reasoning parsers
- OpenAI-compatible API server, plus Anthropic Messages API and gRPC support
- Efficient multi-LoRA support for dense and MoE layers
- Support for NVIDIA GPUs, AMD GPUs, Intel GPUs, and x86/ARM/PowerPC CPUs. Additionally, diverse hardware plugins such as Google TPUs, Intel Gaudi, IBM Spyre, Huawei Ascend, Rebellions NPU, Apple Silicon, MetaX GPU, and more.

vLLM seamlessly supports 200+ model architectures on Hugging Face, including:

- Decoder-only LLMs (e.g., Llama, Qwen, Gemma)
- Mixture-of-Expert LLMs (e.g., Mixtral, DeepSeek-V3, Qwen-MoE, GPT-OSS)
- Hybrid attention and state-space models (e.g., Mamba, Qwen3.5)
- Multi-modal models (e.g., LLaVA, Qwen-VL, Pixtral)
- Embedding and retrieval models (e.g., E5-Mistral, GTE, ColBERT)
- Reward and classification models (e.g., Qwen-Math)

Find the full list of supported models [here](https://docs.vllm.ai/en/latest/models/supported_models.html).

## Getting Started

Install vLLM with [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`:

```bash
uv pip install vllm
```

Or [build from source](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source) for development.

Visit our [documentation](https://docs.vllm.ai/en/latest/) to learn more.

- [Installation](https://docs.vllm.ai/en/latest/getting_started/installation.html)
- [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
- [List of Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

## Contributing

We welcome and value any contributions and collaborations.
Please check out [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/index.html) for how to get involved.

## Citation

If you use vLLM for your research, please cite our [paper](https://arxiv.org/abs/2309.06180):

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

## Contact Us

<!-- --8<-- [start:contact-us] -->
- For technical questions and feature requests, please use GitHub [Issues](https://github.com/vllm-project/vllm/issues)
- For discussing with fellow users, please use the [vLLM Forum](https://discuss.vllm.ai)
- For coordinating contributions and development, please use [Slack](https://slack.vllm.ai)
- For security disclosures, please use GitHub's [Security Advisories](https://github.com/vllm-project/vllm/security/advisories) feature
- For collaborations and partnerships, please contact us at [collaboration@vllm.ai](mailto:collaboration@vllm.ai)
<!-- --8<-- [end:contact-us] -->

## Media Kit

- If you wish to use vLLM's logo, please refer to [our media kit repo](https://github.com/vllm-project/media-kit)
