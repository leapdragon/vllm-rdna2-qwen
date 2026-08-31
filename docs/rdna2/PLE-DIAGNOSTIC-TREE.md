# PLE offload — setup from step 0 and the timeout diagnostic tree

(Condensed hand-out edition; the long-form canonical page is
[PLE-OFFLOAD-SETUP.md](PLE-OFFLOAD-SETUP.md) — keep the two in sync when editing.)

The n-gram (PLE) table is this model's unusual organ: a 51-billion-row embedding table that
lives on the **CPU**, served to the GPUs by a dedicated offload worker every step. Almost
every "PLE timeout" traces back to the environment around that worker, not the code.

## 1. Environment from step 0

### 1.0 What the machine needs

| requirement | minimum | notes |
|---|---|---|
| GPUs | 4× gfx1030 (V620 / RX 6800-class, 32 GB) | TP=4; `--enable-expert-parallel` is mandatory |
| System RAM | 64 GB workable, 96+ comfortable | the int4 sidecar wants to live in page cache (30 GB) next to ~20 GB of process RSS |
| Disk for models | ~110 GB free on an **SSD** | HDD/NFS under the sidecar produces exactly the slow-lookup symptoms in §3 |
| ROCm | TheRock 7.14 at `/opt/rocm` | plus PyTorch/Triton per README §3–4 |

### 1.1 Download — two artifacts, nothing else

```bash
# Backbone (73 GB). Shard 1 is the 102 GB bf16 n-gram table -- do NOT download it.
hf download wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16 --local-dir qwen38-flash-next \
   --exclude "model-00001-of-00005.safetensors"
# n-gram table as an int4 sidecar (30 GB). This REPLACES shard 1.
hf download primitive-ai/Qwen3.8-Flash-Next-PLE-quant --include "ples_int4/*" \
   --local-dir qwen38-flash-next-ple
```

Verify: the weight index references **only shards 2–5 + `model_mtp.safetensors`** (shard 1's
exclusion is by design, not an optimisation); `model_mtp.safetensors` must stay on disk even at
`MTP=0` (it is indexed); the sidecar holds **129 files** and its `META.json` must read
`"layout": "group16_int4_fp16scale_lownibblefirst", "shards": 128, "rows": 320001536,
"width": 160` — anything else is a different sidecar build than we tested.

### 1.2 What we did to the weights: **nothing**

Both repositories are used exactly as downloaded — no conversion, no re-quantisation, no
tokenizer/template/config edits. If your files differ from Hugging Face's checksums, the
difference is yours.

### 1.3 Environment — use the serve script, not raw `vllm serve`

A raw `vllm serve <model>` **cannot work with this checkpoint**: the table is not in the
weight index, so without the offload env the model has no table at all.
`tools/rdna2/serve-qwen38-flash-next.sh` sets: `VLLM_PLE_CPU_OFFLOAD=1`,
`VLLM_PLE_QUANT_DIR=$PLE_INT4`, `VLLM_PLE_OFFLOAD_READY_TIMEOUT=3600` (**vLLM's default is
600 s**), plus `ROCR_VISIBLE_DEVICES`, `HSA_NO_SCRATCH_RECLAIM=1`, `NCCL_P2P_LEVEL=PXB`,
`VLLM_ROCM_USE_AITER=0`, `TORCH_BLAS_PREFER_HIPBLASLT=0`,
`FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`, `PYTORCH_TUNABLEOP_ENABLED=0`,
`VLLM_RDNA_DENSE_INT8=1`, `VLLM_RDNA_AR=1`, and the required CLI
(`--dtype float16 -tp 4 --enable-expert-parallel …`).

**KV-pool interaction you will hit with MTP:** the pool must hold one max-length request. At
90 % on 32 GB cards: MTP=0 text-only ≈ 530k tokens; vision costs ~130k; **MTP=3 costs ~150k
more**. So MTP=3 + vision does **not** fit `MAXLEN=196608` — use 131072 or 65536. A startup
error about max seq len vs KV cache size is this, not PLE.

### 1.4 A healthy first boot prints, in order

`Bound IPC address …; waiting for 4 GPU worker registration(s)` →
`PLE quant table: …, 128 shards mmapped` → 4× `GPU worker N registered` →
`PLE sidecar prefaulted into the page cache: 32.0 GB in ~30 s` →
compile (~12 min cold / ~1 s warm) → `GPU KV cache size: …` → startup complete.
Zero lines matching `Duplicate PLE`, `PLE lookup for launch`, or `did not complete launch`.

## 2. Confirm the tree first

The handshake was rewritten 2026-08-30. Be at **`f6cf11a1f` or later**. Old-tree signatures
that mean "update, don't debug": `'_OpNamespace' … rdna_hc_mix` (warm boot, pre-`a3c5f6846`);
`queue.Full` / `request queue stayed full for 300 s` (mid-rework trees); constant
`Duplicate PLE request … skipping` with garbled output (pre-rework race).

## 3. Diagnostic tree — start from the exact message

**A. `PLE offload worker did not become ready within 600.0s.` (startup)** → not using the
serve script (which sets 3600) *and* slow first-touch. (1) No `PLE quant table: … mmapped`
line → the worker died earlier: bad `VLLM_PLE_QUANT_DIR`, META mismatch, missing shards
(count 129). (2) Line present but timeout anyway → storage: sidecar on HDD/NFS; move to SSD
or raise the timeout. (3) `VLLM_PLE_QUANT_DIR` unset → the single most common cause; set it.

**B. `Waiting for 4 GPU worker registration(s)` forever** → the *GPU* side died first; scroll
up for the real error. Usual: vision OOM (`Tried to allocate 64.00 GiB` — needs the
`max_pixels` cap the serve script applies), `MAXLEN` too big for the pool (§1.3),
a ROCm/torch mismatch.

**C. `PLE lookup for launch N has taken >5 s` (warning)** → page faults on the sidecar. Check
the prefault line's duration; check `free -g` (need ~35 GB headroom); check nothing set
`PLE_OFFLOAD_PREFAULT=0`. Occasional in the first minute on a cold cache = benign; sustained =
storage or RAM pressure, always.

**D. `PLE offload worker did not complete launch N within 600 s (done=M)`** → (1) the worker
crashed — find its traceback above (`PleOffloadWorker pid=…`); report *that*, not the timeout.
(2) `done` climbing slowly → extreme storage latency, same as C. (3) `done` frozen with no
worker error → a genuine protocol hang; we cannot reproduce one at HEAD (full suites at MTP=0
**and** MTP=3, zero warnings) — open an issue with commit hash, full serve log, META.json,
storage type, `free -g`, and whether the serve script was used verbatim.

**E. "It only happens with MTP"** → nothing in the PLE path is MTP-specific (the draft model
runs with PLE off; one lookup per engine step in both modes — verified at HEAD with
byte-identical greedy output under speculative decoding). It is almost always: the KV-pool
shrink (§1.3) throwing startup errors that get pasted next to PLE lines; an existing storage
problem (C) made more visible; or an old tree (§2). Also: on the fixed protocol MTP=3
measured **60–72 t/s vs 62–65 at MTP=0** at ~70 % draft acceptance — a wash. Unless your
workload's acceptance is high, `MTP=0` is the better trade and sidesteps this branch entirely.

Also, a subtlety that matters when MTP=0 works and MTP=3 times out: the host-side PLE wait
sits at the head of every engine step, and step N's lookup request is only *sent* once all of
step N-1's GPU work — including the drafter's extra forwards — has finished. **Any GPU-side
stall in the MTP drafter is therefore reported as a PLE timeout**, with `done` frozen one
behind the launch number. Discriminate in one step: during the stall, check GPU busy
(`rocm-smi`) — pegged high means the drafter is stuck on your hardware and PLE is the
messenger, not the fault; near zero means the worker/storage side really is stuck. Report
busy%, `free -g` in both MTP configurations, the exact `done=M`, and whether it happens at
warmup or mid-serving.

**F. None of the above** → platform, not PLE: `journalctl -k` for amdgpu/PCIe events, the
stability kernel line from CHANGES §4, and never leave a GPU-side stream wait pending on
gfx1030 (CHANGES §8a tells that story).

**G. Suspected board/fabric differences (different mainboard, ACS, chipset-routed or
cross-socket slots)** → the one-shot all-reduce needs healthy GPU↔GPU P2P posted writes.
Since 2026-08-31 it **self-tests at boot** (three verified collectives + a latency bound) and
falls back to RCCL group-wide with `rdna_ar: disabled -- boot self-test failed` if your
board's P2P is broken or slow — so on current trees a bad fabric downgrades performance
instead of corrupting output. Two manual probes: boot once with `VLLM_RDNA_AR=0` (forces
RCCL for the small collectives — if your stalls vanish, your board's P2P is the story), and
check the boot log for the `rdna_ar: one-shot all-reduce active` vs `disabled` lines.
