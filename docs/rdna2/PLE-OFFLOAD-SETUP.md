# PLE offload: environment setup from zero, and a diagnostic tree for timeouts

The n-gram (PLE) table is this model's unusual organ: a 51-billion-row embedding table that
lives on the **CPU**, served to the GPUs by a dedicated offload worker, request by request,
every step. Almost every "PLE timeout" report we have seen traces back to the environment
around that worker — not to the code. A condensed hand-out edition lives at [PLE-DIAGNOSTIC-TREE.md](PLE-DIAGNOSTIC-TREE.md);
keep the two in sync. This page is (1) the environment recipe from step 0,
(2) a statement of exactly what we did and did not do to the weights, and (3) a diagnostic
tree keyed to the exact messages the server prints.

---

## 1. Environment from step 0

### 1.0 What the machine needs

| requirement | minimum | notes |
|---|---|---|
| GPUs | 4× gfx1030 (Radeon PRO V620 / RX 6800-class, 32 GB) | TP=4 is assumed throughout; `--enable-expert-parallel` is mandatory (640 experts / 4 ranks is not divisible by the group size otherwise) |
| System RAM | 64 GB workable, 96+ GB comfortable | the int4 sidecar wants to live in the page cache (30 GB) next to ~20 GB of process RSS |
| Disk for models | ~110 GB free on an **SSD** | backbone 73 GB + sidecar 30 GB. A spinning disk or network filesystem under the sidecar produces exactly the slow-lookup symptoms in §3 |
| ROCm | TheRock 7.14 at `/opt/rocm` (or your build target) | plus the PyTorch/Triton built per README §3–4 |

### 1.1 Download — two artifacts, nothing else

```bash
cd <your-models-dir>

# 1. Backbone (73 GB): AWQ-W4A16. Shard 1 is the 102 GB bf16 n-gram table -- do NOT download it.
hf download wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16 --local-dir qwen38-flash-next \
   --exclude "model-00001-of-00005.safetensors"

# 2. n-gram table as an int4 sidecar (30 GB). This REPLACES shard 1.
hf download primitive-ai/Qwen3.8-Flash-Next-PLE-quant --include "ples_int4/*" \
   --local-dir qwen38-flash-next-ple
```

Facts that make this correct (verify them, don't trust them):

- `model.safetensors.index.json` references **only shards 2–5 and `model_mtp.safetensors`** —
  zero keys point at shard 1. Excluding shard 1 is by design, not an optimization; downloading
  it buys you nothing unless you deliberately run the bf16 disk-offload path (not this recipe).
- `model_mtp.safetensors` (5.2 GB) **must stay on disk even if you run `MTP=0`** — it is in the
  index, and vLLM refuses to start when an indexed file is missing.
- The backbone directory should hold exactly: `model-0000{2,3,4,5}-of-00005.safetensors`,
  `model_mtp.safetensors`, `model.safetensors.index.json`, `config.json`,
  `generation_config.json`, `chat_template.jinja`, `tokenizer.json`, `tokenizer_config.json`,
  `preprocessor_config.json`, `processor_config.json`, `video_preprocessor_config.json`,
  `awq_run_info.json`, `recipe.yaml`.
- The sidecar directory `ples_int4/` should hold **129 files**: `META.json` +
  `shard_0.safetensors` … `shard_127.safetensors`, and META.json must read exactly:

```json
{"layout": "group16_int4_fp16scale_lownibblefirst", "shards": 128, "rows": 320001536, "width": 160, ...}
```

If your META says anything else, you have a different sidecar build than we tested.

### 1.2 What we did to the weights: **nothing**

Both repositories are used **exactly as downloaded from Hugging Face**. No conversion, no
re-quantization, no tokenizer or chat-template edits, no config surgery, no renaming. Every
file in our serving directories carries the download date and matches the HF listing. The only
local artifacts are (a) directory symlinks for convenience and (b) environment variables set
by the serve script. If your files differ from HF's checksums, that difference is yours.

### 1.3 Environment — use the serve script, not raw `vllm serve`

```bash
MODEL=<path>/qwen38-flash-next \
PLE_INT4=<path>/qwen38-flash-next-ple/ples_int4 \
  tools/rdna2/serve-qwen38-flash-next.sh
```

This is the most common failure zone. A raw `vllm serve <model>` **cannot work with this
checkpoint**: the n-gram table is not in the weight index, so without the offload
configuration the model has no table at all. The script sets, and you must not lose:

| variable | value | why |
|---|---|---|
| `VLLM_PLE_CPU_OFFLOAD` | `1` | the table lives on CPU; a dedicated worker process serves it |
| `VLLM_PLE_QUANT_DIR` | `$PLE_INT4` | the int4 sidecar that stands in for shard 1 |
| `VLLM_PLE_OFFLOAD_READY_TIMEOUT` | `3600` | **vLLM's default is 600 s** — a cold first mmap of 30 GB on modest storage can exceed it; see §3-A |
| `ROCR_VISIBLE_DEVICES` | your 4 serving GPUs | |
| `HSA_NO_SCRATCH_RECLAIM=1`, `NCCL_P2P_LEVEL=PXB` | | platform stability (see CHANGES §4) |
| `VLLM_ROCM_USE_AITER=0`, `TORCH_BLAS_PREFER_HIPBLASLT=0`, `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`, `PYTORCH_TUNABLEOP_ENABLED=0` | | gfx1030 kernel routing |
| `VLLM_RDNA_DENSE_INT8=1`, `VLLM_RDNA_AR=1` | | this fork's decode kernels / all-reduce |

Plus the CLI the script builds: `--dtype float16 --tensor-parallel-size 4
--enable-expert-parallel --max-num-seqs 4 --max-num-batched-tokens 2048` and, per knobs,
`MTP=`, `VISION=`, `MAXLEN=`.

**Context-size interaction you will hit with MTP:** the KV pool must hold at least one
max-length request. On 32 GB cards at `GPUUTIL=0.90`, roughly: MTP=0 text-only ≈ 530k tokens
of pool; vision costs ~130k; **MTP=3 costs ~150k more** (head weights + draft graphs). So
`MTP=3` + vision does **not** fit `MAXLEN=196608` — use 131072 or 65536. If you see a startup
error about max seq len vs KV cache size, this is it, not a PLE problem.

### 1.4 What a healthy first boot looks like (in order)

```
Bound IPC address ipc:///tmp/…; waiting for 4 GPU worker registration(s).
PLE quant table: group16_int4_fp16scale_lownibblefirst, 128 shards mmapped from …/ples_int4
GPU worker N registered (dp_rank=0, tp_rank=N, layers=['…layers.1.ple.ple_embedding'])   × 4
PLE sidecar prefaulted into the page cache: 32.0 GB in ~30 s        (SSD; minutes on slow disk)
… torch.compile (~12 min cold, ~1 s warm) … CUDA graph capture …
GPU KV cache size: NNN,NNN tokens
Application startup complete.
```

Cold boot ≈ 15 min (compile dominates); warm boot ≈ 3–4 min. There should be **zero** lines
matching `Duplicate PLE`, `PLE lookup for launch`, or `did not complete launch` — at HEAD we
have run full suites at MTP=0 and MTP=3 with none.

---

## 2. Confirm your tree before diagnosing

`git log --oneline -1` — the PLE handshake was **rewritten on 2026-08-30**. Trees before
`ce9dd91cc` have a different (broken-on-ROCm) synchronization with different failure modes.
Be at `f6cf11a1f` or later before spending time on anything below. Old-tree signatures:

| message | meaning | fix |
|---|---|---|
| `'_OpNamespace' 'vllm' object has no attribute 'rdna_hc_mix'` | warm-boot bug, pre-`a3c5f6846` | update |
| `queue.Full` / `request queue stayed full for 300 s` | mid-rework trees, 2026-08-30 morning | update |
| `Duplicate PLE request … skipping` (constantly, with garbled output) | the pre-rework race, surfaced at MTP=0 | update |

---

## 3. Diagnostic tree — start from the exact message

**A. `TimeoutError: PLE offload worker did not become ready within 600.0s.` (at startup)**
→ You are not running through the serve script (it sets 3600), *and* the worker's first-touch
of the table was slow.
1. Check the log for `PLE quant table: … 128 shards mmapped from …`. **Absent** → the worker
   died earlier; scroll up for its traceback (bad `VLLM_PLE_QUANT_DIR` path, META mismatch,
   missing shards — count the 129 files).
2. Present, but readiness still timed out → storage. `ples_int4` on NVMe/SSD? On HDD or NFS
   the mmap + first reads can take many minutes: raise `VLLM_PLE_OFFLOAD_READY_TIMEOUT`, move
   the sidecar to local SSD.
3. `VLLM_PLE_QUANT_DIR` unset entirely → with this checkpoint the worker tries the bf16
   disk-offload path or fails to find table weights at all (shard 1 is not even in the index).
   Set it. This is the single most common cause.

**B. `Waiting for 4 GPU worker registration(s) ...` forever, then a timeout**
→ The *GPU* side died before registering; the PLE worker is the innocent bystander. Scroll up
for the first real error. Usual suspects: OOM during memory profiling with `VISION=1` and no
image-size cap (`Tried to allocate 64.00 GiB` — see TROUBLESHOOTING §5a; the serve script's
`--mm-processor-kwargs '{"max_pixels": 1638400}'` prevents it); `MAXLEN` too large for the KV
pool with MTP/vision (see §1.3); a ROCm/PyTorch mismatch crashing rank init.

**C. `PLE lookup for launch N has taken >5 s (worker slow or stuck?)` (warning, occasional)**
→ Lookup latency spikes, almost always page faults on the sidecar.
1. Did the prefault line appear (`PLE sidecar prefaulted into the page cache: 32.0 GB in …`)?
   If it reports minutes rather than ~30 s, your storage is the bottleneck.
2. Free RAM < ~35 GB → the page cache cannot hold the sidecar and rows fault to disk forever
   (~0.4 ms each on SATA, worse on HDD). Check `free -g`; anything else large running?
3. `PLE_OFFLOAD_PREFAULT=0` set somewhere → unset it.
Occasional warnings in the first minute on a cold cache are benign; sustained ones are always
storage or memory pressure.

**D. `RuntimeError: PLE offload worker did not complete launch N within 600 s (done=M)`**
→ The worker stopped responding mid-service.
1. Grep the log for the worker's own traceback (`PleOffloadWorker pid=…`) and for
   `Parent exited, shutting down` — if the worker crashed, the *first* error above it is the
   cause; report that line, not the timeout.
2. `done=M` far behind `N` and climbing slowly → extreme storage latency, same as C.
3. `done=M` frozen with no worker error → that would be a genuine protocol hang. We cannot
   reproduce one at HEAD (full suites at MTP=0 **and** MTP=3, zero warnings); open an issue
   with: commit hash, the full serve log, `META.json`, storage type, `free -g`, and whether
   the serve script was used verbatim.

**E. It only happens "with MTP"**
→ There is nothing MTP-specific in the PLE path (the draft model runs with PLE off; one lookup
per engine step in both modes — verified at HEAD with a byte-identical-output determinism test
under speculative decoding). When "MTP broke it", it is almost always one of:
1. **KV pool**: MTP shrinks it ~150k tokens; with a big `MAXLEN` the boot fails or the engine
   thrashes — see §1.3. (This produces startup errors that get pasted next to PLE lines.)
2. Longer engine steps at low acceptance make an existing storage-latency problem (C) cross
   the 5 s warning threshold more visibly.
3. An old tree (§2), where MTP=0 vs MTP=3 really did behave differently.
Also worth knowing: on the fixed protocol MTP=3 measured **60–72 t/s vs 62–65 at MTP=0** on
our hardware — at ~70 % draft acceptance it is a wash. Unless your workload's acceptance is
high, `MTP=0` is the better trade and sidesteps this entire branch.

Also, a subtlety that matters when MTP=0 works and MTP=3 times out: the host-side PLE wait
sits at the head of every engine step, and step N's lookup request is only *sent* once all of
step N-1's GPU work — including the drafter's extra forwards — has finished. **Any GPU-side
stall in the MTP drafter is therefore reported as a PLE timeout**, with `done` frozen one
behind the launch number. Discriminate in one step: during the stall, check GPU busy
(`rocm-smi`) — pegged high means the drafter is stuck on your hardware and PLE is the
messenger, not the fault; near zero means the worker/storage side really is stuck. Report
busy%, `free -g` in both MTP configurations, the exact `done=M`, and whether it happens at
warmup or mid-serving.

**E2. Boots clean, then the FIRST request stalls: one `>5 s` warning at a small launch
number, and the engine dies waiting for `sample_tokens` (graphs + MTP only)**

→ Decode the launch number first: dummy and capture forwards do **not** increment it, so
launch N is the Nth *real* PLE-bearing forward. For a single short request, launch 1 is the
prefill chunk and launch 8 ≈ the 7th decode step — several graph replays already succeeded
before one wedged. That pattern (runs briefly, then a replay freezes) is almost never the PLE
worker: the host-side PLE wait sits at the head of every step, so it is simply the first thing
to *report* a GPU-side wedge (branch E). At MTP=3 the wedging step replays the drafter graphs
and the M=4 verify graph, whose small collectives run the one-shot all-reduce **inside the
replay** — and the boot self-test probes its collectives *outside* graph capture, so a board
whose P2P wedges only under replay cadence passes the self-test and still freezes here.

Note also: the engine's execute-model timeout (default 300 s) kills the workers before the PLE
wait's own 600 s error can print `done=M`. For one diagnostic run,
`export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=900` so the PLE side gets to report first.

Run these in order and report each result:

1. **During the stall**: `rocm-smi` — GPU busy pegged ⇒ a wedged replay (PLE is the
   messenger); ~0 % ⇒ genuinely the worker/storage side (branches C/D).
2. **`VLLM_RDNA_AR=0`**, same command (RCCL takes the small collectives): freeze gone ⇒ your
   board's P2P-under-replay is the story (branch G) — stay on RCCL.
3. Axis bisection: `EAGER=1` with MTP=3 (graphs off); graphs with `MTP=0`; then `MTP=1`.
4. **`PLE_OFFLOAD_DEBUG_TRACE=/tmp/ple-trace.log`** on the serve: after the freeze, does the
   trace show the worker *received and answered* the stalled launch? Answered ⇒ GPU side for
   certain; never received ⇒ transport — report that line.
5. `tools/rdna2/system-report.sh --log <serve log>` and send the report.

For contrast: the `>5 s` warning alone is a stall detector, not a failure. On our hardware at
HEAD, a vision request's first-exposure Triton JIT compiles produced warnings on launches
23–28 (at MTP=0, graphs uninvolved) that all self-resolved, `done=` advancing, serving
uninterrupted. Frozen `done=`, no recovery, engine timeout ⇒ this branch.

**F. None of the above messages — the server just stalls or dies**
→ Platform, not PLE: check the kernel log (`journalctl -k`) for `amdgpu`/queue-eviction/PCIe
events, confirm the T41 stability kernel line and env from CHANGES §4, and never leave a
GPU-side stream wait pending on gfx1030 (CHANGES §8a tells that story).

**G. Suspected board/fabric differences (different mainboard, ACS, chipset-routed or
cross-socket slots)** → the one-shot all-reduce needs healthy GPU↔GPU P2P posted writes.
Since 2026-08-31 it **self-tests at boot** (three verified collectives + a latency bound) and
falls back to RCCL group-wide with `rdna_ar: disabled -- boot self-test failed` if your
board's P2P is broken or slow — so on current trees a bad fabric downgrades performance
instead of corrupting output. Two manual probes: boot once with `VLLM_RDNA_AR=0` (forces
RCCL for the small collectives — if your stalls vanish, your board's P2P is the story), and
check the boot log for the `rdna_ar: one-shot all-reduce active` vs `disabled` lines.
