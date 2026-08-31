#!/usr/bin/env python3
"""Python half of tools/rdna2/system-report.sh: introspection that needs the venv.

Run with the venv's python. Every section is best-effort: a failure prints an
"(unavailable: ...)" line and the report continues. Subcommands:

  packages                      versions of the load-bearing packages
  torch                         torch/HIP build, devices, arch list, properties
  vllm                          vllm version, extension .so files, embedded gfx targets, compile cache
  model MODEL_DIR PLE_DIR        artefact inventory, index cross-check, META, storage class, checksums
  probe [BASE]                  live-server probe (/health, /models, /metrics, tiny completion, vision)
  tests                         idle-GPU platform tests (peer-access matrix, P2P latency)
Options: --checksums (hash the big shards too), --out-json PATH (machine-readable copy)
"""
import hashlib, json, os, re, sys, time

ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
FLAGS = {a for a in sys.argv[1:] if a.startswith("--")}
CHECKSUMS = "--checksums" in FLAGS
REPORT: dict = {}


def p(*a):
    print(*a, flush=True)


def guard(fn):
    def w(*args):
        try:
            return fn(*args)
        except Exception as e:  # noqa: BLE001
            p(f"(unavailable: {type(e).__name__}: {e})")
    return w


# --------------------------------------------------------------------------- packages
@guard
def packages():
    from importlib import metadata
    names = ["vllm", "torch", "triton", "torchvision", "transformers", "tokenizers", "safetensors",
             "numpy", "msgspec", "pyzmq", "pillow", "aiter", "flash_attn", "flash-attn",
             "cuda-bindings", "cuda-python", "cuda_bindings", "amdsmi", "setuptools", "cmake", "ninja"]
    out = {}
    for n in names:
        try:
            out[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            out[n] = None
    for k, v in out.items():
        p(f"  {k:16s} {v if v else '-'}")
    if out.get("cuda-bindings") or out.get("cuda-python") or out.get("cuda_bindings"):
        p("  NOTE: cuda-bindings/cuda-python present -- on ROCm the PLE offload must select the HIP shim by platform (fixed in this fork; matters on older trees)")
    if not out.get("pillow"):
        p("  NOTE: pillow missing -> vision requests will fail")
    p(f"  python           {sys.version.split()[0]}  ({sys.executable})")
    REPORT["packages"] = out


# --------------------------------------------------------------------------- torch
@guard
def torch_info():
    import torch
    info = {"version": torch.__version__, "hip": torch.version.hip, "cuda": torch.version.cuda,
            "arch_list": None, "devices": []}
    p(f"  torch {torch.__version__}  hip={torch.version.hip}  cuda={torch.version.cuda}")
    try:
        info["arch_list"] = torch.cuda.get_arch_list()
        p(f"  compiled arch list: {info['arch_list']}")
    except Exception as e:  # noqa: BLE001
        p(f"  compiled arch list: (unavailable: {e})")
    n = torch.cuda.device_count()
    p(f"  visible devices: {n}  (ROCR_VISIBLE_DEVICES={os.getenv('ROCR_VISIBLE_DEVICES')}, HIP_VISIBLE_DEVICES={os.getenv('HIP_VISIBLE_DEVICES')}, HSA_OVERRIDE_GFX_VERSION={os.getenv('HSA_OVERRIDE_GFX_VERSION')})")
    for i in range(n):
        pr = torch.cuda.get_device_properties(i)
        arch = getattr(pr, "gcnArchName", "?")
        d = {"index": i, "name": pr.name, "arch": arch, "total_mem_gb": round(pr.total_memory / 1e9, 1),
             "multi_processor_count": pr.multi_processor_count}
        info["devices"].append(d)
        p(f"  [{i}] {pr.name}  arch={arch}  vram={d['total_mem_gb']} GB  CUs={pr.multi_processor_count}")
        if not str(arch).startswith("gfx1030"):
            p(f"      NOTE: this fork's kernels are built and tested for gfx1030; {arch} is untested")
    REPORT["torch"] = info


# --------------------------------------------------------------------------- vllm
@guard
def vllm_info():
    import vllm
    root = os.path.dirname(vllm.__file__)
    p(f"  vllm {vllm.__version__}  at {root}")
    info = {"version": vllm.__version__, "root": root, "ext": {}}
    for so in sorted(os.listdir(root)):
        if so.endswith(".so"):
            path = os.path.join(root, so)
            st = os.stat(path)
            gfx = set()
            with open(path, "rb") as f:
                data = f.read()
            gfx = sorted(set(m.decode() for m in re.findall(rb"gfx[0-9a-f]{3,4}", data)))
            info["ext"][so] = {"size_mb": round(st.st_size / 1e6, 1), "mtime": time.strftime("%Y-%m-%d %H:%M", time.localtime(st.st_mtime)), "gfx": gfx}
            p(f"  {so:32s} {info['ext'][so]['size_mb']:8.1f} MB  {info['ext'][so]['mtime']}  gfx targets: {','.join(gfx) or '?'}")
    cache = os.path.expanduser(os.getenv("VLLM_CACHE_ROOT", "~/.cache/vllm")) + "/torch_compile_cache"
    if os.path.isdir(cache):
        entries = sorted(os.listdir(cache))
        total = 0
        for root_, _, files in os.walk(cache):
            for fn in files:
                try:
                    total += os.path.getsize(os.path.join(root_, fn))
                except OSError:
                    pass
        p(f"  torch.compile cache: {len(entries)} entries, {total / 1e9:.1f} GB at {cache}: {entries[:8]}")
        info["compile_cache"] = {"entries": entries, "gb": round(total / 1e9, 1)}
    else:
        p(f"  torch.compile cache: none at {cache}")
    for k in ("VLLM_DISABLE_COMPILE_CACHE", "VLLM_PLE_CPU_OFFLOAD", "VLLM_PLE_QUANT_DIR", "VLLM_PLE_OFFLOAD_READY_TIMEOUT",
              "VLLM_RDNA_AR", "VLLM_RDNA_DENSE_INT8", "VLLM_ROCM_USE_AITER", "VLLM_USE_V2_MODEL_RUNNER"):
        p(f"  env {k}={os.getenv(k)}")
    REPORT["vllm"] = info


# --------------------------------------------------------------------------- model
EXPECTED_BACKBONE = ["model-00002-of-00005.safetensors", "model-00003-of-00005.safetensors",
                     "model-00004-of-00005.safetensors", "model-00005-of-00005.safetensors",
                     "model_mtp.safetensors", "model.safetensors.index.json", "config.json",
                     "generation_config.json", "chat_template.jinja", "tokenizer.json",
                     "tokenizer_config.json", "preprocessor_config.json", "processor_config.json",
                     "video_preprocessor_config.json"]
SMALL = {"config.json", "generation_config.json", "chat_template.jinja", "tokenizer_config.json",
         "model.safetensors.index.json", "preprocessor_config.json", "processor_config.json", "META.json"}


def sha256(path, limit_mb=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 24)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def storage_class(path):
    """filesystem type, mount options, device, rotational flag, free space for the path."""
    real = os.path.realpath(path)
    best = None
    with open("/proc/mounts") as f:
        for line in f:
            dev, mnt, fst, opts = line.split()[:4]
            if real == mnt or real.startswith(mnt.rstrip("/") + "/"):
                if best is None or len(mnt) > len(best[1]):
                    best = (dev, mnt, fst, opts)
    out = {"realpath": real}
    if best:
        dev, mnt, fst, opts = best
        out.update({"device": dev, "mount": mnt, "fstype": fst, "options": opts})
        rot = None
        try:
            bd = os.path.basename(os.path.realpath(dev))
            base = re.sub(r"p?\d+$", "", bd) if not bd.startswith("dm-") else bd
            with open(f"/sys/block/{base}/queue/rotational") as f:
                rot = f.read().strip()
        except OSError:
            pass
        out["rotational"] = rot
    try:
        st = os.statvfs(real)
        out["free_gb"] = round(st.f_bavail * st.f_frsize / 1e9, 1)
    except OSError:
        pass
    return out


@guard
def model_info(model_dir, ple_dir):
    for label, d in (("MODEL", model_dir), ("PLE_INT4", ple_dir)):
        p(f"  {label} = {d}")
        if not d or not os.path.isdir(d):
            p("    (directory missing)")
            continue
        sc = storage_class(d)
        flag = ""
        if sc.get("rotational") == "1":
            flag = "  <-- ROTATIONAL DISK: expect slow lookups / page-fault stalls"
        if sc.get("fstype") in ("nfs", "nfs4", "cifs", "fuse", "fuseblk", "sshfs", "9p"):
            flag += f"  <-- {sc.get('fstype')}: network/FUSE filesystem under the model"
        p(f"    realpath {sc.get('realpath')}")
        p(f"    fs {sc.get('fstype')} on {sc.get('device')} mounted {sc.get('mount')} rotational={sc.get('rotational')} free={sc.get('free_gb')} GB{flag}")
        p(f"    mount options: {sc.get('options')}")
        files = sorted(os.listdir(d))
        total = sum(os.path.getsize(os.path.join(d, f)) for f in files if os.path.isfile(os.path.join(d, f)))
        p(f"    {len(files)} files, {total / 1e9:.1f} GB")
        if label == "MODEL":
            for f in EXPECTED_BACKBONE:
                if f not in files:
                    p(f"    MISSING expected file: {f}" + ("  <-- boot will fail (indexed)" if f == "model_mtp.safetensors" else ""))
            if "model-00001-of-00005.safetensors" in files:
                p("    NOTE: shard 1 (102 GB bf16 n-gram table) is present -- not referenced by the index; wasted disk unless you run the bf16 disk-offload path")
            extra = [f for f in files if f not in EXPECTED_BACKBONE and f not in ("awq_run_info.json", "recipe.yaml", "README.md", "model-00001-of-00005.safetensors") and not f.startswith(".")]
            if extra:
                p(f"    unexpected files: {extra}")
            idx = os.path.join(d, "model.safetensors.index.json")
            if os.path.isfile(idx):
                wm = json.load(open(idx))["weight_map"]
                refd = sorted(set(wm.values()))
                missing = [f for f in refd if f not in files]
                p(f"    index references {len(wm)} tensors in {refd}")
                p(f"    index-referenced files missing: {missing or 'none'}")
                nvis = sum(1 for k in wm if k.startswith("model.visual"))
                nmtp = sum(1 for k in wm if k.startswith("mtp."))
                p(f"    visual tensors: {nvis}   mtp tensors: {nmtp}")
            for f in files:
                path = os.path.join(d, f)
                if f in SMALL or (CHECKSUMS and f.endswith(".safetensors")):
                    p(f"    sha256 {f}: {sha256(path)}  ({os.path.getsize(path)} B)")
        else:
            meta = os.path.join(d, "META.json")
            if os.path.isfile(meta):
                m = json.load(open(meta))
                p(f"    META.json: {json.dumps({k: v for k, v in m.items() if k != 'shards_list'})}")
                if m.get("layout") != "group16_int4_fp16scale_lownibblefirst" or m.get("shards") != 128 or m.get("rows") != 320001536 or m.get("width") != 160:
                    p("    NOTE: META differs from the tested sidecar build (group16_int4_fp16scale_lownibblefirst / 128 / 320001536 / 160)")
                shards = [f for f in files if f.startswith("shard_") and f.endswith(".safetensors")]
                p(f"    shards present: {len(shards)} (expected 128){'  <-- INCOMPLETE' if len(shards) != 128 else ''}")
                p(f"    sha256 META.json: {sha256(meta)}")
            else:
                p("    META.json MISSING -- this is not a valid sidecar directory")
    # RAM headroom vs sidecar
    try:
        mem = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":")
                mem[k] = int(v.split()[0]) // 1024
        p(f"  RAM: total {mem['MemTotal']//1024} GB, available {mem['MemAvailable']//1024} GB, cached {mem.get('Cached',0)//1024} GB  (sidecar wants ~30 GB of page cache to stay resident)")
    except Exception as e:  # noqa: BLE001
        p(f"  RAM: (unavailable: {e})")


# --------------------------------------------------------------------------- probe
@guard
def probe(base="http://localhost:8000"):
    import urllib.request
    def get(path, timeout=5):
        with urllib.request.urlopen(base + path, timeout=timeout) as r:
            return r.status, r.read().decode()
    try:
        st, _ = get("/health", 3)
        p(f"  /health -> {st}")
    except Exception as e:  # noqa: BLE001
        p(f"  /health -> no server ({type(e).__name__}); skipping live probe")
        return
    try:
        p(f"  /version -> {get('/version')[1][:200]}")
    except Exception as e:  # noqa: BLE001
        p(f"  /version -> ({e})")
    try:
        models = json.loads(get("/v1/models")[1])
        p(f"  /v1/models -> {[m.get('id') for m in models.get('data', [])]}  max_model_len={[m.get('max_model_len') for m in models.get('data', [])]}")
    except Exception as e:  # noqa: BLE001
        p(f"  /v1/models -> ({e})")
    try:
        metrics = get("/metrics")[1]
        keep = ("num_requests_running", "num_requests_waiting{", "kv_cache_usage_perc", "prompt_tokens_total",
                "generation_tokens_total", "spec_decode_num_draft_tokens_total", "spec_decode_num_accepted_tokens_total",
                "prefix_cache_queries_total", "prefix_cache_hits_total")
        for line in metrics.splitlines():
            if line.startswith("vllm:") and any(k in line for k in keep):
                p(f"  {line}")
    except Exception as e:  # noqa: BLE001
        p(f"  /metrics -> ({e})")
    # tiny greedy completion with timing
    try:
        body = {"model": None, "messages": [{"role": "user", "content": "What is 17 * 23? Answer with just the number."}],
                "max_tokens": 8, "temperature": 0.0, "chat_template_kwargs": {"enable_thinking": False}, "stream": True}
        body["model"] = json.loads(get("/v1/models")[1])["data"][0]["id"]
        req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        t0 = time.time(); first = None; text = ""; n = 0
        with urllib.request.urlopen(req, timeout=300) as r:
            for line in r:
                line = line.decode().strip()
                if not line.startswith("data:") or line.endswith("[DONE]"):
                    continue
                delta = json.loads(line[5:])["choices"][0].get("delta", {}).get("content") or ""
                if delta and first is None:
                    first = time.time() - t0
                text += delta; n += 1
        dt = time.time() - t0
        p(f"  tiny completion: {text.strip()!r}  ttft {first or 0:.2f}s, {n} chunks in {dt:.2f}s  {'OK' if '391' in text else '<-- WRONG ANSWER'}")
    except Exception as e:  # noqa: BLE001
        p(f"  tiny completion failed: {type(e).__name__}: {str(e)[:200]}")
    # vision probe if enabled
    try:
        from PIL import Image
        import base64, io
        im = Image.new("RGB", (128, 128), (220, 20, 20)); buf = io.BytesIO(); im.save(buf, format="PNG")
        url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        body = {"model": json.loads(get("/v1/models")[1])["data"][0]["id"],
                "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": "What colour is this image? One word."}]}],
                "max_tokens": 8, "temperature": 0.0, "chat_template_kwargs": {"enable_thinking": False}}
        req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=300) as r:
            ans = json.loads(r.read())["choices"][0]["message"]["content"]
        p(f"  vision probe (red square): {ans.strip()!r} in {time.time() - t0:.2f}s  {'OK' if 'red' in ans.lower() else '<-- unexpected'}")
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        if "400" in msg or "language" in msg.lower() or "multimodal" in msg.lower() or "image" in msg.lower():
            p(f"  vision probe: server refused images (vision off?): {msg[:160]}")
        else:
            p(f"  vision probe: skipped/failed: {type(e).__name__}: {msg[:160]}")


# --------------------------------------------------------------------------- tests
@guard
def tests():
    import torch
    n = torch.cuda.device_count()
    p(f"  peer-access matrix ({n} devices):")
    for i in range(n):
        row = []
        for j in range(n):
            row.append("-" if i == j else ("Y" if torch.cuda.can_device_access_peer(i, j) else "N"))
        p(f"    [{i}] {' '.join(row)}")
    if n >= 2:
        p("  P2P copy probe (64 KB and 8 MB, device i -> j, median of 20):")
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                res = []
                for numel in (32768, 4 * 1024 * 1024):
                    src = torch.ones(numel, dtype=torch.float16, device=f"cuda:{i}")
                    dst = torch.empty(numel, dtype=torch.float16, device=f"cuda:{j}")
                    torch.cuda.synchronize(i); torch.cuda.synchronize(j)
                    ts = []
                    for _ in range(20):
                        t0 = time.perf_counter(); dst.copy_(src); torch.cuda.synchronize(j); ts.append(time.perf_counter() - t0)
                    ts.sort(); med = ts[10]
                    res.append(f"{numel*2/1e3:.0f}KB {med*1e6:.0f}us" + (f" ({numel*2/med/1e9:.1f} GB/s)" if numel > 100000 else ""))
                p(f"    {i}->{j}: " + ", ".join(res))


if __name__ == "__main__":
    cmd = ARGS[0] if ARGS else "all"
    if cmd == "packages":
        packages()
    elif cmd == "torch":
        torch_info()
    elif cmd == "vllm":
        vllm_info()
    elif cmd == "model":
        model_info(ARGS[1] if len(ARGS) > 1 else "", ARGS[2] if len(ARGS) > 2 else "")
    elif cmd == "probe":
        probe(ARGS[1] if len(ARGS) > 1 else "http://localhost:8000")
    elif cmd == "tests":
        tests()
    else:
        p(f"unknown subcommand {cmd}")
