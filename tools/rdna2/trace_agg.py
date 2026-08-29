#!/usr/bin/env python3
"""Aggregate a torch-profiler chrome trace: GPU kernel time by name / family, GPU busy vs wall,
per-step wall, and (if shapes were recorded) the CPU op + input dims behind each GEMM kernel."""
import gzip, json, sys, re, collections

path = sys.argv[1]
top = int(sys.argv[2]) if len(sys.argv) > 2 else 30
with gzip.open(path, "rt") as f:
    tr = json.load(f)
ev = tr["traceEvents"]
kern = [e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and "dur" in e]
cpu = {}
for e in ev:
    if e.get("cat") in ("cpu_op",) and "args" in e and "External id" in e["args"]:
        cpu.setdefault(e["args"]["External id"], e)

def fam(n):
    if n.startswith("ncclDevKernel") or "nccl" in n.lower(): return "NCCL"
    if n.startswith("Cijk_"): return "Tensile-GEMM(fp16)"
    if "fused_moe_kernel" in n: return "MoE-triton"
    if n.startswith("execute_context"): return "graph-envelope"
    if "gated_delta" in n or "gdn" in n.lower() or "recurrent" in n: return "GDN"
    if "qsa" in n.lower() or "attention" in n.lower() or "_attn" in n.lower(): return "attention"
    if n.startswith("_hc_"): return "hyperconn"
    if "elementwise" in n or "vectorized" in n or n.startswith("triton_poi") or n.startswith("triton_per") or "index_" in n or "reduce_kernel" in n or "copy" in n.lower() or "fill" in n.lower(): return "glue"
    if "topkGating" in n or "moe_align" in n or "moe_sum" in n or "topk" in n.lower(): return "MoE-glue"
    if "wvSplit" in n or "LLMM" in n or "skinny" in n: return "skinny-GEMV"
    if e_cat(n): return e_cat(n)
    return "other"
def e_cat(n): return None

by_name = collections.defaultdict(lambda: [0.0, 0])
by_fam = collections.defaultdict(lambda: [0.0, 0])
t_min = min(e["ts"] for e in kern); t_max = max(e["ts"] + e["dur"] for e in kern)
env = [e for e in kern if e["name"].startswith("execute_context")]
real = [e for e in kern if not e["name"].startswith("execute_context")]
for e in real:
    by_name[e["name"]][0] += e["dur"]; by_name[e["name"]][1] += 1
    by_fam[fam(e["name"])][0] += e["dur"]; by_fam[fam(e["name"])][1] += 1
# GPU busy: union of kernel intervals (excluding envelopes)
iv = sorted((e["ts"], e["ts"] + e["dur"]) for e in real)
busy = 0.0; cs, ce = None, None
for s, e_ in iv:
    if cs is None: cs, ce = s, e_
    elif s > ce: busy += ce - cs; cs, ce = s, e_
    else: ce = max(ce, e_)
if cs is not None: busy += ce - cs
wall = t_max - t_min
tot = sum(v[0] for v in by_name.values())
print(f"wall {wall/1e3:.1f} ms  kernel-sum {tot/1e3:.1f} ms  GPU-busy(union) {busy/1e3:.1f} ms = {100*busy/wall:.1f}%  kernels {len(real)}  envelopes {len(env)}")
print("\n== by family ==")
for k, (d, c) in sorted(by_fam.items(), key=lambda x: -x[1][0]):
    print(f"  {k:22s} {d/1e3:8.1f} ms {100*d/tot:5.1f}%  n={c:6d}  avg {d/c:7.1f} us")
print(f"\n== top {top} kernels ==")
for k, (d, c) in sorted(by_name.items(), key=lambda x: -x[1][0])[:top]:
    print(f"  {d/1e3:8.1f} ms {100*d/tot:5.1f}%  n={c:6d} avg {d/c:8.1f} us  {k[:90]}")

# shapes behind GEMMs (needs record_shapes)
shp = collections.defaultdict(lambda: [0.0, 0])
have_shapes = False
for e in real:
    if not e["name"].startswith("Cijk_"): continue
    ext = e.get("args", {}).get("External id")
    c = cpu.get(ext)
    if c and "Input Dims" in c.get("args", {}):
        have_shapes = True
        shp[(c["name"], str(c["args"]["Input Dims"]))][0] += e["dur"]; shp[(c["name"], str(c["args"]["Input Dims"]))][1] += 1
if have_shapes:
    print("\n== Tensile GEMM time by (cpu op, input dims) ==")
    for (n, d_), (d, c) in sorted(shp.items(), key=lambda x: -x[1][0])[:25]:
        print(f"  {d/1e3:8.1f} ms n={c:5d} avg {d/c:7.1f} us  {n} {d_[:100]}")
else:
    print("\n(no Input Dims recorded for GEMM kernels)")

# per-step wall: use envelope events (graph replays) as step markers
if env:
    env.sort(key=lambda e: e["ts"])
    durs = [e["dur"] for e in env]
    names = collections.Counter(e["name"] for e in env)
    print(f"\n== graph envelopes: {len(env)}; distinct {len(names)} ==")
    for n, c in names.most_common(8):
        ds = [e["dur"] for e in env if e["name"] == n]
        print(f"  n={c:5d} avg {sum(ds)/len(ds)/1e3:7.2f} ms  {n}")
