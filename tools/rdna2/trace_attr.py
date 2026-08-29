#!/usr/bin/env python3
"""Eager-mode trace attribution: for each GPU kernel, walk the CPU op (External id) and the
python_function stack that encloses it, and aggregate kernel count/time per innermost model
source line (file:line) so fusion targets are ranked by launches per step."""
import gzip, json, sys, collections, bisect

path = sys.argv[1]; steps = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
top = int(sys.argv[3]) if len(sys.argv) > 3 else 40
tr = json.load(gzip.open(path, "rt"))["traceEvents"]
kern = [e for e in tr if e.get("cat") == "kernel"]
cpu_by_ext = collections.defaultdict(list)
for e in tr:
    if e.get("cat") == "cpu_op" and "External id" in e.get("args", {}):
        cpu_by_ext[e["args"]["External id"]].append(e)
# python frames sorted by start; for a given ts find the innermost frame from vllm model code
pyf = [e for e in tr if e.get("cat") == "python_function" and "dur" in e]
pyf.sort(key=lambda e: e["ts"])
starts = [e["ts"] for e in pyf]
def frames_at(ts):
    i = bisect.bisect_right(starts, ts)
    out = []
    for e in pyf[max(0, i - 400):i]:
        if e["ts"] <= ts <= e["ts"] + e["dur"]:
            out.append(e)
    return out
def model_frame(frs):
    best = None
    for e in frs:
        n = e["name"]
        if "/src/vllm/" in n and ("models/" in n or "layers/" in n or "mamba/" in n or "distributed/" in n or "v1/" in n):
            if best is None or e["dur"] < best["dur"]:
                best = e
    return best["name"].replace("/src/vllm/", "") if best else "?"
agg = collections.defaultdict(lambda: [0.0, 0, collections.Counter()])
for e in kern:
    ext = e.get("args", {}).get("External id")
    ops = cpu_by_ext.get(ext, [])
    op = min(ops, key=lambda o: o["dur"])["name"] if ops else "?"
    # use the launching cpu op's timestamp for the python stack
    ts = ops[0]["ts"] if ops else e["ts"]
    site = model_frame(frames_at(ts))
    a = agg[site]; a[0] += e["dur"]; a[1] += 1; a[2][(op[:28], e["name"][:30])] += 1
print(f"kernels {len(kern)} -> {len(kern)/steps:.0f}/step")
for site, (d, c, ops) in sorted(agg.items(), key=lambda x: -x[1][1])[:top]:
    print(f"{c/steps:7.1f}/step {d/steps/1e3:6.2f} ms/step  {site[-85:]}")
    for (op, kn), n in ops.most_common(3):
        print(f"            {n/steps:6.1f}  {op:28s} {kn}")
