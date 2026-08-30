#!/usr/bin/env python3
"""PLE (n-gram sidecar) consistency test for the CPU-offload path.

Background (2026-08-30): on ROCm, hipStreamWaitValue32 is not recorded into HIP graphs, so
CUDA-graph decode steps did not wait for the CPU n-gram lookup and consumed a stale buffer.
Symptoms: garbled long generations (dropped / doubled characters), "Duplicate PLE request"
in the serve log, and 99 % GPU busy at idle. This test makes that visible without a
reference model:

  1. determinism  - two identical greedy generations of a long, structured document must
                    be byte-identical (a stale lookup depends on timing, so runs diverge)
  2. garble scan  - doubled tags / doubled words / unbalanced brackets in the output
  3. log check    - the serve log must not gain "Duplicate PLE request" lines
  4. idle check   - GPU busy must return to ~0 % a few seconds after the run
  5. (optional)   - run once more with PLE_OFFLOAD_DEBUG_DELAY_MS set on the server: the
                    output must stay identical while decode slows down (proves the wait)

Usage: ple_consistency_test.py [--base URL] [--tokens N] [--log serve.log] [--expect TEXTFILE]
Exit 0 only if every check passes. --save FILE stores run A for a later --expect comparison.
"""
import argparse, glob, json, os, re, sys, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="http://localhost:8000")
ap.add_argument("--tokens", type=int, default=1200)
ap.add_argument("--log", default=None, help="serve log to scan (default: newest logs/host-serve-*.log)")
ap.add_argument("--save", default=None)
ap.add_argument("--expect", default=None, help="text file from --save to compare against")
ap.add_argument("--no-idle-check", action="store_true")
ap.add_argument("--trace", default=None,
                help="worker trace file (server started with PLE_OFFLOAD_DEBUG_TRACE=<file>): "
                     "compares the n-gram lookups of runs A and B")
args = ap.parse_args()

PROMPT = (
    "Write a complete, valid HTML5 page for a small bakery called 'Sunrise Loaves'. Requirements: "
    "<!DOCTYPE html>, <html lang=\"en\">, a <head> with <meta charset=\"utf-8\">, a viewport meta tag, "
    "a <title>, an inline <style> block with at least eight CSS rules; a <body> with a <header>, a <nav> "
    "with five links, a <main> containing three <section>s each with an <h2> and two paragraphs, an "
    "ordered list of ten menu items with prices, a <table> of opening hours for all seven days, a "
    "<form> with four labelled inputs and a submit button, and a <footer>. After the HTML, add a fenced "
    "JSON block describing the same menu (name, price, allergens) for all ten items. Output only the "
    "code, no commentary."
)

def generate(max_tokens):
    body = {"model": "qwen38-flash-next", "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": max_tokens, "temperature": 0.0, "seed": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"{args.base}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.loads(r.read())
    dt = time.time() - t0
    text = d["choices"][0]["message"]["content"] or ""
    n = d["usage"]["completion_tokens"]
    return text, n, dt

def garble_report(text):
    issues = []
    for pat, name in [(r">>", "doubled '>'"), (r"<<(?!!)", "doubled '<'"),
                      (r"\b(\w{4,})\1\b", "doubled word fragment (e.g. charsetset)"),
                      (r"<(\w+)(\s[^<>]*)?>\s*<\1(\s[^<>]*)?>", "immediately repeated open tag")]:
        m = re.findall(pat, text)
        if name.startswith("immediately"):   # consecutive void elements (<meta>, <link>, <br>...) are normal HTML
            m = [x for x in m if x[0].lower() not in ("meta", "link", "br", "hr", "input", "img", "td", "th", "li", "option")]
        if m:
            issues.append(f"{name} x{len(m)}")
    opens, closes = text.count("<"), text.count(">")
    if abs(opens - closes) > 2:
        issues.append(f"unbalanced angle brackets ({opens} '<' vs {closes} '>')")
    return issues

def dup_count(path):
    try:
        with open(path, errors="ignore") as f:
            return sum(1 for line in f if "Duplicate PLE request" in line)
    except OSError:
        return -1

def gpu_busy():
    vals = []
    for p in sorted(glob.glob("/sys/class/drm/card?/device/gpu_busy_percent")):
        dev = open(os.path.join(os.path.dirname(p), "device")).read().strip()
        if dev == "0x73a1":  # the V620s
            vals.append(int(open(p).read()))
    return vals

log = args.log or (sorted(glob.glob(os.path.expanduser("~/repos/vllm-rdna2/logs/host-serve-*.log")),
                          key=os.path.getmtime) or [None])[-1]
dups0 = dup_count(log) if log else -1
ok = True
def check(name, cond, detail=""):
    global ok
    print(f"{'PASS' if cond else 'FAIL'} {name}{(' - ' + detail) if detail else ''}")
    ok = ok and cond

print(f"log: {log} (duplicate-PLE lines before: {dups0})")
# warm-up: make the prompt prefix-cache resident so runs A and B see identical cache state
# (a cache miss vs hit changes prefill numerics enough to flip a near-tie token later on)
generate(16)
def trace_lines():
    try:
        return open(args.trace).read().splitlines() if args.trace else []
    except OSError:
        return []
t_before = len(trace_lines())
a, na, dta = generate(args.tokens)
t_mid = len(trace_lines())
b, nb, dtb = generate(args.tokens)
t_after = len(trace_lines())
print(f"run A: {na} tokens in {dta:.1f}s ({na/dta:.1f} tok/s incl. prefill); run B: {nb} tokens in {dtb:.1f}s ({nb/dtb:.1f} tok/s)")
if a != b:
    i = next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    print(f"   first divergence at char {i}: A={a[max(0,i-30):i+30]!r}\n                              B={b[max(0,i-30):i+30]!r}")
check("determinism: two greedy runs identical", a == b)
if args.trace:
    lines = trace_lines()
    ta = [l.split()[1:] for l in lines[t_before:t_mid]]   # drop seq: compare (tokens, reqs, ids, result)
    tb = [l.split()[1:] for l in lines[t_mid:t_after]]
    n = min(len(ta), len(tb))
    first_diff = next((k for k in range(n) if ta[k] != tb[k]), None)
    print(f"   n-gram lookups: run A {len(ta)} requests, run B {len(tb)}; first differing lookup: {first_diff}")
    if first_diff is not None:
        print(f"      A[{first_diff}]={ta[first_diff]}  B[{first_diff}]={tb[first_diff]}")
    # a divergence of the *generated text* at token t must show up as differing input ids at
    # lookup t+1 at the latest; if the lookups differ earlier than the text does, the PLE path
    # itself (ids staged or result produced) is what diverged
    # the first differing lookup must differ in its *input ids* (the text had already
    # diverged upstream); equal ids with a different result = the PLE path itself diverged
    ple_ok = first_diff is None or ta[first_diff][2] != tb[first_diff][2]
    check("PLE path deterministic (lookups differ only after the input ids differ)", ple_ok,
          "identical lookups" if first_diff is None else
          ("ids differ first -> divergence originates upstream of the n-gram path" if ple_ok
           else "same ids, different result -> n-gram path is non-deterministic"))
issues = garble_report(a)
check("garble scan on run A", not issues, "; ".join(issues) if issues else f"{len(a)} chars clean")
if args.expect:
    ref = open(args.expect).read()
    check(f"matches reference {args.expect}", a == ref, f"{len(a)} vs {len(ref)} chars")
if log:
    dups1 = dup_count(log)
    check("no new 'Duplicate PLE request' lines", dups1 == dups0, f"{dups1 - dups0} new")
if not args.no_idle_check:
    time.sleep(4)
    busy = gpu_busy()
    check("GPUs idle after the run", all(v <= 5 for v in busy), f"busy% = {busy}")
if args.save:
    open(args.save, "w").write(a)
    print(f"saved run A to {args.save}")
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
