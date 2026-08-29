#!/usr/bin/env python3
"""Decode t/s at a target context length. Builds a unique pseudo-random prose prompt of ~N
tokens (no prefix-cache hits), streams a completion, reports decode t/s first->last chunk.
Usage: bench_ctx.py <approx_prompt_tokens> [max_tokens] [runs]"""
import json, random, sys, time, urllib.request

target = int(sys.argv[1]); max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 256
runs = int(sys.argv[3]) if len(sys.argv) > 3 else 1
words = ("system memory cache latency bandwidth kernel thread block warp lane register shared "
         "global constant texture stream event graph capture replay launch dispatch queue fence "
         "barrier atomic reduce scatter gather broadcast pipeline stage buffer tile matrix vector "
         "scalar tensor layer head token prompt decode prefill sample logits softmax attention "
         "expert router gate projection residual norm scale bias weight activation gradient").split()

def make_prompt(seed, n_tokens):
    r = random.Random(seed)
    n_words = int(n_tokens / 1.25)
    body = " ".join(r.choice(words) for _ in range(n_words))
    return f"[doc {seed}] " + body + "\n\nSummarize the text above in three sentences."

def one(seed):
    body = {"model": "qwen38-flash-next", "prompt": make_prompt(seed, target),
            "max_tokens": max_tokens, "temperature": 0.0, "stream": True,
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request("http://localhost:8000/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); t_first = t_last = None; n_chunks = 0; usage = None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:"): continue
            p = line[5:].strip()
            if p == "[DONE]": break
            d = json.loads(p)
            if d.get("usage"): usage = d["usage"]
            ch = d.get("choices") or []
            if ch and ch[0].get("text"):
                n_chunks += 1; t_last = time.time()
                if t_first is None: t_first = t_last
    pt = usage["prompt_tokens"] if usage else -1; ct = usage["completion_tokens"] if usage else n_chunks
    ds = t_last - t_first
    print(f"ctx {pt} tok: {ct} out, decode {(ct-1)/ds:.2f} t/s ({ct-1}/{ds:.2f}s), ttft {t_first-t0:.1f}s "
          f"= prefill {pt/(t_first-t0):.0f} t/s, {ct/max(n_chunks,1):.2f} tok/step", flush=True)

for i in range(runs):
    one(1000 * target + i)
