#!/usr/bin/env python3
"""Decode-throughput bench: streams one completion, measures tokens/s between first and
last chunk (excludes prefill), reports completion tokens from the server's usage block.
Usage: bench.py [runs] [max_tokens] [prompt_words]"""
import json, sys, time, urllib.request

runs = int(sys.argv[1]) if len(sys.argv) > 1 else 3
max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 256
words = int(sys.argv[3]) if len(sys.argv) > 3 else 40
prompt = ("Write a detailed, multi-paragraph technical explanation of how a hash table "
          "handles collisions, covering open addressing and chaining, with examples. " * 8)[: words * 6]

def one(seed):
    body = {"model": "qwen38-flash-next", "prompt": prompt + f" (variant {seed})",
            "max_tokens": max_tokens, "temperature": 0.0, "stream": True,
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request("http://localhost:8000/v1/completions",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time(); t_first = None; t_last = None; n_chunks = 0; usage = None; text = ""
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            d = json.loads(payload)
            if d.get("usage"):
                usage = d["usage"]
            ch = d.get("choices") or []
            if ch and ch[0].get("text"):
                text += ch[0]["text"]
                n_chunks += 1
                t_last = time.time()
                if t_first is None:
                    t_first = t_last
    ct = usage["completion_tokens"] if usage else n_chunks
    decode_s = (t_last - t_first) if (t_first and t_last and t_last > t_first) else float("nan")
    return ct, decode_s, t_first - t0, n_chunks, text[:60].replace("\n", " ")

for i in range(runs):
    ct, ds, ttft, nc, head = one(i)
    print(f"run {i}: {ct} tok, decode {ct - 1}/{ds:.2f}s = {(ct - 1) / ds:.2f} t/s  "
          f"(ttft {ttft:.2f}s, {nc} chunks, tok/chunk {ct / max(nc, 1):.2f})  '{head}'", flush=True)
