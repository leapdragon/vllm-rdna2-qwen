#!/usr/bin/env python3
"""Greedy sanity checks against a running server (chat completions, thinking off, T=0).
Exit code 0 only if every expected answer is found. Usage: validate.py [base_url]"""
import json, sys, urllib.request

base = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
checks = [
    ("What is 17 * 23? Answer with just the number.", "391"),
    ("What is the capital of Australia? One word.", "Canberra"),
    ("List the first five prime numbers, comma-separated.", "2, 3, 5, 7, 11"),
    ("Explain in two sentences why the sky is blue.", "scatter"),
]
ok = True
for q, want in checks:
    body = {"model": "qwen38-flash-next", "messages": [{"role": "user", "content": q}],
            "max_tokens": 120, "temperature": 0.0, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"{base}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        text = json.loads(r.read())["choices"][0]["message"]["content"].strip().replace("\n", " ")
    hit = want.lower() in text.lower()
    ok &= hit
    print(f"[{'ok' if hit else 'FAIL'}] {q}\n      -> {text[:140]}")
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
