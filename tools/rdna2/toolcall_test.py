#!/usr/bin/env python3
"""Check OpenAI-style tool calling with tool_choice "auto" (what agentic clients such as
Kilocode/Cline/Roo send). Exit 0 only if the model returns a well-formed tool call.
Usage: toolcall_test.py [base_url]"""
import json, sys, urllib.request

base = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
}]
body = {
    "model": "qwen38-flash-next",
    "messages": [{"role": "user", "content": "What's the weather in Vancouver right now? Use the tool."}],
    "tools": tools,
    "tool_choice": "auto",
    "temperature": 0.0,
    "max_tokens": 300,
}
req = urllib.request.Request(f"{base}/v1/chat/completions", data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=600) as r:
        resp = json.loads(r.read())
except urllib.error.HTTPError as e:
    print("HTTP", e.code, e.read().decode()[:300]); sys.exit(1)
msg = resp["choices"][0]["message"]
calls = msg.get("tool_calls") or []
print("finish_reason:", resp["choices"][0].get("finish_reason"))
print("reasoning_content:", (msg.get("reasoning_content") or "")[:120].replace("\n", " "))
print("content:", (msg.get("content") or "")[:120].replace("\n", " "))
for c in calls:
    print("tool_call:", c["function"]["name"], c["function"]["arguments"])
ok = bool(calls) and calls[0]["function"]["name"] == "get_weather" and "Vancouver" in calls[0]["function"]["arguments"]
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
