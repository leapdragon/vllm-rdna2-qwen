#!/bin/bash
# soak_fabric_watch.sh — sustained generation while watching the kernel log for PCIe-fabric
# symptoms (2026-08-30). On this machine, fabric stress shows up as the SAS HBA's tape drive
# re-initialising (mpt2sas "detecting: handle") and, in the bad case, as amdgpu "device lost
# from bus". Runs back-to-back long greedy generations for DURATION seconds and reports:
# tokens generated, decode rate, kernel-log lines matching mpt2sas / amdgpu / pcie / AER during
# the run, and GPU busy % a few seconds after it ends.
#
# Usage: soak_fabric_watch.sh [duration_s=300] [max_tokens=1024] [base=http://localhost:8000]
set -uo pipefail
DUR="${1:-300}"; TOK="${2:-1024}"; BASE="${3:-http://localhost:8000}"
OUT="${OUT:-/tmp/soak-$(date +%H%M).log}"
KLOG="$OUT.kernel"
PROMPTS=(
  "Write a long, detailed technical design document (at least 1500 words) for a distributed key-value store: architecture, replication, failure handling, API, and a worked example with code."
  "Write a complete single-file HTML5 page with embedded CSS and JavaScript implementing a working todo app with local storage, filtering, and keyboard shortcuts. Output only code."
  "Explain, step by step and at length, how a modern CPU pipeline handles branch misprediction, with concrete cycle-by-cycle examples and a discussion of speculative execution hazards."
  "Write a JSON document describing 40 fictional products (name, sku, price, tags, description of two sentences each). Output only the JSON."
)
echo "soak: ${DUR}s, max_tokens=${TOK}, log=$OUT, kernel-log capture=$KLOG"
: > "$OUT"; : > "$KLOG"
journalctl -k -f -n 0 --no-pager 2>/dev/null | grep --line-buffered -iE "mpt2sas|amdgpu|pcieport|AER|device lost|reset" > "$KLOG" &
JPID=$!
t0=$(date +%s); n=0; total=0; gen_s=0
while [ $(( $(date +%s) - t0 )) -lt "$DUR" ]; do
  p="${PROMPTS[$((n % ${#PROMPTS[@]}))]}"
  body=$(python3 -c 'import json,sys;print(json.dumps({"model":"qwen38-flash-next","messages":[{"role":"user","content":sys.argv[1]}],"max_tokens":int(sys.argv[2]),"temperature":0.0,"chat_template_kwargs":{"enable_thinking":False}}))' "$p" "$TOK")
  s=$(date +%s.%N)
  resp=$(curl -s -m 900 "$BASE/v1/chat/completions" -H "Content-Type: application/json" -d "$body")
  e=$(date +%s.%N)
  ct=$(echo "$resp" | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d.get("usage",{}).get("completion_tokens",0))' 2>/dev/null || echo 0)
  dt=$(python3 -c "print(f'{$e-$s:.1f}')")
  total=$((total + ct)); gen_s=$(python3 -c "print($gen_s+$e-$s)")
  printf "%s  run %2d: %4s tokens in %ss  (%.0f tok/s)  kernel-log lines so far: %s\n" "$(date +%H:%M:%S)" "$n" "$ct" "$dt" "$(python3 -c "print($ct/max($e-$s,0.001))")" "$(wc -l < "$KLOG")" | tee -a "$OUT"
  n=$((n + 1))
  [ "$ct" = "0" ] && { echo "  empty response — server down?"; break; }
done
sleep 5
kill $JPID 2>/dev/null
busy=$(for c in /sys/class/drm/card?/device; do [ "$(cat $c/device 2>/dev/null)" = "0x73a1" ] && printf "%s " "$(cat $c/gpu_busy_percent 2>/dev/null)"; done)
echo "=== soak done: $n runs, $total tokens, $(python3 -c "print(f'{$total/max($gen_s,1):.1f}')") tok/s overall; idle busy% after: $busy" | tee -a "$OUT"
echo "=== kernel log during the soak ($(wc -l < "$KLOG") lines):" | tee -a "$OUT"
sort "$KLOG" | uniq -c | sort -rn | head -12 | tee -a "$OUT"
