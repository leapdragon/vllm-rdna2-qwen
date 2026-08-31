#!/usr/bin/env bash
# system-report.sh — collect everything we need to debug a report about this fork on your
# machine, into ./system-report.log (review it, then email it to us).
#
#   tools/rdna2/system-report.sh [--probe] [--tests] [--log PATH] [--checksums] [--no-redact] [--out FILE]
#
#   --probe      also query a running server (/health, /models, /metrics, one tiny completion,
#                one tiny image if vision is on) -- harmless, a few seconds
#   --tests      also run idle-GPU platform tests (peer-access matrix, P2P copy latency);
#                only with no server running
#   --log PATH   digest this serve log instead of the newest logs/host-serve-*.log found
#   --checksums  sha256 the big safetensors shards too (slow: ~110 GB of reads)
#   --no-redact  keep home paths, user name, hostname and IPs in the report
#
# Never aborts: every command runs under a timeout and a failure is recorded as text. Runs in
# under a minute without --tests. Nothing is modified. Secrets (anything named *KEY*, *TOKEN*,
# *SECRET*, *PASS*) are masked; home paths, user name, hostname and IPs are redacted by default.
set -uo pipefail

PROBE=0; TESTS=0; LOG=""; CHECKSUMS=""; REDACT=1; OUT="system-report.log"
while [ $# -gt 0 ]; do
  case "$1" in
    --probe) PROBE=1 ;; --tests) TESTS=1 ;; --log) LOG="$2"; shift ;;
    --checksums) CHECKSUMS="--checksums" ;; --no-redact) REDACT=0 ;; --out) OUT="$2"; shift ;;
    -h|--help) sed -n 2,20p "$0"; exit 0 ;;
    *) echo "unknown option $1"; exit 2 ;;
  esac; shift
done

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
TMP="$(mktemp)"; trap 'rm -f "$TMP"' EXIT
T=20   # per-command timeout, seconds

section() { printf '\n\n==================== %s ====================\n' "$1" >> "$TMP"; echo "  [$1]"; }
run() {  # run "label" cmd args...  -> "$ label" + output, never fails the script
  local label="$1"; shift
  { printf '\n$ %s\n' "$label"; timeout "$T" "$@" 2>&1 || printf '(exit %s)\n' "$?"; } >> "$TMP"
}
sh_run() { run "$1" bash -c "$2"; }   # for pipelines

# ---- resolve the environment the same way the serve script / launcher would --------------
VENV_PY=""
for cand in "${VENV:-}/bin/python3" "${VIRTUAL_ENV:-}/bin/python3" "$HOME/venvs/vllm-rdna2-qwen/bin/python3" "$(command -v python3 || true)"; do
  [ -x "$cand" ] && "$cand" -c "import vllm" >/dev/null 2>&1 && { VENV_PY="$cand"; break; }
done
[ -z "$VENV_PY" ] && VENV_PY="$(command -v python3 || echo python3)"
SITE_CONF="${VLLM_QWEN38_CONF:-$HOME/.config/vllm-qwen38-flash-next.env}"
if [ -f "$SITE_CONF" ]; then
  # KEY=VALUE shell syntax; explicit env wins (same precedence as the launcher)
  eval "$(bash -c "set -a; . '$SITE_CONF' >/dev/null 2>&1; for k in MODEL PLE_INT4 MTP MAXLEN GPUUTIL VISION MM_LIMIT MM_PROCESSOR_KWARGS CHAT_KWARGS COMPILE_CACHE_OFF DENSE_INT8 TOOLS GPUS PORT; do [ -n \"\${!k+x}\" ] && printf 'SC_%s=%q\n' \"\$k\" \"\${!k}\"; done" 2>/dev/null)"
fi
MODEL="${MODEL:-${SC_MODEL:-$REPO/models/qwen38-flash-next}}"
PLE_INT4="${PLE_INT4:-${SC_PLE_INT4:-$REPO/models/qwen38-flash-next-ple/ples_int4}}"
PORT="${PORT:-${SC_PORT:-8000}}"

echo "system-report: writing $OUT (repo $REPO, python $VENV_PY)"
{
  echo "vllm-rdna2-qwen system report"
  echo "generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)   tool: system-report.sh 2026-08-31"
  echo "options: probe=$PROBE tests=$TESTS log=${LOG:-auto} checksums=${CHECKSUMS:-no} redact=$REDACT"
} > "$TMP"

# ---- 0. code ------------------------------------------------------------------------------
section "0. fork tree"
run "git -C REPO log -3 --oneline"        git -C "$REPO" log -3 --oneline
run "git -C REPO status --short | head"   bash -c "git -C '$REPO' status --short | head -20"
run "git -C REPO remote -v"               git -C "$REPO" remote -v
run "git -C REPO describe --always --dirty" git -C "$REPO" describe --always --dirty

# ---- 1. host ------------------------------------------------------------------------------
section "1. host platform"
run "uname -a" uname -a
run "/proc/cmdline" cat /proc/cmdline
sh_run "os-release" "grep -E '^(PRETTY_NAME|VERSION_ID)=' /etc/os-release"
sh_run "lscpu (model/cores/NUMA)" "lscpu | grep -E 'Model name|^CPU\(s\)|Socket|NUMA node\(s\)|NUMA node[0-9]'"
run "free -g" free -g
sh_run "swap / overcommit / memlock / ptrace" "echo \"swap devices: \$(swapon --show --noheadings 2>/dev/null | wc -l)\"; echo vm.overcommit_memory=\$(cat /proc/sys/vm/overcommit_memory); echo memlock ulimit=\$(ulimit -l); echo ptrace_scope=\$(cat /proc/sys/kernel/yama/ptrace_scope 2>/dev/null)"
sh_run "IOMMU" "ls /sys/class/iommu 2>/dev/null | head; journalctl -k -b --no-pager 2>/dev/null | grep -iE 'AMD-Vi|DMAR|iommu' | head -5"
sh_run "tmux / systemd user" "tmux -V 2>/dev/null; systemctl --user is-system-running 2>/dev/null; loginctl show-user \$USER -p Linger 2>/dev/null"
run "uptime" uptime -p

# ---- 2. PCIe ------------------------------------------------------------------------------
section "2. PCIe topology and links"
run "lspci -tv" lspci -tv
sh_run "AMD GPUs (lspci)" "lspci -nn | grep -iE 'VGA|Display|3D' "
sh_run "per-GPU link caps vs status" "for d in \$(lspci -Dn | awk '/1002:/ && (\$2 ~ /^03/) {print \$1}'); do echo \"-- \$d\"; lspci -vvs \$d 2>/dev/null | grep -E 'LnkCap:|LnkSta:|LnkCtl:' | sed 's/^[ \t]*//'; done"
sh_run "ACS on bridges (needs root; may be unavailable)" "if sudo -n true 2>/dev/null; then sudo -n lspci -vvv 2>/dev/null | grep -E '^[0-9a-f:.]+ |ACSCtl' | grep -B1 ACSCtl | grep -vE '^--' | head -60; else echo 'no passwordless sudo: ACS state not collected (run: sudo lspci -vvv | grep -B1 ACSCtl)'; fi"
sh_run "amd-smi pcie metrics" "amd-smi metric --pcie 2>/dev/null | grep -E '^GPU|WIDTH|SPEED|REPLAY|RECOVERY|NAK' | sed -E 's/\s+/ /g' | head -60"
sh_run "other devices on the fabric (storage/net controllers)" "lspci -nn | grep -iE 'SAS|RAID|SATA|NVMe|Ethernet|Network|USB controller' | head -20"

# ---- 3. GPUs ------------------------------------------------------------------------------
section "3. GPUs (rocm-smi / rocminfo / sysfs)"
run "rocm-smi" rocm-smi
run "rocm-smi --showhw" rocm-smi --showhw
run "rocm-smi --showbus --showvbios" rocm-smi --showbus --showvbios
run "rocm-smi --showmeminfo vram --showuse" rocm-smi --showmeminfo vram --showuse
run "rocm-smi --showtemp --showpower --showmaxpower" rocm-smi --showtemp --showpower --showmaxpower
run "rocm-smi --showperflevel --showclocks" rocm-smi --showperflevel --showclocks
run "rocm-smi --showpids" rocm-smi --showpids
sh_run "rocminfo (agents + gfx targets)" "rocminfo 2>/dev/null | grep -E 'Marketing Name|^\s*Name:|Uuid|gfx|Compute Unit|Wavefront|Max Waves' | grep -vE 'Vendor' | head -80"
sh_run "kfd topology (gfx_target_version, cu count per node)" "for n in /sys/class/kfd/kfd/topology/nodes/*; do echo \"-- \$(basename \$n): \$(grep -E 'gfx_target_version|simd_count|cu_per_simd_array|max_engine_clk_fcompute' \$n/properties 2>/dev/null | tr '\n' ' ')\"; done"
sh_run "drm sysfs (device id, busy%, perf level, power cap)" "for c in /sys/class/drm/card?/device; do [ -f \$c/device ] || continue; printf '%s dev=%s busy=%s%% perf=%s cap=%sW\n' \"\$(basename \$(dirname \$c))\" \"\$(cat \$c/device)\" \"\$(cat \$c/gpu_busy_percent 2>/dev/null)\" \"\$(cat \$c/power_dpm_force_performance_level 2>/dev/null)\" \"\$(( \$(cat \$c/hwmon/hwmon*/power1_cap 2>/dev/null | head -1 || echo 0) / 1000000 ))\"; done"
sh_run "amdgpu module version + key params" "cat /sys/module/amdgpu/version 2>/dev/null; for p in pcie_gen_cap aspm runpm noretry gpu_recovery ras_enable ppfeaturemask gartsize; do printf '%s=%s ' \$p \"\$(cat /sys/module/amdgpu/parameters/\$p 2>/dev/null)\"; done; echo"
sh_run "kfd queues per process" "for pd in /sys/class/kfd/kfd/proc/*; do [ -d \$pd/queues ] && echo \"pid \$(basename \$pd): \$(ls \$pd/queues | wc -l) queues (\$(ps -o comm= -p \$(basename \$pd) 2>/dev/null))\"; done"

# ---- 4. ROCm ------------------------------------------------------------------------------
section "4. ROCm installation"
sh_run "/opt/rocm version" "cat /opt/rocm/.info/version 2>/dev/null; ls -ld /opt/rocm /opt/rocm-* 2>/dev/null"
sh_run "apt-installed rocm/hip packages (mixed-install check)" "dpkg -l 2>/dev/null | awk '/^ii/ && (\$2 ~ /rocm|hip|hsa|amdgpu/) {print \$2, \$3}' | head -40"
run "hipcc --version" bash -c "/opt/rocm/bin/hipcc --version 2>&1 | head -3"
sh_run "which libamdhip64 torch loads" "\"$VENV_PY\" - <<'EOF'
import torch, os, subprocess
lib = [l for l in os.listdir(os.path.join(os.path.dirname(torch.__file__), 'lib')) if 'hip' in l.lower() or 'amdhip' in l.lower()]
print('torch/lib hip libs:', lib[:8])
out = subprocess.run(['ldd', os.path.join(os.path.dirname(torch.__file__), 'lib', 'libtorch_hip.so')], capture_output=True, text=True).stdout
print('\n'.join(l.strip() for l in out.splitlines() if 'amdhip' in l or 'hsa' in l or 'rocm' in l))
EOF"
sh_run "ROCm-related env" "env | grep -E '^(ROCM_PATH|ROCR_|HSA_|HIP_|GPU_|AMD_|NCCL_|RCCL_|TORCH_BLAS|FLASH_ATTENTION|PYTORCH_|MIOPEN)' | sort"

# ---- 5. python / torch / vllm ---------------------------------------------------------------
section "5. Python, PyTorch, vLLM build"
run "python + packages" "$VENV_PY" "$HERE/system_report_probe.py" packages
run "torch devices" "$VENV_PY" "$HERE/system_report_probe.py" torch
run "vllm build + compile cache" "$VENV_PY" "$HERE/system_report_probe.py" vllm

# ---- 6. model artefacts ---------------------------------------------------------------------
section "6. model artefacts and storage"
run "inventory / index / META / storage class" "$VENV_PY" "$HERE/system_report_probe.py" model "$MODEL" "$PLE_INT4" $CHECKSUMS
sh_run "df -hT for both" "df -hT \"$(readlink -f "$MODEL" 2>/dev/null || echo /)\" \"$(readlink -f "$PLE_INT4" 2>/dev/null || echo /)\" 2>/dev/null"

# ---- 7. serving configuration --------------------------------------------------------------
section "7. serving configuration"
sh_run "site config ($SITE_CONF)" "if [ -f '$SITE_CONF' ]; then grep -vE '^\s*#|^\s*$' '$SITE_CONF'; else echo '(none)'; fi"
DRYENV=(env DRYRUN=1 "MODEL=$MODEL" "PLE_INT4=$PLE_INT4")
for k in MTP MAXLEN VISION GPUUTIL CHAT_KWARGS COMPILE_CACHE_OFF DENSE_INT8 TOOLS GPUS MM_LIMIT MM_PROCESSOR_KWARGS; do
  v="SC_$k"; [ -n "${!v+x}" ] && DRYENV+=("$k=${!v}")
done
run "serve script dry run (what would launch, with the site config applied)" "${DRYENV[@]}" bash -c "cd '$REPO' && tools/rdna2/serve-qwen38-flash-next.sh 2>&1 | head -60"
sh_run "running vLLM processes (pid ppid rss-MB elapsed cmd)" "ps -eo pid,ppid,rss,etime,args | grep -E 'openai\.api_server|VLLM::|ple_offload|spawn_main|manage-gpu-fans|llama-server|vllm serve' | grep -vE 'grep|system-report' | awk '{printf \"%s %s %.0fMB %s \", \$1, \$2, \$3/1024, \$4; for(i=5;i<=NF&&i<12;i++) printf \"%s \", \$i; print \"\"}'"
sh_run "duplicate-instance check" "for pat in 'openai\.api_server' 'manage-gpu-fans' 'PleOffloadWorker|ple_offload/worker'; do n=\$(ps -eo args | grep -E \"\$pat\" | grep -vcE 'grep|system-report'); echo \"\$pat: \$n\"; done"
sh_run "running api_server environment (filtered)" "p=\$(pgrep -f 'openai\.api_server' | head -1); if [ -n \"\$p\" ]; then tr '\0' '\n' < /proc/\$p/environ | grep -E '^(VLLM_|ROCR_|HSA_|HIP_|NCCL_|PLE_|MM_|PYTORCH_|TORCH_|FLASH_|MTP|MAXLEN|GPUUTIL|VISION)' | sort; ps -o args= -p \$p | tr ' ' '\n' | grep -nE '^--' | head -40; else echo '(no server running)'; fi"
sh_run "listening ports" "ss -ltnp 2>/dev/null | grep -E ':$PORT |python' | head -5"

# ---- 8. live probe ----------------------------------------------------------------------------
section "8. live server probe"
if [ "$PROBE" = 1 ]; then
  run "probe http://localhost:$PORT" timeout 600 "$VENV_PY" "$HERE/system_report_probe.py" probe "http://localhost:$PORT"
  sh_run "GPU snapshot right after the probe" "rocm-smi --showuse --showpower --showtemp 2>/dev/null | grep -E 'GPU\[[0-9]\].*(GPU use|Average|edge)' | sed -E 's/\s+/ /g'"
else
  echo "(skipped: run with --probe to query a running server)" >> "$TMP"
fi

# ---- 9. serve log digest ------------------------------------------------------------------------
section "9. serve log digest"
if [ -z "$LOG" ]; then
  LOG="$(ls -t "$REPO"/logs/host-serve-*.log "$HOME"/repos/vllm-rdna2/logs/host-serve-*.log 2>/dev/null | head -1)"
fi
if [ -n "$LOG" ] && [ -f "$LOG" ]; then
  echo "log: $LOG ($(du -h "$LOG" | cut -f1), modified $(stat -c %y "$LOG" | cut -c1-19))" >> "$TMP"
  sh_run "engine config / non-default args" "grep -m1 -oE 'Initializing a V1 LLM engine \(v[^)]*\) with config: model=.{0,400}' '$LOG'; grep -m1 -oE 'non-default args: .*' '$LOG' | cut -c1-1500"
  sh_run "boot milestones" "grep -E 'Bound IPC address|GPU worker [0-9] registered|PLE quant table|prefaulted|Loading weights took|Using cache directory|compile cache is disabled|Dynamo bytecode transform|Compiling a graph|Graph capturing finished|init engine|Initial profiling|GPU KV cache size|Maximum concurrency|rdna_ar:|Using .* for ViT|Encoder cache|Application startup complete|Elided' '$LOG' | sed -E 's/^\([A-Za-z_0-9 =]+\) //' | cut -c1-200 | head -60"
  sh_run "WARNING/ERROR lines (deduplicated, with counts)" "grep -E 'WARNING|ERROR|Traceback|Error' '$LOG' | sed -E 's/^\([A-Za-z_0-9 =]+\) //; s/[0-9]{2}-[0-9]{2} [0-9:]{8} //; s/pid=[0-9]+/pid=N/g' | cut -c1-220 | sort | uniq -c | sort -rn | head -40"
  sh_run "known signatures" "grep -cE 'Duplicate PLE request' '$LOG' | sed 's/^/Duplicate PLE request: /'; grep -cE 'PLE lookup for launch' '$LOG' | sed 's/^/PLE lookup >5s: /'; grep -cE 'did not complete launch' '$LOG' | sed 's/^/did not complete launch: /'; grep -cE 'did not become ready' '$LOG' | sed 's/^/did not become ready: /'; grep -cE 'queue.Full|stayed full' '$LOG' | sed 's/^/queue.Full: /'; grep -cE 'rdna_hc_mix' '$LOG' | sed 's/^/rdna_hc_mix missing: /'; grep -cE 'Tried to allocate' '$LOG' | sed 's/^/OOM (Tried to allocate): /'; grep -cE 'max seq len|max_model_len' '$LOG' | sed 's/^/max seq len mentions: /'; grep -cE 'self-test failed|rdna_ar: disabled' '$LOG' | sed 's/^/rdna_ar disabled: /'; grep -cE 'MISMATCH' '$LOG' | sed 's/^/fused PLE mismatch: /'; grep -cE 'device lost|GPU reset' '$LOG' | sed 's/^/gpu lost or reset: /'"
  sh_run "PLE timing / host-wait lines (last 4)" "grep -E 'PLE offload (timing|host wait)' '$LOG' | sed -E 's/^\([A-Za-z_0-9 =]+\) //' | tail -4"
  sh_run "last 150 lines" "tail -150 '$LOG' | cut -c1-240"
else
  echo "(no serve log found; pass --log PATH)" >> "$TMP"
fi

# ---- 10. kernel log -----------------------------------------------------------------------------
section "10. kernel log (this boot and the previous one)"
sh_run "this boot: amdgpu/kfd/pcie/AER/reset/lost/oversubscribed/hung" "journalctl -k -b --no-pager 2>/dev/null | grep -iE 'amdgpu|kfd|pcieport|AER|mpt[23]sas|reset|lost|oversubscri|preempt|hung|lockup|Vi\b|nmi' | grep -vE 'UFW' | tail -150 || dmesg 2>/dev/null | grep -iE 'amdgpu|kfd|pcie|reset|lost' | tail -100"
sh_run "previous boot: same filter (the crash you rebooted out of)" "journalctl -k -b -1 --no-pager 2>/dev/null | grep -iE 'amdgpu|kfd|pcieport|AER|mpt[23]sas|reset|lost|oversubscri|preempt|hung|lockup|nmi' | grep -vE 'UFW' | tail -100"
sh_run "thermal (sensors)" "sensors 2>/dev/null | head -60"

# ---- 11. optional platform tests ------------------------------------------------------------------
section "11. platform tests (idle GPUs only)"
if [ "$TESTS" = 1 ]; then
  if pgrep -f 'openai\.api_server' >/dev/null; then
    echo "(skipped: a server is running; stop it before --tests)" >> "$TMP"
  else
    run "peer access + P2P copy probe" timeout 300 "$VENV_PY" "$HERE/system_report_probe.py" tests
    [ -f "$HERE/ple_coherence_test.py" ] && run "cross-process DMA coherence (1 GPU, 500 rounds)" timeout 300 "$VENV_PY" "$HERE/ple_coherence_test.py" 500
  fi
else
  echo "(skipped: run with --tests on idle GPUs)" >> "$TMP"
fi

# ---- redact + write ---------------------------------------------------------------------------------
if [ "$REDACT" = 1 ]; then
  HN="$(hostname 2>/dev/null || echo host)"
  sed -E \
    -e "s#$HOME#~#g" \
    -e "s/\b$USER\b/\$USER/g" \
    -e "s/\b$HN\b/<host>/g" \
    -e 's/\b((25[0-5]|2[0-4][0-9]|1?[0-9]?[0-9])\.){3}(25[0-5]|2[0-4][0-9]|1?[0-9]?[0-9])\b/x.x.x.x/g' \
    -e 's/127\.0\.0\.1/127.0.0.1/g' \
    -e 's/^([A-Za-z_]*(KEY|TOKEN|SECRET|PASS)[A-Za-z_]*=).*/\1<redacted>/' \
    "$TMP" > "$OUT"
else
  cp "$TMP" "$OUT"
fi
echo
echo "system-report: wrote $OUT ($(du -h "$OUT" | cut -f1)). Please skim it for anything you do not want to share, then email it with a one-paragraph description of the problem and the exact error text."
