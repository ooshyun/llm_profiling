#!/usr/bin/env bash
# Step 5 — post-reboot verification, including the Riva/NeMo regression check
# against the baseline captured before the upgrade.
# Writes $HOME/backup/post_upgrade/ for pulling back into the repo.
set -uo pipefail
cd "$(dirname "$0")" && . ./lib.sh

OUT="$BACKUP_DIR/post_upgrade"
mkdir -p "$OUT"
REPORT="$OUT/verify_report.txt"
BASE="$BACKUP_DIR/pre_upgrade_container_baseline.txt"
RC=0

exec > >(tee "$REPORT") 2>&1

hdr "1. release + toolchain"
say "L4T: $(l4t_rev)   (want REVISION: 5.0)"
l4t_rev | grep -q '5\.0' && ok "on r36.5" || { fail "not on r36.5"; RC=1; }
/usr/local/cuda/bin/nvcc --version 2>&1 | tail -2
dpkg -l 2>/dev/null | grep -E 'libcudnn[0-9]|^ii  tensorrt |nvidia-l4t-core|nvidia-container-toolkit' | awk '{print "  ", $2, $3}'

hdr "2. power mode (upgrades can reset it — every later benchmark depends on this)"
nvpmodel -q 2>&1 | tail -2
if nvpmodel -q 2>/dev/null | grep -q 'MODE_30W'; then
  ok "MODE_30W preserved"
else
  warn "NOT MODE_30W — re-asserting so measurements stay comparable to the CUDA 12.2 baseline"
  $SUDO -n nvpmodel -m 2 2>&1 || warn "could not set (needs sudo); do it before benchmarking"
  sleep 2; nvpmodel -q 2>&1 | tail -2
fi

hdr "3. boot slots"
nvbootctrl dump-slots-info 2>/dev/null | grep -E 'Current bootloader slot|Active bootloader slot|num_slots' | sed 's/^/  /'
nvbootctrl -t rootfs dump-slots-info 2>&1 | head -1 | sed 's/^/  /'

hdr "4. data intact"
for p in "$HOME/models" "$HOME/hf" "$HOME/serving_bench" "$HOME/llama.cpp-build"; do
  if [ -d "$p" ]; then ok "$(basename "$p"): $(du -sh "$p" 2>/dev/null | cut -f1)"
  else fail "MISSING $p"; RC=1; fi
done
df -h / | tail -1 | sed 's/^/  /'
free -g | head -2 | sed 's/^/  /'

hdr "5. docker + nvidia runtime"
docker images --format '{{.Repository}}:{{.Tag}}  {{.ID}}  {{.Size}}' | sort | sed 's/^/  /'
if timeout 60 docker run --rm --runtime nvidia ubuntu:22.04 true 2>/dev/null; then
  ok "nvidia runtime usable"
else
  warn "trivial --runtime nvidia run failed (may just be a missing ubuntu:22.04 image)"
fi

hdr "6. REGRESSION CHECK vs pre-upgrade baseline"
say "baseline recorded: $(grep -m1 'PRE-UPGRADE CONTAINER BASELINE' "$BASE" 2>/dev/null || echo '(baseline file missing!)')"
say ""
say "--- what the baseline said ---"
grep -E 'RIVA_CONTAINER_STARTED|NEMO_CONTAINER_STARTED|^torch |Orin \(nvgpu\)|riva exit|nemo exit' "$BASE" 2>/dev/null | sed 's/^/  /'
say ""

say "--- Riva now ---"
riva_out=$(timeout 150 docker run --rm --runtime nvidia \
  nvcr.io/nvidia/riva/riva-speech:2.19.0 bash -c \
  'echo RIVA_CONTAINER_STARTED; nvidia-smi --query-gpu=name --format=csv,noheader 2>&1 | head -1' 2>&1 | tail -6)
printf '%s\n' "$riva_out" | sed 's/^/  /'
if printf '%s' "$riva_out" | grep -q RIVA_CONTAINER_STARTED; then
  printf '%s' "$riva_out" | grep -q 'Orin' && ok "Riva: starts AND sees GPU (same as baseline)" \
    || { warn "Riva: starts but GPU not visible — REGRESSION"; RC=1; }
else fail "Riva: container failed to start — REGRESSION"; RC=1; fi

say ""
say "--- NeMo now (baseline had: torch 2.1.0 cuda True) ---"
nemo_out=$(timeout 240 docker run --rm --runtime nvidia \
  dustynv/nemo:r36.2.0 bash -c \
  'echo NEMO_CONTAINER_STARTED; nvidia-smi --query-gpu=name --format=csv,noheader 2>&1 | head -1; python3 -c "import torch; print(\"torch\", torch.__version__, \"cuda\", torch.cuda.is_available())" 2>&1 | tail -1' 2>&1 | tail -6)
printf '%s\n' "$nemo_out" | sed 's/^/  /'
if printf '%s' "$nemo_out" | grep -q 'cuda True'; then
  ok "NeMo: torch still sees CUDA (same as baseline)"
elif printf '%s' "$nemo_out" | grep -q NEMO_CONTAINER_STARTED; then
  warn "NeMo: starts but torch.cuda.is_available() is False — REGRESSION (was True)"; RC=1
else fail "NeMo: container failed to start — REGRESSION"; RC=1; fi

hdr "verdict"
if [ $RC -eq 0 ]; then ok "verification PASSED — report at $REPORT"
else fail "verification found problems — see above; report at $REPORT"; fi
say "Pull it back with:  rsync -a home.orin.ts:~/backup/post_upgrade/ results/orin_upgrade/post_upgrade/"
exit $RC
