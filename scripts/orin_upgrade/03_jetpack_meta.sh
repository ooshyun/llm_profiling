#!/usr/bin/env bash
# Step 3 — install the nvidia-jetpack metapackage so the whole CUDA 12.6 /
# cuDNN 9 / TensorRT 10 stack lands consistently rather than piecemeal.
set -uo pipefail
cd "$(dirname "$0")" && . ./lib.sh
need_sudo

STATUS="$UPGRADE_DIR/.upgrade_status"
[ -f "$STATUS" ] && grep -q DONE "$STATUS" || halt "step 2 has not completed cleanly — check ./02b_watch.sh"

hdr "candidate"
apt-cache policy nvidia-jetpack | head -4

LOG="$LOG_DIR/03_jetpack.log"
hdr "installing nvidia-jetpack (several GB, ~10-20 min)"
DEBIAN_FRONTEND=noninteractive sudo apt-get -y \
  -o Dpkg::Options::="--force-confold" install nvidia-jetpack 2>&1 | tee "$LOG" | tail -20

if grep -qE '^E:' "$LOG"; then
  fail "apt reported errors (full log: $LOG)"; grep -E '^E:' "$LOG" | head; exit 1
fi

hdr "stack versions now installed"
dpkg -l | grep -E 'libcudnn[0-9]|^ii  tensorrt |nvidia-l4t-core|cuda-toolkit' | awk '{print $2, $3}'
ok "done — next: ./04_reboot.sh  (read its warning first)"
