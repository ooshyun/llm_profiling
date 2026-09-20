#!/usr/bin/env bash
# Follow a running (or finished) step-2 upgrade. Ctrl-C is safe — it only stops
# watching, never the upgrade.
set -uo pipefail
cd "$(dirname "$0")" && . ./lib.sh

STATUS="$UPGRADE_DIR/.upgrade_status"
LOG="$BACKUP_DIR/dist-upgrade.log"
[ -f "$STATUS" ] || halt "no upgrade has been started — run 02_dist_upgrade.sh"

hdr "watching (Ctrl-C to stop watching; the upgrade continues)"
while true; do
  s=$(cat "$STATUS" 2>/dev/null || echo "UNKNOWN")
  case "$s" in
    RUNNING*)
      pkgs=$(grep -cE '^(Setting up|Unpacking) ' "$LOG" 2>/dev/null || echo 0)
      printf '\r  running… %s package actions, log %s lines   ' \
        "$pkgs" "$(wc -l < "$LOG" 2>/dev/null || echo 0)"
      sleep 15
      ;;
    DONE*)
      printf '\n'; tail -25 "$LOG"
      ok "upgrade finished cleanly. L4T now: $(l4t_rev)"
      say "Next: ./03_jetpack_meta.sh"
      exit 0
      ;;
    DTB_FAILED*)
      printf '\n'
      grep -inE 'procedure for .* update failed|[AB]_kernel(-dtb)?.*fail|kernel-dtb.*fail' "$LOG" | tail -15
      halt "$(cat <<'EOT'
A boot-chain partition update FAILED (see the matching lines above —
A_kernel-dtb, B_kernel-dtb, A_kernel and cpu-bootloader all land here).

DO NOT reboot. DO NOT run the parted commands from NVIDIA's doc — they target
/dev/mmcblk0 with eMMC offsets, and this box boots from NVMe where
A_kernel-dtb (p3) starts at 134 MB and A_reserved_on_user (p4) spans
135-168 MB. Copying those constants would shrink p3 below its own start and
destroy the partition table — turning a recoverable state into a reflash.

Collect and report:
  lsblk -o NAME,SIZE,PARTLABEL /dev/nvme0n1
  grep -nE 'A_kernel-dtb|FAILED' ~/backup/dist-upgrade.log
EOT
)"
      ;;
    FAILED*)
      printf '\n'; tail -40 "$LOG"
      fail "upgrade returned a non-zero status. Full log: $LOG"
      say "The box still boots (slot A kernel is untouched). Most such failures are"
      say "apt-level and fixable forward; report the tail above before retrying."
      exit 1
      ;;
    *) printf '\n'; fail "unrecognised status: $s"; exit 1 ;;
  esac
done
