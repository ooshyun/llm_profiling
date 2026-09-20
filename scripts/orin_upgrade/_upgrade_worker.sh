#!/usr/bin/env bash
# Inner worker for step 2. Launched detached AND AS ROOT by 02_dist_upgrade.sh
# — never run directly.
#
# Detached on purpose: a dropped SSH connection must not leave dpkg
# half-finished, which is far harder to recover than either a clean success or
# a clean failure. Root on purpose: the upgrade outlives sudo's ~15 min
# credential cache and has no TTY to re-prompt on.
set -uo pipefail
cd "$(dirname "$0")" && . ./lib.sh

# Files are created while root; hand them back so the non-root steps that read
# and rewrite them (02b_watch, a later re-run of 02) still can.
restore_ownership() {
  [ -n "${RUN_AS_UID:-}" ] || return 0
  chown "$RUN_AS_UID:${RUN_AS_GID:-$RUN_AS_UID}" \
    "$STATUS" "$LOG" 2>/dev/null || true
}
trap restore_ownership EXIT

STATUS="$UPGRADE_DIR/.upgrade_status"
LOG="$BACKUP_DIR/dist-upgrade.log"
mkdir -p "$BACKUP_DIR"
echo "RUNNING $(date -Is)" > "$STATUS"

# The log is appended across runs, so scan only THIS run's lines — otherwise a
# retry after a fixed failure would keep matching the previous run's error.
START_LINE=$(wc -l < "$LOG" 2>/dev/null || echo 0)
this_run() { tail -n "+$((START_LINE + 1))" "$LOG" 2>/dev/null; }

# Any boot-chain partition update that fails leaves a half-written boot chain;
# rebooting after one is the move that can make the box unbootable. Catch the
# whole family, not just A_kernel-dtb: B_kernel-dtb, A_kernel and
# cpu-bootloader failures are the same class of risk.
FATAL_RE='procedure for .* update failed|[AB]_kernel(-dtb)?.*(update )?fail(ed)?|kernel-dtb.*fail'

{
  echo "########## dist-upgrade started $(date -Is) ##########"
  echo "L4T before: $(l4t_rev)"

  # --force-confold: keep existing config files. Without it a conffile prompt
  # blocks forever in a detached run.
  DEBIAN_FRONTEND=noninteractive $SUDO apt-get -y \
      -o Dpkg::Options::="--force-confold" \
      -o Dpkg::Options::="--force-confdef" \
      dist-upgrade
  rc_dist=$?
  echo "########## dist-upgrade exit=$rc_dist ##########"

  echo "########## fix-broken $(date -Is) ##########"
  DEBIAN_FRONTEND=noninteractive $SUDO apt-get -y -f \
      -o Dpkg::Options::="--force-overwrite" install
  rc_fix=$?
  echo "########## fix-broken exit=$rc_fix ##########"
  echo "L4T after: $(l4t_rev)"
  echo "OVERALL rc_dist=$rc_dist rc_fix=$rc_fix"
} >> "$LOG" 2>&1

# The one failure that must never be followed by a reboot or by improvised
# parted commands: the DTB partition is too small and NVIDIA's documented fix
# uses eMMC offsets that are wrong for this NVMe-booted box.
if this_run | grep -qiE "$FATAL_RE"; then
  {
    echo "########## FATAL boot-chain failure detected, matching lines: ##########"
    this_run | grep -inE "$FATAL_RE" | head -20
  } >> "$LOG" 2>&1
  echo "DTB_FAILED $(date -Is)" > "$STATUS"
  exit 90
fi

if this_run | grep -qE '^OVERALL rc_dist=0 rc_fix=0'; then
  echo "DONE $(date -Is)" > "$STATUS"
  exit 0
fi

echo "FAILED $(date -Is)" > "$STATUS"
exit 1
