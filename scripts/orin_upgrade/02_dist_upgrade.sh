#!/usr/bin/env bash
# Step 2 — the upgrade itself (40-70 min). Launches detached, then follows.
# Safe to Ctrl-C the follow: the upgrade keeps running. Re-attach with
# ./02b_watch.sh
set -uo pipefail
cd "$(dirname "$0")" && . ./lib.sh
ensure_sudo

STATUS="$UPGRADE_DIR/.upgrade_status"

if [ -f "$STATUS" ] && grep -q RUNNING "$STATUS"; then
  warn "an upgrade is already running ($(cat "$STATUS")) — following it instead"
  exec ./02b_watch.sh
fi

grep -q "$TARGET_REL" "$APT_SRC" || halt "apt sources are not on $TARGET_REL — run 01_switch_repo.sh first."

hdr "about to upgrade"
say "  $(l4t_rev) -> $TARGET_REL   (this is NOT reversible without a reflash)"
say "  log: $BACKUP_DIR/dist-upgrade.log"
say ""
say "  Reboot comes later, in step 4 — not here."
printf 'Type UPGRADE to proceed: '
read -r reply
[ "$reply" = "UPGRADE" ] || { say "aborted"; exit 1; }

# Launched AS ROOT, not as a user process that calls sudo internally: the
# upgrade runs 40-70 min while sudo's credential cache expires after ~15, and a
# setsid'd process has no TTY to re-prompt on. One elevation up front removes
# that failure mode entirely.
#
# setsid detaches from this terminal's session, so neither Ctrl-C nor a dropped
# SSH connection can interrupt dpkg mid-transaction.
# HOME must be passed explicitly: sudo resets it to root's, which would send
# $BACKUP_DIR/$UPGRADE_DIR to /root and orphan the log and status files.
$SUDO env HOME="$HOME" RUN_AS_UID="$(id -u)" RUN_AS_GID="$(id -g)" \
  setsid nohup ./_upgrade_worker.sh >/dev/null 2>&1 < /dev/null &
sleep 3
ok "launched detached, running as root"
exec ./02b_watch.sh
