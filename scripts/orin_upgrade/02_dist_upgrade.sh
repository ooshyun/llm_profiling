#!/usr/bin/env bash
# Step 2 — the upgrade itself (40-70 min). Launches detached, then follows.
# Safe to Ctrl-C the follow: the upgrade keeps running. Re-attach with
# ./02b_watch.sh
set -uo pipefail
cd "$(dirname "$0")" && . ./lib.sh
need_sudo

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

# setsid detaches from this terminal's session entirely, so neither Ctrl-C nor
# a dropped SSH connection can interrupt dpkg mid-transaction.
setsid nohup ./_upgrade_worker.sh >/dev/null 2>&1 < /dev/null &
sleep 3
ok "launched detached (worker pid ~$!)"
exec ./02b_watch.sh
