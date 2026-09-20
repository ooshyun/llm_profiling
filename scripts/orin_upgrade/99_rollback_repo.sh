#!/usr/bin/env bash
# Undo step 1 ONLY — restores the apt sources to r36.3.
#
# This is not an upgrade rollback. Once step 2 has run, packages are already
# replaced and pointing apt back at r36.3 does NOT return the system to
# JetPack 6.0 — apt downgrades across L4T releases are unsupported. This script
# is only useful between step 1 and step 2, e.g. if `apt update` came back dirty.
set -uo pipefail
cd "$(dirname "$0")" && . ./lib.sh
need_sudo

latest_bak=$(ls -1t "$APT_SRC".bak.* 2>/dev/null | head -1 || true)
[ -n "$latest_bak" ] || halt "no $APT_SRC.bak.* found — nothing to restore."

if [ -f "$UPGRADE_DIR/.upgrade_status" ] && grep -qE 'DONE|FAILED|DTB_FAILED' "$UPGRADE_DIR/.upgrade_status"; then
  warn "step 2 has already run ($(cat "$UPGRADE_DIR/.upgrade_status"))."
  warn "Restoring the repo list will NOT undo the upgrade. Continuing anyway only"
  warn "changes which repo future apt commands read from."
  printf 'Type YES to continue: '; read -r r; [ "$r" = YES ] || exit 1
fi

hdr "restoring from $latest_bak"
sudo cp -a "$latest_bak" "$APT_SRC"
grep -v '^#' "$APT_SRC" | grep -v '^$'
sudo apt-get update 2>&1 | tail -5
ok "apt sources restored"
