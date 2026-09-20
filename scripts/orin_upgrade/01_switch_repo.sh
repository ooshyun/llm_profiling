#!/usr/bin/env bash
# Step 1 — point the L4T apt sources at r36.5 and refresh the index.
# Reversible: keeps a .bak and 99_rollback_repo.sh restores it.
set -uo pipefail
cd "$(dirname "$0")" && . ./lib.sh
ensure_sudo

hdr "before"
grep -v '^#' "$APT_SRC" | grep -v '^$'

if ! grep -q "$CURRENT_REL" "$APT_SRC"; then
  warn "no '$CURRENT_REL' lines found — already switched? Nothing to do."
else
  hdr "rewriting $CURRENT_REL -> $TARGET_REL"
  $SUDO cp -a "$APT_SRC" "$APT_SRC.bak.$(date +%Y%m%d-%H%M%S)"
  $SUDO sed -i "s/${CURRENT_REL//./\\.}/$TARGET_REL/g" "$APT_SRC"
  ok "rewritten (timestamped .bak kept next to it)"
fi

hdr "after"
grep -v '^#' "$APT_SRC" | grep -v '^$'
left=$(grep -c "$CURRENT_REL" "$APT_SRC" || true)
[ "$left" -eq 0 ] || halt "still $left line(s) on $CURRENT_REL — sed did not take. Inspect $APT_SRC."

hdr "apt update"
LOG="$LOG_DIR/01_apt_update.log"
$SUDO apt-get update 2>&1 | tee "$LOG" | tail -20

if grep -qiE '^E:|404 +Not Found|NO_PUBKEY|Failed to fetch' "$LOG"; then
  fail "apt update reported errors (full log: $LOG)"
  grep -iE '^E:|404 +Not Found|NO_PUBKEY|Failed to fetch' "$LOG" | head -10
  halt "Do not run step 2 with a broken index. Restore with ./99_rollback_repo.sh if needed."
fi

hdr "what the upgrade would pull (dry run)"
$SUDO apt-get -s dist-upgrade 2>&1 | tail -5

ok "repo switched and index clean — proceed to 02_dist_upgrade.sh"
