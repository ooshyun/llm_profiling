#!/usr/bin/env bash
# Shared helpers for the JetPack 6.0 -> 6.2 (r36.3 -> r36.5) upgrade scripts.
# Sourced by every step; not meant to be run directly.

UPGRADE_DIR="$HOME/orin_upgrade"
LOG_DIR="$UPGRADE_DIR/logs"
BACKUP_DIR="$HOME/backup"
TARGET_REL="r36.5"
CURRENT_REL="r36.3"
APT_SRC="/etc/apt/sources.list.d/nvidia-l4t-apt-source.list"

mkdir -p "$LOG_DIR"

c_red=$'\033[31m'; c_grn=$'\033[32m'; c_yel=$'\033[33m'; c_bld=$'\033[1m'; c_off=$'\033[0m'

say()  { printf '%s\n' "$*"; }
ok()   { printf '%s[ OK ]%s %s\n'   "$c_grn" "$c_off" "$*"; }
warn() { printf '%s[WARN]%s %s\n'   "$c_yel" "$c_off" "$*"; }
fail() { printf '%s[FAIL]%s %s\n'   "$c_red" "$c_off" "$*"; }
hdr()  { printf '\n%s=== %s ===%s\n' "$c_bld" "$*" "$c_off"; }

# Hard stop: print a loud banner and exit non-zero. Use when continuing could
# make things worse rather than just fail.
halt() {
  printf '\n%s%s' "$c_red" "$c_bld"
  printf '################################################################\n'
  printf '#  STOP — do not run the next step, do not reboot.              #\n'
  printf '################################################################%s\n' "$c_off"
  printf '%s\n\n' "$*"
  exit 90
}

need_sudo() {
  if ! sudo -n true 2>/dev/null; then
    fail "passwordless sudo is not available."
    say  "Grant it with (validates syntax before installing):"
    say  "  echo \"\$USER ALL=(ALL) NOPASSWD:ALL\" > /tmp/nopasswd && \\"
    say  "  sudo visudo -c -f /tmp/nopasswd && \\"
    say  "  sudo install -m 440 -o root -g root /tmp/nopasswd /etc/sudoers.d/99-\$USER-nopasswd"
    say  "Revoke after the upgrade with:  sudo rm /etc/sudoers.d/99-\$USER-nopasswd"
    exit 1
  fi
}

l4t_rev() { grep -oE 'REVISION: [0-9]+\.[0-9]+' /etc/nv_tegra_release 2>/dev/null | head -1; }

# ps without the self-match trap: `pgrep -f X` and `pkill -f X` match their own
# command line over ssh and have already cost this project three false
# positives and two killed shells. The bracket trick avoids it.
running_advisor() { ps -eo pid,args | grep '[a]dvisor\.gateway' || true; }
