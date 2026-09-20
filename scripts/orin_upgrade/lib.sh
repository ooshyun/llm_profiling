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

# SUDO is "" when already root, else "sudo". Use "$SUDO cmd" everywhere so the
# same script works whether invoked normally or already elevated.
if [ "$(id -u)" -eq 0 ]; then SUDO=""; else SUDO="sudo"; fi

# Make sure root is usable, prompting for a password if needed.
# Returns 0 when subsequent "$SUDO cmd" calls will work.
ensure_sudo() {
  [ -z "$SUDO" ] && return 0                 # already root
  sudo -n true 2>/dev/null && return 0       # cached or NOPASSWD

  if [ -t 0 ] || [ -t 1 ]; then
    warn "sudo needs your password."
    if sudo -v; then ok "sudo authenticated"; return 0; fi
    fail "sudo authentication failed."; exit 1
  fi

  fail "sudo needs a password but there is no terminal to ask on."
  say  "Run this script from an interactive shell, e.g.:"
  say  "    ssh -t home.orin.ts '~/orin_upgrade/$(basename "$0")'"
  say  "(the -t flag allocates a TTY so sudo can prompt)"
  exit 1
}

# Report only — used by preflight, which must not fail just because a password
# will be needed.
sudo_mode() {
  if [ "$(id -u)" -eq 0 ]; then echo "root"
  elif sudo -n true 2>/dev/null; then echo "passwordless"
  else echo "password"; fi
}

l4t_rev() { grep -oE 'REVISION: [0-9]+\.[0-9]+' /etc/nv_tegra_release 2>/dev/null | head -1; }

# ps without the self-match trap: `pgrep -f X` and `pkill -f X` match their own
# command line over ssh and have already cost this project three false
# positives and two killed shells. The bracket trick avoids it.
running_advisor() { ps -eo pid,args | grep '[a]dvisor\.gateway' || true; }
