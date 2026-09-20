#!/usr/bin/env bash
# Step 0 — preflight. Read-only: changes nothing, only reports go/no-go.
set -uo pipefail
cd "$(dirname "$0")" && . ./lib.sh

RC=0
hdr "current release"
say "L4T: $(l4t_rev)  (expect REVISION: 3.0 before upgrading)"
say "target: $TARGET_REL"
if [ -z "$(l4t_rev)" ]; then fail "cannot read /etc/nv_tegra_release"; RC=1; fi
if l4t_rev | grep -q '5\.0'; then ok "already on r36.5 — upgrade appears done"; fi

hdr "sudo"
case "$(sudo_mode)" in
  root)         ok "running as root" ;;
  passwordless) ok "passwordless sudo available" ;;
  password)     ok "sudo available — you will be prompted for your password"
                say "    Run the steps over an SSH session with a TTY so sudo can ask:"
                say "      ssh -t home.orin.ts '~/orin_upgrade/0X_...sh'"
                say "    Step 2 elevates once up front and runs the whole upgrade as root,"
                say "    so the 15-minute sudo cache cannot expire mid-upgrade." ;;
esac

hdr "apt sources"
if [ -f "$APT_SRC" ]; then
  grep -v '^#' "$APT_SRC" | grep -v '^$' || true
  n=$(grep -c "$CURRENT_REL" "$APT_SRC" || true)
  say "lines still on $CURRENT_REL: $n  (expect 3 before step 1)"
else fail "missing $APT_SRC"; RC=1; fi

hdr "repo reachability for $TARGET_REL"
for r in common t234 ffmpeg; do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 15 \
    "https://repo.download.nvidia.com/jetson/$r/dists/$TARGET_REL/Release" || echo 000)
  if [ "$code" = 200 ]; then ok "$r -> HTTP 200"; else fail "$r -> HTTP $code"; RC=1; fi
done

hdr "disk (need >= 20 GB free)"
avail=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
df -h / | tail -1
if [ "${avail:-0}" -ge 20 ]; then ok "${avail}G free"; else fail "${avail}G free — too little"; RC=1; fi

hdr "reboot blockers — live user workloads"
adv="$(running_advisor)"
if [ -n "$adv" ]; then
  fail "advisor.gateway is RUNNING — a reboot kills it and it is not a systemd service:"
  printf '%s\n' "$adv" | cut -c1-140
  say  "Stop it deliberately (owner's call) before step 4."
  RC=1
else ok "no advisor.gateway process"; fi

for svc in llama-server cloudflared; do
  if pgrep -x "$svc" >/dev/null 2>&1; then
    warn "$svc is running — it will die on reboot (expected; restart it afterwards)"
  else ok "$svc not running"; fi
done

hdr "backups present"
for f in "$BACKUP_DIR/pre_upgrade/system_state.txt" "$BACKUP_DIR/pre_upgrade/dpkg_full.txt" \
         "$BACKUP_DIR/pre_upgrade_container_baseline.txt"; do
  if [ -s "$f" ]; then ok "$(basename "$f")"; else fail "missing $f"; RC=1; fi
done

hdr "boot layout (informational — why step 2 must stop on a DTB failure)"
say "boot storage: $(grep TEGRA_BOOT_STORAGE /etc/nv_boot_control.conf | awk '{print $2}')"
say "rootfs A/B  : $(nvbootctrl -t rootfs dump-slots-info 2>&1 | head -1)"
say "bootloader  : slot $(nvbootctrl dump-slots-info 2>/dev/null | awk -F': ' '/Current bootloader slot/{print $2}')  (A/B failover exists)"
say "NOTE: NVIDIA's A_kernel-dtb recovery doc targets /dev/mmcblk0 with eMMC offsets."
say "      This box boots from NVMe where those offsets differ — copying them would"
say "      destroy the partition table. Step 2 halts instead; do not improvise parted."

hdr "verdict"
if [ $RC -eq 0 ]; then ok "preflight PASSED — proceed to 01_switch_repo.sh"
else fail "preflight FAILED — fix the items above first"; fi
exit $RC
