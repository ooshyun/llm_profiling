#!/usr/bin/env bash
# Step 4 — reboot into the new kernel. The one step that needs a human who can
# physically reach the box.
set -uo pipefail
cd "$(dirname "$0")" && . ./lib.sh
need_sudo

hdr "pre-reboot checks"
adv="$(running_advisor)"
if [ -n "$adv" ]; then
  fail "advisor.gateway is still running — a reboot kills it permanently"
  printf '%s\n' "$adv" | cut -c1-140
  halt "Stop it (owner's call) or accept the loss, then re-run this step."
fi
ok "no advisor.gateway running"

for svc in llama-server cloudflared; do
  pgrep -x "$svc" >/dev/null 2>&1 && warn "$svc running — expected to die; restart after reboot"
done

say ""
say "Bootloader slot: $(nvbootctrl dump-slots-info 2>/dev/null | awk -F': ' '/Current bootloader slot/{print $2}') of $(nvbootctrl dump-slots-info 2>/dev/null | awk -F': ' '/num_slots/{print $2}')"
say "If the new kernel does not boot, UEFI fails over to the other slot after"
say "3 consecutive failed boots. RootFS has no such copy — $(nvbootctrl -t rootfs dump-slots-info 2>&1 | head -1)"
say ""
warn "Be physically near the box. If SSH is absent 5 minutes after the reboot,"
warn "power-cycle; if that fails, a monitor or serial console is the only way in."
say ""
printf 'Type REBOOT to proceed: '
read -r reply
[ "$reply" = "REBOOT" ] || { say "aborted"; exit 1; }

say "rebooting…  reconnect with:  ssh home.orin.ts 'uptime; head -1 /etc/nv_tegra_release'"
say "then run:  ~/orin_upgrade/05_verify.sh"
sudo reboot
