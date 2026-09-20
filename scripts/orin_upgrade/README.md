# Orin JetPack 6.0 → 6.2 upgrade scripts

Run these **on the Orin**, in order, from `~/orin_upgrade/`.
Plan and risk analysis: `docs/superpowers/plans/2026-09-20-jetpack-upgrade-and-vllm.md`.

```
ssh home.orin.ts
cd ~/orin_upgrade
./00_preflight.sh        # read-only go/no-go
./01_switch_repo.sh      # r36.3 -> r36.5, apt update
./02_dist_upgrade.sh     # the upgrade (40-70 min) — asks you to type UPGRADE
./03_jetpack_meta.sh     # nvidia-jetpack metapackage (CUDA 12.6 stack)
./04_reboot.sh           # asks you to type REBOOT — be near the box
# ... reconnect ...
./05_verify.sh           # versions, power mode, data, Riva/NeMo regression
```

Then tell Claude, which continues with the llama.cpp rebuild, the CUDA 12.6
re-baseline, and vLLM bring-up.

## What each step guarantees

| Script | Changes anything? | Notes |
|---|---|---|
| `00_preflight.sh` | no | Fails if sudo is unavailable, disk < 20 GB, a repo is unreachable, backups are missing, or `advisor.gateway` is running |
| `01_switch_repo.sh` | yes, reversible | Timestamped `.bak`; halts if `apt update` reports 404/NO_PUBKEY/E: |
| `02_dist_upgrade.sh` | **yes, not reversible** | Runs detached (see below); halts hard on a DTB failure |
| `02b_watch.sh` | no | Re-attach to a running upgrade; Ctrl-C only stops watching |
| `03_jetpack_meta.sh` | yes | Refuses to run unless step 2 reported DONE |
| `04_reboot.sh` | yes | Refuses while `advisor.gateway` is running |
| `05_verify.sh` | re-asserts MODE_30W only | Writes `~/backup/post_upgrade/verify_report.txt` |
| `99_rollback_repo.sh` | undoes step 1 only | **Does not undo the upgrade** |

## Why step 2 runs detached

`setsid nohup` puts the upgrade outside this terminal's session, so a dropped
SSH connection or a stray Ctrl-C cannot interrupt `dpkg` mid-transaction. A
half-finished dpkg is much harder to recover than a clean success or a clean
failure. Watch it with `./02b_watch.sh`; status lives in
`~/orin_upgrade/.upgrade_status` (`RUNNING` / `DONE` / `FAILED` / `DTB_FAILED`)
and the full log in `~/backup/dist-upgrade.log`.

## 🛑 The one error that must not be improvised

```
ERROR. Procedure for A_kernel-dtb update FAILED
```

The scripts halt here and **do not reboot**. NVIDIA's documented fix runs
`parted /dev/mmcblk0 -s resizepart 3 67.9MB` — eMMC offsets. This box boots
from **NVMe** (`TEGRA_BOOT_STORAGE nvme0n1`), where the same partitions sit
elsewhere:

| Partition | eMMC (doc assumes) | NVMe (this box) |
|---|---|---|
| p3 `A_kernel-dtb` | 768 KB | 768 KB, starts **134 MB** |
| p4 `A_reserved_on_user` | 31.6 MB | 33 MB, **135 → 168 MB** |

Running the doc's constants against NVMe would resize p3 below its own start
offset and destroy the partition table — converting a recoverable state into a
reflash. Report the halt instead.

## Recovery reality

Three tiers, verified on this box with `nvbootctrl`:

1. **Kernel/DTB/bootloader bad, won't boot** — recoverable. Two bootloader
   slots exist and UEFI fails over after 3 failed boots.
2. **Userspace broken but boots** — recoverable by hand with apt. No slot helps.
3. **RootFS corrupted** — not recoverable. `RootFS A/B is not enabled`; there is
   no second copy of `/`. Reflash, which also destroys `~/workspace`, `~/incar`,
   `~/smoke`, `~/cochl*` — none of which this project backed up.

An upgraded, working system cannot be returned to JetPack 6.0 except by
reflashing. Nothing here changes that.
