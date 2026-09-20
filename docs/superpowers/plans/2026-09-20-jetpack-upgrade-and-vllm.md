# JetPack 6.0 → 6.2 Upgrade + vLLM Bring-up — Plan

> Date: 2026-09-20
> Spec: `docs/superpowers/specs/2026-08-27-serving-framework-eval-design.md` §7 (runbook), §11 (open questions)
> Supersedes spec §7's risk section — the on-device investigation below replaces its assumptions.
> Scope: user asked for steps 1–3 (upgrade → llama.cpp rebuild → vLLM) plus Riva/NeMo verification after the upgrade, with the two §7 risks investigated in detail first.

---

## 1. Risk investigation (done 2026-09-20, on-device + upstream)

### Risk A — "rollback is reflash only"

**Investigated.** Partially wrong, and the correction matters.

`nvbootctrl` on this box reports:
```
Current bootloader slot: A      Active bootloader slot: A      num_slots: 2
Capsule update status: 0
RootFS A/B is not enabled.
```
And both eMMC and NVMe carry the full redundant layout: `A_kernel`, `A_kernel-dtb`,
`A_reserved_on_user`, `B_kernel`, `B_kernel-dtb`, `B_reserved_on_user`, plus `recovery`/
`recovery_alt` and `esp`/`esp_alt`.

So the honest risk model is **three tiers, not one**:

| Failure | Recoverable? | How |
|---|---|---|
| Bootloader / kernel / DTB bad → won't boot | **Yes** | UEFI fails over to slot B after 3 consecutive failed boots (Jetson default); or force it manually from a working shell with `nvbootctrl set-active-boot-slot 1` |
| Userspace broken (apt half-upgraded, driver/CUDA mismatch) | **Yes, but manual** | The box still boots — slot A's kernel is fine. Fix with apt. No slot helps here. |
| Rootfs itself corrupted, or both slots bad | **No** | Reflash. `RootFS A/B is not enabled` → there is no second copy of `/`. |

**What this changes:** the scary-sounding tier (won't boot at all) is the *recoverable* one.
The unrecoverable tier is rootfs corruption, which `apt dist-upgrade` does not typically
cause — it replaces packages, not the filesystem. Net: risk is lower than spec §7.7 stated,
but **not zero, and there is still no way to "undo" the upgrade** — an upgraded, booting,
working system cannot be returned to 6.0 except by reflashing. That asymmetry stands.

**Consequence for the plan:** back up what a reflash would destroy, and treat "it boots but
something is broken" as the expected bad case (fix forward), not "it's bricked".

### Risk B — NVIDIA's kernel-dtb recovery procedure assumes eMMC; we boot from NVMe

**Investigated.** The layouts are identical, but the documented numbers are not transferable.

NVIDIA's JetPack 6.2 install doc gives this recovery for `ERROR. Procedure for A_kernel-dtb
update FAILED`:
```
sudo parted /dev/mmcblk0 -s rm 4
sudo parted /dev/mmcblk0 -s resizepart 3 67.9MB
sudo parted /dev/mmcblk0 mkpart A_reserved_on_user 67.9MB 101MB
```
It deletes `A_reserved_on_user` (p4) and grows `A_kernel-dtb` (p3) into the freed space,
because the new DTB no longer fits in 768 KB.

This box (`TEGRA_BOOT_STORAGE nvme0n1`) has the same partition names and numbers on NVMe,
but **different offsets**:

| Partition | eMMC (doc's target) | NVMe (our boot device) |
|---|---|---|
| p3 `A_kernel-dtb` | 768 KB | 768 KB, **starts 134 MB** |
| p4 `A_reserved_on_user` | 31.6 MB | 33 MB, **starts 135 MB, ends ~168 MB** |

So `resizepart 3 67.9MB` — a literal copy — would **shrink p3 below its own start offset on
NVMe and destroy the partition table.** The doc's constants are eMMC-specific.

**Consequence for the plan — this is a hard stop, not an improvisation point.** If that error
appears, the plan STOPS and reports. It does not run parted with recomputed numbers on a
live boot device. Rationale: a wrong offset here is the one action in this entire plan that
converts a fixable state into a reflash, and NVIDIA's own text warns "Do NOT reboot between
running these commands and retrying" — meaning the system is in a half-updated state where a
mistake is unrecoverable. Recomputing offsets is a decision for the user with NVIDIA forum
confirmation, not something to derive mid-run.

Secondary note: upstream reports of `apt dist-upgrade` on NVMe-booted Orins are mixed —
successes exist, and the documented failures cluster on custom carrier boards. This is an
official devkit (`NVIDIA Jetson AGX Orin Developer Kit`, TNSPEC `3701-500-0005`), which is
the configuration NVIDIA supports for the apt path.

### Decisions from the investigation

| # | Decision | Rationale |
|---|---|---|
| D1 | Target **r36.5** (JetPack 6.2.x), not r36.4 | The staged containers are `...-r36.5.tegra-aarch64-cp312-cu129-...`; matching exactly beats relying on minor-version compat. Both repos verified live (HTTP 200 on `common` and `t234`). One upgrade instead of two. |
| D2 | `A_kernel-dtb` failure = **stop and report**, never improvise parted | Risk B. Only action in the plan that can turn recoverable → reflash. |
| D3 | Baseline Riva/NeMo **before** the upgrade | "Verify after upgrade" is meaningless without a known-good before. The Riva container last exited with status 1 five months ago — it may already be broken, and we must not attribute a pre-existing failure to the upgrade. |
| D4 | Never `rm -rf` anything; rename (`mv`) instead | User's standing rule. Applies to `build-cuda/` on rebuild. |
| D5 | Stop before the upgrade for a human gate | Needs sudo (none passwordless) and someone physically near the box. |

---

## 2. Blocker that gates execution

**`sudo` requires a password on this box** (`sudo -n true` → "a password is required";
`SUDO_ASKPASS` unset; no `~/.sudo_askpass`). Every upgrade step needs root. I cannot supply
a password and will not go looking for one.

Also, the upgrade requires **a reboot with someone able to reach the box physically** if it
does not come back — Tailscale only helps once the network is up.

So Tasks 0–3 run now; **Task 4 onward needs the user to unblock.**

---

## 3. Task list

### Task 0 — Pre-work that must land before any re-measurement *(no sudo)*
- [ ] `s2_turns.csv` carries no `runtime` column (`metrics.py` `s2_turns` dict, `bench.py`
      DictWriter field list). After the CUDA 12.6 rerun the per-turn TTFT rows for 12.2 and
      12.6 would overlay indistinguishably. Add `runtime` to both, keep `engine_version`.
- [ ] Add a test asserting `s2_turns` rows carry `runtime`.
- [ ] Full suite green, commit.

### Task 1 — Baseline the demo containers BEFORE touching anything *(no sudo)*
- [ ] Record current state of `riva-speech:2.19.0` and `dustynv/nemo:r36.2.0`: image IDs,
      the `riva-model-repo` volume (20.6 GB, linked to the exited `riva-speech` container),
      and whether each image can start a trivial command under `--runtime nvidia` today.
- [ ] Save to `results/orin_upgrade/pre_upgrade_container_baseline.txt`, commit.
- [ ] **Expected**: the `riva-speech` container exited with status 1 five months ago, so a
      failure now is pre-existing, not upgrade-caused. Record exactly what happens either way.

### Task 2 — Backups + pre-flight capture *(no sudo)*
- [ ] `~/backup/` on the Orin: `dpkg -l`, `apt-mark showmanual`, the apt source list,
      `/boot/extlinux/extlinux.conf`, `nv_boot_control.conf`, `nvbootctrl` output,
      `lsblk -o NAME,SIZE,PARTLABEL` for both `mmcblk0` and `nvme0n1`, partition start/size
      from `/sys/block/*/`, `df -h`, `free -g`, `docker images`, `pip3 list`, versions of
      cuda/cudnn/tensorrt, `nvpmodel -q`.
- [ ] Copy the same bundle into the repo at `results/orin_upgrade/pre_upgrade/`, commit.
      (A reflash wipes the Orin copy; the repo copy is the one that survives.)
- [ ] Confirm what a reflash would cost and that nothing irreplaceable is only on the Orin:
      `~/models/` 26 GB + `~/hf/` 39 GB are re-downloadable; `~/workspace/`, `~/smoke/`,
      `~/cochl*`, `~/riva_*` are **user data this plan does not back up** — flag to the user.

### Task 3 — Stop the live services cleanly *(no sudo)*
- [ ] The public endpoint (`vllm1.ooshyun.cc` → `llama-server` on :8000 + `cloudflared`)
      dies on reboot anyway. Stop both deliberately and record how to bring them back.
- [ ] Use `pkill -x llama-server` / `pkill -x cloudflared` — **not** `pkill -f`, which
      self-matches its own ssh command line and kills the shell (observed twice this project).

### 🚦 GATE — user must unblock before Task 4
Needs: (a) sudo usable non-interactively, (b) someone near the box, (c) go-ahead.

### Task 4 — The upgrade *(sudo; ~40–70 min + reboot)*
- [ ] Re-read NVIDIA's current JetPack 6.2 install page immediately before running — confirm
      the documented steps have not changed.
- [ ] `sed` the three lines of `/etc/apt/sources.list.d/nvidia-l4t-apt-source.list`
      `r36.3` → `r36.5`; show the file; `apt update`.
- [ ] `DEBIAN_FRONTEND=noninteractive apt-get -y -o Dpkg::Options::="--force-confold" dist-upgrade`
      — `--force-confold` keeps existing config files, so a prompt cannot silently stall a
      detached run. Tee the whole thing to `~/backup/dist-upgrade.log`.
- [ ] `apt install -f -o Dpkg::Options::="--force-overwrite"`.
- [ ] **If `A_kernel-dtb ... FAILED` appears → STOP** (D2). Do not reboot. Do not run parted.
      Capture the log, `lsblk`, and both devices' partition tables, and report.
- [ ] Optionally `apt install nvidia-jetpack` to pull the full CUDA 12.6 stack.
- [ ] Reboot; wait up to 5 min for SSH; if absent, report and hand to the user (physical).

### Task 5 — Post-upgrade verification *(some sudo)*
- [ ] L4T is r36.5; `nvcc` 12.6; cuDNN 9.x; TensorRT 10.x; `nvbootctrl` slot still A.
- [ ] **Power mode may have reset** — re-assert MODE_30W (`nvpmodel -m 2`) or every later
      measurement is confounded. Capture `nvpmodel -q` before any benchmark.
- [ ] `nvidia-container-toolkit` still functional: run a trivial container with `--runtime nvidia`.
- [ ] `~/models/`, `~/hf/`, `~/serving_bench/` intact; disk/RAM sane.
- [ ] **Riva/NeMo re-verification against Task 1's baseline** — same commands, diff the
      outcome. Report per-image: worked before/after, broke, or was already broken.
- [ ] Commit `results/orin_upgrade/post_upgrade/`.

### Task 6 — llama.cpp rebuild + CUDA 12.6 re-baseline *(no sudo)*
- [ ] `mv ~/llama.cpp-build/build-cuda ~/llama.cpp-build/build-cuda.cu122` (D4 — keep, don't delete).
- [ ] Rebuild with `~/rebuild_llama.sh` (same source, same commit `6fdd0ac` — only the CUDA
      version changes, so the comparison isolates one variable).
- [ ] `llama-bench` on 35B-A3B and 8B; compare to CUDA 12.2 (35B tg32 10.8, 8B 7.63).
- [ ] Harness S1/S2/S3 for both models with `--runtime cu12.6`, cold-start protocol from the
      README (restart before S1, restart before S2, no probes). Commit JSONL + summary.

### Task 7 — vLLM bring-up *(no sudo)*
- [ ] Ports: llama-server must be **off** before vLLM claims :8000 (single-engine rule, and
      the tunnel maps `vllm1.ooshyun.cc` → `localhost:8000`).
- [ ] **HF cache path**: models live at `~/hf/models--Qwen--...`, i.e. `~/hf` IS the cache
      root (`HF_HOME`-style layout), not `~/.cache/huggingface`. Mount `-v ~/hf:/hf -e HF_HOME=/hf`
      and pass the repo id, or mount and pass the snapshot dir directly. Verify vLLM resolves
      it **offline** — a wrong mount silently triggers a 23 GB re-download.
- [ ] 8B bf16 first (proves the container runs at all), then 35B GPTQ-Int4 with
      `--quantization moe_wna16 --reasoning-parser qwen3 --max-model-len 8192
      --gpu-memory-utilization 0.5`.
- [ ] Unified memory: 0.5 ≈ 31 GB. If OOM, step down 0.45 → 0.40 and record the value that
      worked — it is a result, not a nuisance.
- [ ] Harness S1/S2/S3 with `--engine vllm --runtime cu12.6 --fmt gptq-int4
      --chat-template-kwargs '{"enable_thinking": false}'`. Verify no `<think>` and that
      `reasoning_deltas` stays 0 (the warm-up guard added after the Phase-0 review).
- [ ] If the prebuilt image fails: capture the error, report, and stop — `jetson-containers
      build vllm` is 2–4 h and a separate decision.

### Task 8 — Results + docs
- [ ] Extend `claudedocs/serving_framework_eval_20260915.md` with the CUDA 12.6 rows and the
      vLLM rows; state plainly which comparisons are confounded (GGUF Q4_K_M vs GPTQ-Int4).
- [ ] Update `CLAUDE.md`: JetPack/CUDA versions, Riva/NeMo verdict, new baselines, Next up.
- [ ] Restore the public endpoint (llama.cpp or vLLM, user's choice) and note it.

---

## 4. What this plan will not do

- Run `parted` on a boot device (D2).
- Delete anything (D4) — renames only.
- Reflash, or start anything that requires physical access.
- Flip to MAXN. All measurement stays MODE_30W so the CUDA 12.2 → 12.6 delta stays readable.
- Back up `~/workspace/`, `~/smoke/`, `~/cochl*`, `~/riva_*` — out of this project's scope,
  and a reflash would destroy them. Flagged to the user before the gate.
