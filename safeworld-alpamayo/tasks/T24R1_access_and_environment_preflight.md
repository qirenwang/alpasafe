# T24R1 — Access, download, and environment preflight

Status: completed (GO for HF access; infra sub-blockers recorded)
Date: 2026-06-21 (UTC)
Run dir: `artifacts/safeworld_alpasim_t24_retry/20260621T014528Z/`
Owner: Claude Code

## Goal

Capture the baseline environment, verify authenticated metadata access to the three required HF
repositories, and produce the access matrix / download plan / preflight report before any download.

## Result

- **HF access: GO.** `nvidia/PhysicalAI-Autonomous-Vehicles-NuRec` (dataset, gated=auto, OK),
  `nvidia/Alpamayo-1.5-10B` (model, OK, cached), `nvidia/Cosmos-Reason2-8B` (model, gated=auto, OK,
  config cached). Authenticated as user `StevenW77` (token never printed).
- **Key correction:** prior T22 said the Alpamayo checkpoint was not local; it IS fully cached under
  `HF_HOME=/home/qiren/storage/hf_cache` (T22 looked at the wrong cache path).
- **Infra sub-blockers found (for T24R2):** no `cargo`/Rust; no `docker compose`/buildx plugins;
  Docker daemon not accessible without sudo (`qiren` not in `docker` group); sudo is password-gated.
- **Large-download flag:** `SAFEWORLD_ALLOW_LARGE_DOWNLOADS` is UNSET — no >1 GB downloads performed.

## Files created

- `manifests/asset_access_matrix.json`
- `manifests/download_plan.json`
- `reports/T24R1_preflight_report.md`
- `logs/baseline_preflight.log`, `logs/hf_access_check.log`, `logs/cache_inventory.log`

## Commands (evidence in logs/)

Baseline capture (git, python, uv, cargo, docker, nvidia-smi, torch, disk, HF cache, groups);
`HfApi.repo_info` metadata checks for the three repos; HF cache inventory.

## Acceptance

- Baseline captured ✓; three access checks succeed ✓; access matrix + download plan + preflight
  report created ✓; no HF_TOKEN printed ✓; no large download ✓.
- HF access did not fail, so the HF HUMAN_ACTION_REQUIRED stop does not apply; the infra blocker is
  carried into T24R2.
