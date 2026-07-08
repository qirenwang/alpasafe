# T22 - AlpaSim/NuRec candidate-rollout capability audit

Status: completed
Started: 2026-06-20 00:35 EDT
Completed: 2026-06-20 00:56 EDT
Owner: Claude Code

## Goal

Determine whether the installed AlpaSim and NuRec stack can evaluate each
Alpamayo-native candidate trajectory independently.

## Scientific reason

The next scientific bottleneck is obtaining independent future outcomes per
candidate. Without an independent rollout/evaluation backend, candidate-
conditioned labels are impossible and training cannot start.

## Inputs and assumptions

- Inspect local environment only. No large (>1 GB) downloads (decision #12).
- No training (decision #11). Do not invent results.

## Files expected to be created or modified

- outputs/audits/alpasim_nurec_capability_audit.md
- outputs/audits/environment_manifest.json
- outputs/audits/candidate_rollout_api_map.md
- outputs/audits/metric_availability_matrix.csv

## Implementation plan

1. Record versions/commits (Alpamayo, AlpaSim, NuRec, CUDA/torch).
2. Probe for AlpaSim/NuRec packages and APIs; inspect `physical_ai_av`.
3. Verify coordinate/horizon conventions from Alpamayo source.
4. Emit GO/BLOCKED decision with exact missing items.

## Acceptance criteria

- Every conclusion points to an inspected file/command/API. [met]
- No large downloads. [met] No training. [met]
- This log complete. [met]

## Attempt log

### Attempt 1 - 2026-06-20 00:35 EDT
- Commands run:
  - `git -C /home/qiren/alpamayo1.5 rev-parse HEAD` -> `a5bd40c0...`
  - `pip list | grep -iE 'alpa|nurec|torch|cuda|transformers'`
  - `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`
    -> `2.8.0+cu128 True`
  - `nvidia-smi --query-gpu=name,memory.total --format=csv` -> RTX PRO 6000 Blackwell, 97887 MiB
  - `ls a1_5_venv/lib/python3.12/site-packages | grep -iE 'alpasim|nurec|nuplan|nuscenes|carla'` -> NONE
  - `find .../physical_ai_av -name '*.py'`; `grep -rliE 'simulat|collision|drivable|rollout|reactive|metric' .../physical_ai_av` -> no match
  - inspected `src/alpamayo1_5/models/alpamayo1_5.py`, `load_physical_aiavdataset.py`,
    `README.md` (grep alpasim|nurec|closed-loop|simulat|rollout|replay)
- Files changed: the four audit outputs above.
- Tests run: n/a (audit).
- Results: AlpaSim absent, NuRec absent, `physical_ai_av` is data-only; Alpamayo
  repo is inference-only. Only `logged_executed_action_only` is producible.
- Problems: Alpamayo checkpoint not local (download >1 GB, not authorized).
- Decisions: decision = `BLOCKED_MISSING_ASSETS_OR_API`.
- Next action: T23 schemas/contracts; T24 will be BLOCKED.

## Completion summary

Completed: 2026-06-20 00:56 EDT
Git diff summary: +outputs/audits/{alpasim_nurec_capability_audit.md,
  environment_manifest.json, candidate_rollout_api_map.md,
  metric_availability_matrix.csv}.
Commands run: see attempt log.
Test results: n/a.
Generated artifacts: four audit files.
Known limitations: cannot confirm AlpaSim/NuRec semantics that are not installed;
  conclusions limited to inspectable local state.
Final decision: **BLOCKED_MISSING_ASSETS_OR_API**.
Next task recommendation: T23.
