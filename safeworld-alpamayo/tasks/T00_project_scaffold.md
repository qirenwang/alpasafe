# T00 - Project scaffold

Status: completed
Started: 2026-06-03 16:08 EDT
Completed: 2026-06-03 16:19 EDT
Owner: Codex

## Goal

Create the SafeWorld-Alpamayo repository skeleton, Python package, configs,
scripts, tests, outputs folders, and project-root task log location.

## Scientific reason

This task supports the SafeWorld-Alpamayo hypothesis that a frozen VLA proposal
should be evaluated through action-conditioned future consequence prediction and
calibrated safety auditing before execution.

## Inputs and assumptions

- Use licensed Alpamayo/PhysicalAI-AV data only.
- Use sample/dry-run mode first.
- Record model/data/config provenance.

## Files expected to be created or modified

- `README.md`
- `pyproject.toml`
- `configs/`
- `src/safeworld/`
- `scripts/`
- `tests/`
- `outputs/`
- `tasks/`

## Implementation plan

1. Create the subproject under `safeworld-alpamayo/`.
2. Add package folders, scripts, configs, tests, and outputs directories.
3. Run tests.
4. Record task status in `tasks/INDEX.md`.

## Acceptance criteria

A clean importable Python package exists and `pytest` discovers tests.

## Attempt log

### Attempt 1 - 2026-06-03 16:08 EDT
- Commands run: `mkdir -p safeworld-alpamayo/...`; `chmod +x safeworld-alpamayo/scripts/*.sh`; `pytest`.
- Files changed: created `safeworld-alpamayo/README.md`, `pyproject.toml`, `.gitignore`, `configs/`, `src/safeworld/`, `scripts/`, `tests/`, `outputs/`, `tasks/`.
- Tests run: `pytest`.
- Results: Python package is importable; 7 unit tests passed.
- Problems: initial task-log placement was confusing because a duplicate `researchs/.../tasks` source directory existed.
- Decisions: project-root `safeworld-alpamayo/tasks/` is now the canonical log location.
- Next action: implement configs and dry-run data flow.

## Completion summary

Completed: 2026-06-03 16:19 EDT
Git diff summary: new `safeworld-alpamayo/` subtree.
Commands run: `pytest`.
Test results: 7 passed.
Generated artifacts: project scaffold, runnable scripts, `tasks/`.
Known limitations: no real NVIDIA data/model loading in scaffold step.
Next task recommendation: T01 environment and config.

