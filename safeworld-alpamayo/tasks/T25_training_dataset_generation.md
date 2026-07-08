# T25 - Training dataset generation

Status: planned (placeholder only — do not implement this phase)
Started: not started
Completed: not completed
Owner: unassigned

## Goal

Generate the candidate-conditioned training dataset once a GO rollout backend
exists (T22 must be GO_*).

## Scientific reason

Training W_theta / G_phi (T26) requires independent candidate-conditioned future
targets across scenes and strata, with leakage prevention and versioning.

## Plan (not implemented)

- Scene-level splits (no scene crosses train/val/test).
- Candidate rollout generation: K in {2,5,10} Alpamayo-native candidates per
  scene, each independently rolled out via the T22-approved backend; store as
  `CandidateRolloutRecord` + `FutureConsequenceTarget`.
- Long-tail / OOD strata: intersection, pedestrian crossing, cut-in, blocked
  lane, construction, low light, adverse weather, ambiguous nav, OOD geometry.
- Leakage prevention: observation history must not derive from candidate future;
  `validate_no_logged_label_duplication`; predicted vs evaluation reward stored
  in separate files.
- Storage estimates: per-candidate future state seq + events; estimate GB/scene
  before bulk generation; respect the >1 GB authorization rule.
- Dataset versioning: dataset hash recorded in `WeightProvenance`.

## Acceptance criteria (future)

- Reproducible, versioned dataset with documented strata and split integrity.

## Attempt log

(none — placeholder; do not begin this phase)

## Completion summary

Not started. Blocked on T22 = GO and T24 real smoke success.
