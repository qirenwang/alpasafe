# SafeWorld-Alpamayo

SafeWorld-Alpamayo is a research prototype for pre-execution safety auditing of
Alpamayo 1.5 trajectory proposals. Alpamayo is treated as a frozen reasoning and
trajectory proposal model. SafeWorld predicts action-conditioned future
consequences, checks safety predicates, scores reasoning-action-world
consistency, calibrates risk, and gates execution.

This first pass is intentionally dry-run friendly. It uses synthetic/sample
records unless licensed NVIDIA PhysicalAI-AV data and Alpamayo credentials are
provided.

## Quick dry run

```bash
PYTHONPATH=src python -m safeworld.data.build_index --config configs/data_physicalai.yaml --dry-run --limit 12
PYTHONPATH=src python -m safeworld.alpamayo.run_inference --config configs/inference_alpamayo.yaml --dry-run --limit 12
PYTHONPATH=src python -m safeworld.data.scenario_mining --config configs/data_physicalai.yaml --limit 12
PYTHONPATH=src python -m safeworld.world_model.dataset --config configs/world_model_v1.yaml --limit 64
PYTHONPATH=src python -m safeworld.world_model.train --config configs/world_model_v1.yaml
PYTHONPATH=src python -m safeworld.eval.open_loop_eval --config configs/eval_open_loop.yaml
```

Generated reports are written under `outputs/reports/`.

## Optional: AlpaSim (for real rollouts / closed-loop simulation)

The dry run above only needs synthetic/sample data. Real rollouts and
closed-loop simulation (`rollout_backend="alpasim"`,
`use_alpasim_if_available`) depend on [NVlabs/alpasim](https://github.com/NVlabs/alpasim),
which is a separate project and is **not** vendored in this repo
(`external/` is gitignored). To use it:

```bash
git clone https://github.com/NVlabs/alpasim.git ../external/alpasim
```

adjusting the destination path to match whatever `external/alpasim` path
your configs/scripts expect. See AlpaSim's own README for its setup and
licensing requirements.

## Logging

Task attempts and completion summaries are recorded only in this project root:
`tasks/`. The proposal/source package under `../researchs/SafeWorld_Alpamayo_Codex`
is kept as input documentation and should not contain duplicate experiment logs.
