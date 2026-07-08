#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHONPATH=src python -m safeworld.eval.open_loop_eval --config configs/eval_open_loop.yaml

