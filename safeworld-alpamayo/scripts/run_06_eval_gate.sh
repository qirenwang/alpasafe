#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHONPATH=src python -m safeworld.eval.closed_loop_eval --config configs/eval_closed_loop.yaml

