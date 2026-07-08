#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHONPATH=src python -m safeworld.world_model.train --config configs/world_model_v1.yaml

