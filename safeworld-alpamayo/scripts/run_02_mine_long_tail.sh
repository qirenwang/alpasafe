#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHONPATH=src python -m safeworld.data.scenario_mining --config configs/data_physicalai.yaml --limit "${1:-12}"

