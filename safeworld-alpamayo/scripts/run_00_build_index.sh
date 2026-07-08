#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHONPATH=src python -m safeworld.data.build_index --config configs/data_physicalai.yaml --dry-run --limit "${1:-12}"

