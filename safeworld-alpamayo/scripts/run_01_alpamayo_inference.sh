#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHONPATH=src python -m safeworld.alpamayo.run_inference --config configs/inference_alpamayo.yaml --dry-run --limit "${1:-12}"

