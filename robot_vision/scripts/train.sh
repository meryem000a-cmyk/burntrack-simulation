#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
PY=./not_important/flora_env/bin/python
NP=./not_important

echo "========================================"
echo "  NanoFloraVLM — Train on Balanced Data"
echo "========================================"

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$NP"

echo "[1/4] Building flora tokenizer..."
$PY "$NP/build_flora_tokenizer.py"

echo "[2/4] Generating teacher pairs from balanced dataset..."
$PY "$NP/generate_pairs_from_balanced.py"

echo "[3/4] Training NanoFloraVLM..."
$PY "$NP/train_nano_vlm.py"

echo "[4/4] Exporting model..."
$PY "$NP/export_nano_vlm.py"

echo "  Done!"
