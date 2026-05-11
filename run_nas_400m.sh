#!/bin/bash
# Run NAS 400M POC on l40sx1 GPU
set -euo pipefail

cd /app/HelixLM

python nas_400m_poc.py \
  --n-trials 15 \
  --epochs 1 \
  --max-samples 10000 \
  --output-dir ./nas_400m_results \
  --dataset-repo david-thrower/HelixLM-tiny-400.0Mt-730000pt-57143it-20260430
