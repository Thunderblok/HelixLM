#!/bin/bash
# Run NAS 400M POC on l40sx1 GPU
set -euo pipefail

cd /app/HelixLM

python nas_400m_poc.py \
  --n-trials 15 \
  --epochs 1 \
  --max-samples 10000 \
  --output-dir ./nas_400m_results \
  --dataset-repo https://huggingface.co/datasets/david-thrower/HelixLM-small-50.0Mt-91250pt-7143it-20260427

