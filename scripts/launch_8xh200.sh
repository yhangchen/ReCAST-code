#!/usr/bin/env bash
set -euo pipefail

: "${PROFILES:?Set PROFILES to the density-profile NPZ file}"
: "${PROMPTS:?Set PROMPTS to a plain-text or JSONL prompt file}"

CONFIG=${CONFIG:-configs/h200_8gpu.toml}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/stage1-h200}

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=8 \
  train_recast.py \
  --config "${CONFIG}" \
  --profiles "${PROFILES}" \
  --prompts "${PROMPTS}" \
  --output-dir "${OUTPUT_DIR}" \
  "$@"
