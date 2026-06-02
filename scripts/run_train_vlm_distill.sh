#!/usr/bin/env bash
# Stage 2 (Environment A / HAMMER env): single-GPU HAMMER+VLM training on RTX 5080.
#
# Usage:
#   bash scripts/run_train_vlm_distill.sh [CONFIG]
# CONFIG defaults to configs/train_vlm_distill.yaml (= experiment E2).
# For the controlled comparison, run all three with the SAME schedule:
#   E0 baseline : edit config -> vlm_distill: false
#   E1 MLC only : edit config -> loss_vlm_tmg_wgt: 0.0
#   E2 MLC+TMG  : the default config
set -e

EXPID=$(date +"%Y%m%d_%H%M%S")
HOST='127.0.0.1'
PORT='1'
NUM_GPU=1
CONFIG="${1:-configs/train_vlm_distill.yaml}"

echo "Training with config=${CONFIG}  log_num=${EXPID}"

python train.py \
  --config "${CONFIG}" \
  --output_dir 'results' \
  --checkpoint 'ALBEF_4M.pth' \
  --text_encoder bert-base-uncased \
  --launcher pytorch \
  --rank 0 \
  --log_num ${EXPID} \
  --dist-url tcp://${HOST}:1003${PORT} \
  --token_momentum \
  --world_size $NUM_GPU \
  --model_save_epoch 100

echo "Done. Training dir: results/log${EXPID}"
echo "Evaluate with:"
echo "  bash scripts/run_eval_vlm_distill.sh log${EXPID} best"
