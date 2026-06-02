#!/usr/bin/env bash
# Stage 1 (Environment B / Qwen env): build the offline VLM evidence cache.
#
# This sample run does 50 samples to smoke-test Qwen2.5-VL on the RTX 5080 and
# verify the JSON parses. Then build the full subset cache (commented below).
#
# IMPORTANT: --max-words 50 and --lowercase MUST match the training config
# (configs/train_vlm_distill.yaml: max_words 50, bert => lowercase). Otherwise
# the cache keys won't match at training time and every lookup will miss.
set -e

DATA_ROOT="../../datasets"
META="${DATA_ROOT}/DGM4/metadata/train.json"
MODEL="Qwen/Qwen2.5-VL-3B-Instruct"

mkdir -p vlm_cache

# --- 50-sample smoke test ---
python tools/build_vlm_cache.py \
  --ann "${META}" \
  --image-root "${DATA_ROOT}" \
  --out vlm_cache/qwen25_3b_train_sample.jsonl \
  --split train --dataset-division 5 --limit 50 \
  --backend qwen --model-id "${MODEL}" \
  --max-words 50 --lowercase --resume

python tools/validate_vlm_cache.py \
  --cache vlm_cache/qwen25_3b_train_sample.jsonl \
  --ann "${META}" --dataset-division 5 --limit 50 \
  --max-words 50 --lowercase

python tools/inspect_vlm_cache.py \
  --cache vlm_cache/qwen25_3b_train_sample.jsonl \
  --ann "${META}" --image-root "${DATA_ROOT}" \
  --n 5 --only-flagged --max-words 50 --lowercase

# --- FULL subset cache (~41.6k samples, division=5). Resumable. ---
# python tools/build_vlm_cache.py \
#   --ann "${META}" \
#   --image-root "${DATA_ROOT}" \
#   --out vlm_cache/qwen25_3b_train_div5.jsonl \
#   --split train --dataset-division 5 \
#   --backend qwen --model-id "${MODEL}" \
#   --max-words 50 --lowercase --resume
