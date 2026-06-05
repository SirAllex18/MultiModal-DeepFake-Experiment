#!/usr/bin/env bash
# Stage 1 (Environment B / Qwen env): build the offline VLM evidence cache.
#
# This sample run does 200 samples to (a) smoke-test Qwen2.5-VL-7B in 4-bit on
# the RTX 5080 and (b) MEASURE whether the teacher signal aligns with DGM4 GT
# (tools/vlm_signal_alignment.py) BEFORE committing to the ~30h full build.
#
# 7B does not fit in 16 GB at bf16 (~14-15 GB weights), so we load it in 4-bit
# (needs bitsandbytes in the qwen env). The prompt (prompt_version v2) focuses
# purely on image-text contradiction for text_swap/text_attribute.
#
# IMPORTANT: --max-words 50 and --lowercase MUST match the training config
# (configs/train_vlm_distill.yaml: max_words 50, bert => lowercase). Otherwise
# the cache keys won't match at training time and every lookup will miss.
set -e

DATA_ROOT="../../datasets"
META="${DATA_ROOT}/DGM4/metadata/train.json"
MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
SAMPLE_OUT="vlm_cache/qwen25_7b_v3_sample.jsonl"

mkdir -p vlm_cache

# --- 200-sample smoke (7B, 4-bit) ---
python tools/build_vlm_cache.py \
  --ann "${META}" \
  --image-root "${DATA_ROOT}" \
  --out "${SAMPLE_OUT}" \
  --split train --dataset-division 5 --limit 200 \
  --backend qwen --model-id "${MODEL}" --load-in-4bit \
  --max-words 50 --lowercase --resume

python tools/validate_vlm_cache.py \
  --cache "${SAMPLE_OUT}" \
  --ann "${META}" --dataset-division 5 --limit 200 \
  --max-words 50 --lowercase

# --- the decisive check: does the signal separate GT? ---
python tools/vlm_signal_alignment.py \
  --cache "${SAMPLE_OUT}" \
  --ann "${META}" --dataset-division 5 --limit 200 \
  --max-words 50 --lowercase

python tools/inspect_vlm_cache.py \
  --cache "${SAMPLE_OUT}" \
  --ann "${META}" --image-root "${DATA_ROOT}" \
  --n 6 --only-flagged --max-words 50 --lowercase

# --- FULL subset cache (~41.6k samples, division=5). Resumable. ~30h on 16 GB. ---
# Only run this if the alignment report above shows distillable signal.
# python tools/build_vlm_cache.py \
#   --ann "${META}" \
#   --image-root "${DATA_ROOT}" \
#   --out vlm_cache/qwen25_7b_train_div5.jsonl \
#   --split train --dataset-division 5 \
#   --backend qwen --model-id "${MODEL}" --load-in-4bit \
#   --max-words 50 --lowercase --resume
