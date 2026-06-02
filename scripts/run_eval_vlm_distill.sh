#!/usr/bin/env bash
# Stage 2 eval (Environment A / HAMMER env): evaluate a trained checkpoint on the
# FULL DGM4 test set. The VLM cache is NOT used at test time, so these numbers
# are directly comparable to results_all_bert.txt.
#
# Usage:
#   bash scripts/run_eval_vlm_distill.sh LOG_NUM [TEST_EPOCH]
# where LOG_NUM is the results/log<LOG_NUM> dir produced by training.
set -e

if [ -z "$1" ]; then
  echo "usage: bash scripts/run_eval_vlm_distill.sh LOG_NUM [TEST_EPOCH]"; exit 1
fi
LOG_NUM="$1"
TEST_EPOCH="${2:-best}"
HOST='127.0.0.1'
PORT='1'
NUM_GPU=1

# train.py writes results/log<EXPID> (it prepends 'log', train.py:459), but
# test.py reads results/<log_num> verbatim (test.py:311). Accept either the
# full dir name (log<EXPID>) or the bare <EXPID> and pick whichever actually
# holds the checkpoint, so the contract isn't a silent file-not-found.
RESOLVED=""
for cand in "${LOG_NUM}" "log${LOG_NUM}"; do
  if [ -f "results/${cand}/checkpoint_${TEST_EPOCH}.pth" ]; then
    RESOLVED="${cand}"; break
  fi
done
if [ -z "${RESOLVED}" ]; then
  echo "error: checkpoint not found. Looked for:"
  echo "  results/${LOG_NUM}/checkpoint_${TEST_EPOCH}.pth"
  echo "  results/log${LOG_NUM}/checkpoint_${TEST_EPOCH}.pth"
  echo "Training creates results/log<EXPID>/ — pass that dir name (or the bare EXPID)."
  exit 1
fi
echo "Evaluating results/${RESOLVED}/checkpoint_${TEST_EPOCH}.pth"

python test.py \
  --config 'configs/test_vlm_distill.yaml' \
  --output_dir 'results' \
  --text_encoder bert-base-uncased \
  --launcher pytorch \
  --rank 0 \
  --log_num "${RESOLVED}" \
  --dist-url tcp://${HOST}:1003${PORT} \
  --token_momentum \
  --world_size $NUM_GPU \
  --test_epoch "${TEST_EPOCH}"

echo "Eval written under results/${RESOLVED}/evaluation/"
