#!/bin/bash
#
# One-time setup for BCT hyperparameter sweep:
#   1. Generate 2000 BCT samples from MMLU (suggested_answer bias) for Llama
#   2. Copy existing instruct samples into the same directory for the instruct-mix variant
#
# Usage: bash scripts/tinker_training/setup_bct_sweep_data.sh

set -euo pipefail

cd /home/prakharg/cot-transparency
source /home/prakharg/miniconda3/etc/profile.d/conda.sh
conda activate cot
export PYTHONPATH="/home/prakharg/cot-transparency:${PYTHONPATH:-}"

OUTPUT_NAME="sa-mmlu-hp"
TRAIN_DIR="dataset_dumps/train_seed_42/llama-3-1-8b-instruct/${OUTPUT_NAME}"
EXISTING_INSTRUCT="dataset_dumps/train_seed_42/llama-3-1-8b-instruct/instruct_samples.jsonl"

echo "=== Step 1: Generate BCT data from MMLU ==="
if [[ -f "${TRAIN_DIR}/bct_cot.jsonl" ]]; then
    echo "  Already exists: ${TRAIN_DIR}/bct_cot.jsonl ($(wc -l < ${TRAIN_DIR}/bct_cot.jsonl) lines)"
    echo "  Skipping data generation."
else
    python scripts/tinker_training/generate_bct_from_test.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --bias suggested_answer \
        --datasets mmlu \
        --limits 2000 \
        --output-name "${OUTPUT_NAME}" \
        --batch-size 64
fi

echo ""
echo "=== Step 2: Copy instruct samples into the same dir ==="
if [[ -f "${TRAIN_DIR}/instruct.jsonl" ]]; then
    echo "  Already exists: ${TRAIN_DIR}/instruct.jsonl ($(wc -l < ${TRAIN_DIR}/instruct.jsonl) lines)"
else
    cp "${EXISTING_INSTRUCT}" "${TRAIN_DIR}/instruct.jsonl"
    echo "  Copied $(wc -l < ${TRAIN_DIR}/instruct.jsonl) lines"
fi

echo ""
echo "=== Setup complete ==="
ls -la "${TRAIN_DIR}/"
