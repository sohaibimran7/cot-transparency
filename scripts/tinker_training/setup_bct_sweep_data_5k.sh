#!/bin/bash
#
# One-time setup for extending the BCT n_datapoints sweep to nd=5000.
# Per Sohaib (Apr 21): "continue this and do one for 5,000".
# Uses the new mmlu_7000samples_suggested_answer.jsonl dataset he pushed.
#
# Output: dataset_dumps/train_seed_42/llama-3-1-8b-instruct/sa-mmlu-hp-5k/bct_cot.jsonl
#         (5000 samples, different from existing sa-mmlu-hp/ which has 2000)
#
# Usage: bash scripts/tinker_training/setup_bct_sweep_data_5k.sh

set -euo pipefail

cd /home/prakharg/cot-transparency
source /home/prakharg/miniconda3/etc/profile.d/conda.sh
conda activate cot
export PYTHONPATH="/home/prakharg/cot-transparency:${PYTHONPATH:-}"

OUTPUT_NAME="sa-mmlu-hp-5k"
TRAIN_DIR="dataset_dumps/train_seed_42/llama-3-1-8b-instruct/${OUTPUT_NAME}"

echo "=== Generate 5000 BCT samples from MMLU 7K dataset ==="
if [[ -f "${TRAIN_DIR}/bct_cot.jsonl" ]]; then
    echo "  Already exists: ${TRAIN_DIR}/bct_cot.jsonl ($(wc -l < ${TRAIN_DIR}/bct_cot.jsonl) lines)"
    echo "  Skipping data generation."
else
    python scripts/tinker_training/generate_bct_from_test.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --bias suggested_answer \
        --datasets mmlu_7000samples \
        --limits 5000 \
        --output-name "${OUTPUT_NAME}" \
        --batch-size 64
fi

echo ""
echo "=== Setup complete ==="
ls -la "${TRAIN_DIR}/"
