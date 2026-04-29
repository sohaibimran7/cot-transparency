#!/bin/bash
#
# Organize eval logs by sweep into per-sweep dirs, then run analysis per-sweep.
# This produces clean, readable plots with only that sweep's models.
#
# Run as SLURM job: sbatch scripts/tinker_training/analyze_by_sweep.sh

#SBATCH --account=eecs542w26s001_class
#SBATCH --job-name=analyze-sweeps
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --export=ALL
#SBATCH --partition=standard
#SBATCH --output=logs/analyze-sweeps-%j.log

set -euo pipefail

cd /home/prakharg/cot-transparency
source /home/prakharg/miniconda3/etc/profile.d/conda.sh
conda activate cot

TINKER=sycophancy_eval_inspect/logs/tinker_evals
# Per Sohaib (Apr 21 meeting): use the 659-sample base evals for tighter error bars.
# This dir has 5 biased + 1 unbiased eval on TruthfulQA at 659 samples each.
BASE_EVAL_DIR=sycophancy_eval_inspect/logs/llama_base_tqa_659samples
BASE_MODEL_NAME=llama-base-tqa-659

# Map sweep -> model prefixes (link into per-sweep dirs for filtered analysis)
# Using BASE_MODEL_NAME (659-sample) instead of plain llama-base (200-sample).
declare -A SWEEPS=(
    [rollouts]="$BASE_MODEL_NAME llama-rlct-nr32 llama-rlct-nr64 llama-rlct-nr128"
    [n-consistency]="$BASE_MODEL_NAME llama-rlct-nc16 llama-rlct-nc32 llama-rlct-nc64"
    [kl-coef]="$BASE_MODEL_NAME llama-rlct-kl0.01 llama-rlct-kl0.05 llama-rlct-kl0.1 llama-rlct-kl0.5"
    [refresh-every]="$BASE_MODEL_NAME llama-rlct-re1 llama-rlct-re5 llama-rlct-re10"
    [n-datapoints]="$BASE_MODEL_NAME llama-rlct-nd64 llama-rlct-nd128 llama-rlct-nd256"
    [bct-instruct]="$BASE_MODEL_NAME llama-bct-noinst llama-bct-inst"
    [bct-batch-size]="$BASE_MODEL_NAME llama-bct-bs32 llama-bct-bs64 llama-bct-bs128 llama-bct-bs256"
    [bct-n-datapoints]="$BASE_MODEL_NAME llama-bct-nd500 llama-bct-nd1000 llama-bct-nd2000 llama-bct-nd5000"
)

# Resolve a model name to its actual logs directory (base eval lives in a different
# parent dir than other training runs).
resolve_model_dir() {
    local model="$1"
    if [[ "$model" == "$BASE_MODEL_NAME" && -d "$BASE_EVAL_DIR/$model" ]]; then
        echo "$BASE_EVAL_DIR/$model"
    elif [[ -d "$TINKER/$model" ]]; then
        echo "$TINKER/$model"
    fi
}

for sweep in "${!SWEEPS[@]}"; do
    echo ""
    echo "========================================"
    echo "=== Analyzing sweep: $sweep ==="
    echo "========================================"

    # Create per-sweep logs dir
    sweep_dir="sycophancy_eval_inspect/logs/sweep_${sweep//-/_}_view"
    rm -rf "$sweep_dir"
    mkdir -p "$sweep_dir"

    # Symlink only the relevant model dirs
    for model in ${SWEEPS[$sweep]}; do
        src=$(resolve_model_dir "$model")
        [[ -n "$src" ]] && ln -sf "../../../$src" "$sweep_dir/$model"
        # Also include -ctrl variants (always from tinker_evals; no base ctrl)
        [[ -d "$TINKER/${model}-ctrl" ]] && ln -sf "../../../$TINKER/${model}-ctrl" "$sweep_dir/${model}-ctrl"
    done

    echo "Models included:"
    ls "$sweep_dir"

    # Run analysis with just this sweep's models
    out_dir="plots/${sweep}"
    mkdir -p "$out_dir"
    python -m sycophancy_eval_inspect.visualize_results \
        --log-dir "$sweep_dir" \
        --output-dir "$out_dir" \
        --bir --plot \
        --save "$out_dir/bir_table" \
        2>&1 | tail -5
    echo "Saved plots to: $out_dir"
done

echo ""
echo "=== ALL SWEEPS ANALYZED ==="
for sweep in "${!SWEEPS[@]}"; do
    echo ""
    echo "### $sweep ###"
    ls "plots/${sweep}/" 2>/dev/null | head -10
done
