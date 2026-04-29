#!/bin/bash
#
# Single SLURM job that runs all anchor weight configs sequentially.
# Each config runs the full pipeline: train (main+control) → eval → analysis.
#
# Usage: sbatch scripts/tinker_training/sweep_anchor_weight.sh
#        bash scripts/tinker_training/sweep_anchor_weight.sh  # local (no SLURM)

#SBATCH --account=eecs542w26s001_class
#SBATCH --job-name=sweep-aw
#SBATCH --mail-user=prakharg@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --export=ALL
#SBATCH --partition=standard
#SBATCH --output=logs/sweep-anchor-weight/sweep-slurm-%j.log

set -euo pipefail

cd /home/prakharg/cot-transparency
source /home/prakharg/miniconda3/etc/profile.d/conda.sh
conda activate cot

CONFIGS_DIR="scripts/tinker_training/experiment_configs"
ALPHAS=(0.0 0.25 0.5 0.75 1.0)

mkdir -p logs/sweep-anchor-weight

echo "=== Anchor Weight Sweep: ${#ALPHAS[@]} configs ==="
echo "Started: $(date)"
echo ""

for i in "${!ALPHAS[@]}"; do
    ALPHA="${ALPHAS[$i]}"
    CONFIG="${CONFIGS_DIR}/sweep_anchor_weight_${ALPHA}.yaml"
    echo "========================================"
    echo "[$(($i+1))/${#ALPHAS[@]}] anchor_weight=${ALPHA}"
    echo "Config: ${CONFIG}"
    echo "Time: $(date)"
    echo "========================================"

    python scripts/tinker_training/run_experiment.py "${CONFIG}" || {
        echo "FAILED: anchor_weight=${ALPHA} (exit $?)"
        echo "Continuing to next config..."
    }

    echo ""
done

echo "=== Sweep complete: $(date) ==="
