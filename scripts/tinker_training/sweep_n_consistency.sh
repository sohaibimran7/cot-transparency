#!/bin/bash
#
# Single SLURM job that runs all n_consistency_rollouts configs sequentially.
# Each config runs the full pipeline: train (main+control) → eval → analysis.
#
# Usage: sbatch scripts/tinker_training/sweep_n_consistency.sh
#        bash scripts/tinker_training/sweep_n_consistency.sh  # local (no SLURM)

#SBATCH --account=eecs542w26s001_class
#SBATCH --job-name=sweep-nc
#SBATCH --mail-user=prakharg@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --export=ALL
#SBATCH --partition=standard
#SBATCH --output=logs/sweep-n-consistency/sweep-slurm-%j.log

set -euo pipefail

cd /home/prakharg/cot-transparency
source /home/prakharg/miniconda3/etc/profile.d/conda.sh
conda activate cot

CONFIGS_DIR="scripts/tinker_training/experiment_configs"
NCS=(16 32 64)

mkdir -p logs/sweep-n-consistency

echo "=== N Consistency Rollouts Sweep: ${#NCS[@]} configs ==="
echo "Started: $(date)"
echo ""

for i in "${!NCS[@]}"; do
    NC="${NCS[$i]}"
    CONFIG="${CONFIGS_DIR}/sweep_n_consistency_${NC}.yaml"
    echo "========================================"
    echo "[$(($i+1))/${#NCS[@]}] n_consistency_rollouts=${NC}"
    echo "Config: ${CONFIG}"
    echo "Time: $(date)"
    echo "========================================"

    python scripts/tinker_training/run_experiment.py "${CONFIG}" || {
        echo "FAILED: n_consistency_rollouts=${NC} (exit $?)"
        echo "Continuing to next config..."
    }

    echo ""
done

echo "=== Sweep complete: $(date) ==="
