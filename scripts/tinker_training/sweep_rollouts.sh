#!/bin/bash
#
# Single SLURM job that runs all rollout count configs sequentially.
# Each config runs the full pipeline: train (main+control) → eval → analysis.
#
# NOTE: Run AFTER the anchor weight sweep. Update anchor_weight in the
#       sweep_rollouts_*.yaml configs to the best value first.
#
# Usage: sbatch scripts/tinker_training/sweep_rollouts.sh
#        bash scripts/tinker_training/sweep_rollouts.sh  # local (no SLURM)

#SBATCH --account=eecs542w26s001_class
#SBATCH --job-name=sweep-nr
#SBATCH --mail-user=prakharg@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --export=ALL
#SBATCH --partition=standard
#SBATCH --output=logs/sweep-rollouts/sweep-slurm-%j.log

set -euo pipefail

cd /home/prakharg/cot-transparency
source /home/prakharg/miniconda3/etc/profile.d/conda.sh
conda activate cot

CONFIGS_DIR="scripts/tinker_training/experiment_configs"
NROLLS=(32 64 128)

mkdir -p logs/sweep-rollouts

echo "=== Rollout Count Sweep: ${#NROLLS[@]} configs ==="
echo "Started: $(date)"
echo ""

for i in "${!NROLLS[@]}"; do
    NR="${NROLLS[$i]}"
    CONFIG="${CONFIGS_DIR}/sweep_rollouts_${NR}.yaml"
    echo "========================================"
    echo "[$(($i+1))/${#NROLLS[@]}] n_rollouts=${NR}"
    echo "Config: ${CONFIG}"
    echo "Time: $(date)"
    echo "========================================"

    python scripts/tinker_training/run_experiment.py "${CONFIG}" || {
        echo "FAILED: n_rollouts=${NR} (exit $?)"
        echo "Continuing to next config..."
    }

    echo ""
done

echo "=== Sweep complete: $(date) ==="
