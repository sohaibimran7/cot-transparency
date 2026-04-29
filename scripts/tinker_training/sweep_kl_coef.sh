#!/bin/bash
#
# Single SLURM job that runs all kl_coef configs sequentially.
# Each config runs the full pipeline: train (main+control) → eval → analysis.
#
# Usage: sbatch scripts/tinker_training/sweep_kl_coef.sh
#        bash scripts/tinker_training/sweep_kl_coef.sh  # local (no SLURM)

#SBATCH --account=eecs542w26s001_class
#SBATCH --job-name=sweep-kl
#SBATCH --mail-user=prakharg@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --export=ALL
#SBATCH --partition=standard
#SBATCH --output=logs/sweep-kl-coef/sweep-slurm-%j.log

set -euo pipefail

cd /home/prakharg/cot-transparency
source /home/prakharg/miniconda3/etc/profile.d/conda.sh
conda activate cot

CONFIGS_DIR="scripts/tinker_training/experiment_configs"
KLS=(0.01 0.05 0.1 0.5)

mkdir -p logs/sweep-kl-coef

echo "=== KL Coef Sweep: ${#KLS[@]} configs ==="
echo "Started: $(date)"
echo ""

for i in "${!KLS[@]}"; do
    KL="${KLS[$i]}"
    CONFIG="${CONFIGS_DIR}/sweep_kl_coef_${KL}.yaml"
    echo "========================================"
    echo "[$(($i+1))/${#KLS[@]}] kl_coef=${KL}"
    echo "Config: ${CONFIG}"
    echo "Time: $(date)"
    echo "========================================"

    python scripts/tinker_training/run_experiment.py "${CONFIG}" || {
        echo "FAILED: kl_coef=${KL} (exit $?)"
        echo "Continuing to next config..."
    }

    echo ""
done

echo "=== Sweep complete: $(date) ==="
