#!/bin/bash
#
# Single SLURM job that runs all refresh_every configs sequentially.
# Each config runs the full pipeline: train (main+control) → eval → analysis.
#
# Usage: sbatch scripts/tinker_training/sweep_refresh_every.sh
#        bash scripts/tinker_training/sweep_refresh_every.sh  # local (no SLURM)

#SBATCH --account=eecs542w26s001_class
#SBATCH --job-name=sweep-re
#SBATCH --mail-user=prakharg@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --export=ALL
#SBATCH --partition=standard
#SBATCH --output=logs/sweep-refresh-every/sweep-slurm-%j.log

set -euo pipefail

cd /home/prakharg/cot-transparency
source /home/prakharg/miniconda3/etc/profile.d/conda.sh
conda activate cot

CONFIGS_DIR="scripts/tinker_training/experiment_configs"
REFRESH=(1 5 10)

mkdir -p logs/sweep-refresh-every

echo "=== Refresh Every Sweep: ${#REFRESH[@]} configs ==="
echo "Started: $(date)"
echo ""

for i in "${!REFRESH[@]}"; do
    RE="${REFRESH[$i]}"
    CONFIG="${CONFIGS_DIR}/sweep_refresh_every_${RE}.yaml"
    echo "========================================"
    echo "[$(($i+1))/${#REFRESH[@]}] refresh_every=${RE}"
    echo "Config: ${CONFIG}"
    echo "Time: $(date)"
    echo "========================================"

    python scripts/tinker_training/run_experiment.py "${CONFIG}" || {
        echo "FAILED: refresh_every=${RE} (exit $?)"
        echo "Continuing to next config..."
    }

    echo ""
done

echo "=== Sweep complete: $(date) ==="
