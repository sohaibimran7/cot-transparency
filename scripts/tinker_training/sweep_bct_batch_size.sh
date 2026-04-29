#!/bin/bash
#
# BCT batch_size sweep (4 configs: 32, 64, 128, 256).
# PREREQUISITE: Run setup_bct_sweep_data.sh first.
#
# Usage: sbatch scripts/tinker_training/sweep_bct_batch_size.sh

#SBATCH --account=eecs542w26s001_class
#SBATCH --job-name=sweep-bct-bs
#SBATCH --mail-user=prakharg@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --export=ALL
#SBATCH --partition=standard
#SBATCH --output=logs/sweep-bct-batch-size/sweep-slurm-%j.log

set -euo pipefail

cd /home/prakharg/cot-transparency
source /home/prakharg/miniconda3/etc/profile.d/conda.sh
conda activate cot

mkdir -p logs/sweep-bct-batch-size

CONFIGS=(
    sweep_bct_batch_size_32
    sweep_bct_batch_size_64
    sweep_bct_batch_size_128
    sweep_bct_batch_size_256
)

echo "=== BCT Batch Size Sweep: ${#CONFIGS[@]} configs ==="
echo "Started: $(date)"
echo ""

for i in "${!CONFIGS[@]}"; do
    cfg="${CONFIGS[$i]}"
    CONFIG_PATH="scripts/tinker_training/experiment_configs/${cfg}.yaml"
    echo "========================================"
    echo "[$(($i+1))/${#CONFIGS[@]}] ${cfg}"
    echo "Time: $(date)"
    echo "========================================"

    python scripts/tinker_training/run_experiment.py "${CONFIG_PATH}" || {
        echo "FAILED: ${cfg} (exit $?)"
        echo "Continuing to next config..."
    }
    echo ""
done

echo "=== Sweep complete: $(date) ==="
