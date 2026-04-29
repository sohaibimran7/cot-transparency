#!/bin/bash
#
# BCT n_datapoints sweep (4 configs: 500, 1000, 2000, 5000).
# PREREQUISITES:
#   1. Run setup_bct_sweep_data.sh (for nd=500/1000/2000 — 2000-sample MMLU)
#   2. Run setup_bct_sweep_data_5k.sh (for nd=5000 — 7000-sample MMLU)
#
# Usage: sbatch scripts/tinker_training/sweep_bct_n_datapoints.sh

#SBATCH --account=eecs542w26s001_class
#SBATCH --job-name=sweep-bct-nd
#SBATCH --mail-user=prakharg@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --export=ALL
#SBATCH --partition=standard
#SBATCH --output=logs/sweep-bct-n-datapoints/sweep-slurm-%j.log

set -euo pipefail

cd /home/prakharg/cot-transparency
source /home/prakharg/miniconda3/etc/profile.d/conda.sh
conda activate cot

mkdir -p logs/sweep-bct-n-datapoints

CONFIGS=(
    sweep_bct_n_datapoints_500
    sweep_bct_n_datapoints_1000
    sweep_bct_n_datapoints_2000
    sweep_bct_n_datapoints_5000
)

echo "=== BCT N Datapoints Sweep: ${#CONFIGS[@]} configs ==="
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
