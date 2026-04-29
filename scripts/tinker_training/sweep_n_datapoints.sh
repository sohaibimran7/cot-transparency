#!/bin/bash
#
# Single SLURM job that runs all n_datapoints configs sequentially.
# Each config runs the full pipeline: train (main+control) → eval → analysis.
#
# Usage: sbatch scripts/tinker_training/sweep_n_datapoints.sh
#        bash scripts/tinker_training/sweep_n_datapoints.sh  # local (no SLURM)

#SBATCH --account=eecs542w26s001_class
#SBATCH --job-name=sweep-nd
#SBATCH --mail-user=prakharg@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --export=ALL
#SBATCH --partition=standard
#SBATCH --output=logs/sweep-n-datapoints/sweep-slurm-%j.log

set -euo pipefail

cd /home/prakharg/cot-transparency
source /home/prakharg/miniconda3/etc/profile.d/conda.sh
conda activate cot

CONFIGS_DIR="scripts/tinker_training/experiment_configs"
NDS=(64 128 256)

mkdir -p logs/sweep-n-datapoints

echo "=== N Datapoints Sweep: ${#NDS[@]} configs ==="
echo "Started: $(date)"
echo ""

for i in "${!NDS[@]}"; do
    ND="${NDS[$i]}"
    CONFIG="${CONFIGS_DIR}/sweep_n_datapoints_${ND}.yaml"
    echo "========================================"
    echo "[$(($i+1))/${#NDS[@]}] n_datapoints=${ND}"
    echo "Config: ${CONFIG}"
    echo "Time: $(date)"
    echo "========================================"

    python scripts/tinker_training/run_experiment.py "${CONFIG}" || {
        echo "FAILED: n_datapoints=${ND} (exit $?)"
        echo "Continuing to next config..."
    }

    echo ""
done

echo "=== Sweep complete: $(date) ==="
