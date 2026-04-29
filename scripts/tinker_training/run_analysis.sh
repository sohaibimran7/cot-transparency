#!/bin/bash
#
# Run combined analysis as a SLURM job (login node may run OOM for large log dirs).
#
# Usage: sbatch scripts/tinker_training/run_analysis.sh <sweep_name>
# Example: sbatch scripts/tinker_training/run_analysis.sh refresh-every

#SBATCH --account=eecs542w26s001_class
#SBATCH --job-name=analysis
#SBATCH --mail-user=prakharg@umich.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --export=ALL
#SBATCH --partition=standard
#SBATCH --output=logs/analysis-%j.log

set -euo pipefail

SWEEP="${1:-}"
if [[ -z "$SWEEP" ]]; then
    echo "Usage: sbatch scripts/tinker_training/run_analysis.sh <sweep_name>"
    echo "  Where <sweep_name> is: rollouts, n-consistency, kl-coef, refresh-every, n-datapoints, all"
    exit 1
fi

cd /home/prakharg/cot-transparency
source /home/prakharg/miniconda3/etc/profile.d/conda.sh
conda activate cot

bash scripts/tinker_training/combined_analysis.sh "$SWEEP"
