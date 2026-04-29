#!/bin/bash
#
# Extract unbiased-eval accuracy for each model in each sweep.
# Per Sohaib (Apr 21 meeting): check if training degrades model performance
# on the unbiased baseline.
#
# Reads existing sweep view dirs created by analyze_by_sweep.sh.
# Output: plots/<sweep>/accuracy_unbiased.txt per sweep + combined table.
#
# Submit as SLURM job (memory-intensive due to loading .eval files):
#   sbatch scripts/tinker_training/analyze_accuracy.sh

#SBATCH --account=eecs542w26s001_class
#SBATCH --job-name=analyze-acc
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --export=ALL
#SBATCH --partition=standard
#SBATCH --output=logs/analyze-accuracy-%j.log

set -euo pipefail

cd /home/prakharg/cot-transparency
source /home/prakharg/miniconda3/etc/profile.d/conda.sh
conda activate cot

python scripts/tinker_training/analyze_accuracy.py
