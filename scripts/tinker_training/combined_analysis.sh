#!/bin/bash
#
# Run combined analysis across all completed sweeps to produce comparison plots.
# Runs locally (no SLURM needed) since analysis just reads eval logs.
# Plots are moved to plots/<sweep_name>/ to prevent overwriting between sweeps.
#
# Usage: bash scripts/tinker_training/combined_analysis.sh <sweep_name>
#
# Where <sweep_name> is one of:
#   rollouts, n-consistency, kl-coef, refresh-every, n-datapoints, anchor-weight, all

set -euo pipefail

SWEEP="${1:-}"
CONFIGS_DIR="scripts/tinker_training/experiment_configs"

case "$SWEEP" in
    rollouts)
        CONFIGS="${CONFIGS_DIR}/sweep_rollouts_*.yaml"
        ;;
    n-consistency)
        CONFIGS="${CONFIGS_DIR}/sweep_n_consistency_*.yaml"
        ;;
    kl-coef)
        CONFIGS="${CONFIGS_DIR}/sweep_kl_coef_*.yaml"
        ;;
    refresh-every)
        CONFIGS="${CONFIGS_DIR}/sweep_refresh_every_*.yaml"
        ;;
    n-datapoints)
        CONFIGS="${CONFIGS_DIR}/sweep_n_datapoints_*.yaml"
        ;;
    anchor-weight)
        CONFIGS="${CONFIGS_DIR}/sweep_anchor_weight_*.yaml"
        ;;
    all)
        CONFIGS="${CONFIGS_DIR}/sweep_*.yaml"
        ;;
    *)
        echo "Usage: $0 <sweep_name>"
        echo ""
        echo "Where <sweep_name> is one of:"
        echo "  rollouts       - Combined plot for rollouts sweep"
        echo "  n-consistency  - Combined plot for n_consistency_rollouts sweep"
        echo "  kl-coef        - Combined plot for kl_coef sweep"
        echo "  refresh-every  - Combined plot for refresh_every sweep"
        echo "  n-datapoints   - Combined plot for n_datapoints sweep"
        echo "  anchor-weight  - Combined plot for anchor_weight sweep (skipped per Sohaib)"
        echo "  all            - Combined plot across all sweeps"
        exit 1
        ;;
esac

cd /home/prakharg/cot-transparency
source /home/prakharg/miniconda3/etc/profile.d/conda.sh
conda activate cot

echo "Running combined analysis for sweep: ${SWEEP}"
echo "Configs: ${CONFIGS}"
echo ""

python scripts/tinker_training/run_experiment.py ${CONFIGS} \
    --stages analysis --force

# Move plots to sweep-specific folder so they don't get overwritten by future sweeps
DEST="plots/${SWEEP}"
mkdir -p "${DEST}"
moved=0
for f in plots/llama_*.png plots/gpt-*.png; do
    [[ -f "$f" ]] || continue
    mv "$f" "${DEST}/"
    moved=$((moved + 1))
done

echo ""
if [[ $moved -gt 0 ]]; then
    echo "Moved ${moved} plot(s) to ${DEST}/"
    ls "${DEST}/"
else
    echo "No plots found in plots/ to move (check if analysis succeeded)."
fi
