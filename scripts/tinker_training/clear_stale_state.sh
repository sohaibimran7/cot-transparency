#!/bin/bash
#
# Clear stale experiment state files for experiments where all tasks are
# still "running" (meaning they were killed mid-execution and won't resume
# cleanly). Leaves experiments with any completed stages intact.
#
# Usage: bash scripts/tinker_training/clear_stale_state.sh
#        bash scripts/tinker_training/clear_stale_state.sh --dry-run

set -euo pipefail

DRY_RUN="${1:-}"
EXPERIMENTS_DIR="experiments"

if [[ ! -d "$EXPERIMENTS_DIR" ]]; then
    echo "No experiments directory found."
    exit 0
fi

echo "Checking for stale experiment state files..."
echo ""

for exp_dir in "$EXPERIMENTS_DIR"/*/; do
    state_file="${exp_dir}state.json"
    [[ ! -f "$state_file" ]] && continue

    # Check if any stage or task is completed
    if python3 -c "
import json, sys
with open('$state_file') as f:
    state = json.load(f)
stages = state.get('stages', {})
tasks = state.get('tasks', {})
# Stale if no completed stages and no completed tasks
has_completed = any(s.get('status') == 'completed' for s in stages.values()) or \
                any(t.get('status') == 'completed' for t in tasks.values())
sys.exit(0 if has_completed else 1)
" 2>/dev/null; then
        echo "  KEEP  $exp_dir (has completed stages)"
    else
        echo "  STALE $exp_dir (no completed stages)"
        if [[ "$DRY_RUN" != "--dry-run" ]]; then
            rm -rf "$exp_dir"
            echo "        Deleted."
        fi
    fi
done

echo ""
if [[ "$DRY_RUN" == "--dry-run" ]]; then
    echo "Dry run complete. Re-run without --dry-run to actually delete."
else
    echo "Stale state files cleared."
fi
