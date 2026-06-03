#!/bin/bash
# Cross-factor / cross-env generalization: did training on F6 (jailbreak framing) close the gap on
# OTHER cues the model never trained on? Compare base vs the F6-closed checkpoint
# (qwen-f6-deploy-300 step150: F6 +0.083, EAB-cap 0%) across held-out slices.
#   F1  placeholder entities     (base +0.11)
#   F8  explicit eval disclosure (base +0.13)
#   nat naturalized-heldout val  (base +0.06)
#   AM  agentic-misalignment env, cue=train_cue (base -0.02)
# If F1/F8 gaps shrink too → general eval-invariance; if only F6 shrank → F6-specific.
set -uo pipefail
cd /Users/jannes/dev/MATS/rmct
PY=.venv/bin/python
QWEN="Qwen/Qwen3-30B-A3B-Instruct-2507"
STEP150="tinker://d8fc6026-5f96-5376-aa26-d18ddd90e095:train:0/sampler_weights/evalaware_rlct_qwen-f6-deploy-300_step150"
EC=dataset_dumps/eval_awareness/eval_cues
RES=/tmp/crossfactor_results.txt
echo "crossfactor sweep start $(date)" > $RES

geval() {  # name model_spec data extra_args
  local NAME="$1" SPEC="$2" DATA="$3"; shift 3
  $PY scripts/eval_awareness/calibrate.py --models "$SPEC" --data "$DATA" \
      --behavior-judge misalignment --no-recognition --limit 80 --n-samples 3 \
      --max-new-tokens 1024 --concurrency 6 --out /tmp/cf_${NAME}.json "$@" \
      > /tmp/cf_${NAME}.log 2>&1 || echo "[$NAME] EVAL FAIL" >> $RES
  local NC; NC=$(grep -oE "natural=[0-9.]+ cued=[0-9.]+ gap=[+-][0-9.]+" /tmp/cf_${NAME}.log | tail -1)
  echo "[$NAME] $NC" >> $RES
}

# Each slice: base then step150 (separate out files — calibrate keys by model-id).
for slice in "F1::$EC/evalawarebench_F1.jsonl::" \
             "F8::$EC/evalawarebench_F8.jsonl::" \
             "nat::dataset_dumps/eval_awareness/val_indist/naturalized_heldout.jsonl::" \
             "AM::dataset_dumps/eval_awareness/eval_env/agentic_misalignment.jsonl::--cue train_cue"; do
  S="${slice%%::*}"; rest="${slice#*::}"; DATA="${rest%%::*}"; EXTRA="${rest#*::}"
  echo ">>> SLICE $S @ $(date +%T)" >> $RES
  geval "${S}_base"  "$QWEN"            "$DATA" $EXTRA &
  geval "${S}_s150"  "${QWEN}::${STEP150}" "$DATA" $EXTRA &
  wait
done
echo "=== CROSSFACTOR SWEEP DONE $(date) ===" >> $RES
