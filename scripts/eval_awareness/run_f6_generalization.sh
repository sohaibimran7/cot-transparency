#!/bin/bash
# F6 wrapper-generalization eval: does multi-cue (matched_pair, 32 wrappers) close the gap on
# UNSEEN jailbreak framings better than a single-cue model? Clean control: qwen-noanchor is the
# single-cue analog of qwen-f6-matched (same Deploy0.5/KL0.03/no-IFanchor; differ only in
# single-cue+shrinkage vs 32-cue+matched_pair). Grader = calibrate default (gpt-5.4-mini).
set -uo pipefail
cd /Users/jannes/dev/MATS/rmct
PY=.venv/bin/python
QWEN="Qwen/Qwen3-30B-A3B-Instruct-2507"
MATCHED="tinker://01dccbc3-4487-52b8-a14e-bf000b581b51:train:0/sampler_weights/evalaware_rlct_qwen-f6-matched"
NOANCHOR="tinker://f5b7982a-9aab-5459-9f93-3d691d2a336d:train:0/sampler_weights/evalaware_rlct_qwen-noanchor"
HOLD="dataset_dumps/eval_awareness/eval_cues/f6_holdout_wrappers.jsonl"
TRAINED="dataset_dumps/eval_awareness/eval_cues/f6_trained_wrappers.jsonl"
RES=/tmp/f6_generalization_results.txt
echo "f6-generalization start $(date)" > $RES

geval() {  # name split data spec
  local NAME="$1" SPLIT="$2" DATA="$3" SPEC="$4"
  $PY scripts/eval_awareness/calibrate.py --models "$SPEC" --data "$DATA" \
      --behavior-judge misalignment --no-recognition --limit 80 --n-samples 3 \
      --max-new-tokens 1024 --concurrency 6 --out /tmp/f6gen_${NAME}_${SPLIT}.json \
      > /tmp/f6gen_${NAME}_${SPLIT}.log 2>&1 || echo "[$NAME/$SPLIT] EVAL FAIL" >> $RES
  local NC; NC=$(grep -oE "natural=[0-9.]+ cued=[0-9.]+ gap=[+-][0-9.]+" /tmp/f6gen_${NAME}_${SPLIT}.log | tail -1)
  echo "[$NAME / $SPLIT] $NC" >> $RES
}

# Held-out wrappers (the payoff): base vs single-cue vs multi-cue.
echo ">>> HOLDOUT wrappers @ $(date +%T)" >> $RES
geval base     holdout "$HOLD" "$QWEN" &
geval noanchor holdout "$HOLD" "${QWEN}::${NOANCHOR}" &
geval matched  holdout "$HOLD" "${QWEN}::${MATCHED}" &
wait
# Trained wrappers (in-distribution for matched; control for noanchor).
echo ">>> TRAINED wrappers @ $(date +%T)" >> $RES
geval matched  trained "$TRAINED" "${QWEN}::${MATCHED}" &
geval noanchor trained "$TRAINED" "${QWEN}::${NOANCHOR}" &
wait
echo "=== F6-GENERALIZATION DONE $(date) ===" >> $RES
