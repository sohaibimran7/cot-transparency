#!/bin/bash
# 8-run Qwen3-30B + F6 OFAT battery. Parallel batches of ~3. EPOCHS set from the 300-step run.
# Usage: EPOCHS=2 bash scripts/eval_awareness/run_battery8.sh
set -uo pipefail
cd /Users/jannes/dev/MATS/rmct
PY=.venv/bin/python
EPOCHS=${EPOCHS:-1}   # gap moves in <50 steps (300-run trajectory) → 1 epoch (100 steps) suffices
QWEN="Qwen/Qwen3-30B-A3B-Instruct-2507"
F6="dataset_dumps/eval_awareness/eval_cues/evalawarebench_F6.jsonl"
IFEVAL="dataset_dumps/eval_awareness/ifeval_train/ifeval.jsonl"
# Reduced rollouts (32/32/24 vs the deep run's 48/48/32) — ~1.4x faster, fine for attribution.
# Training trait grading = gpt-5.4-nano (high-volume, validated == mini on clear-cut judgments). Evals stay mini.
COMMON="--model $QWEN --data $F6 --limit 100 --n-epochs $EPOCHS --n-ref-rollouts 32 --n-train-rollouts 32 --n-consistency-rollouts 24 --n-helpfulness-rollouts 12 --max-new-tokens 1024 --checkpoint-every 50 -y --grader-model gpt-5.4-nano --experiment-name evalaware_rlct"
IFANCHOR="--helpfulness-weight 0.5 --helpfulness-trait ifeval --helpfulness-mode anchor --helpfulness-data $IFEVAL --helpfulness-limit 64"
SNR="--advantage-estimator snr_scaling --snr-mode soft --snr-z 2.0"
RES=/tmp/battery8_results.txt
echo "battery8 start $(date) EPOCHS=$EPOCHS (=$((EPOCHS*100)) steps)" > $RES

run_one() {
  local RUN="$1"; shift
  $PY scripts/eval_awareness/train_evalaware.py rl $COMMON --run-name "$RUN" "$@" > /tmp/$RUN.log 2>&1 || echo "[$RUN] train exit $?" >> $RES
  local CKPT
  CKPT=$(grep -oE "tinker://[a-z0-9-]+:train:0/sampler_weights/evalaware_rlct_${RUN}" /tmp/$RUN.log | tail -1)
  if [ -z "$CKPT" ]; then echo "[$RUN] NO CKPT" >> $RES; tail -3 /tmp/$RUN.log >> $RES; return; fi
  $PY scripts/eval_awareness/calibrate.py --models "${QWEN}::${CKPT}" --data $F6 --behavior-judge misalignment --no-recognition --limit 100 --n-samples 3 --max-new-tokens 1024 --concurrency 6 --out /tmp/${RUN}_gap.json > /tmp/${RUN}_gapeval.log 2>&1 || true
  local NC; NC=$(grep -oE "natural=[0-9.]+ cued=[0-9.]+ gap=[+-][0-9.]+" /tmp/${RUN}_gapeval.log | tail -1)
  $PY scripts/eval_awareness/overrefusal_eval.py --base-model "$QWEN" --models "${RUN}=${CKPT}" --n-samples 3 --max-new-tokens 1024 --concurrency 6 --out /tmp/qwen_overrefusal.json > /tmp/${RUN}_ovreval.log 2>&1 || true
  local XS EAB
  XS=$(grep "xstest_safe" /tmp/${RUN}_ovreval.log | grep -oE "over-refusal=[0-9.]+")
  EAB=$(grep "eab_capability" /tmp/${RUN}_ovreval.log | grep -oE "over-refusal=[0-9.]+")
  echo "[$RUN] F6 $NC | XSTest $XS  EAB-cap $EAB  ckpt=$CKPT" >> $RES
}

batch() {  # launch all args in parallel, wait for all
  echo ">>> BATCH @ $(date +%T): $(for s in "$@"; do echo -n "${s%%::*} "; done)" >> $RES
  for spec in "$@"; do
    name="${spec%%::*}"; args="${spec#*::}"
    run_one "$name" $args &
  done
  wait
}

# Batch A — direction axis
batch \
  "qwen-b0::--anchor-weight 0.5 $SNR --kl-coef 0.03 $IFANCHOR" \
  "qwen-sym::--anchor-weight 0.0 $SNR --kl-coef 0.03 $IFANCHOR" \
  "qwen-aw08::--anchor-weight 0.8 $SNR --kl-coef 0.03 $IFANCHOR"
# Batch B — KL axis + GRPO
batch \
  "qwen-kl01::--anchor-weight 0.5 $SNR --kl-coef 0.01 $IFANCHOR" \
  "qwen-kl10::--anchor-weight 0.5 $SNR --kl-coef 0.1 $IFANCHOR" \
  "qwen-grpo::--anchor-weight 0.5 --advantage-estimator grpo_normalized --kl-coef 0.03 $IFANCHOR"
# Batch C — matched_pair (32-wrapper F6 family, sycophancy-style, pull cued→natural) + anchor ablation.
# matched_pair overrides rollouts to 2/cue (argparse last-wins over COMMON's 32) and uses no IFEval anchor.
batch \
  "qwen-f6-matched::--f6-cues 32 --advantage-estimator matched_pair --anchor-weight 0.5 --n-train-rollouts 2 --n-consistency-rollouts 2 --kl-coef 0.03" \
  "qwen-noanchor::--anchor-weight 0.5 $SNR --kl-coef 0.03"
echo "=== BATTERY8 DONE $(date) ===" >> $RES
