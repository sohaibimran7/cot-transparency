#!/bin/bash
# Follow-up to battery8. Clean recipe = Deploy 0.5 + snr_scaling + KL, NO IFEval anchor (§3b finding).
# Arms:
#   qwen-const : --lr-schedule constant   → tests "snr_scaling ⇒ no LR schedule needed" (user hypothesis)
#   qwen-kl01n : KL 0.01  (re-run wifi-killed, now no-IFanchor → comparable to noanchor)
#   qwen-kl10n : KL 0.1   (re-run wifi-killed, now no-IFanchor)
# Reference midpoint already done: qwen-noanchor (KL 0.03, linear LR) = +0.190.
set -uo pipefail
cd /Users/jannes/dev/MATS/rmct
PY=.venv/bin/python
EPOCHS=${EPOCHS:-1}
QWEN="Qwen/Qwen3-30B-A3B-Instruct-2507"
F6="dataset_dumps/eval_awareness/eval_cues/evalawarebench_F6.jsonl"
COMMON="--model $QWEN --data $F6 --limit 100 --n-epochs $EPOCHS --n-ref-rollouts 32 --n-train-rollouts 32 --n-consistency-rollouts 24 --n-helpfulness-rollouts 12 --max-new-tokens 1024 --checkpoint-every 50 -y --grader-model gpt-5.4-nano --experiment-name evalaware_rlct"
SNR="--advantage-estimator snr_scaling --snr-mode soft --snr-z 2.0"
RES=/tmp/lrkl_results.txt
echo "lr-kl batch start $(date) EPOCHS=$EPOCHS" > $RES

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

echo ">>> BATCH @ $(date +%T): const kl01n kl10n" >> $RES
run_one "qwen-const" --anchor-weight 0.5 $SNR --lr-schedule constant --kl-coef 0.03 &
run_one "qwen-kl01n" --anchor-weight 0.5 $SNR --kl-coef 0.01 &
run_one "qwen-kl10n" --anchor-weight 0.5 $SNR --kl-coef 0.1 &
wait
echo "=== LR-KL BATCH DONE $(date) ===" >> $RES
