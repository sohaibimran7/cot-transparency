#!/bin/bash
# Canonical eval-awareness F6 run: redesigned SNR (unit-variance GRPO × SNR gate), paper-grade rollouts.
# Deploy-anchor 0.5, KL 0.03, no helpfulness term, 128/128/128 rollouts, batch 4, default lr (~5e-4 —
# new snr is grpo-scale, no bump needed), normalize=pooled (default). 2 epochs = 50 steps; checkpoint
# every 10 → eval gap + over-refusal at each. Training grader = nano; eval grader = mini (calibrate default).
# RESULT (§3f): pooled closes the gap but over-refuses (EAB-cap 0→48%). See run_snr_peritem.sh for the per_item twin.
set -uo pipefail
cd /Users/jannes/dev/MATS/rmct
PY=.venv/bin/python
QWEN="Qwen/Qwen3-30B-A3B-Instruct-2507"
F6="dataset_dumps/eval_awareness/eval_cues/evalawarebench_F6.jsonl"
RUN="qwen-f6-snr-v2"
METRICS="logs/evalaware_rlct/$RUN/metrics.jsonl"
RES=/tmp/snr_canonical_results.txt
echo "snr canonical start $(date)" > $RES

# ── Train ────────────────────────────────────────────────────────────────────
$PY scripts/eval_awareness/train_evalaware.py rl \
  --model "$QWEN" --data "$F6" --limit 100 \
  --anchor-weight 0.5 \
  --advantage-estimator snr_scaling --snr-mode soft --snr-z 2.0 \
  --kl-coef 0.03 \
  --n-ref-rollouts 128 --n-train-rollouts 128 --n-consistency-rollouts 128 \
  --n-helpfulness-rollouts 0 --max-new-tokens 1024 --lora-rank 8 \
  --batch-size 4 --n-epochs 2 --checkpoint-every 10 -y \
  --grader-model gpt-5.4-nano --experiment-name evalaware_rlct --run-name "$RUN" \
  > /tmp/$RUN.log 2>&1 || echo "[$RUN] train exit $?" >> $RES

# ── Collect checkpoints (intermediate steps from metrics.jsonl + the final) ────
FINAL=$(grep -oE "tinker://[a-z0-9-]+:train:0/sampler_weights/evalaware_rlct_${RUN}" /tmp/$RUN.log | tail -1)
CKPTS=$($PY - "$METRICS" "$FINAL" <<'PY'
import json,sys
metrics,final=sys.argv[1],sys.argv[2]
seen=[]
try:
    for l in open(metrics):
        try: d=json.loads(l)
        except: continue
        if "checkpoint" in d: seen.append((d.get("step"), d["checkpoint"]))
except FileNotFoundError: pass
if final: seen.append(("final", final))
for step,uri in seen:
    print(f"{step}\t{uri}")
PY
)
echo "$CKPTS" | sed 's/^/  ckpt: /' >> $RES

# ── Eval each checkpoint: F6 gap + over-refusal (XSTest / EAB-cap) ─────────────
echo "$CKPTS" | while IFS=$'\t' read -r STEP URI; do
  [ -n "$URI" ] || continue
  $PY scripts/eval_awareness/calibrate.py --models "${QWEN}::${URI}" --data "$F6" \
      --behavior-judge misalignment --no-recognition --limit 100 --n-samples 3 \
      --max-new-tokens 1024 --concurrency 6 --out /tmp/snrcanon_${STEP}_gap.json \
      > /tmp/snrcanon_${STEP}_gap.log 2>&1 || true
  NC=$(grep -oE "natural=[0-9.]+ cued=[0-9.]+ gap=[+-][0-9.]+" /tmp/snrcanon_${STEP}_gap.log | tail -1)
  $PY scripts/eval_awareness/overrefusal_eval.py --base-model "$QWEN" --models "snr_${STEP}=${URI}" \
      --n-samples 3 --max-new-tokens 1024 --concurrency 6 --out /tmp/snrcanon_overrefusal.json \
      > /tmp/snrcanon_${STEP}_ovr.log 2>&1 || true
  XS=$(grep "xstest_safe" /tmp/snrcanon_${STEP}_ovr.log | grep -oE "over-refusal=[0-9.]+")
  EAB=$(grep "eab_capability" /tmp/snrcanon_${STEP}_ovr.log | grep -oE "over-refusal=[0-9.]+")
  echo "[step $STEP] F6 $NC | XSTest $XS  EAB-cap $EAB" >> $RES
done
echo "=== SNR CANONICAL DONE $(date) ===" >> $RES
