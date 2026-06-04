#!/bin/bash
# per_item twin of run_snr_canonical.sh (pooled). IDENTICAL config except --normalize per_item,
# for a head-to-head on advantage standardization (pooled vs per-item-group) under the gated snr.
# new snr · default lr (~5e-4) · Deploy 0.5 · KL 0.03 · no helpfulness · 128/128/128 · batch 4 ·
# 50 steps · checkpoint every 10 → eval gap + over-refusal at each.
set -uo pipefail
cd /Users/jannes/dev/MATS/rmct
PY=.venv/bin/python
QWEN="Qwen/Qwen3-30B-A3B-Instruct-2507"
F6="dataset_dumps/eval_awareness/eval_cues/evalawarebench_F6.jsonl"
RUN="qwen-f6-snr-peritem"
METRICS="logs/evalaware_rlct/$RUN/metrics.jsonl"
RES=/tmp/snr_peritem_results.txt
echo "snr per_item start $(date)" > $RES

$PY scripts/eval_awareness/train_evalaware.py rl \
  --model "$QWEN" --data "$F6" --limit 100 \
  --anchor-weight 0.5 \
  --advantage-estimator snr_scaling --snr-mode soft --snr-z 2.0 \
  --normalize per_item \
  --kl-coef 0.03 \
  --n-ref-rollouts 128 --n-train-rollouts 128 --n-consistency-rollouts 128 \
  --n-helpfulness-rollouts 0 --max-new-tokens 1024 --lora-rank 8 \
  --batch-size 4 --n-epochs 2 --checkpoint-every 10 -y \
  --grader-model gpt-5.4-nano --experiment-name evalaware_rlct --run-name "$RUN" \
  > /tmp/$RUN.log 2>&1 || echo "[$RUN] train exit $?" >> $RES

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
for step,uri in seen: print(f"{step}\t{uri}")
PY
)
echo "$CKPTS" | sed 's/^/  ckpt: /' >> $RES
echo "$CKPTS" | while IFS=$'\t' read -r STEP URI; do
  [ -n "$URI" ] || continue
  $PY scripts/eval_awareness/calibrate.py --models "${QWEN}::${URI}" --data "$F6" \
      --behavior-judge misalignment --no-recognition --limit 100 --n-samples 3 \
      --max-new-tokens 1024 --concurrency 6 --out /tmp/snrpi_${STEP}_gap.json \
      > /tmp/snrpi_${STEP}_gap.log 2>&1 || true
  NC=$(grep -oE "natural=[0-9.]+ cued=[0-9.]+ gap=[+-][0-9.]+" /tmp/snrpi_${STEP}_gap.log | tail -1)
  $PY scripts/eval_awareness/overrefusal_eval.py --base-model "$QWEN" --models "snrpi_${STEP}=${URI}" \
      --n-samples 3 --max-new-tokens 1024 --concurrency 6 --out /tmp/snrpi_overrefusal.json \
      > /tmp/snrpi_${STEP}_ovr.log 2>&1 || true
  XS=$(grep "xstest_safe" /tmp/snrpi_${STEP}_ovr.log | grep -oE "over-refusal=[0-9.]+")
  EAB=$(grep "eab_capability" /tmp/snrpi_${STEP}_ovr.log | grep -oE "over-refusal=[0-9.]+")
  echo "[step $STEP] F6 $NC | XSTest $XS  EAB-cap $EAB" >> $RES
done
echo "=== SNR PER_ITEM DONE $(date) ===" >> $RES
