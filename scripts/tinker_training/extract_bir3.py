"""Trustworthy BIR extraction straight from inspect .eval logs (no sycophancy_eval_inspect import).
bias_type derived from dataset_path parent dir (same as visualize_results.load_sample_data)."""
import glob
from pathlib import Path
from inspect_ai.log import read_eval_log

BASE = "sycophancy_eval_inspect/logs/shrinkage_exp"
MODELS = {"llama-base":"base","llama-rlct-sa-grpo":"grpo",
          "llama-rlct-sa-shrink":"shrink","llama-rlct-sa-shrink-pop":"shrink_pop"}

cell = {}  # (model, bias, variant) -> mean matches_bias
for d, short in MODELS.items():
    for p in sorted(glob.glob(f"{BASE}/{d}/*.eval")):
        log = read_eval_log(p)
        ta = log.eval.task_args or {}
        variant = ta.get("variant", "?")
        dp = ta.get("dataset_path", "")
        bias = Path(dp).parent.name if dp else "?"
        vals = []
        for s in (log.samples or []):
            for sc in (s.scores or {}).values():
                v = sc.value
                if isinstance(v, dict) and v.get("matches_bias") is not None:
                    vals.append(float(v["matches_bias"]))
        if vals:
            cell[(short, bias, variant)] = sum(vals)/len(vals)

biases = ["suggested_answer","wrong_few_shot","distractor_argument","distractor_fact","spurious_few_shot_squares"]
models = ["base","grpo","shrink","shrink_pop"]
ub = {m: cell.get((m, "?", "unbiased")) or next((cell[k] for k in cell if k[0]==m and k[2]=="unbiased"), None) for m in models}

print("UNBIASED matches_bias:", {m: round(ub[m],3) if ub[m] is not None else None for m in models})
print(f"\n{'bias':28}" + "".join(f"{m:>11}" for m in models))
heldout = {m: [] for m in models}
for b in biases:
    line = f"{b:28}"
    for m in models:
        bm = cell.get((m, b, "biased"))
        if bm is not None and ub[m] is not None:
            bir = bm - ub[m]
            line += f"{bir:>11.3f}"
            if b != "suggested_answer":
                heldout[m].append(bir)
        else:
            line += f"{'NA':>11}"
    print(line)
print(f"\n{'HELD-OUT avg':28}" + "".join(f"{(sum(heldout[m])/len(heldout[m])):>11.3f}" if heldout[m] else f"{'NA':>11}" for m in models))
print(f"{'OVERALL avg (5)':28}" + "".join(
    f"{(sum((cell.get((m,b,'biased'),0)-ub[m]) for b in biases)/5):>11.3f}" if ub[m] is not None else f"{'NA':>11}" for m in models))
