"""Paper-style figure for the SNR-scaling RLCT experiment.

Eval results from logs/snr_scaling_exp (Llama-3.1-8B, cot, TruthfulQA, limit 200,
5 biases; trained on MMLU suggested_answer). BIR numbers are computed LIVE from the
.eval logs via extract_bir3.compute_snr_scaling_bir() — net population BIR
(mean biased - mean unbiased, per-question matched). Requires logs/snr_scaling_exp present.
See RESULTS.md for provenance.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# extract_bir3 is a sibling script; put its dir on the path so the import resolves
# regardless of the caller's cwd / invocation style.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_bir3 import BIASES, compute_snr_scaling_bir, heldout_avg

OUT = Path("sycophancy_eval_inspect/plots/snr_scaling_exp")
OUT.mkdir(parents=True, exist_ok=True)

models = ["Base", "GRPO", "SNR", "SNR+Pop"]
colors = {"Base": "#999999", "GRPO": "#ff7f0e", "SNR": "#1f77b4", "SNR+Pop": "#2ca02c"}
mc = [colors[m] for m in models]

# --- BIR by bias type (trained: Sugg. Answer; rest held-out) ---
# Single source of truth: computed from .eval logs via the same functions the
# extract_bir3 table prints (incl. heldout_avg), so figure and table can never disagree.
biases = ["Sugg.\nAnswer*", "Wrong\nFS", "Argument", "Fact", "Squares", "Held-out\nAvg"]
_bir, _ub = compute_snr_scaling_bir()
_SHORT = {"Base": "base", "GRPO": "grpo", "SNR": "snr", "SNR+Pop": "snr_pop"}


def _row(short):
    row = _bir[short]
    # Missing (model, bias) cells come back as None; use NaN so matplotlib renders a gap
    # instead of crashing, mirroring the table's "NA".
    per_bias = [row[b] if row.get(b) is not None else np.nan for b in BIASES]  # canonical order
    ho = heldout_avg(row)
    return per_bias + [ho if ho is not None else np.nan]                       # + held-out avg column


bir = {label: _row(short) for label, short in _SHORT.items()}
# --- capability / stability ---
accuracy = {"Base": 0.438, "GRPO": 0.288, "SNR": 0.449, "SNR+Pop": 0.457}
kl_base  = {"Base": 0.0,   "GRPO": 0.101, "SNR": 0.020, "SNR+Pop": 0.0078}

fig, (axB, axA, axK) = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={"width_ratios": [3, 1.25, 1.25]})

# Panel 1: BIR grouped bars
x = np.arange(len(biases)); w = 0.2
for i, m in enumerate(models):
    axB.bar(x + (i - 1.5) * w, bir[m], w, label=m, color=mc[i], edgecolor="white", linewidth=0.4)
axB.axvspan(-0.5, 0.5, color="0.92", zorder=0)  # highlight trained bias
axB.set_xticks(x); axB.set_xticklabels(biases, fontsize=9)
axB.set_ylabel("BIR  (bias influence rate)", fontsize=11)
axB.set_title("Bias influence rate by bias type  (lower = less sycophantic)", fontsize=11)
axB.legend(frameon=False, fontsize=9, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.16))
axB.axhline(0, color="0.6", lw=0.8)
axB.set_ylim(-0.06, 0.34); axB.grid(axis="y", alpha=0.25)
axB.annotate("trained\nbias", (0, 0.32), ha="center", va="top", fontsize=8, color="0.4")

# Panel 2: Accuracy (capability)
xa = np.arange(len(models))
axA.bar(xa, [accuracy[m] for m in models], color=mc, edgecolor="white")
axA.axhline(accuracy["Base"], ls="--", color="0.5", lw=1)
axA.set_xticks(xa); axA.set_xticklabels(models, rotation=30, ha="right", fontsize=9)
axA.set_ylabel("Accuracy", fontsize=11); axA.set_ylim(0, 0.55)
axA.set_title("Task accuracy\n(GRPO collapses)", fontsize=11)
for i, m in enumerate(models):
    axA.text(i, accuracy[m] + 0.01, f"{accuracy[m]:.3f}", ha="center", fontsize=8)
axA.grid(axis="y", alpha=0.25)

# Panel 3: KL drift from base (stability)
axK.bar(xa, [kl_base[m] for m in models], color=mc, edgecolor="white")
axK.set_xticks(xa); axK.set_xticklabels(models, rotation=30, ha="right", fontsize=9)
axK.set_ylabel("KL(policy ‖ base), final", fontsize=11)
axK.set_title("Drift from base\n(lower = more stable)", fontsize=11)
for i, m in enumerate(models):
    axK.text(i, kl_base[m] + 0.002, f"{kl_base[m]:.3f}", ha="center", fontsize=8)
axK.grid(axis="y", alpha=0.25)

fig.suptitle("RLCT advantage estimator: GRPO (sign-only) vs. SNR-scaling  —  Llama-3.1-8B, suggested_answer→TruthfulQA",
             fontsize=12, y=1.02)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"snr_scaling_summary.{ext}", dpi=150, bbox_inches="tight")
print("wrote", OUT / "snr_scaling_summary.png")
