"""Paper-style figure for the shrinkage RLCT experiment.

Numbers are the eval results from logs/shrinkage_exp (Llama-3.1-8B, cot, TruthfulQA,
limit 200, 5 biases; trained on MMLU suggested_answer). Hardcoded here so the figure
is reproducible without re-reading .eval files. See RESULTS.md for provenance.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path("sycophancy_eval_inspect/plots/shrinkage_exp")
OUT.mkdir(parents=True, exist_ok=True)

models = ["Base", "GRPO", "Shrink", "Shrink+Pop"]
colors = {"Base": "#999999", "GRPO": "#ff7f0e", "Shrink": "#1f77b4", "Shrink+Pop": "#2ca02c"}
mc = [colors[m] for m in models]

# --- BIR by bias type (trained: Sugg. Answer; rest held-out) ---
# Verified from .eval logs via scripts/tinker_training/extract_bir3.py
# (BIR = mean matches_bias[biased] - mean matches_bias[unbiased]).
biases = ["Sugg.\nAnswer*", "Wrong\nFS", "Argument", "Fact", "Squares", "Held-out\nAvg"]
bir = {
    "Base":       [0.251, 0.299, 0.013, 0.144, 0.263, 0.180],
    "GRPO":       [-0.013, 0.094, -0.014, 0.026, 0.100, 0.051],
    "Shrink":     [0.071, 0.167, -0.012, 0.048, 0.103, 0.077],
    "Shrink+Pop": [0.089, 0.154, -0.001, 0.117, 0.183, 0.113],
}
# --- capability / stability ---
accuracy = {"Base": 0.438, "GRPO": 0.288, "Shrink": 0.449, "Shrink+Pop": 0.457}
kl_base  = {"Base": 0.0,   "GRPO": 0.101, "Shrink": 0.020, "Shrink+Pop": 0.0078}

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

fig.suptitle("RLCT advantage estimator: GRPO (sign-only) vs. SNR-shrinkage  —  Llama-3.1-8B, suggested_answer→TruthfulQA",
             fontsize=12, y=1.02)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"shrinkage_summary.{ext}", dpi=150, bbox_inches="tight")
print("wrote", OUT / "shrinkage_summary.png")
