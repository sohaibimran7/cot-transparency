"""Standalone over-refusal comparison plot (reads overrefusal.json → overrefusal.png).

Self-contained (no dependency on the shared visualization code) so it can be regenerated
any time overrefusal_eval.py adds a checkpoint. Grouped bars: XSTest-safe + EAB-capability
over-refusal rate per model; lower = better; base line drawn for reference.

Usage:
  python scripts/eval_awareness/plot_overrefusal.py
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
LOG = PROJECT_ROOT / "sycophancy_eval_inspect" / "logs" / "eval_awareness" / "overrefusal.json"
PLOT = PROJECT_ROOT / "sycophancy_eval_inspect" / "plots" / "eval_awareness" / "overrefusal.png"

# display order + nicer labels
ORDER = [
    ("llama-base", "base"),
    ("llama-bct", "BCT\n(+Alpaca)"),
    ("llama-rlct-deploy", "RLCT\nplain"),
    ("llama-rlct-symmetric", "RLCT\nsymmetric"),
    ("llama-rlct-deploy-help", "RLCT\n+help(0.5)"),
    ("llama-rlct-deploy-help-matched", "RLCT\n+help(1.0)\nmatched"),
]
SRC = [("xstest_safe", "XSTest-safe (250)", "#4292c6"),
       ("eab_capability", "EAB-capability", "#fd8d3c")]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="inp", default=str(LOG))
    p.add_argument("--out", default=str(PLOT))
    args = p.parse_args()

    data = json.loads(Path(args.inp).read_text())
    models = [(k, lbl) for k, lbl in ORDER if k in data]
    x = np.arange(len(models)); w = 0.38
    fig, ax = plt.subplots(figsize=(max(9, 1.6 * len(models)), 6))
    for i, (key, label, color) in enumerate(SRC):
        vals = [data[k].get(key, {}).get("over_refusal_rate", float("nan")) for k, _ in models]
        pos = x + (i - 0.5) * w
        bars = ax.bar(pos, vals, w, label=label, color=color, edgecolor="black", linewidth=0.8)
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    base = data.get("llama-base", {}).get("xstest_safe", {}).get("over_refusal_rate")
    if base is not None:
        ax.axhline(base, color="#4292c6", ls="--", lw=1, alpha=0.6)
    ax.set_xticks(x); ax.set_xticklabels([lbl for _, lbl in models], fontsize=8)
    ax.set_ylabel("Over-refusal rate on benign prompts (lower = better)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Over-refusal: consistency-RL collapses to refuse-everything;\n"
                 "helpfulness cuts generic (XSTest) but not on-twin (EAB-cap) over-refusal",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(args.out, dpi=150); plt.close(fig)
    print(f"wrote {args.out}")
    for k, lbl in models:
        xs = data[k].get("xstest_safe", {}).get("over_refusal_rate")
        cap = data[k].get("eab_capability", {}).get("over_refusal_rate")
        print(f"  {k:32s} XSTest={xs:.3f}  EAB-cap={cap:.3f}")


if __name__ == "__main__":
    main()
