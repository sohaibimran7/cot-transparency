"""Plot the cross-model eval-awareness calibration (behaviour gap vs recognition).

Consumes the JSON files written by calibrate.py (keyed by model id, each value has
compliance_natural/compliance_cued/behavior_gap/recognition_natural/recognition_cued/n).
Each input is tagged with an *environment* label (e.g. ``val`` = naturalised-harmful
held-out set, ``AM`` = Agentic Misalignment), so a model can appear once per env.

Reproduces the paper's decomposition (recognition ≠ behaviour change):
  1. calib_scatter.png   — x = behaviour gap (natural − cued), y = recognition(natural).
                           Far-right + low-y = "recognises eval yet still changes behaviour".
  2. calib_behavior.png  — behaviour gap per model, grouped by environment (with 95% CI).
  3. calib_recognition.png — recognition rate natural vs cued per model.
  + calibration_summary.md table.

Usage:
  python scripts/eval_awareness/plot_calibration.py \
      --inputs val:sycophancy_eval_inspect/logs/eval_awareness/calibration.json \
               val:/tmp/calibration_kimi.json \
               AM:/tmp/calibration_am.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
PLOT_ROOT = PROJECT_ROOT / "sycophancy_eval_inspect" / "plots" / "eval_awareness"

ENV_COLORS = {"val": "#4292c6", "AM": "#fd8d3c"}
DEFAULT_COLORS = ["#4292c6", "#fd8d3c", "#a1d99b", "#9467bd", "#e377c2"]


def _short(model: str) -> str:
    return model.split("/")[-1].replace("-Instruct", "").replace("meta-llama-", "")


def _binom_se(p: float, n: int) -> float:
    if not n or p != p:  # nan check
        return float("nan")
    return math.sqrt(max(p * (1 - p), 0.0) / n)


def load_inputs(specs: list[str]) -> list[dict]:
    """specs: ['env:path', ...] → flat records {env, model, ...}."""
    recs = []
    for spec in specs:
        env, _, path = spec.partition(":")
        p = Path(path)
        if not p.exists():
            print(f"  skip (missing): {spec}")
            continue
        data = json.loads(p.read_text())
        for model, r in data.items():
            recs.append({"env": env, "model": model, "short": _short(model), **r})
    return recs


def _env_order(recs):
    seen = []
    for r in recs:
        if r["env"] not in seen:
            seen.append(r["env"])
    return seen


def _model_order(recs):
    seen = []
    for r in recs:
        if r["short"] not in seen:
            seen.append(r["short"])
    return seen


def plot_scatter(recs, out: Path):
    fig, ax = plt.subplots(figsize=(9, 6.5))
    envs = _env_order(recs)
    markers = {e: m for e, m in zip(envs, ["o", "s", "^", "D"])}
    for r in recs:
        c = ENV_COLORS.get(r["env"], "#666")
        ax.scatter(r["behavior_gap"], r["recognition_natural"], s=140,
                   color=c, marker=markers[r["env"]], edgecolor="black", linewidth=1.0, zorder=3)
        ax.annotate(r["short"], (r["behavior_gap"], r["recognition_natural"]),
                    xytext=(6, 5), textcoords="offset points", fontsize=8)
    ax.axvline(0, color="gray", lw=1.0)
    ax.set_xlabel("Behaviour gap  (harmful rate: natural − eval-cued)")
    ax.set_ylabel("Recognition (natural)  — reads as an evaluation")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Eval-awareness decomposition: recognition vs behaviour change",
                 fontsize=12, fontweight="bold")
    handles = [plt.Line2D([0], [0], marker=markers[e], color="w", markerfacecolor=ENV_COLORS.get(e, "#666"),
                          markeredgecolor="black", markersize=11, label=f"{e} env") for e in envs]
    ax.legend(handles=handles, fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  wrote {out}")


def _grouped_bar(recs, value_key, se_fn, ylabel, title, out: Path, hline0=True):
    models = _model_order(recs)
    envs = _env_order(recs)
    by = {(r["short"], r["env"]): r for r in recs}
    x = np.arange(len(models)); k = len(envs); width = 0.8 / max(k, 1)
    fig, ax = plt.subplots(figsize=(max(8, 1.7 * len(models)), 6))
    for i, e in enumerate(envs):
        pos = x + (i - (k - 1) / 2) * width
        vals = [by.get((m, e), {}).get(value_key, float("nan")) for m in models]
        errs = [se_fn(by.get((m, e), {})) for m in models]
        ax.bar(pos, vals, width, label=f"{e} env", color=ENV_COLORS.get(e, DEFAULT_COLORS[i % len(DEFAULT_COLORS)]),
               edgecolor="black", linewidth=0.8)
        ax.errorbar(pos, vals, yerr=[1.96 * s if s == s else 0 for s in errs],
                    fmt="none", capsize=3, ecolor="black", alpha=0.7)
    if hline0:
        ax.axhline(0, color="gray", lw=1.0)
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=9, rotation=15, ha="right")
    ax.set_ylabel(ylabel); ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  wrote {out}")


def plot_recognition(recs, out: Path):
    """Per-model recognition: natural vs cued, faceted is overkill — one env at a time inline."""
    models = _model_order(recs)
    envs = _env_order(recs)
    by = {(r["short"], r["env"]): r for r in recs}
    fig, axes = plt.subplots(1, len(envs), figsize=(max(6, 4.2 * len(envs)), 5), squeeze=False)
    for ax, e in zip(axes[0], envs):
        x = np.arange(len(models)); w = 0.38
        nat = [by.get((m, e), {}).get("recognition_natural", float("nan")) for m in models]
        cue = [by.get((m, e), {}).get("recognition_cued", float("nan")) for m in models]
        ax.bar(x - w/2, nat, w, label="natural", color="#fd8d3c", edgecolor="black", linewidth=0.8)
        ax.bar(x + w/2, cue, w, label="eval-cued", color="#4292c6", edgecolor="black", linewidth=0.8)
        ax.set_xticks(x); ax.set_xticklabels(models, fontsize=8, rotation=20, ha="right")
        ax.set_ylim(0, 1.02); ax.set_title(f"{e} env", fontsize=10); ax.grid(axis="y", alpha=0.25); ax.legend(fontsize=8)
    axes[0][0].set_ylabel("Recognition rate (reads as eval)")
    fig.suptitle("Eval recognition: natural vs eval-cued", fontsize=12, fontweight="bold")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  wrote {out}")


def write_table(recs, out: Path):
    lines = ["# Eval-awareness calibration (base models, no training)\n",
             "Behaviour gap = harmful rate natural − eval-cued. Recognition = third-person eval-recognition judge.\n",
             "| model | env | beh gap | comp nat | comp cued | rec nat | rec cued | n |",
             "|---|---|---|---|---|---|---|---|"]
    for r in sorted(recs, key=lambda r: (r["env"], -r.get("behavior_gap", 0))):
        lines.append(
            f"| {r['short']} | {r['env']} | {r['behavior_gap']:+.3f} | {r['compliance_natural']:.3f} | "
            f"{r['compliance_cued']:.3f} | {r['recognition_natural']:.3f} | {r['recognition_cued']:.3f} | {r['n']} |")
    out.write_text("\n".join(lines) + "\n")
    print(f"  wrote {out}")
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", nargs="+", required=True, help="env:path.json specs (env e.g. val / AM)")
    p.add_argument("--out-dir", default=str(PLOT_ROOT))
    args = p.parse_args()

    recs = load_inputs(args.inputs)
    if not recs:
        raise SystemExit("No calibration records loaded.")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    plot_scatter(recs, out_dir / "calib_scatter.png")
    _grouped_bar(recs, "behavior_gap",
                 lambda r: math.sqrt(_binom_se(r.get("compliance_natural", float("nan")), r.get("n", 0)) ** 2
                                     + _binom_se(r.get("compliance_cued", float("nan")), r.get("n", 0)) ** 2),
                 "Behaviour gap (natural − cued)", "Eval-awareness behaviour gap by model",
                 out_dir / "calib_behavior.png")
    plot_recognition(recs, out_dir / "calib_recognition.png")
    write_table(recs, out_dir / "calibration_summary.md")


if __name__ == "__main__":
    main()
