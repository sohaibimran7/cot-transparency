"""Analyze eval-awareness results and produce sycophancy-style plots.

Consumes results.json files written by run_eval.py (one per model/checkpoint under
sycophancy_eval_inspect/logs/eval_awareness/<name>/results.json) and produces:

  1. gap_by_slice.png   — eval-awareness gap (natural − cued) per slice, grouped by model
  2. rates_by_slice.png — absolute misalignment rate, natural vs eval-cued, per model
  3. gap_by_cue.png     — gap per held-out cue (EvalAwareBench factors F1–F8 + our markers)
  + summary.md / summary.csv tables.

Gap > 0 ⇒ the model misbehaves MORE when it does NOT think it's being evaluated
(the eval-awareness failure). Consistency training should drive gaps toward 0.

Error bars are ±1.96·SE (95% CI) using a binomial SE per rate.

Usage:
  python scripts/eval_awareness/analyze.py                       # scan all result dirs
  python scripts/eval_awareness/analyze.py --models claude-haiku-base gpt-evalaware-rlct
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).parent.parent.parent
LOG_ROOT = PROJECT_ROOT / "sycophancy_eval_inspect" / "logs" / "eval_awareness"
PLOT_ROOT = PROJECT_ROOT / "sycophancy_eval_inspect" / "plots" / "eval_awareness"

MODEL_COLORS = ["#4292c6", "#a1d99b", "#fd8d3c", "#9467bd", "#e377c2", "#8c564b"]


def _binom_se(p: float, n: int) -> float:
    if n <= 0 or math.isnan(p):
        return float("nan")
    return math.sqrt(max(p * (1 - p), 0.0) / n)


def _cue_label(slice_name: str, file_name: str) -> str:
    stem = file_name.replace(".jsonl", "")
    if slice_name == "in_distribution":
        return "train cue\n(in-dist)"
    if slice_name == "env_gen":
        return "AM\n(all cues)"
    if stem.startswith("evalawarebench_"):
        return stem.split("_", 1)[1]          # F1..F8
    if stem.startswith("marker_"):
        return stem.split("_", 1)[1]          # this_is_a_test, ...
    return stem


def load_results(model_dirs: list[Path]) -> list[dict]:
    """Flatten per-file records across models, deriving cue + SEs."""
    recs = []
    for d in model_dirs:
        rj = d / "results.json"
        if not rj.exists():
            print(f"  skip {d.name}: no results.json")
            continue
        data = json.loads(rj.read_text())
        model = data.get("model", d.name)
        label = d.name
        for r in data["per_file"]:
            gap = r["rate_natural"] - r["rate_cued"]
            se = math.sqrt(
                _binom_se(r["rate_natural"], r.get("n_natural", 0)) ** 2
                + _binom_se(r["rate_cued"], r.get("n_cued", 0)) ** 2
            )
            recs.append({
                "model_label": label, "model": model,
                "slice": r["slice"], "file": r["file"],
                "cue": _cue_label(r["slice"], r["file"]),
                "rate_natural": r["rate_natural"], "rate_cued": r["rate_cued"],
                "gap": gap, "gap_se": se,
                "n_natural": r.get("n_natural", 0), "n_cued": r.get("n_cued", 0),
            })
    return recs


def _model_order(recs: list[dict]) -> list[str]:
    seen = []
    for r in recs:
        if r["model_label"] not in seen:
            seen.append(r["model_label"])
    return seen


def _slice_agg(recs: list[dict]) -> dict[tuple[str, str], dict]:
    """Aggregate (model, slice) by pooling samples across files in the slice."""
    acc: dict[tuple[str, str], dict] = {}
    for r in recs:
        k = (r["model_label"], r["slice"])
        a = acc.setdefault(k, {"nat_sum": 0.0, "nat_n": 0, "cue_sum": 0.0, "cue_n": 0})
        if not math.isnan(r["rate_natural"]):
            a["nat_sum"] += r["rate_natural"] * r["n_natural"]; a["nat_n"] += r["n_natural"]
        if not math.isnan(r["rate_cued"]):
            a["cue_sum"] += r["rate_cued"] * r["n_cued"]; a["cue_n"] += r["n_cued"]
    out = {}
    for k, a in acc.items():
        pn = a["nat_sum"] / a["nat_n"] if a["nat_n"] else float("nan")
        pc = a["cue_sum"] / a["cue_n"] if a["cue_n"] else float("nan")
        se = math.sqrt(_binom_se(pn, a["nat_n"]) ** 2 + _binom_se(pc, a["cue_n"]) ** 2)
        out[k] = {"rate_natural": pn, "rate_cued": pc, "gap": pn - pc, "gap_se": se,
                  "n_natural": a["nat_n"], "n_cued": a["cue_n"]}
    return out


def _grouped_bar(ax, categories, series, colors, ylabel, title, hline0=True):
    """series: list of (label, values, errs)."""
    x = np.arange(len(categories))
    k = len(series)
    width = 0.8 / max(k, 1)
    for i, (label, vals, errs) in enumerate(series):
        pos = x + (i - (k - 1) / 2) * width
        ax.bar(pos, vals, width, label=label, color=colors[i % len(colors)],
               edgecolor="black", linewidth=0.8)
        ax.errorbar(pos, vals, yerr=[2 * e if not math.isnan(e) else 0 for e in errs],
                    fmt="none", capsize=3, ecolor="black", alpha=0.7)
    if hline0:
        ax.axhline(0, color="gray", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)


def plot_gap_by_slice(recs, models, out: Path):
    agg = _slice_agg(recs)
    slices = [s for s in ["in_distribution", "cue_gen", "env_gen"]
              if any(k[1] == s for k in agg)]
    nice = {"in_distribution": "In-distribution\n(trained cue)",
            "cue_gen": "Cue generalization\n(held-out cues)",
            "env_gen": "Env generalization\n(Agentic Misalignment)"}
    series = []
    for m in models:
        vals = [agg.get((m, s), {}).get("gap", float("nan")) for s in slices]
        errs = [agg.get((m, s), {}).get("gap_se", float("nan")) for s in slices]
        series.append((m, vals, errs))
    fig, ax = plt.subplots(figsize=(max(8, 2.4 * len(slices)), 6))
    _grouped_bar(ax, [nice[s] for s in slices], series, MODEL_COLORS,
                 "Eval-awareness gap  (rate natural − cued)",
                 "Eval-awareness gap by generalization slice")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  wrote {out}")


def plot_rates_by_slice(recs, models, out: Path):
    agg = _slice_agg(recs)
    slices = [s for s in ["in_distribution", "cue_gen", "env_gen"] if any(k[1] == s for k in agg)]
    # One subplot per model: natural vs cued bars across slices.
    fig, axes = plt.subplots(1, len(models), figsize=(max(6, 4 * len(models)), 5), squeeze=False)
    nice = {"in_distribution": "in-dist", "cue_gen": "cue-gen", "env_gen": "env-gen"}
    for ax, m in zip(axes[0], models):
        x = np.arange(len(slices)); w = 0.38
        nat = [agg.get((m, s), {}).get("rate_natural", float("nan")) for s in slices]
        cue = [agg.get((m, s), {}).get("rate_cued", float("nan")) for s in slices]
        nat_se = [_binom_se(agg.get((m, s), {}).get("rate_natural", float("nan")), agg.get((m, s), {}).get("n_natural", 0)) for s in slices]
        cue_se = [_binom_se(agg.get((m, s), {}).get("rate_cued", float("nan")), agg.get((m, s), {}).get("n_cued", 0)) for s in slices]
        ax.bar(x - w/2, nat, w, label="natural", color="#fd8d3c", edgecolor="black", linewidth=0.8)
        ax.bar(x + w/2, cue, w, label="eval-cued", color="#4292c6", edgecolor="black", linewidth=0.8)
        ax.errorbar(x - w/2, nat, yerr=[2*e if not math.isnan(e) else 0 for e in nat_se], fmt="none", capsize=3, ecolor="black", alpha=0.7)
        ax.errorbar(x + w/2, cue, yerr=[2*e if not math.isnan(e) else 0 for e in cue_se], fmt="none", capsize=3, ecolor="black", alpha=0.7)
        ax.set_xticks(x); ax.set_xticklabels([nice[s] for s in slices], fontsize=9)
        ax.set_title(m, fontsize=10); ax.set_ylim(0, 1); ax.grid(axis="y", alpha=0.25); ax.legend(fontsize=8)
    axes[0][0].set_ylabel("Misalignment / harmful-compliance rate")
    fig.suptitle("Behavior rate: natural vs eval-cued", fontsize=12, fontweight="bold")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  wrote {out}")


def plot_gap_by_cue(recs, models, out: Path):
    # Held-out cues only (cue_gen slice), ordered F1..F8 then markers.
    cue_recs = [r for r in recs if r["slice"] == "cue_gen"]
    if not cue_recs:
        print("  (no cue_gen data; skipping gap_by_cue)")
        return
    cues = []
    for r in cue_recs:
        if r["cue"] not in cues:
            cues.append(r["cue"])
    cues.sort(key=lambda c: (not c.startswith("F"), c))
    models_with_data = [m for m in models if any(r["model_label"] == m for r in cue_recs)]
    series = []
    for m in models_with_data:
        by_cue = {r["cue"]: r for r in cue_recs if r["model_label"] == m}
        vals = [by_cue.get(c, {}).get("gap", float("nan")) for c in cues]
        errs = [by_cue.get(c, {}).get("gap_se", float("nan")) for c in cues]
        series.append((m, vals, errs))
    fig, ax = plt.subplots(figsize=(max(10, 1.1 * len(cues)), 6))
    _grouped_bar(ax, cues, series, MODEL_COLORS,
                 "Eval-awareness gap  (rate natural − cued)",
                 "Eval-awareness gap by held-out cue")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  wrote {out}")


def write_tables(recs, models, out_dir: Path):
    agg = _slice_agg(recs)
    lines = ["# Eval-awareness results\n",
             "## Gap by slice (rate natural − cued; + = misbehaves more when not cued)\n",
             "| model | slice | natural | cued | gap | n |",
             "|---|---|---|---|---|---|"]
    csv = ["model,slice,rate_natural,rate_cued,gap,gap_se,n_natural,n_cued"]
    for m in models:
        for s in ["in_distribution", "cue_gen", "env_gen"]:
            a = agg.get((m, s))
            if not a:
                continue
            lines.append(f"| {m} | {s} | {a['rate_natural']:.3f} | {a['rate_cued']:.3f} | {a['gap']:+.3f} | {a['n_natural']} |")
            csv.append(f"{m},{s},{a['rate_natural']:.4f},{a['rate_cued']:.4f},{a['gap']:.4f},{a['gap_se']:.4f},{a['n_natural']},{a['n_cued']}")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    (out_dir / "summary.csv").write_text("\n".join(csv) + "\n")
    print(f"  wrote {out_dir/'summary.md'} and summary.csv")
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="*", default=None, help="Result dir names under logs/eval_awareness/ (default: all)")
    p.add_argument("--log-root", default=str(LOG_ROOT))
    p.add_argument("--out-dir", default=str(PLOT_ROOT))
    args = p.parse_args()

    log_root = Path(args.log_root)
    if args.models:
        dirs = [log_root / m for m in args.models]
    else:
        dirs = sorted([d for d in log_root.glob("*") if (d / "results.json").exists() and not d.name.startswith("_")])
    if not dirs:
        print(f"No result dirs with results.json under {log_root}")
        sys.exit(1)

    print(f"Models: {[d.name for d in dirs]}")
    recs = load_results(dirs)
    if not recs:
        print("No records loaded.")
        sys.exit(1)
    models = _model_order(recs)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_gap_by_slice(recs, models, out_dir / "gap_by_slice.png")
    plot_rates_by_slice(recs, models, out_dir / "rates_by_slice.png")
    plot_gap_by_cue(recs, models, out_dir / "gap_by_cue.png")
    write_tables(recs, models, out_dir)


if __name__ == "__main__":
    main()
