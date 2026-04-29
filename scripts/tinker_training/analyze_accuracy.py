#!/usr/bin/env python3
"""
Compute accuracy on unbiased eval samples per model, grouped by sweep.

Per Sohaib (Apr 21 meeting): check if training degrades model performance on
unbiased eval — comparing base vs trained variants within each sweep.

Reads existing sweep view dirs created by analyze_by_sweep.sh.
Output: per-sweep accuracy tables + combined report.

Usage:
    python scripts/tinker_training/analyze_accuracy.py
"""

from __future__ import annotations

import glob
from pathlib import Path

from inspect_ai.log import read_eval_log


ROOT = Path("/home/prakharg/cot-transparency")
SWEEPS = [
    "rollouts",
    "n-consistency",
    "kl-coef",
    "refresh-every",
    "n-datapoints",
    "bct-instruct",
    "bct-batch-size",
    "bct-n-datapoints",
]


def compute_model_accuracy(model_dir: Path) -> dict:
    """For one model's eval dir, return {'n': int, 'correct_rate': float}
    computed ONLY on variant=unbiased samples."""
    correct_count = 0
    total = 0
    for f in sorted(model_dir.glob("*.eval")):
        try:
            log = read_eval_log(str(f))
        except Exception as e:
            print(f"  WARN: failed to read {f.name}: {e}")
            continue
        for s in log.samples or []:
            variant = s.metadata.get("variant")
            if variant != "unbiased":
                continue
            sc = (s.scores or {}).get("mcq_bias_scorer")
            if not sc or not isinstance(sc.value, dict):
                continue
            correct = sc.value.get("correct")
            if correct is None:
                continue
            total += 1
            if correct == 1.0:
                correct_count += 1
    return {"n": total, "correct_rate": correct_count / total if total > 0 else float("nan")}


def analyze_sweep(sweep_name: str) -> list[tuple[str, int, float]]:
    """Return list of (model_name, n, accuracy) for each model in the sweep view."""
    view_dir = ROOT / f"sycophancy_eval_inspect/logs/sweep_{sweep_name.replace('-', '_')}_view"
    if not view_dir.exists():
        print(f"  SKIP {sweep_name}: view dir not found ({view_dir})")
        return []

    rows = []
    for model_dir in sorted(view_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        stats = compute_model_accuracy(model_dir)
        rows.append((model_dir.name, stats["n"], stats["correct_rate"]))
    return rows


def format_table(rows: list[tuple[str, int, float]]) -> str:
    if not rows:
        return "  (no data)\n"
    out = [f"  {'model':<30s} {'n':>5s} {'accuracy':>10s}"]
    out.append("  " + "-" * 47)
    # Sort: base first, then alphabetically
    def sort_key(r):
        is_base = "base" in r[0]
        return (0 if is_base else 1, r[0])
    for name, n, acc in sorted(rows, key=sort_key):
        acc_str = f"{acc:.3f}" if acc == acc else "nan"
        out.append(f"  {name:<30s} {n:>5d} {acc_str:>10s}")
    return "\n".join(out) + "\n"


def main():
    print("=" * 72)
    print("Unbiased-eval accuracy by sweep (variant=unbiased only)")
    print("Per Sohaib (Apr 21): check if training degrades accuracy on unbiased.")
    print("=" * 72)

    combined_lines = []
    combined_lines.append("# Unbiased-eval accuracy by sweep")
    combined_lines.append("# Per Sohaib (Apr 21): check for accuracy degradation vs base")
    combined_lines.append("")

    for sweep in SWEEPS:
        print(f"\n### {sweep} ###")
        rows = analyze_sweep(sweep)
        table = format_table(rows)
        print(table)

        # Save per-sweep
        out_dir = ROOT / f"plots/{sweep}"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "accuracy_unbiased.txt").write_text(
            f"# Unbiased-eval accuracy for sweep: {sweep}\n"
            f"# variant=unbiased only, mcq_bias_scorer.correct\n\n"
            + table
        )

        combined_lines.append(f"### {sweep} ###")
        combined_lines.append(table)

    # Save combined
    combined_path = ROOT / "plots/accuracy_all_sweeps.txt"
    combined_path.write_text("\n".join(combined_lines))
    print(f"\nSaved combined report to: {combined_path}")


if __name__ == "__main__":
    main()
