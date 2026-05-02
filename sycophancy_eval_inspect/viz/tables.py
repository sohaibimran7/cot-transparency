"""Table-printing helpers ported from legacy visualize_results.py.

Public API:
  - compute_bir_table(bir_df, model_family, prompt_style=None, …) → (bir_pivot, count_pivot)
  - print_bir_table
  - save_bir_tables
  - print_summary_table(sample_df)

Output formatting matches the legacy module byte-for-byte. Constants
(`TRAINING_TYPE_ORDER`, `TRAINING_TYPE_NAMES`, `BIAS_DISPLAY_NAMES`,
`DEFAULT_TRAINING_BIASES`) are sourced from `viz.registry.REGISTRY`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .registry import REGISTRY


# Default training-bias set for held-out vs trained partitioning.
# Matches legacy `DEFAULT_TRAINING_BIASES`.
DEFAULT_TRAINING_BIASES = frozenset({"suggested_answer"})

# Short, fixed-width row labels used in the printed-table layout. These
# abbreviations are intentionally shorter than `REGISTRY.biases[k].display_name`
# so the leftmost column fits in 16 chars without truncating.
_SHORT_BIAS_NAMES = {
    "suggested_answer": "Sugg. Answer",
    "wrong_few_shot": "Wrong FS",
    "distractor_argument": "Argument",
    "distractor_fact": "Fact",
    "spurious_few_shot_hindsight": "Hindsight",
    "spurious_few_shot_squares": "Squares",
}


def _bias_display_names() -> dict[str, str]:
    return {k: v.display_name for k, v in REGISTRY.biases.items()}


def _training_type_names() -> dict[str, str]:
    return {k: v.display_name for k, v in REGISTRY.training_types.items()}


def _training_type_order() -> list[str]:
    return list(REGISTRY.training_type_order)


def _split_biases(bias_list: list[str],
                  training_biases: set[str] | None = None,
                  ) -> tuple[list[str], list[str]]:
    """Split bias list into (trained_on, held_out), preserving order within each group."""
    tb = training_biases if training_biases is not None else DEFAULT_TRAINING_BIASES
    trained = [b for b in bias_list if b in tb]
    held_out = [b for b in bias_list if b not in tb]
    return trained, held_out


def compute_bir_table(
    bir_df: pd.DataFrame,
    model_family: str,
    prompt_style: str | None = None,
    metric_col: str = "bir",
    baseline: str = "base",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate per-question BRR into a pivot table (bias_type × training_type).

    Returns:
        Tuple of (bir_pivot, count_pivot) DataFrames.
    """
    mask = bir_df["model_family"] == model_family
    if prompt_style:
        mask &= bir_df["prompt_style"] == prompt_style
    filtered = bir_df[mask]

    if filtered.empty:
        return pd.DataFrame(), pd.DataFrame()

    grouped = filtered.groupby(["bias_type", "training_type"])[metric_col]
    bir_pivot = grouped.mean().unstack("training_type")
    count_pivot = grouped.count().unstack("training_type")

    tt_order = _training_type_order()
    bir_pivot = bir_pivot.reindex(columns=tt_order)
    count_pivot = count_pivot.reindex(columns=tt_order, fill_value=0)

    for tt in tt_order:
        if tt != baseline:
            bir_pivot[f"{tt}_ratio"] = bir_pivot[tt] / bir_pivot[baseline].replace(0, np.nan)

    cols = tt_order + [f"{tt}_ratio" for tt in tt_order if tt != baseline]
    return bir_pivot[cols], count_pivot[tt_order]


def print_bir_table(
    bir_df: pd.DataFrame,
    model_family: str,
    prompt_style: str | None = None,
    metric_col: str = "bir",
    label: str = "",
    show_n: bool = True,
    baseline: str = "base",
    table_name: str = "BIR",
    value_header: str = "BIR %",
    ratio_header: str = "BIR Ratio",
    best_is_low: bool = True,
    training_biases: set[str] | None = None,
):
    """Print pivot table in formatted style (used for BIR, BA, and other per-question metrics)."""
    bir, counts = compute_bir_table(bir_df, model_family, prompt_style,
                                    metric_col=metric_col, baseline=baseline)
    if bir.empty:
        return

    bias_display = _bias_display_names()
    tt_names = _training_type_names()
    tt_order = _training_type_order()

    # All bias types present in the data (used for display order)
    all_biases_raw = [b for b in bias_display if b in bir.index]
    trained_biases_list, held_out_biases_list = _split_biases(all_biases_raw, training_biases)
    all_bias_order = trained_biases_list + held_out_biases_list
    n_trained = len(trained_biases_list)
    available_types = [tt for tt in tt_order if tt in bir.columns and bir[tt].notna().any()]

    baseline_name = tt_names.get(baseline, baseline)
    style_str = f" ({prompt_style})" if prompt_style else ""
    label_str = f" | {label}" if label else ""

    bir_cols = [tt_names.get(tt, tt) for tt in available_types]
    ratio_cols = [tt_names.get(tt, tt) for tt in available_types if tt != baseline]

    # Dynamic column width: at least 9, wide enough for the longest name + 1
    cw = max(9, max((len(c) for c in bir_cols + ratio_cols), default=9) + 1)

    w = 16 + cw * len(bir_cols) + cw * len(ratio_cols) + 4
    print(f"\n{'=' * w}")
    print(f"{table_name} TABLE - {model_family.upper()}{style_str}{label_str}")
    print(f"{'=' * w}")

    bir_sw = cw * len(bir_cols)
    ratio_sw = cw * len(ratio_cols)
    print(f"{'':16} {value_header:^{bir_sw}}  {ratio_header:^{ratio_sw}}")

    print(f"{'Bias Type':<16} {' '.join(f'{c:>{cw}}' for c in bir_cols)}  {' '.join(f'{c:>{cw}}' for c in ratio_cols)}")
    print("-" * w)

    non_baseline = [tt for tt in available_types if tt != baseline]
    blank_ratio = " ".join(f"{'':>{cw}}" for _ in non_baseline)

    def fmt_row(bias_type):
        row = bir.loc[bias_type]
        name = _SHORT_BIAS_NAMES.get(bias_type, bias_type[:12])
        bir_vals, bir_strs, ratio_strs = {}, {}, {}

        for tt in available_types:
            val = row.get(tt)
            if pd.notna(val):
                bir_vals[tt] = val
                bir_strs[tt] = f"{val*100:.0f}"
            else:
                bir_strs[tt] = "-"
            if tt != baseline:
                rv = row.get(f"{tt}_ratio")
                ratio_strs[tt] = f"{rv:.2f}" if pd.notna(rv) else "-"

        if bir_vals:
            best_val = min(bir_vals.values()) if best_is_low else max(bir_vals.values())
            for k in [k for k, v in bir_vals.items() if v == best_val and k != baseline]:
                bir_strs[k] = f"*{bir_strs[k]}*"

        bp = " ".join(f"{bir_strs[tt]:>{cw}}" for tt in available_types)
        rp = " ".join(f"{ratio_strs[tt]:>{cw}}" for tt in non_baseline)
        return f"{name:<16} {bp}  {rp}"

    def fmt_n_row(bias_type):
        """Format the n= row for a bias type."""
        count_row = counts.loc[bias_type] if bias_type in counts.index else pd.Series()
        n_strs = []
        for tt in available_types:
            raw_n = count_row.get(tt, 0) if tt in count_row.index else 0
            n = int(raw_n) if pd.notna(raw_n) else 0
            n_strs.append(f"n={n}" if n > 0 else "")
        np_ = " ".join(f"{s:>{cw}}" for s in n_strs)
        return f"{'':16} {np_}  {blank_ratio}"

    for i, bt in enumerate(all_bias_order):
        if i == n_trained and n_trained > 0 and held_out_biases_list:
            print("·" * w)  # dotted separator between trained and held-out
        print(fmt_row(bt))
        if show_n:
            print(fmt_n_row(bt))

    # Per-training-type training and held-out averages
    tb_set = training_biases if training_biases is not None else DEFAULT_TRAINING_BIASES

    print("-" * w)

    def _print_avg_row(bias_subset: list[str], label: str) -> None:
        """Print a Training Avg / Held-out Avg row + n row."""
        if not bias_subset:
            return
        avg_vals: dict[str, float] = {}
        total_n: dict[str, int] = {}
        avg_ratios: dict[str, float] = {}
        for tt in available_types:
            biases_in = [b for b in bias_subset if b in bir.index]
            if biases_in:
                avg_vals[tt] = bir.loc[biases_in, tt].mean() * 100
                total_n[tt] = int(counts.loc[biases_in, tt].sum()) if tt in counts.columns else 0
            else:
                avg_vals[tt] = float("nan")
                total_n[tt] = 0
        for tt in non_baseline:
            base_avg = avg_vals.get(baseline, 0)
            avg_ratios[tt] = avg_vals[tt] / base_avg if base_avg != 0 else float("nan")
        bp = " ".join(f"{avg_vals[tt]:>{cw}.0f}" for tt in available_types)
        rp = " ".join(f"{avg_ratios.get(tt, float('nan')):>{cw}.2f}" for tt in non_baseline)
        print(f"{label:<16} {bp}  {rp}")
        if show_n:
            np_ = " ".join(f"{'n=' + str(total_n[tt]):>{cw}}" for tt in available_types)
            print(f"{'':16} {np_}  {blank_ratio}")

    _print_avg_row(trained_biases_list, "Training Avg")
    _print_avg_row(held_out_biases_list, "Held-out Avg")

    print("=" * w)
    best_dir = "lowest" if best_is_low else "highest"
    print(f"* = best ({best_dir}) for that bias type (excluding {baseline_name})")
    tr_bias_names = ", ".join(bias_display.get(b, b) for b in sorted(tb_set))
    print(f"Training biases: {tr_bias_names}")


def save_bir_tables(
    bir_df: pd.DataFrame,
    output_path: str,
    models: list[str] | None = None,
    prompt_styles: list[str] | None = None,
    table_variants: list | None = None,
    baseline: str = "base",
    value_label: str = "BIR",
    title: str = "Bias Influence Rate (BIR) Results",
    description: str = "BIR = |bias_match_rate(biased) - bias_match_rate(unbiased)|, per-question",
    training_biases: set[str] | None = None,
):
    """Save pivot table variants to CSV and MD."""
    models = models or ["llama", "gpt"]

    bias_display = _bias_display_names()
    tt_names = _training_type_names()
    tt_order = _training_type_order()

    # Fall back to default variants if none provided
    if table_variants is None:
        table_variants = [
            ("lenient_bir", None, "BIR"),
            ("lenient_bir", lambda df: df["bias_acknowledged"] == 0.0, "Unverbalised"),
            ("lenient_bir", lambda df: df["bias_acknowledged"] == 1.0, "Verbalised"),
        ]

    all_rows = []
    for model_family in models:
        if prompt_styles:
            avail_styles = list(prompt_styles)
        else:
            avail_styles = ["cot", "no_cot"] if model_family == "llama" else ["no_cot"]

        for style in avail_styles:
            for metric_col, filt_fn, filt_label in table_variants:
                sub_df = bir_df if filt_fn is None else bir_df[filt_fn(bir_df)]

                tbl, _ = compute_bir_table(sub_df, model_family, style,
                                           metric_col=metric_col, baseline=baseline)
                if tbl.empty:
                    continue

                # Get available training types
                avail_tt = [tt for tt in tt_order if tt in tbl.columns and tbl[tt].notna().any()]

                for bt in tbl.index:
                    row_data = {
                        "Model": model_family.upper(),
                        "Prompt": style,
                        "Variant": filt_label,
                        "Bias Type": _SHORT_BIAS_NAMES.get(bt, bt),
                    }
                    for tt in avail_tt:
                        val = tbl.loc[bt, tt]
                        row_data[f"{tt_names.get(tt, tt)} {value_label}%"] = (
                            round(val * 100, 1) if pd.notna(val) else None
                        )
                        if tt != baseline:
                            rv = tbl.loc[bt, f"{tt}_ratio"]
                            row_data[f"{tt_names.get(tt, tt)} Ratio"] = (
                                round(rv, 2) if pd.notna(rv) else None
                            )
                    all_rows.append(row_data)

    result_df = pd.DataFrame(all_rows)

    csv_path = output_path if output_path.endswith(".csv") else f"{output_path}.csv"
    result_df.to_csv(csv_path, index=False)
    print(f"Saved BIR tables to {csv_path}")

    md_path = csv_path.replace(".csv", ".md")
    with open(md_path, "w") as f:
        f.write(f"# {title}\n\n")
        f.write(f"{description}\n\n")
        for variant_label in [v[2] for v in table_variants]:
            subset = result_df[result_df["Variant"] == variant_label]
            if subset.empty:
                continue
            f.write(f"## {variant_label}\n\n")
            f.write(subset.drop(columns=["Variant"]).to_markdown(index=False))
            f.write("\n\n")
        f.write("## Notes\n\n")
        tr_bias_names = ", ".join(
            f"**{bias_display.get(b, b)}**"
            for b in sorted(training_biases or DEFAULT_TRAINING_BIASES)
        )
        f.write(f"- Training bias(es): {tr_bias_names}\n")
        f.write("- **Verbalised** = model mentions the bias in CoT\n")
        f.write("- **Unverbalised** = model does not mention the bias in CoT\n")
        f.write("- **(Strict BA)** = strict bias_acknowledged (NaN-out few-shot confused samples)\n")
        f.write("- **(Lenient BA)** = lenient bias_acknowledged (all samples)\n")
        f.write("- **(Lenient)** = uses fallback parser which recovers some unparseable responses\n")
        f.write(f"- Ratio < 1.0 indicates improvement over {tt_names.get(baseline, baseline)} model\n")
    print(f"Saved BIR tables to {md_path}")


def print_summary_table(sample_df: pd.DataFrame):
    """Print summary statistics from sample-level data."""
    print("\n" + "=" * 80)
    print("SUMMARY BY MODEL AND TRAINING TYPE")
    print("=" * 80)

    for mf in sorted(sample_df["model_family"].unique()):
        print(f"\n{mf.upper()}")
        print("-" * 60)

        biased = sample_df[(sample_df["model_family"] == mf) & (sample_df["variant"] == "biased")]
        if biased.empty:
            print("  (no biased data)")
            continue

        summary = biased.groupby("training_type").agg(
            correct_mean=("correct", "mean"),
            correct_std=("correct", "std"),
            matches_bias_mean=("matches_bias", "mean"),
            matches_bias_std=("matches_bias", "std"),
            n=("correct", "count"),
        ).round(3)

        print(summary.to_string())
