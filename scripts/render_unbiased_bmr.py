"""Render unbiased plots — one bar per (model, training_type), no bias x-axis.

The unbiased prompt is bias-agnostic so the plot collapses bias_type entirely:
two panels (gpt-oss-20b on top, Llama on bottom), three bars per panel
(Base/BCT/RMCT, aggregated across LRs). For each metric in (matches_bias,
correct) we render both a raw plot and a ratio plot (% Δ vs base). n is
shown above each bar on the raw plots; suppressed on ratio plots since the
delta-method SE depends on both numerator and denominator counts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sycophancy_eval_inspect.viz.aggregations import aggregate_training_types
from sycophancy_eval_inspect.viz.loaders import load_samples
from sycophancy_eval_inspect.viz.theme import PUBLICATION_THEME

MODEL_DISPLAY = {
    "gpt-oss-20b": "OpenAI GPT OSS 20B",
    "llama": "Meta Llama 3.1 8B Instruct",
}
MODEL_ORDER = ["gpt-oss-20b", "llama"]
TT_ORDER = [("base", "Base"), ("bct-da-lravg", "BCT"), ("rlct-da-aw0-r128b4", "RMCT")]
METRICS = [("matches_bias", "Bias Match Rate"), ("correct", "Accuracy")]


def _agg_per_tt(unb, mf: str, metric: str):
    """Yield (training_type, label, mean, se, n) tuples for one model panel."""
    mf_unb = unb[unb["model_family"] == mf]
    for tt, label in TT_ORDER:
        grp = mf_unb[mf_unb["training_type"] == tt]
        vals = grp[metric].dropna().to_numpy()
        n = len(vals)
        if n == 0:
            continue
        mean = float(vals.mean())
        se = float((mean * (1 - mean) / n) ** 0.5) if n > 0 else 0.0
        yield tt, label, mean, se, n


def _draw_panel(ax, theme, panel_data, *, ratio: bool, ylabel: str, model_label: str, ylim, show_n: bool):
    """Draw one model panel given pre-aggregated rows.

    `panel_data` already excludes 'base' on ratio plots (base is the
    denominator, always 0%, so it's elided from the x-axis).
    """
    xticks = []
    xticklabels = []
    for x, (tt, label, val, se, n) in enumerate(panel_data):
        style = theme.bar_style_for(tt)
        bars = ax.bar(
            x,
            val,
            0.55,
            facecolor=style["facecolor"],
            edgecolor=style["edgecolor"],
            linewidth=style["linewidth"],
            zorder=2,
        )
        if style["linewidth"]:
            for b in bars:
                b.set_clip_path(b)
        ax.errorbar(
            x,
            val,
            yerr=2 * se,
            fmt="none",
            ecolor=theme.error_color,
            capsize=theme.error_capsize,
            linewidth=theme.error_linewidth,
            alpha=theme.error_alpha,
            zorder=3,
        )
        if show_n:
            ax.text(
                x,
                val + 2 * se + 0.02 * (ylim[1] - ylim[0]),
                f"n={n}",
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=theme.n_label_fontsize,
                color=theme.error_color,
                zorder=4,
            )
        xticks.append(x)
        xticklabels.append(label)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontsize=theme.xtick_label_fontsize)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel, fontsize=theme.ylabel_fontsize)
    ax.tick_params(axis="y", labelsize=theme.ytick_label_fontsize)
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", color="#ececee", linewidth=0.6, zorder=0)
    if ratio:
        ax.axhline(y=0, color=theme.zero_line_color, linewidth=1, zorder=1)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0f}%" if v != 0 else "0"))
    ax.set_title(
        model_label, loc="left", fontsize=theme.panel_title_fontsize, fontweight="semibold", color="#141518", pad=1.5
    )


def _ratio_rows(panel_rows: list[tuple]) -> list[tuple]:
    """Convert raw rows → % Δ vs base via the delta method.

    Drops the base row; baseline value=0 → row dropped (undefined ratio).
    """
    base = next((r for r in panel_rows if r[0] == "base"), None)
    if base is None or base[2] == 0:
        return []
    bv, be = base[2], base[3]
    out = []
    for tt, label, val, se, n in panel_rows:
        if tt == "base":
            continue
        ratio = val / bv
        pct = (ratio - 1.0) * 100
        # Delta method in derivative form; unlike the relative-error form,
        # this remains defined when the numerator value is exactly zero.
        ratio_var = (se / bv) ** 2 + (val * be / (bv**2)) ** 2
        se_pct = ratio_var**0.5 * 100
        out.append((tt, label, pct, se_pct, n))
    return out


def _render_one(unb, *, metric: str, ylabel: str, ratio: bool, output_dir: Path) -> Path | None:
    theme = PUBLICATION_THEME
    theme.apply()

    # First pass: collect aggregates for all panels so we can pick a shared ylim.
    per_panel = []
    for mf in MODEL_ORDER:
        rows = list(_agg_per_tt(unb, mf, metric))
        if ratio:
            rows = _ratio_rows(rows)
        per_panel.append((mf, rows))

    if all(not rows for _, rows in per_panel):
        return None

    # ylim: shared across panels, derived from data extents (dynamic).
    # Headroom is proportional to the data range so the axis tracks the data
    # tightly instead of getting pinned to an absolute padding constant.
    extents = [v + 2 * se for _, rows in per_panel for (_, _, v, se, _) in rows] + [
        v - 2 * se for _, rows in per_panel for (_, _, v, se, _) in rows
    ]
    if ratio:
        lo, hi = min(extents, default=0.0), max(extents, default=0.0)
        span = max(hi - lo, 1.0)
        ymin = min(lo, 0.0) - 0.10 * span
        ymax = max(hi, 0.0) + 0.10 * span
    else:
        lo, hi = min(extents, default=0.0), max(extents, default=1.0)
        span = max(hi - lo, 0.05)
        # Top headroom needs room for the rotated "n=NNN" labels — bigger pad
        # at the top than the bottom; both proportional to span.
        ymin = max(0.0, lo - 0.10 * span)
        ymax = hi + 0.40 * span
    ylim = (ymin, ymax)

    fig, axes = plt.subplots(len(MODEL_ORDER), 1, figsize=(3.6, 2.0 * len(MODEL_ORDER)), sharey=True, squeeze=False)
    axes = axes.flatten()
    for idx, (mf, rows) in enumerate(per_panel):
        _draw_panel(
            axes[idx],
            theme,
            rows,
            ratio=ratio,
            ylabel=ylabel,
            model_label=MODEL_DISPLAY.get(mf, mf),
            ylim=ylim,
            show_n=not ratio,
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    fname = f"unbiased_{metric}_ratio.png" if ratio else f"unbiased_{metric}.png"
    out = output_dir / fname
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    return out


def render(*, log_dirs: list[str], output_dir: Path, dedup: str = "last") -> int:
    samples = load_samples(log_dirs, dedup=dedup)
    samples = aggregate_training_types(samples)
    unb = samples[(samples["variant"] == "unbiased") & (samples["model_family"].isin(MODEL_ORDER))]
    if unb.empty:
        print("No unbiased samples.")
        return 0
    # Dedup by sample_id within each model_name (not training_type): the
    # loader's dedup keeps one row per (sample_id, bias_type), but for
    # unbiased samples bias_type is a spurious label (the unbiased prompt is
    # bias-agnostic). Keying on model_name preserves the 3 LR variants of a
    # trained family as 3 independent evals while collapsing legacy base
    # rows that re-ran the same questions across multiple bias_type
    # campaigns under one model_name.
    unb = unb.sort_values("created").drop_duplicates(subset=["model_name", "sample_id", "prompt_style"], keep="last")

    n = 0
    for metric, label in METRICS:
        # Raw + ratio
        if _render_one(unb, metric=metric, ylabel=label, ratio=False, output_dir=output_dir):
            n += 1
        if _render_one(unb, metric=metric, ylabel=f"% Δ {label}", ratio=True, output_dir=output_dir):
            n += 1
    return n


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--log-dir", action="append", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--dedup", choices=("last", "mean", "none"), default="last")
    args = p.parse_args()
    n = render(log_dirs=args.log_dir, output_dir=Path(args.output_dir), dedup=args.dedup)
    print(f"Wrote {n} figure(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
