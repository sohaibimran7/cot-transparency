"""bar_plot — single grouped-bars renderer for the publication pipeline.

Replaces both `plot_grouped_bars_publication` and `plot_metric_ratio_publication`
from the legacy module. Differences (ratio vs grouped):

  - ratio: pass `zero_line=True`, `yformatter=...`, derive ylim from data
  - grouped: pass `baseline_hline=...`, `random_line=0.25`, fixed ylim

Same code path otherwise: panels, cluster offsets, cream bands, hairlines,
top-right legend.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .registry import REGISTRY, training_type_info
from .theme import Theme
from .transforms import (
    Faceter,
    Panel,
    add_held_out_avg,
    facet_by_training_biases,
    no_facet,
    order_biases_by_panels,
)


_FAMILY_ORDER = ("base", "bct", "rlct", "vft", "other")


def _cluster_offsets(panel_types: list[str], bar_w: float, gap: float
                     ) -> dict[str, float]:
    """X offsets per training type, clustered by method family.

    Bars within a family sit flush; controls follow treatments. An extra `gap`
    separates families. Centered around 0.
    """
    by_family: dict[str, list[str]] = {}
    for tt in panel_types:
        info = training_type_info(tt)
        by_family.setdefault(info.method, []).append(tt)
    # Treatment first, then controls
    for fam in by_family:
        by_family[fam].sort(key=lambda t: training_type_info(t).is_control)

    ordered_families = [f for f in _FAMILY_ORDER if f in by_family]
    ordered_families += [f for f in by_family if f not in _FAMILY_ORDER]

    n_bars = sum(len(by_family[f]) for f in ordered_families)
    n_gaps = max(0, len(ordered_families) - 1)
    total = n_bars * bar_w + n_gaps * gap

    cursor = -total / 2 + bar_w / 2
    out: dict[str, float] = {}
    for fi, fam in enumerate(ordered_families):
        for tt in by_family[fam]:
            out[tt] = cursor
            cursor += bar_w
        if fi < len(ordered_families) - 1:
            cursor += gap
    return out


def _simplified_legend(training_types: list[str], theme: Theme):
    """One legend entry per (method, is_control). Drops the data-scale shade.

    Order: Base, BCT, BCT Control, RLCT, RLCT Control, VFT, VFT Control.
    """
    from matplotlib.patches import Patch
    DISPLAY = {
        ("base", False): "Base",
        ("bct", False): "BCT",
        ("bct", True): "BCT Control",
        ("rlct", False): "RLCT",
        ("rlct", True): "RLCT Control",
        ("vft", False): "VFT",
        ("vft", True): "VFT Control",
    }
    ORDER = ["Base", "BCT", "BCT Control", "RLCT", "RLCT Control", "VFT", "VFT Control"]
    seen: dict[str, Patch] = {}
    for tt in training_types:
        info = training_type_info(tt)
        label = DISPLAY.get((info.method, info.is_control), info.display_name)
        if label in seen:
            continue
        style = theme.bar_style_for(tt)
        seen[label] = Patch(
            facecolor=style["facecolor"],
            edgecolor=style["edgecolor"],
            linewidth=max(style["linewidth"], 0.5),
            label=label,
        )
    return [seen[k] for k in ORDER if k in seen] + [
        v for k, v in seen.items() if k not in ORDER
    ]


def _bias_label(bias_key: str, held_out_label: str) -> str:
    """Single-line display name for rotated 45° x-tick labels.

    Mirrors legacy publication path: uses BIAS_DISPLAY_NAMES (not the multi-line
    PUBLICATION_BIAS_LABELS, which was defined but unused by the rotated path).
    """
    if bias_key == held_out_label:
        return bias_key
    info = REGISTRY.biases.get(bias_key)
    if info is not None:
        return info.display_name
    return bias_key


def bar_plot(
    agg: pd.DataFrame,
    *,
    metric: str,
    model_family: str | None = None,
    prompt_style: str | None = None,
    theme: Theme,
    faceter: Faceter | None = None,
    output_path: str | Path | None = None,
    ylabel: str = "",
    ylim: tuple[float, float] | None = None,
    yformatter: Callable[[float, int], str] | None = None,
    baseline_hline: float | None = None,
    baseline_band: tuple[float, float] | None = None,
    random_line: float | None = None,
    zero_line: bool = False,
    drop_biases: tuple[str, ...] = ("are_you_sure",),
    show_controls: bool = True,
    write_n_csv: bool = True,
    held_out_avg_label: str = "Held-out Avg",
    auto_held_out: bool = True,
    n_labels: bool = False,
) -> None:
    """Render a grouped-bars figure from a long-form aggregated frame.

    `agg` columns required: training_type, bias_type, metric, value, stderr, n
    (plus whatever the faceter needs).

    Faceting is delegated to a `Faceter` callable. Default is
    `facet_by_training_biases`, which mirrors the legacy publication layout
    (one panel per unique training-bias set, cream-band highlights for trained
    biases). Pass `facet_by_column("model_family")` (etc.) to facet by any
    other dimension.

    Parameters affecting which slice of `agg` is plotted:
      metric        — required filter on the `metric` column
      model_family  — optional filter; pass None to facet by it via faceter
      prompt_style  — optional filter; pass None to facet by it via faceter

    `n_labels=True` draws "n=NNN" above each bar (off by default for
    publication style; on for non-publication parity).
    """
    theme.apply()
    if faceter is None:
        faceter = facet_by_training_biases

    # ── 1. Filter to the slice we're plotting ─────────────────────────────
    sel = (agg["metric"] == metric)
    if model_family is not None and "model_family" in agg.columns:
        sel = sel & (agg["model_family"] == model_family)
    if prompt_style is not None and "prompt_style" in agg.columns:
        sel = sel & (agg["prompt_style"] == prompt_style)
    sub = agg[sel].copy()
    if drop_biases:
        sub = sub[~sub["bias_type"].isin(drop_biases)]
    if sub.empty:
        return

    # ── 2. Pick training types (registry order, optionally drop controls) ──
    tts_in_data = set(sub["training_type"].unique())
    training_types = [t for t in REGISTRY.training_type_order if t in tts_in_data]
    if not show_controls:
        training_types = [t for t in training_types
                          if not training_type_info(t).is_control]
    if not training_types:
        return
    sub = sub[sub["training_type"].isin(training_types)]

    # ── 3. Panel split via the faceter, then bias ordering ────────────────
    all_biases = [b for b in REGISTRY.biases.keys() if b in sub["bias_type"].values]
    panels = faceter(sub)
    if not panels:
        panels = no_facet(sub)

    bias_order, held_out_biases = order_biases_by_panels(all_biases, panels)
    if auto_held_out and held_out_biases:
        sub = add_held_out_avg(sub, held_out_biases, label=held_out_avg_label)
        # Held-out-avg rows need to be reflected in each panel's data too,
        # since panels carry their own data slices.
        panels = [Panel(label=p.label, training_types=p.training_types,
                        data=add_held_out_avg(
                            p.data if p.data is not None else sub,
                            held_out_biases, label=held_out_avg_label),
                        highlighted_biases=p.highlighted_biases)
                  for p in panels]
        bias_order = bias_order + [held_out_avg_label]
    if not bias_order:
        return

    # Build wide pivots for fast lookup + n.csv emission. Use pivot (not
    # pivot_table) so NaN-only groups remain in the index — matching legacy
    # behavior where bias columns appear even with no data.
    pivot_val = sub.pivot(index="bias_type", columns="training_type", values="value")
    pivot_err = sub.pivot(index="bias_type", columns="training_type", values="stderr")
    pivot_n = sub.pivot(index="bias_type", columns="training_type", values="n")
    pivot_val = pivot_val.reindex(bias_order)
    pivot_err = pivot_err.reindex(bias_order)
    pivot_n = pivot_n.reindex(bias_order)

    # ── 4. ylim ───────────────────────────────────────────────────────────
    if ylim is None:
        # For grouped bars: ymin=min(0, min−.05), ymax=max+.10
        # For ratio (when zero_line): include errorbar extents, pad ±10
        vals = pivot_val.to_numpy(dtype=float).flatten()
        vals = vals[~np.isnan(vals)]
        if zero_line:
            vals_flat = pivot_val.to_numpy(dtype=float).flatten()
            errs_flat = pivot_err.to_numpy(dtype=float).flatten()
            errs_flat = np.where(np.isnan(errs_flat), 0, errs_flat)
            extents = np.concatenate([vals_flat - 2 * errs_flat,
                                      vals_flat + 2 * errs_flat])
            extents = extents[~np.isnan(extents)]
            if extents.size:
                ymin = min(float(extents.min()), 0.0) - 10
                ymax = max(float(extents.max()), 0.0) + 10
            else:
                ymin, ymax = -10.0, 10.0
        else:
            if vals.size:
                ymin = min(0.0, float(vals.min()) - 0.05)
                ymax = float(vals.max()) + 0.10
            else:
                ymin, ymax = 0.0, 1.05
        ylim = (ymin, ymax)

    # ── 5. Geometry ───────────────────────────────────────────────────────
    n_panels = len(panels)
    bias_spacing = theme.bias_spacing
    bar_w = theme.bar_width
    cluster_gap = theme.cluster_gap
    fig_w = max(theme.figure_width_min,
                theme.figure_width_per_bias * len(bias_order)
                + theme.figure_width_intercept)
    fig_h = theme.figure_height_per_panel * n_panels + theme.figure_height_intercept
    fig, axes = plt.subplots(n_panels, 1, figsize=(fig_w, fig_h),
                             sharex=True, sharey=True, squeeze=False)
    axes = axes.flatten()

    bias_x = np.arange(len(bias_order)) * bias_spacing
    half_span = bias_spacing / 2
    legend_handles = _simplified_legend(training_types, theme)

    # ── 6. Per-panel rendering ────────────────────────────────────────────
    for ax_idx, panel in enumerate(panels):
        ax = axes[ax_idx]
        panel_types = [t for t in panel.training_types if t in pivot_val.columns]
        if not panel_types:
            continue

        offsets = _cluster_offsets(panel_types, bar_w, cluster_gap)

        # Per-panel pivots: each panel's data slice may differ when the
        # faceter splits the frame (e.g. facet_by_column). For training-bias
        # faceting, panel.data == sub, so the local pivots equal the global ones.
        if panel.data is not None and panel.data is not sub:
            p_val = panel.data.pivot(index="bias_type", columns="training_type",
                                     values="value").reindex(bias_order)
            p_err = panel.data.pivot(index="bias_type", columns="training_type",
                                     values="stderr").reindex(bias_order)
            p_n = panel.data.pivot(index="bias_type", columns="training_type",
                                   values="n").reindex(bias_order)
        else:
            p_val, p_err, p_n = pivot_val, pivot_err, pivot_n

        # Background bands: cream for highlighted biases, grey for held-out summary
        for j, bt in enumerate(bias_order):
            x_center = bias_x[j]
            if bt in panel.highlighted_biases:
                ax.axvspan(x_center - half_span, x_center + half_span,
                           color=theme.panel_bg_trained, alpha=0.7, zorder=0)
            elif bt == held_out_avg_label:
                ax.axvspan(x_center - half_span, x_center + half_span,
                           color=theme.panel_bg_summary, alpha=0.9, zorder=0)

        # Hairlines between bias columns
        for j in range(1, len(bias_order)):
            ax.axvline(x=bias_x[j] - half_span, color=theme.hairline_color,
                       linewidth=theme.hairline_lw, zorder=1)

        # Bars + error bars + optional n labels
        for tt in panel_types:
            offset = offsets[tt]
            style = theme.bar_style_for(tt)
            for j, x_val in enumerate(bias_x):
                bt = bias_order[j]
                if bt not in p_val.index or tt not in p_val.columns:
                    continue
                val = p_val.loc[bt, tt]
                err = (p_err.loc[bt, tt]
                       if (tt in p_err.columns and bt in p_err.index)
                       else np.nan)
                if pd.isna(val):
                    continue
                xp = x_val + offset
                ax.bar(xp, val, bar_w,
                       facecolor=style["facecolor"],
                       edgecolor=style["edgecolor"],
                       linewidth=style["linewidth"],
                       zorder=2)
                if pd.notna(err) and err > 0:
                    ax.errorbar(xp, val, yerr=2 * err, fmt="none",
                                ecolor=theme.error_color,
                                capsize=theme.error_capsize,
                                linewidth=theme.error_linewidth,
                                alpha=theme.error_alpha, zorder=3)
                if n_labels and tt in p_n.columns and bt in p_n.index:
                    n_val = p_n.loc[bt, tt]
                    if pd.notna(n_val):
                        text_y = val + (2 * err if pd.notna(err) and err > 0 else 0) \
                                 + 0.02 * (ylim[1] - ylim[0]) if ylim else val
                        ax.text(xp, text_y, f"n={int(n_val)}",
                                ha="center", va="bottom", fontsize=6,
                                rotation=90, zorder=4)

        # Axes formatting
        ax.set_xticks(bias_x)
        ax.set_xticklabels(
            [_bias_label(b, held_out_avg_label) for b in bias_order],
            rotation=45, ha="right", fontsize=theme.xtick_label_fontsize,
        )
        ax.set_xlim(bias_x[0] - half_span, bias_x[-1] + half_span)
        ax.set_ylim(*ylim)
        ax.set_ylabel(ylabel, fontsize=theme.ylabel_fontsize)
        ax.tick_params(axis="y", labelsize=theme.ytick_label_fontsize)
        ax.grid(axis="x", visible=False)
        ax.grid(axis="y", color="#ececee", linewidth=0.6, zorder=0)
        if yformatter is not None:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(yformatter))
        if panel.label:
            ax.set_title(panel.label, loc="left",
                         fontsize=theme.panel_title_fontsize,
                         fontweight="semibold", color="#141518", pad=1.5)

        # Reference lines
        if baseline_hline is not None:
            ax.axhline(y=baseline_hline, color=theme.baseline_color,
                       linestyle=":", linewidth=1, alpha=0.6, zorder=1)
            if baseline_band is not None:
                ax.axhspan(baseline_band[0], baseline_band[1],
                           color=theme.baseline_color, alpha=0.08, zorder=0)
        if random_line is not None:
            ax.axhline(y=random_line, color=theme.baseline_color,
                       linestyle="--", alpha=0.5, linewidth=0.8, zorder=1)
        if zero_line:
            ax.axhline(y=0, color=theme.zero_line_color, linewidth=1, zorder=1)

    # ── 7. Layout + legend + save ─────────────────────────────────────────
    fig.tight_layout(rect=[0, 0.0, 1, 0.96], pad=0.15, h_pad=1.4)
    fig.legend(handles=legend_handles, loc="upper right",
               bbox_to_anchor=(1.0, 1.0),
               ncol=len(legend_handles),
               frameon=False, fontsize=7,
               handlelength=1.0, columnspacing=0.9,
               labelspacing=0.2, borderpad=0.2)

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved: {out}")
        if write_n_csv:
            n_csv = out.with_name(out.stem + "_n.csv")
            pivot_n.to_csv(n_csv)
    else:
        plt.show()
    plt.close(fig)
