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
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .registry import REGISTRY, training_type_info
from .theme import Theme
from .transforms import (
    Faceter,
    Panel,
    add_held_out_avg,
    add_pct_change_held_out_avg,
    facet_by_training_biases,
    no_facet,
    one_sample_p,
    order_biases_by_panels,
    significance_marker,
    two_sample_p,
)

_FAMILY_ORDER = ("base", "bct", "rlct", "vft", "other")


def _cluster_offsets(panel_types: list[str], bar_w: float, gap: float) -> dict[str, float]:
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

    Order: Base, BCT, BCT Control, RMCT, RMCT Control, VFT, VFT Control.
    """
    from matplotlib.patches import Patch

    DISPLAY = {
        ("base", False): "Base",
        ("bct", False): "BCT",
        ("bct", True): "BCT Control",
        ("rlct", False): "RMCT",
        ("rlct", True): "RMCT Control",
        ("vft", False): "VFT",
        ("vft", True): "VFT Control",
    }
    ORDER = ["Base", "BCT", "BCT Control", "RMCT", "RMCT Control", "VFT", "VFT Control"]
    seen: dict[str, Patch] = {}
    for tt in training_types:
        info = training_type_info(tt)
        label = DISPLAY.get((info.method, info.is_control), info.display_name)
        if label in seen:
            continue
        style = theme.bar_style_for(tt)
        # Bars use a doubled linewidth + clip-to-self so the visible inside
        # stroke is half the nominal width. Legend swatches aren't clipped, so
        # halve here to keep the swatch outline visually equivalent.
        legend_lw = style["linewidth"] / 2 if style["linewidth"] else 0.0
        seen[label] = Patch(
            facecolor=style["facecolor"],
            edgecolor=style["edgecolor"],
            linewidth=max(legend_lw, 0.5),
            label=label,
        )
    return [seen[k] for k in ORDER if k in seen] + [v for k, v in seen.items() if k not in ORDER]


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
    held_out_avg_label: str = "Held-out Avg",
    auto_held_out: bool = True,
    n_labels: bool = False,
    show_significance: bool = True,
    significance_baseline: str = "base",
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
    sel = agg["metric"] == metric
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
        training_types = [t for t in training_types if not training_type_info(t).is_control]
    if not training_types:
        return
    sub = sub[sub["training_type"].isin(training_types)]

    # ── 3. Panel split via the faceter, then bias ordering ────────────────
    all_biases = [b for b in REGISTRY.biases.keys() if b in sub["bias_type"].values]
    panels = faceter(sub)
    if not panels:
        panels = no_facet(sub)

    bias_order, held_out_biases = order_biases_by_panels(all_biases, panels)
    if auto_held_out:
        # Held-out is computed *per panel*: each panel's held-out set is every
        # bias not in its own training. A bias trained on by some-but-not-all
        # panels (a "partial" bias in `order_biases_by_panels`) is held-out
        # for the panels that didn't train on it, so it belongs in their
        # held-out average — even though it still gets its own bias column on
        # the shared x-axis. Both panels share one "Held-out Avg" column at
        # the right but the value differs per panel.
        new_panels = []
        any_added = False
        add_summary = add_pct_change_held_out_avg if zero_line else add_held_out_avg
        for p in panels:
            held_for_p = [b for b in all_biases if b not in p.highlighted_biases]
            p_data = p.data if p.data is not None else sub
            if held_for_p:
                p_data = add_summary(p_data, held_for_p, label=held_out_avg_label)
                any_added = True
            new_panels.append(
                Panel(
                    label=p.label, training_types=p.training_types, data=p_data, highlighted_biases=p.highlighted_biases
                )
            )
        panels = new_panels
        if any_added:
            bias_order = bias_order + [held_out_avg_label]
    if not bias_order:
        return

    # Detect ambiguous duplicates *per panel* — more than one row per
    # (bias_type, training_type) within a panel means bars would overlap in
    # identical style. The check runs against each panel's own data slice so
    # that a faceter splitting on model_family / prompt_style suppresses the
    # error correctly (the global frame may carry dups that the faceter then
    # separates into distinct panels).
    for _p in panels:
        _panel_df = _p.data if _p.data is not None else sub
        _dup_check_cols = [
            c for c in ("model_family", "prompt_style") if c in _panel_df.columns and _panel_df[c].nunique() > 1
        ]
        if not _dup_check_cols:
            continue
        per_cluster = _panel_df.groupby(["bias_type", "training_type"], dropna=False).size()
        if (per_cluster > 1).any():
            raise ValueError(
                f"Panel {_p.label!r}: multiple rows per (bias_type, "
                f"training_type). Column(s) {_dup_check_cols} have >1 unique "
                f"value within the panel. Either filter (e.g. "
                f"prompt_style='cot') or facet on it via "
                f"`faceter=facet_by_column('{_dup_check_cols[0]}')`."
            )

    # Build wide pivots for fast lookup + n.csv emission. Use pivot_table with
    # aggfunc='first' to tolerate dups in the global frame when panels split
    # them on model_family / prompt_style (each panel pivots its own slice
    # downstream anyway). Behavior matches legacy `pivot` when there are no
    # cross-panel dups: NaN-only groups remain in the index because we reindex
    # to `bias_order` below.
    def _safe_pivot(value_col: str) -> pd.DataFrame:
        return sub.pivot_table(
            index="bias_type", columns="training_type", values=value_col, aggfunc="first", dropna=False
        )

    pivot_val = _safe_pivot("value")
    pivot_err = _safe_pivot("stderr")
    pivot_n = _safe_pivot("n")
    pivot_val = pivot_val.reindex(bias_order)
    pivot_err = pivot_err.reindex(bias_order)
    pivot_n = pivot_n.reindex(bias_order)

    # ── 4. ylim ───────────────────────────────────────────────────────────
    if ylim is None:
        # For grouped bars: ymin=min(0, min−.05), ymax=max+.10
        # For ratio (when zero_line): include errorbar extents, pad ±10
        # Derive from raw `sub` rather than pivot_val: pivot_val uses
        # aggfunc='first' to tolerate cross-panel dups (multi-model frames),
        # so its cells may not reflect the panel-level maxima.
        vals = sub["value"].to_numpy(dtype=float)
        vals = vals[~np.isnan(vals)]
        if zero_line:
            vals_flat = sub["value"].to_numpy(dtype=float)
            errs_flat = sub["stderr"].to_numpy(dtype=float)
            errs_flat = np.where(np.isnan(errs_flat), 0, errs_flat)
            extents = np.concatenate([vals_flat - 2 * errs_flat, vals_flat + 2 * errs_flat])
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
        # Reserve extra headroom when n_labels and/or significance markers are
        # drawn — vertical "n=NNN" text needs room above the tallest bar+
        # errorbar; significance stars sit just outside the whisker. For
        # ratio plots where bars cross 0, headroom is added on whichever
        # side(s) have bars (negative bars annotate below their bottom
        # whisker, so they need bottom headroom rather than top).
        has_pos = bool(vals.size) and float(vals.max()) >= 0 if vals.size else True
        has_neg = bool(vals.size) and float(vals.min()) < 0 if vals.size else False
        pad_frac = theme.n_label_headroom if n_labels else theme.sig_headroom if show_significance else 0.0
        if pad_frac > 0:
            span = ymax - ymin
            if has_pos:
                ymax = ymax + pad_frac * span
            if has_neg:
                ymin = ymin - pad_frac * span
        ylim = (ymin, ymax)

    # ── 5. Geometry ───────────────────────────────────────────────────────
    # Auto-grow bias_spacing when any panel's cluster (n_bars × bar_w + gaps)
    # would overflow theme.bias_spacing — keeps clusters from colliding when
    # a panel has many training types or many method families.
    n_panels = len(panels)
    bar_w = theme.bar_width
    cluster_gap = theme.cluster_gap

    def _cluster_width(types: tuple[str, ...]) -> float:
        families: dict[str, int] = {}
        for t in types:
            families.setdefault(training_type_info(t).method, 0)
            families[training_type_info(t).method] += 1
        n_bars = sum(families.values())
        n_gaps = max(0, len(families) - 1)
        return n_bars * bar_w + n_gaps * cluster_gap

    max_cw = max((_cluster_width(p.training_types) for p in panels), default=0.0)
    bias_spacing = max(theme.bias_spacing, max_cw + theme.cluster_pad)
    spacing_ratio = bias_spacing / theme.bias_spacing

    fig_w = max(
        theme.figure_width_min,
        theme.figure_width_per_bias * len(bias_order) * spacing_ratio + theme.figure_width_intercept,
    )
    fig_h = theme.figure_height_per_panel * n_panels + theme.figure_height_intercept
    fig, axes = plt.subplots(n_panels, 1, figsize=(fig_w, fig_h), sharex=True, sharey=True, squeeze=False)
    axes = axes.flatten()

    bias_x = np.arange(len(bias_order)) * bias_spacing
    half_span = bias_spacing / 2
    legend_handles = _simplified_legend(training_types, theme)

    # ── 6. Per-panel rendering ────────────────────────────────────────────
    any_significance = False
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
            p_val = panel.data.pivot(index="bias_type", columns="training_type", values="value").reindex(bias_order)
            p_err = panel.data.pivot(index="bias_type", columns="training_type", values="stderr").reindex(bias_order)
            p_n = panel.data.pivot(index="bias_type", columns="training_type", values="n").reindex(bias_order)
        else:
            p_val, p_err, p_n = pivot_val, pivot_err, pivot_n

        # Background bands: cream for highlighted biases, grey for held-out summary
        for j, bt in enumerate(bias_order):
            x_center = bias_x[j]
            if bt in panel.highlighted_biases:
                ax.axvspan(
                    x_center - half_span, x_center + half_span, color=theme.panel_bg_trained, alpha=0.7, zorder=0
                )
            elif bt == held_out_avg_label:
                ax.axvspan(
                    x_center - half_span, x_center + half_span, color=theme.panel_bg_summary, alpha=0.9, zorder=0
                )

        # Hairlines between bias columns
        for j in range(1, len(bias_order)):
            ax.axvline(x=bias_x[j] - half_span, color=theme.hairline_color, linewidth=theme.hairline_lw, zorder=1)

        # Per-panel base row used by the absolute-value significance test.
        # For ratio plots (zero_line=True) the test is one-sample vs 0 and
        # doesn't need a base.
        has_base = (significance_baseline in p_val.columns) if not zero_line else False

        # Bars + error bars + optional significance markers
        y_range = (ylim[1] - ylim[0]) if ylim else 1.0
        # `sig_offset` is the distance (in y-range fractions) from the
        # whisker to the closest star — fixed regardless of how many stars
        # appear above it.
        sig_offset = 0.02
        for tt in panel_types:
            offset = offsets[tt]
            style = theme.bar_style_for(tt)
            for j, x_val in enumerate(bias_x):
                bt = bias_order[j]
                if bt not in p_val.index or tt not in p_val.columns:
                    continue
                val = p_val.loc[bt, tt]
                err = p_err.loc[bt, tt] if (tt in p_err.columns and bt in p_err.index) else np.nan
                if pd.isna(val):
                    continue
                xp = x_val + offset
                bars = ax.bar(
                    xp,
                    val,
                    bar_w,
                    facecolor=style["facecolor"],
                    edgecolor=style["edgecolor"],
                    linewidth=style["linewidth"],
                    zorder=2,
                )
                # Clip the stroke to the bar's own path so the outline renders
                # inside the bar (otherwise half the stroke sits outside the
                # bar's nominal width and controls look wider than treatments).
                if style["linewidth"]:
                    for _b in bars:
                        _b.set_clip_path(_b)
                err_abs = float(err) if pd.notna(err) and err > 0 else 0.0
                if err_abs > 0:
                    ax.errorbar(
                        xp,
                        val,
                        yerr=2 * err_abs,
                        fmt="none",
                        ecolor=theme.error_color,
                        capsize=theme.error_capsize,
                        linewidth=theme.error_linewidth,
                        alpha=theme.error_alpha,
                        zorder=3,
                    )
                # Pick the "outside" anchor + direction once: positive bars
                # annotate above the top whisker, negative bars below the
                # bottom whisker. This keeps stars and n-labels visible on
                # ratio plots where bars cross 0.
                if val >= 0:
                    outside_y = val + 2 * err_abs
                    direction = 1
                    text_va = "bottom"
                else:
                    outside_y = val - 2 * err_abs
                    direction = -1
                    text_va = "top"

                # Significance star (skip for the baseline itself).
                marker = ""
                if show_significance and tt != significance_baseline:
                    if zero_line:
                        # Ratio plot — test % change ≠ 0 with the bar's own SE
                        p = one_sample_p(float(val), err_abs)
                    elif has_base:
                        base_val = p_val.loc[bt, significance_baseline] if bt in p_val.index else np.nan
                        base_err = (
                            p_err.loc[bt, significance_baseline]
                            if (bt in p_err.index and significance_baseline in p_err.columns)
                            else 0.0
                        )
                        if pd.notna(base_val):
                            p = two_sample_p(
                                float(val),
                                err_abs,
                                float(base_val),
                                float(base_err) if pd.notna(base_err) else 0.0,
                            )
                        else:
                            p = None
                    else:
                        p = None
                    marker = significance_marker(p)
                    if marker:
                        any_significance = True
                        # Stack stars vertically via newlines, anchored at
                        # the bar-facing edge (va="bottom" for positive bars
                        # → bottom of last line at the anchor; va="top" for
                        # negative bars → top of first line at the anchor).
                        # The closest star to the whisker is therefore always
                        # at the same offset (`sig_offset`) regardless of how
                        # many stars; additional stars stack outward.
                        ax.text(
                            xp,
                            outside_y + direction * sig_offset * y_range,
                            "\n".join(marker),
                            ha="center",
                            va=text_va,
                            fontsize=theme.sig_fontsize,
                            linespacing=0.3,
                            color=theme.error_color,
                            zorder=4,
                        )

                if n_labels and tt in p_n.columns and bt in p_n.index:
                    n_val = p_n.loc[bt, tt]
                    if pd.notna(n_val):
                        # Put n beyond any significance stack so the two
                        # annotations never overlap. With no marker this
                        # reduces to the legacy 2%-of-range offset.
                        marker_clearance = 0.03 * len(marker) if marker else 0.0
                        n_offset = (0.02 + marker_clearance) * y_range
                        ax.text(
                            xp,
                            outside_y + direction * n_offset,
                            f"n={int(n_val)}",
                            ha="center",
                            va=text_va,
                            fontsize=theme.n_label_fontsize,
                            rotation=90,
                            color=theme.error_color,
                            zorder=4,
                        )

        # Axes formatting
        ax.set_xticks(bias_x)
        ax.set_xticklabels(
            [_bias_label(b, held_out_avg_label) for b in bias_order],
            rotation=45,
            ha="right",
            fontsize=theme.xtick_label_fontsize,
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
            ax.set_title(
                panel.label,
                loc="left",
                fontsize=theme.panel_title_fontsize,
                fontweight="semibold",
                color="#141518",
                pad=1.5,
            )

        # Reference lines
        if baseline_hline is not None:
            ax.axhline(y=baseline_hline, color=theme.baseline_color, linestyle=":", linewidth=1, alpha=0.6, zorder=1)
            if baseline_band is not None:
                ax.axhspan(baseline_band[0], baseline_band[1], color=theme.baseline_color, alpha=0.08, zorder=0)
        if random_line is not None:
            ax.axhline(y=random_line, color=theme.baseline_color, linestyle="--", alpha=0.5, linewidth=0.8, zorder=1)
        if zero_line:
            ax.axhline(y=0, color=theme.zero_line_color, linewidth=1, zorder=1)

    # ── 7. Layout + legend + save ─────────────────────────────────────────
    # Lay out axes first to use the full figure top, then anchor the legend
    # *above* the figure (y > 1.0) so it doesn't overlap panel titles. The
    # `bbox_inches="tight"` on savefig crops the canvas back to include the
    # legend.
    # Reserve a narrow figure-level band for the significance key. Without
    # this, a long left-aligned panel title can run directly into the key at
    # the top-right (especially for single-panel Llama figures).
    top = 0.94 if any_significance else 1.0
    fig.tight_layout(pad=0.15, h_pad=1.4, rect=(0.0, 0.0, 1.0, top))
    fig.legend(
        handles=legend_handles,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.0),
        ncol=len(legend_handles),
        frameon=False,
        fontsize=7,
        handlelength=1.0,
        columnspacing=0.9,
        labelspacing=0.2,
        borderpad=0.2,
    )

    # Significance-marker key, anchored to the top-right inside the figure
    # so it sits just below the bar legend (which lives above y=1.0). Drawn
    # only when stars are actually drawn, so non-significance plots aren't
    # littered with a stray annotation.
    if any_significance:
        fig.text(
            0.99,
            1.0,
            "* p<0.05   ** p<0.01   *** p<0.001",
            ha="right",
            va="top",
            fontsize=6.5,
            color=theme.error_color,
            transform=fig.transFigure,
        )

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved: {out}")
    else:
        plt.show()
    plt.close(fig)
