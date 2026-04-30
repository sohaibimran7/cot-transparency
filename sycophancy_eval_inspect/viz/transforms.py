"""Transforms: MetricFrame → MetricFrame.

Each function is a pure pandas transform. Compose with `.pipe()`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .registry import REGISTRY, training_type_info


def filter_frame(frame: pd.DataFrame, **selectors) -> pd.DataFrame:
    """Filter rows by exact column equality. None values skip the filter."""
    out = frame
    for col, val in selectors.items():
        if val is None or col not in out.columns:
            continue
        out = out[out[col] == val]
    return out.copy()


def _associated_controls(trained_tt: str, all_tts: list[str]) -> list[str]:
    """Controls matched to a trained tt (by `_ctrl` / `_control` suffix)."""
    return [c for c in (trained_tt + "_ctrl", trained_tt + "_control")
            if c in all_tts]


def _is_associated_control(tt: str, all_tts: list[str]) -> bool:
    if tt.endswith("_ctrl"):
        base = tt[:-5]
    elif tt.endswith("_control"):
        base = tt[:-8]
    else:
        return False
    return base in all_tts and bool(training_type_info(base).training_biases)


@dataclass(frozen=True)
class Panel:
    """One panel of a faceted plot.

    The renderer uses these fields:
      label              — header text drawn above the panel
      training_types     — ordered subset of training types whose bars to draw
      data               — rows belonging to this panel (can be the full frame
                           when faceting only by training_types)
      highlighted_biases — bias keys to draw with a cream background band
                           (used by `facet_by_training_biases` for trained-on
                            biases; empty for other faceters)
    """
    label: str
    training_types: tuple[str, ...]
    data: pd.DataFrame | None = None
    highlighted_biases: frozenset[str] = frozenset()


# A Faceter takes the long-form aggregated frame and returns the panel split.
# All faceters compose the same way; bar_plot is agnostic to which is used.
from typing import Callable
Faceter = Callable[[pd.DataFrame], list[Panel]]


def facet_by_training_biases(frame: pd.DataFrame) -> list[Panel]:
    """Default faceter: one panel per unique training-bias set.

    Mirrors the legacy publication layout: each panel contains its trained
    types + their associated controls + any untrained shared types (base,
    etc.) not pinned to a specific trained type. Panels ordered by
    (size of bias set, sorted bias keys). Highlighted biases = the trained
    biases of that panel — drives cream-band rendering.

    Returns an empty list if no training_type in `frame` has training_biases.
    """
    tts_in_data = set(frame["training_type"].unique())
    training_types = [t for t in REGISTRY.training_type_order if t in tts_in_data]

    groups: dict[frozenset[str], list[str]] = {}
    for tt in training_types:
        biases = training_type_info(tt).training_biases
        if biases:
            groups.setdefault(biases, []).append(tt)
    if not groups:
        return []

    shared = [tt for tt in training_types
              if not training_type_info(tt).training_biases
              and not _is_associated_control(tt, training_types)]

    sorted_groups = sorted(groups.items(),
                           key=lambda x: (len(x[0]), tuple(sorted(x[0]))))
    panels: list[Panel] = []
    for biases, trained_tts in sorted_groups:
        ordered: list[str] = []
        for tt in trained_tts:
            ordered.append(tt)
            for ctrl in _associated_controls(tt, training_types):
                if ctrl not in ordered:
                    ordered.append(ctrl)
        for tt in shared:
            if tt not in ordered:
                ordered.append(tt)
        ordered = [t for t in REGISTRY.training_type_order if t in ordered]
        bias_label = " + ".join(
            REGISTRY.biases[b].display_name if b in REGISTRY.biases else b
            for b in sorted(biases)
        )
        panels.append(Panel(
            label=f"Trained on {bias_label}",
            training_types=tuple(ordered),
            data=frame,                      # full frame; tts subset narrows it
            highlighted_biases=biases,
        ))
    return panels


def facet_by_column(column: str, *, label_template: str = "{value}") -> Faceter:
    """Faceter factory: one panel per unique value of `column`.

    The panel's data is the slice where frame[column] == value. All training
    types present in the slice are rendered in registry order. No bias
    highlighting (publication cream bands are training-bias-specific).

    Use for non-default faceting dimensions: model_family, prompt_style, seed.
    """
    def faceter(frame: pd.DataFrame) -> list[Panel]:
        if column not in frame.columns:
            return []
        out: list[Panel] = []
        for value in sorted(frame[column].dropna().unique()):
            sub = frame[frame[column] == value]
            tts_in = set(sub["training_type"].unique())
            tts = tuple(t for t in REGISTRY.training_type_order if t in tts_in)
            if not tts:
                continue
            out.append(Panel(
                label=label_template.format(value=value, column=column),
                training_types=tts,
                data=sub,
            ))
        return out
    return faceter


def no_facet(frame: pd.DataFrame) -> list[Panel]:
    """Single-panel faceter: all data in one panel, no header."""
    tts_in = set(frame["training_type"].unique())
    tts = tuple(t for t in REGISTRY.training_type_order if t in tts_in)
    return [Panel(label="", training_types=tts, data=frame)]


# Back-compat alias for callers that imported `panels_for_training_types`.
def panels_for_training_types(training_types: list[str]) -> list[Panel]:
    """Deprecated: build panels for a flat training_types list.

    Retained so older callers keep working. New code should use a Faceter.
    """
    # Build a minimal stub frame so facet_by_training_biases can run
    stub = pd.DataFrame({"training_type": list(training_types)})
    return facet_by_training_biases(stub)


def order_biases_by_panels(all_biases: list[str], panels: list[Panel]
                           ) -> tuple[list[str], list[str]]:
    """Order: panel-intersection-trained → partial-trained → held-out.

    Returns (ordered_biases, held_out_biases).
    """
    if not panels:
        return list(all_biases), list(all_biases)
    sets = [set(p.highlighted_biases) for p in panels]
    intersection: set[str] = set.intersection(*sets)
    union: set[str] = set.union(*sets)
    intersect = [b for b in all_biases if b in intersection]
    partial = [b for b in all_biases if b in union and b not in intersection]
    held = [b for b in all_biases if b not in union]
    return intersect + partial + held, held


def add_held_out_avg(agg: pd.DataFrame,
                     held_out_biases: list[str],
                     label: str = "Held-out Avg",
                     ) -> pd.DataFrame:
    """Append synthetic 'Held-out Avg' rows averaged over held-out biases.

    SE of mean = sqrt(sum SE_i^2) / k. n = sum.
    Operates per (model_family, training_type, prompt_style, metric) group.
    """
    if not held_out_biases:
        return agg
    sub = agg[agg["bias_type"].isin(held_out_biases)]
    if sub.empty:
        return agg
    group_cols = [c for c in ("model_family", "training_type", "prompt_style", "metric")
                  if c in agg.columns]

    rows = []
    for keys, grp in sub.groupby(group_cols, dropna=False):
        vals = grp["value"].dropna()
        errs = grp["stderr"].dropna()
        ns = grp["n"].dropna()
        if vals.empty:
            continue
        k = len(vals)
        rec = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        rec.update(
            bias_type=label,
            value=float(vals.mean()),
            stderr=float(((errs ** 2).sum() ** 0.5) / k),
            n=int(ns.sum()),
        )
        rows.append(rec)
    if not rows:
        return agg
    return pd.concat([agg, pd.DataFrame(rows)], ignore_index=True)


def pct_change_vs_baseline(agg: pd.DataFrame,
                           baseline_training_type: str = "base",
                           ) -> pd.DataFrame:
    """For each (mf, ps, bt, metric), produce % change rows vs baseline tt.

    Drops the baseline rows and rows with baseline value 0 / NaN. SE is
    propagated via the standard ratio formula:
        SE(R) = |R| * sqrt((SE_a/a)² + (SE_b/b)²)
        SE(% change) = SE(R) * 100

    Uses the `uncertainties` package when available; falls back to manual math
    otherwise (manual is what the legacy code did).
    """
    try:
        from uncertainties import ufloat
        _have_uncertainties = True
    except ImportError:
        _have_uncertainties = False

    group_cols = [c for c in ("model_family", "prompt_style", "bias_type", "metric")
                  if c in agg.columns]
    out_rows = []
    for keys, grp in agg.groupby(group_cols, dropna=False):
        base_row = grp[grp["training_type"] == baseline_training_type]
        base_val = float(base_row["value"].iloc[0]) if not base_row.empty else np.nan
        base_err = float(base_row["stderr"].iloc[0]) if not base_row.empty else np.nan
        base_ok = (not base_row.empty
                   and np.isfinite(base_val) and base_val != 0)
        for _, row in grp.iterrows():
            tt = row["training_type"]
            if tt == baseline_training_type:
                continue
            tt_val = float(row["value"]) if pd.notna(row["value"]) else np.nan
            tt_err = float(row["stderr"]) if pd.notna(row["stderr"]) else 0.0

            if base_ok and pd.notna(tt_val):
                if _have_uncertainties:
                    base_u = ufloat(base_val, base_err if pd.notna(base_err) else 0.0)
                    tt_u = ufloat(tt_val, tt_err)
                    ratio_u = tt_u / base_u
                    pct = (ratio_u - 1.0) * 100
                    pct_val = pct.nominal_value
                    pct_err = pct.std_dev
                else:
                    ratio = tt_val / base_val
                    pct_val = (ratio - 1.0) * 100
                    rel_a = (tt_err / tt_val) ** 2 if tt_val != 0 else 0
                    rel_b = (base_err / base_val) ** 2 if base_val != 0 else 0
                    pct_err = abs(ratio) * (rel_a + rel_b) ** 0.5 * 100
            else:
                # Preserve the (bt, tt) row with NaN so the bias type column
                # still renders (matching legacy behavior).
                pct_val = np.nan
                pct_err = np.nan
            new_row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
            new_row.update(
                training_type=tt, value=pct_val, stderr=pct_err,
                n=int(row["n"]) if pd.notna(row["n"]) else 0,
            )
            out_rows.append(new_row)
    return pd.DataFrame(out_rows) if out_rows else agg.iloc[:0].copy()


def add_pct_change_held_out_avg(pct: pd.DataFrame,
                                held_out_biases: list[str],
                                label: str = "Held-out Avg",
                                ) -> pd.DataFrame:
    """For pct-change rows, add a Held-out Avg row per (mf, ps, tt, metric).

    Uses simple mean of pct values; SE propagated as sqrt(sum SE_i²)/k. This
    matches legacy `plot_metric_ratio_publication` behavior — averaging the
    per-bias % changes (not re-deriving from averaged means/baselines).
    """
    return add_held_out_avg(pct, held_out_biases, label=label)
