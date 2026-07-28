"""Transforms: MetricFrame → MetricFrame.

Each function is a pure pandas transform. Compose with `.pipe()`.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, sqrt
from typing import Callable

import numpy as np
import pandas as pd

from .registry import REGISTRY, training_type_info


def two_sided_p(z: float) -> float:
    """Two-sided p-value for a standard-normal test statistic.

    p = 2 * (1 - Phi(|z|)) = 1 - erf(|z| / sqrt(2)).
    """
    if not np.isfinite(z):
        return 1.0
    return float(1.0 - erf(abs(z) / sqrt(2.0)))


def significance_marker(p: float | None) -> str:
    """Return ``*``/``**``/``***`` for p<0.05/0.01/0.001 thresholds. ``""`` otherwise."""
    if p is None or not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def two_sample_p(val_a: float, se_a: float, val_b: float, se_b: float) -> float | None:
    """Two-sample z-test p-value for ``val_a − val_b``.

    Returns ``None`` if any input is NaN or the combined SE is non-positive
    (no test can be performed).
    """
    if not (np.isfinite(val_a) and np.isfinite(val_b)):
        return None
    se_a = 0.0 if not np.isfinite(se_a) else float(se_a)
    se_b = 0.0 if not np.isfinite(se_b) else float(se_b)
    se = (se_a**2 + se_b**2) ** 0.5
    if se <= 0:
        return None
    return two_sided_p((val_a - val_b) / se)


def one_sample_p(val: float, se: float, mu: float = 0.0) -> float | None:
    """Single-sample z-test p-value for ``val`` vs the constant ``mu``.

    Used for ratio plots (test % change ≠ 0). Returns ``None`` when SE is
    missing or non-positive.
    """
    if not np.isfinite(val) or not np.isfinite(se) or se <= 0:
        return None
    return two_sided_p((val - mu) / float(se))


def filter_frame(frame: pd.DataFrame, **selectors) -> pd.DataFrame:
    """Filter rows by exact column equality. None values skip the filter."""
    out = frame
    for col, val in selectors.items():
        if val is None or col not in out.columns:
            continue
        out = out[out[col] == val]
    return out.copy()


def _associated_controls(trained_tt: str, all_tts: list[str]) -> list[str]:
    """Controls in `all_tts` whose registry `control_for` points at trained_tt."""
    return [tt for tt in all_tts if training_type_info(tt).control_for == trained_tt]


def _is_associated_control(tt: str, all_tts: list[str]) -> bool:
    """True iff `tt` is registered as a control for some trained type that
    is present in `all_tts` and itself carries training_biases."""
    info = training_type_info(tt)
    base = info.control_for
    if base is None or base not in all_tts:
        return False
    return bool(training_type_info(base).training_biases)


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
Faceter = Callable[[pd.DataFrame], list[Panel]]


def facet_by_column(
    column: str,
    *,
    label: str | Callable[[object], str] = "{value}",
    annotate: Callable[[object], dict] | None = None,
    sort_key: Callable[[object], object] | None = None,
) -> Faceter:
    """Faceter: one panel per unique value of `column`.

    The fundamental panel split. All other faceters (training-biases,
    no-facet) are configurations of this one.

    Parameters:
      column        — the frame column to partition on.
      label         — str template (`{value}`/`{column}` placeholders) OR a
                      callable `value → str` for non-string panel keys.
      annotate      — callable `value → dict` for per-panel annotations
                      (e.g. `highlighted_biases` for cream bands).
      sort_key      — explicit sort key for panel ordering. Required when
                      values aren't natively sortable (e.g. frozensets).

    Used directly for model_family / prompt_style / seed faceting; used via
    composition with `with_training_biases_column` for the publication
    training-bias-set faceter.
    """

    def faceter(frame: pd.DataFrame) -> list[Panel]:
        if column not in frame.columns:
            return []
        values = list(frame[column].dropna().unique())
        try:
            values.sort(key=sort_key) if sort_key else values.sort()
        except TypeError:
            pass  # leave unsorted if not orderable
        out: list[Panel] = []
        for value in values:
            sub = frame[frame[column] == value]
            tts_in = set(sub["training_type"].unique())
            tts = tuple(t for t in REGISTRY.training_type_order if t in tts_in)
            if not tts:
                continue
            label_str = label(value) if callable(label) else label.format(value=value, column=column)
            anno = annotate(value) if annotate else {}
            out.append(
                Panel(
                    label=label_str,
                    training_types=tts,
                    data=sub,
                    highlighted_biases=frozenset(anno.get("highlighted_biases", ())),
                )
            )
        return out

    return faceter


def with_training_biases_column(
    frame: pd.DataFrame,
    *,
    key_column: str = "training_biases_set",
) -> pd.DataFrame:
    """Add a `training_biases_set` column for column-based faceting.

    For each row, decide which training-bias-set panel(s) it belongs to:
      - trained types         → one row, panel_key = its training_biases set
      - associated controls   → one row, panel_key = the trained's bias set
      - shared types (base)   → duplicated, one row per panel_key set in data

    Result: `facet_by_column(key_column)` on the enriched frame produces the
    legacy publication panel layout. Returns an unchanged frame if no
    training_type carries any training_biases (i.e. nothing to facet by).
    """
    tts_in_data = list(frame["training_type"].unique())
    bias_sets: set[frozenset[str]] = {
        info.training_biases for info in (training_type_info(t) for t in tts_in_data) if info.training_biases
    }
    if not bias_sets:
        return frame  # no faceting possible

    def panel_keys_for(tt: str) -> list[frozenset[str]]:
        info = training_type_info(tt)
        if info.training_biases:
            return [info.training_biases]
        if info.control_for and info.control_for in tts_in_data:
            return [training_type_info(info.control_for).training_biases]
        # Shared (base, etc.): duplicate into every panel
        return list(bias_sets)

    pieces = []
    for tt, group in frame.groupby("training_type", sort=False):
        keys = panel_keys_for(tt)
        for key in keys:
            piece = group.copy()
            piece[key_column] = [key] * len(piece)
            pieces.append(piece)
    return pd.concat(pieces, ignore_index=True)


def _format_bias_set(biases: frozenset[str]) -> str:
    return " + ".join(REGISTRY.biases[b].display_name if b in REGISTRY.biases else b for b in sorted(biases))


def facet_by_training_biases(frame: pd.DataFrame) -> list[Panel]:
    """Default publication faceter: one panel per unique training-bias set.

    Now expressed as a composition of `with_training_biases_column` (which
    handles the shared-types-in-every-panel + controls-go-with-trained
    semantics via row duplication) and `facet_by_column` (the generic
    column-based panel split).
    """
    enriched = with_training_biases_column(frame)
    if "training_biases_set" not in enriched.columns:
        return []
    return facet_by_column(
        "training_biases_set",
        label=lambda biases: f"Trained on {_format_bias_set(biases)}",
        annotate=lambda biases: {"highlighted_biases": biases},
        sort_key=lambda biases: (len(biases), tuple(sorted(biases))),
    )(enriched)


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


def order_biases_by_panels(all_biases: list[str], panels: list[Panel]) -> tuple[list[str], list[str]]:
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


def add_held_out_avg(
    agg: pd.DataFrame,
    held_out_biases: list[str],
    label: str = "Held-out Avg",
) -> pd.DataFrame:
    """Append synthetic 'Held-out Avg' rows pooled over held-out biases.

    Treats the union of held-out biases as one big sample:
      mean   = sum(n_i * m_i) / sum(n_i)                # n-weighted mean
      var    = sum(n_i * (sigma_i² + (m_i − mean)²)) / N   # law of total variance
      stderr = sqrt(var / N)
      n      = sum(n_i)

    Equivalent to recomputing nanmean / SE on the pooled per-question
    values. Gives the binomial Wald SE when each bias is binary, and the
    correct sample SEM when the metric is continuous. When n_i are equal,
    `value` matches the unweighted mean of per-bias means; `stderr`
    converges to sqrt((sum SE_i²)/k²) only in the equal-mean case.

    Operates per (model_family, training_type, prompt_style, metric) group.
    """
    if not held_out_biases:
        return agg
    sub = agg[agg["bias_type"].isin(held_out_biases)]
    if sub.empty:
        return agg
    group_cols = [c for c in ("model_family", "training_type", "prompt_style", "metric") if c in agg.columns]

    rows = []
    for keys, grp in sub.groupby(group_cols, dropna=False):
        valid = grp.dropna(subset=["value", "n"])
        if valid.empty:
            continue
        m_i = valid["value"].to_numpy(dtype=float)
        n_i = valid["n"].to_numpy(dtype=float)
        se_i = valid["stderr"].fillna(0.0).to_numpy(dtype=float)
        N = float(n_i.sum())
        if N <= 0:
            continue
        pooled_mean = float((n_i * m_i).sum() / N)
        # sigma_i² (per-element variance) recovered from SE_i: sigma² = SE² · n
        sigma_sq_i = (se_i**2) * n_i
        # Variance of the pooled sample:
        #   Var(pool) = (1/N) · sum_i n_i · (sigma_i² + (m_i − pooled_mean)²)
        pooled_var = float((n_i * (sigma_sq_i + (m_i - pooled_mean) ** 2)).sum() / N)
        pooled_se = float((pooled_var / N) ** 0.5) if N > 0 else 0.0
        rec = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        rec.update(
            bias_type=label,
            value=pooled_mean,
            stderr=pooled_se,
            n=int(N),
        )
        rows.append(rec)
    if not rows:
        return agg
    return pd.concat([agg, pd.DataFrame(rows)], ignore_index=True)


def pct_change_vs_baseline(
    agg: pd.DataFrame,
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

    group_cols = [c for c in ("model_family", "prompt_style", "bias_type", "metric") if c in agg.columns]
    out_rows = []
    for keys, grp in agg.groupby(group_cols, dropna=False):
        base_row = grp[grp["training_type"] == baseline_training_type]
        base_val = float(base_row["value"].iloc[0]) if not base_row.empty else np.nan
        base_err = float(base_row["stderr"].iloc[0]) if not base_row.empty else np.nan
        base_ok = not base_row.empty and np.isfinite(base_val) and base_val != 0
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
                    # Delta method written in derivative form so the
                    # numerator may be exactly zero. The relative-error form
                    # (SE_a / a) is undefined there and used to incorrectly
                    # collapse the propagated uncertainty to zero.
                    base_err_f = base_err if pd.notna(base_err) else 0.0
                    ratio_var = (tt_err / base_val) ** 2 + (tt_val * base_err_f / (base_val**2)) ** 2
                    pct_err = ratio_var**0.5 * 100
            else:
                # Preserve the (bt, tt) row with NaN so the bias type column
                # still renders (matching legacy behavior).
                pct_val = np.nan
                pct_err = np.nan
            new_row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
            new_row.update(
                training_type=tt,
                value=pct_val,
                stderr=pct_err,
                n=int(row["n"]) if pd.notna(row["n"]) else 0,
            )
            out_rows.append(new_row)
    return pd.DataFrame(out_rows) if out_rows else agg.iloc[:0].copy()


def add_pct_change_held_out_avg(
    pct: pd.DataFrame,
    held_out_biases: list[str],
    label: str = "Held-out Avg",
) -> pd.DataFrame:
    """For pct-change rows, add a Held-out Avg row per (mf, ps, tt, metric).

    Uses simple mean of pct values; SE propagated as sqrt(sum SE_i²)/k. This
    matches legacy `plot_metric_ratio_publication` behavior — averaging the
    per-bias % changes (not re-deriving from averaged means/baselines).
    """
    if not held_out_biases:
        return pct
    sub = pct[pct["bias_type"].isin(held_out_biases)]
    if sub.empty:
        return pct

    group_cols = [c for c in ("model_family", "training_type", "prompt_style", "metric") if c in pct.columns]
    rows = []
    for keys, grp in sub.groupby(group_cols, dropna=False):
        valid = grp.dropna(subset=["value"])
        if valid.empty:
            continue
        values = valid["value"].to_numpy(dtype=float)
        errors = valid["stderr"].fillna(0.0).to_numpy(dtype=float)
        rec = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        rec.update(
            bias_type=label,
            value=float(values.mean()),
            stderr=float((errors**2).sum() ** 0.5 / len(values)),
            n=int(valid["n"].fillna(0).sum()),
        )
        rows.append(rec)
    if not rows:
        return pct
    return pd.concat([pct, pd.DataFrame(rows)], ignore_index=True)
