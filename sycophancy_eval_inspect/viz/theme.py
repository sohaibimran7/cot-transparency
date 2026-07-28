"""Theme: rcParams + per-training-type bar styling.

Replaces the dual hatched-vs-outlined style systems in the legacy code.
A Theme is just a dataclass exposing:
  - apply()                 : install rcParams
  - bar_style_for(tt)       : returns dict for `ax.bar(**style)`
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .registry import training_type_info


@dataclass(frozen=True)
class Theme:
    rcparams: dict
    bar_style_for: Callable[[str], dict]
    ylabel_fontsize: float = 9.0
    panel_title_fontsize: float = 6.0
    xtick_label_fontsize: float = 7.0
    ytick_label_fontsize: float = 7.0
    bias_spacing: float = 0.6
    bar_width: float = 0.13
    cluster_gap: float = 0.05  # gap between method families inside a cluster
    cluster_pad: float = 0.12  # gap between adjacent clusters when
    # bias_spacing is auto-grown
    n_label_fontsize: float = 5.0  # "n=NNN" text above each bar
    n_label_headroom: float = 0.20  # extra ylim headroom when n_labels=True
    # (fraction of data range) — sized for
    # *** stack + n=… cleared above
    sig_fontsize: float = 7.0  # significance star (*, **, ***) text
    # — newline-stacked (each `*` upright,
    # centered on the bar), so the only
    # crowding direction is vertical
    sig_headroom: float = 0.16  # ylim headroom when significance is
    # drawn but no n_labels — sized for
    # *** stacked + a small visual buffer
    figure_width_intercept: float = 0.9
    figure_width_per_bias: float = 0.36
    figure_width_min: float = 3.4
    figure_height_per_panel: float = 2.1
    figure_height_intercept: float = 0.3
    panel_bg_trained: str = "#fbf6ea"
    panel_bg_summary: str = "#f5f4ef"
    hairline_color: str = "#d8dadf"
    hairline_lw: float = 0.6
    error_color: str = "#141518"
    error_capsize: float = 1.2
    error_linewidth: float = 0.6
    error_alpha: float = 0.7
    baseline_color: str = "#6b7280"
    zero_line_color: str = "#141518"

    def apply(self) -> None:
        import logging
        import matplotlib as mpl

        logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
        mpl.rcParams.update(self.rcparams)


# ── Publication theme ─────────────────────────────────────────────────────
# Mirrors PUBLICATION_RCPARAMS + PUBLICATION_PALETTE + _light_publication_style_for
# from the legacy module. Color choice ignores the training type's data_scale
# (panel headers convey "Trained on …"), so all bars within a method family
# share the lighter shade.

PUBLICATION_RCPARAMS = {
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "font.family": ["IBM Plex Sans", "Inter", "DejaVu Sans", "sans-serif"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "semibold",
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#ececee",
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "axes.edgecolor": "#141518",
    "axes.linewidth": 0.8,
}

PUBLICATION_PALETTE = {
    "base": "#9aa0a6",
    "bct_da": "#7aa7d9",
    "bct_dawfs": "#2e5f9a",
    "rlct_da": "#8cc39a",
    "rlct_dawfs": "#2f7a4d",
    "vft": "#9e9ac8",
}


def _publication_bar_style(training_type: str) -> dict:
    """Light shade per family (DA shade), regardless of data scale.

    Mirrors legacy `_light_publication_style_for`. Controls drawn as outlined
    bars: white facecolor with colored edge. The bar renderer clips the stroke
    to the bar's own path so the outline sits *inside* the bar's bounding box;
    `linewidth` here is doubled so the visible inside-half matches a 1.5pt
    stroke. Without this, controls' visual width = bar_width + linewidth and
    they look exaggerated relative to treatment bars.
    """
    info = training_type_info(training_type)
    method = info.method
    if method == "base":
        color = PUBLICATION_PALETTE["base"]
    elif method == "bct":
        color = PUBLICATION_PALETTE["bct_da"]
    elif method == "rlct":
        color = PUBLICATION_PALETTE["rlct_da"]
    elif method == "vft":
        color = PUBLICATION_PALETTE["vft"]
    else:
        color = info.color or "#888888"
    return {
        "facecolor": "white" if info.is_control else color,
        "edgecolor": color,
        "linewidth": 3.0 if info.is_control else 0.0,
    }


PUBLICATION_THEME = Theme(
    rcparams=PUBLICATION_RCPARAMS,
    bar_style_for=_publication_bar_style,
    ylabel_fontsize=7.5,
)
