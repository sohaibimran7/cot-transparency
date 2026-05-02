"""Visualization package for sycophancy eval results.

Pipeline: raw_logs → MetricFrame (long-form) → transforms → bar_plot(theme).

`BIAS_DISPLAY_NAMES` is re-exported for `scripts/plot_model_comparison.py`,
which uses it as a sort key on bias columns. New display names added to the
registry (`experiments.toml` / YAML viz_registration) appear automatically.
"""
from .registry import REGISTRY


BIAS_DISPLAY_NAMES = {k: v.display_name for k, v in REGISTRY.biases.items()}
