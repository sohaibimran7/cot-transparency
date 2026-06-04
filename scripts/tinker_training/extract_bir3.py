"""Population net BIR for the SNR-scaling RLCT experiment, computed from .eval logs.

Single source of truth: this routes through
visualize_results.compute_per_question_bir + collapse_to_population_bir(signed=True)
(per-question *matched* net BIR), instead of a private re-implementation of the loop.
`make_snr_scaling_figure.py` imports compute_snr_scaling_bir() from here so the table and
the figure can never disagree.

bias_type is derived from dataset_path's parent dir (see eval_log_loader.iter_eval_samples).
"""
from pathlib import Path


from sycophancy_eval_inspect.visualize_results import (  # noqa: E402
    collapse_to_population_bir,
    compute_per_question_bir,
)

# Absolute so the figure/table work regardless of the caller's cwd.
BASE = str(Path(__file__).resolve().parents[2] / "sycophancy_eval_inspect" / "logs" / "snr_scaling_exp")
# Eval-log dir name -> short model label used by the table/figure.
MODELS = {
    "llama-base": "base",
    "llama-rlct-sa-grpo": "grpo",
    "llama-rlct-sa-snr": "snr",
    "llama-rlct-sa-snr-pop": "snr_pop",
}
BIASES = ["suggested_answer", "wrong_few_shot", "distractor_argument",
          "distractor_fact", "spurious_few_shot_squares"]
TRAINED_BIAS = "suggested_answer"  # the rest are held-out


def heldout_avg(model_bir: dict) -> float | None:
    """Mean net BIR over held-out (non-trained) biases for one model; None if none present."""
    vals = [model_bir[b] for b in BIASES if b != TRAINED_BIAS and model_bir.get(b) is not None]
    return sum(vals) / len(vals) if vals else None


def overall_avg(model_bir: dict) -> float | None:
    """Mean net BIR over all present biases for one model; None if none present."""
    vals = [model_bir[b] for b in BIASES if model_bir.get(b) is not None]
    return sum(vals) / len(vals) if vals else None


def compute_snr_scaling_bir(base_dir: str = BASE):
    """Compute net population BIR for the SNR-scaling experiment.

    Returns (bir, ub):
      bir[short_label][bias_type] -> net population BIR (mean biased - mean unbiased), or None
      ub[short_label]             -> unbiased baseline BMR (mean unbiased_bmr), or None
    """
    df = compute_per_question_bir(base_dir)
    pop = collapse_to_population_bir(df, signed=True)

    bir: dict[str, dict[str, float | None]] = {}
    ub: dict[str, float | None] = {}
    for dir_name, short in MODELS.items():
        sub = pop[pop["model"] == dir_name]
        bir[short] = {}
        for bt in BIASES:
            cell = sub[sub["bias_type"] == bt]["bir"]
            bir[short][bt] = float(cell.mean()) if not cell.empty else None
        # One unbiased value per question: the per-question frame repeats it across
        # bias_type rows, so dedupe by hash to avoid coverage-weighting the baseline.
        dfm = df[df["model"] == dir_name].drop_duplicates("hash")["unbiased_bmr"].dropna()
        ub[short] = float(dfm.mean()) if not dfm.empty else None
    return bir, ub


def _print_table(bir, ub):
    models = list(MODELS.values())

    def _fmt(x):
        return f"{x:>11.3f}" if x is not None else f"{'NA':>11}"

    print("UNBIASED matches_bias:", {m: round(ub[m], 3) if ub[m] is not None else None for m in models})
    print(f"\n{'bias':28}" + "".join(f"{m:>11}" for m in models))
    for b in BIASES:
        print(f"{b:28}" + "".join(_fmt(bir[m].get(b)) for m in models))
    print(f"\n{'HELD-OUT avg':28}" + "".join(_fmt(heldout_avg(bir[m])) for m in models))
    print(f"{'OVERALL avg':28}" + "".join(_fmt(overall_avg(bir[m])) for m in models))


if __name__ == "__main__":
    _print_table(*compute_snr_scaling_bir())
