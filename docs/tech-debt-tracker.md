# Tech-debt tracker

Refactor opportunities found during the 2026-06-01 audit but **not** done, so they stay
legible instead of being re-discovered. Line numbers are approximate (they drift); search
by symbol name. Effort S/M/L, Payoff low/med/high.

## Held — planned, gated on the user

These were in the approved refactor plan but deferred (see `.claude/plans/`):

- **D — eval_awareness sampling/IO duplication.** *Partly done (2026-06-03):* the shared
  `cot_transparency/eval_awareness/samplers.py` (`TinkerSampler`) now exists and is adopted by
  6 scripts (`calibrate.py`, `overrefusal_eval.py`, `run_eval.py`, `ifeval_gap.py`,
  `show_refusals.py`, `show_am_transcripts.py`) — the old "`calibrate.py` imports a private
  `_TinkerSampler` from `run_eval` via a `sys.path` hack" is resolved. **Residual:**
  `train_evalaware.py` still rolls its own `_text` token-parse on the *training* path (should use
  the shared sampler / `TinkerSamplingClient`), and JSONL/EAB-baseline loaders are still
  duplicated ~6× → `eval_awareness/io.py`. **Effort S (was M), Payoff med.**
- **H — installable package.** No `[project]` table in `pyproject.toml`; 17 `sys.path.insert`
  hacks stand in for it. Add the table + `uv pip install -e .` and remove the hacks.
  **Effort M, Payoff high (compounding).** *(Held: highest blast radius.)*

## Not yet scoped

### God functions (decompose into named phases)
- `scripts/tinker_training/run_experiment.py` — **`build_task_graph`** (~330 lines after
  dead-code removal): mixes data-gen / data-prep / train (main+control cross-product) /
  eval with ~7 repeated closure-factories. Extract `_add_{data_gen,data_prep,training,eval}_tasks`
  and a single `_static_builder(cmd)`. Main-vs-control builders are near-identical —
  parameterize on `is_control`. **Effort L, Payoff high.**
- `cot_transparency/apis/tinker/rl_training.py` — **`_train_loop_inner`** (~280 lines):
  prefetch pipeline + reward/advantage + KL + logging + checkpointing via nested closures.
  Lift the prefetch queue and anchor-resolution out; consider a `_RolloutSampler` /
  `PrefetchPipeline`. Also `_build_training_batch` interleaves reward/advantage/shrinkage/metric
  concerns and returns a positional 5-tuple → small dataclass. **Effort L, Payoff high.**
  *(Note: this file is under active development — coordinate before refactoring.)*
- `sycophancy_eval_inspect/visualize_results.py` — `main()` (~290 lines) → argparse
  subcommands; `plot_metric_ratio` (~240) and `plot_grouped_bars` (~220) share a
  bar/hatch/errorbar skeleton → extract `_grouped_bar(ax, ...)`. **Effort L, Payoff med-high.**

### Duplication
- **Checkpoint save/finalize** copy-pasted across `finetune.py` (~217-258) and
  `rl_training.py` (~1024-1069): intermediate-save + final-checkpoint blocks,
  `paths.get("sampler_path") or paths.get("state_path")` 4×. Pull
  `save_intermediate_checkpoint()` / `finalize_checkpoint()` into `apis/tinker/common.py`.
  **Effort M, Payoff high.**
- **Config→CLI arg translation** in `run_experiment.py`: the
  `if k in args: cmd += ["--k", str(args[k])]` + CSV-coercion pattern is hand-written in
  several builders. A declarative `{config_key: cli_flag}` table + `add_arg()` helper would
  collapse it. **Effort M, Payoff med.**
- **`get_renderer_for_model`** (`inference.py`) duplicates `common.get_renderer_and_tokenizer`;
  `types.SamplingParams(...)` built in 3 places. Centralize in `common.py`. **Effort S, Payoff med.**
- **Matplotlib styling / palettes** re-declared across `visualize_results.py`,
  `plot_model_comparison.py`, `eval_awareness/analyze.py`, `make_shrinkage_figure.py`
  (`MODEL_COLORS`/`STYLE_MAP`, `grid(axis="y", alpha=0.25)`, `savefig(dpi=150)`). A
  `plot_style.py` with `style_axis()`/`save_fig()` + one palette. `analyze.py` already has a
  clean reusable `_grouped_bar` worth promoting. **Effort S, Payoff med.**
- **`judge.py`** (`cot_transparency/eval_awareness/`): 4 near-identical yes/no judges + 3
  identical `make_*` factory closures → one `_yes_no_judge()` + one `_bind()` factory.
  **Effort S, Payoff low-med.**
- **eval-awareness batch drivers** (`scripts/eval_awareness/*.sh`): the train→calibrate→overrefusal
  scaffold (`run_one()`) is copy-pasted in `run_battery8.sh` + `run_lr_kl_batch.sh`, and the
  calibrate wrapper (`geval()`) in `run_crossfactor_sweep.sh` + `run_f6_generalization.sh` — 4
  drivers sharing the same `$PY calibrate.py … | grep gap` / `overrefusal_eval.py` plumbing.
  Extract a sourced `_eval_lib.sh` with `run_one`/`geval`. Also `_prepend`/`_prepend_wrapper`
  (message-wrapping) is duplicated `build_f6_wrapper_eval.py` ↔ `train_evalaware.py`.
  **Effort S, Payoff low-med.**

### Known failing tests (re-verified 2026-06-03)
Most of the 2026-06-01 list now passes (the rl-training fixes landed): `test_rl_training.py`,
`test_rl_pipelining.py`, `test_trace_info.py`, `test_trace_splitting.py` are **green**. Remaining
failures, both in legacy `cot_transparency` (unrelated to eval-awareness):
- `tests/test_eval_pipeline.py::test_ba_scorer_template_exists[are_you_sure]`
- `tests/test_prompt_sensitivity_analysis.py::{test_fleiss_kappa_on_total_disagreement,
  test_fleiss_kappa_on_total_greement}` (DataFrame shape mismatch, 4×4 vs 4×5).
