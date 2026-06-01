# Tech-debt tracker

Refactor opportunities found during the 2026-06-01 audit but **not** done, so they stay
legible instead of being re-discovered. Line numbers are approximate (they drift); search
by symbol name. Effort S/M/L, Payoff low/med/high.

## Held — planned, gated on the user

These were in the approved refactor plan but deferred (see `.claude/plans/`):

- **D — eval_awareness sampling/IO duplication.** Tinker token-parse logic
  reimplemented 4× (`scripts/eval_awareness/run_eval.py` `_resp_text`, `train_evalaware.py`
  `_text`, `overrefusal_eval.py`, `build_alpaca_interleave.py`) — should use
  `TinkerSamplingClient.sample()/.sample_text()` (`cot_transparency/apis/tinker/inference.py`).
  Samplers belong in a shared `cot_transparency/eval_awareness/samplers.py` (calibrate.py
  imports a private `_TinkerSampler` from run_eval via a `sys.path` hack). JSONL/EAB-baseline
  loaders duplicated ~6× → `eval_awareness/io.py`. **Effort M, Payoff high.** *(Held: files
  under active edit.)*
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

### Known failing tests (as of 2026-06-01, unrelated to the BIR/loader refactor)
Failing in committed code (`f3ccf72` shrinkage scaffolding), worth a look:
`tests/test_rl_training.py` (`RateEstimationConfig` validation, `BatchItem.p_hat_counts`),
`tests/test_rl_pipelining.py`, `tests/test_eval_pipeline.py`, `tests/test_trace_info.py`,
`tests/test_trace_splitting.py`, `tests/test_prompt_sensitivity_analysis.py`.
