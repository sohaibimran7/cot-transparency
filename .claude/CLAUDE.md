# Project Instructions

This is a map, not a manual — it points to where things live. Keep it short.

## What this repo is
Anti-sycophancy / eval-awareness **consistency training** on the Tinker API (BCT = SFT,
RLCT = RL), plus an Inspect-AI **sycophancy eval** stack. Core metric: **BIR / bias-switch
rate** (how much a bias cue flips the model's answer).

## Environment
- Setup, the `uv` flow, and gotchas (grugstream, inspect-ai, the harmless `conda`
  message): **`docs/ENVIRONMENT.md`**. Do NOT follow `README_FOR_CODE.md` for setup —
  it documents the legacy `cot_transparency` stage_one pipeline.
- Run with the project venv: `.venv/bin/python` or `uv run`.

## How to run things — use the skills (don't reinvent)
- `/run-experiment` — end-to-end pipeline from a YAML in `scripts/tinker_training/experiment_configs/`.
- `/train-tinker` — launch SFT (BCT) or RL (RLCT) training.
- `/run-evals` — sycophancy evals on a checkpoint (5 biases + unbiased × 2 datasets).
- `/analyze-evals` — plots + BIR tables (`python -m sycophancy_eval_inspect.visualize_results`).
- `/generate-bct-data`, `/generate-vft-data` — build training data.
- `/cleanup-checkpoints` — delete Tinker checkpoints not referenced by any eval log.
- `/model-apis` — the standard model-id set for eval runs.

## Where things live
- `cot_transparency/apis/tinker/` — Tinker wrappers: `rl_training.py` (RLCT),
  `finetune.py` (BCT/SFT), `inference.py` (sampling), `common.py`.
- `cot_transparency/eval_awareness/` — eval-awareness cues + LLM judges.
- `sycophancy_eval_inspect/` — Inspect-AI eval tasks/scorers + analysis. Run as a module
  (`python -m sycophancy_eval_inspect.visualize_results`).
- `scripts/tinker_training/`, `scripts/eval_awareness/` — runnable scripts.
- `.eval` logs: `sycophancy_eval_inspect/logs/<exp>/<model-dir>/`. New model dirs must be
  registered in `sycophancy_eval_inspect/model_registry.json` to appear in plots/tables.
- Data: `dataset_dumps/`, `data/`. **LFS budget is exceeded** — fresh clones can't fetch
  `.jsonl` data (regenerate via `build_*.py`); `.eval` logs are recoverable from the pack.

## BIR — single source of truth
- Loading `.eval` logs: `sycophancy_eval_inspect/eval_log_loader.py`
  (`iter_eval_samples`, `extract_bias_metrics`). Don't re-roll the glob/read/parse loop.
- BIR computation: `visualize_results.compute_per_question_bir` (per-question, paper
  default) and `collapse_to_population_bir(signed=)` (population net/abs). `extract_bir3.py`
  + `make_shrinkage_figure.py` consume these — don't hardcode BIR numbers.
- The shrinkage figure/table use **per-question matched net** BIR; the methodology change
  and its result impact are documented in **`docs/bir-methodology-change.md`**.

## Tests
```bash
pytest            # offline subset (default)
pytest -m tinker  # live-Tinker tests
pytest -m ""      # everything
```
Mark live-API / GPU / Tinker tests with `@pytest.mark.{network,gpu,tinker}` so the default
loop stays fast and offline. Markers are defined in `pyproject.toml`.

## Known issues / backlog
- Deferred refactors (god functions, checkpoint-save duplication, etc.): **`docs/tech-debt-tracker.md`**.
