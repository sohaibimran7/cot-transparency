# Project Instructions

Single entry-point doc for this repo: guardrails, setup, and a map of where things live.
(`AGENTS.md` is a symlink to this file.)

## Training & Eval Runs (hard rule)
- **ALWAYS** show the user the exact command and parameters for every training run and
  eval run **before** executing.
- If multiple tasks must run in sequence (e.g., train then eval), present **all**
  commands upfront.
- **Do not execute** until the user explicitly approves.

## What this repo is
Anti-sycophancy / eval-awareness **consistency training** on the Tinker API (BCT = SFT,
RLCT = RL), plus an Inspect-AI **sycophancy eval** stack. Core metric: **BIR / bias-switch
rate** (how much a bias cue flips the model's answer).

## Environment
Uses **`uv`** (not the legacy `pyenv`/`pip` flow — see the legacy section at the bottom).
Run things with `uv run <cmd>` or activate the venv (`source .venv/bin/activate`); the venv
is Python 3.12 (3.11+ works). `requirements.txt` is the dependency source of truth — **no
`uv.lock`** is checked in.

```bash
uv venv                                                        # 1. create the venv
grep -v '^grugstream' requirements.txt | uv pip install -r /dev/stdin  # 2. deps, EXCLUDING grugstream
uv pip install inspect-ai                                      # 3. inspect-ai is NOT in requirements.txt
uv run python -c "import nltk; nltk.download('punkt')"         # 4. one-time NLTK data (some formatters)
uv pip install -e . --no-deps                                  # 5. editable install of this repo (see below)
```

- **grugstream**: in `requirements.txt` but must be excluded (step 2) — it fails to
  resolve/build here and nothing in the active Tinker/eval path imports it.
- **inspect-ai**: the `sycophancy_eval_inspect/` stack needs it, but it's installed
  separately (step 3), not pinned in `requirements.txt`.
- **editable install** (step 5): makes the first-party packages (`cot_transparency`,
  `sycophancy_eval_inspect`, `data`, `dataset_dumps`, `scripts`, `tests`) importable from
  anywhere — new scripts just `from cot_transparency... import ...`, **no `sys.path` hacks**.
  Configured in `pyproject.toml` (`[project]` + `setuptools.packages.find`, namespaces=true).
- **`zsh: command not found: conda`** printed before shell commands is harmless (a stale
  conda init in the shell, not from this repo) — ignore it.
- **API keys**: put credentials in a `.env` at the repo root (loaded via `python-dotenv`):
  `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and the Tinker API key. `.env` is gitignored —
  never commit it.

## How to run things — use the skills (don't reinvent)
- `/run-experiment` — end-to-end pipeline from a YAML in `scripts/tinker_training/experiment_configs/`.
- `/train-tinker` — launch SFT (BCT) or RL (RLCT) training.
- `/run-evals` — sycophancy evals on a checkpoint (5 biases + unbiased × 2 datasets).
- `/analyze-evals` — plots + BIR tables (`python -m sycophancy_eval_inspect.visualize_results`).
- `/generate-bct-data`, `/generate-vft-data` — build training data.
- `/cleanup-checkpoints` — delete Tinker checkpoints not referenced by any eval log.
- `/scorer-eyeball` — audit `bias_acknowledged` (BA) scorer behavior across eval logs for false pos/neg.
- `/model-apis` — the standard model-id set for eval runs.

## Where things live
- `cot_transparency/apis/tinker/` — Tinker wrappers: `rl_training.py` (RLCT),
  `finetune.py` (BCT/SFT), `inference.py` (sampling), `common.py` (shared config +
  checkpoint save/finalize helpers — both training loops delegate here, don't re-roll).
- `cot_transparency/eval_awareness/` — eval-awareness cues + LLM judges.
- `sycophancy_eval_inspect/` — Inspect-AI eval tasks/scorers + analysis. Run as a module
  (`python -m sycophancy_eval_inspect.visualize_results`).
- `scripts/tinker_training/`, `scripts/eval_awareness/` — runnable scripts.
- `.eval` logs: `sycophancy_eval_inspect/logs/<exp>/<model-dir>/`. New model dirs must be
  registered in `sycophancy_eval_inspect/model_registry.json` to appear in plots/tables.
- Data: `dataset_dumps/`, `data/`. **LFS budget is exceeded** — fresh clones can't fetch
  `.jsonl` data (regenerate via `scripts/eval_awareness/build_*.py`); `.eval` logs are
  recoverable from the pack.

## BIR — single source of truth
- Loading `.eval` logs: `sycophancy_eval_inspect/eval_log_loader.py`
  (`iter_eval_samples`, `extract_bias_metrics`). Don't re-roll the glob/read/parse loop.
- BIR computation: `visualize_results.compute_per_question_bir` (per-question, paper
  default) and `collapse_to_population_bir(signed=)` (population net/abs). `extract_bir3.py`
  + `make_shrinkage_figure.py` (in `scripts/tinker_training/`) consume these — don't
  hardcode BIR numbers.
- The shrinkage figure/table use **per-question matched net** BIR; the methodology change
  and its result impact are documented in **`docs/bir-methodology-change.md`**.

## Tests & lint
```bash
pytest            # offline subset (default; excludes network/gpu/tinker markers)
pytest -m tinker  # tests needing the live Tinker service / a checkpoint
pytest -m ""      # everything
make check        # black + ruff + pyright via pre-commit (.pre-commit-config.yaml)
```
Mark live-API / GPU / Tinker tests with `@pytest.mark.{network,gpu,tinker}` so the default
loop stays fast and offline. Markers are defined in `pyproject.toml`.

## Legacy `stage_one` pipeline (out of date)
The original `cot_transparency` `stage_one` flow predates the Tinker / eval-awareness
workflow above. Its scripts still exist but are **not** the current path — don't use them
for new work; kept here only for the old experiment flow. Needs `OPENAI_API_KEY` /
`ANTHROPIC_API_KEY` in `.env`.
```bash
# generate samples (20 per bbh task; sycophancy vs unbiased)
python stage_one.py --exp_dir experiments/dummy_run --models "['text-davinci-003']" \
  --formatters "['ZeroShotCOTUnbiasedFormatter', 'ZeroShotCOTSycophancyFormatter']" \
  --repeats_per_question 1 --batch 10 --example_cap 20   # writes json under experiments/
python analysis.py accuracy --exp_dir experiments/dummy_run   # accuracy
python viewer.py --exp_dir experiments/dummy_run              # view samples (--n_compare 2 = side-by-side)
streamlit run streamlit_viewer.py experiments/dummy_run      # streamlit viewer (stage_one only)
```
Other legacy bits: `make hooks` installs pre-commit hooks; git-LFS auto-tracks `.json`,
add more with `git lfs track "path/to/file"`.
