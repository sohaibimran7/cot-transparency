---
name: run-evals
description: Run sycophancy evaluations on Tinker model checkpoints. Defaults to 5 core biases + unbiased across 2 datasets (12 eval files per checkpoint).
argument-hint: [checkpoint-path(s)]
---

# Run Sycophancy Evaluations

Run the sycophancy eval suite on one or more Tinker model checkpoints.

## Arguments

- `$ARGUMENTS` — One or more checkpoint paths (tinker://...) or "base" for base model only

## Default configuration (5 biases x 2 datasets = 12 evals per checkpoint)

**Default biases** (the 5 most informative):
- `suggested_answer` — Direct answer suggestion
- `distractor_argument` — Misleading reasoning argument
- `distractor_fact` — Misleading factual claim
- `wrong_few_shot` — Incorrect few-shot examples
- `spurious_few_shot_squares` — Spurious pattern (squares)

**Default datasets**: `hellaswag, logiqa`

**Default prompt style**: `cot` only (use `no_cot` for gpt-oss-120b)

**Variants**: Both biased + unbiased (unbiased runs once per dataset, biased runs per bias type)

This gives: 5 biased + 1 unbiased per dataset x 2 datasets = **12 eval files per checkpoint**.

## Model-specific notes

### gpt-oss-120b
- Use `--prompt-styles no_cot` — the model has internal CoT via channel tags (`<|channel|>analysis`), so requesting explicit CoT would double-prompt reasoning
- Use `--base-model openai/gpt-oss-120b`

## Before running

**Confirm with the user:**
1. Which checkpoint(s) to evaluate
2. Whether to use default biases or customize
3. Base model (default: `meta-llama/Llama-3.1-8B-Instruct`, for gpt use `openai/gpt-oss-120b`)
4. Prompt styles: `cot`, `no_cot`, or both (gpt-oss-120b should use `no_cot` only)
5. Sample limit (default: no limit, use `--limit N` for quick tests)
6. Whether to do a `--dry-run` first

**Explain the command** before executing, showing:
- The full command
- Number of eval tasks that will run
- Checkpoint being evaluated
- The eval artifact root, preferably `artifacts/runs/EXPERIMENT_NAME/eval_logs`

## Command template

**Step 1 — precompute the hash file** (required). This is a one-time setup step, takes ~2 seconds, and lets you run multiple eval commands in parallel against the same hash set:

```bash
python -m sycophancy_eval_inspect.generate_hash_file \
    --datasets hellaswag,logiqa \
    --bias-types suggested_answer,distractor_argument,distractor_fact,wrong_few_shot,spurious_few_shot_squares \
    --limit 200 \
    --output artifacts/runs/EXPERIMENT_NAME/eval_logs/common_hashes.json
```

The command is idempotent (no-op if the file already exists with identical content, errors out if it exists with different content unless you pass `--overwrite`).

**Step 2 — run evals** using `--hash-file` (required unless `--skip-hash-filter`):

### Single checkpoint
```bash
python -m sycophancy_eval_inspect.run_tinker_evals \
    --checkpoint "CHECKPOINT_PATH" \
    --base-model meta-llama/Llama-3.1-8B-Instruct \
    --name "MODEL_NAME" \
    --bias-types suggested_answer,distractor_argument,distractor_fact,wrong_few_shot,spurious_few_shot_squares \
    --datasets hellaswag,logiqa \
    --prompt-styles cot,no_cot \
    --limit 200 \
    --hash-file artifacts/runs/EXPERIMENT_NAME/eval_logs/common_hashes.json \
    --log-dir artifacts/runs/EXPERIMENT_NAME/eval_logs
```

### Base model (no checkpoint)
```bash
python -m sycophancy_eval_inspect.run_tinker_evals \
    --base-model meta-llama/Llama-3.1-8B-Instruct \
    --name "llama-base" \
    --bias-types suggested_answer,distractor_argument,distractor_fact,wrong_few_shot,spurious_few_shot_squares \
    --datasets hellaswag,logiqa \
    --prompt-styles cot,no_cot \
    --limit 200 \
    --hash-file artifacts/runs/EXPERIMENT_NAME/eval_logs/common_hashes.json \
    --log-dir artifacts/runs/EXPERIMENT_NAME/eval_logs
```

### Multiple checkpoints in parallel
Since the hash file is precomputed, all checkpoint evals can run simultaneously:
```bash
python -m sycophancy_eval_inspect.run_tinker_evals --checkpoint CKPT1 --name NAME1 ... --hash-file $HASH_FILE &
python -m sycophancy_eval_inspect.run_tinker_evals --checkpoint CKPT2 --name NAME2 ... --hash-file $HASH_FILE &
wait
```

## Available bias types (all 8)

| Bias Type | Type | Description |
|-----------|------|-------------|
| `suggested_answer` | Single-turn | "I think the answer is X" |
| `distractor_argument` | Single-turn | Misleading reasoning |
| `distractor_fact` | Single-turn | Misleading factual claim |
| `wrong_few_shot` | Single-turn | Incorrect few-shot examples |
| `spurious_few_shot_squares` | Single-turn | Spurious pattern (squares) |
| `spurious_few_shot_hindsight` | Single-turn | Spurious pattern (hindsight) |
| `post_hoc` | Multi-turn | Post-hoc rationalization (pre-filled wrong answer) |
| `are_you_sure` | Multi-turn | "Are you sure?" follow-up (forced first answer + 2 on-policy generations) |

### Multi-turn bias notes
- **post_hoc**: Has a pre-filled assistant message with a wrong answer, then asks the model to explain. Single generation.
- **are_you_sure**: Uses `multi_turn_generate` solver. First answer is teacher-forced with the correct ground truth letter, then challenged with "Are you sure?", then asked for final answer (2 on-policy generations).

### Reasoning model notes
For reasoning models (gpt-oss-120b, o1, etc.), the eval pipeline uses Inspect's native `reasoning_history=False` to strip reasoning/thinking tokens from prior assistant turns before each subsequent generation. This prevents thinking from leaking into conversation history in multi-turn biases.

## Available datasets

`mmlu`, `truthfulqa`, `hellaswag`, `logiqa`

`hindsight_neglect` — **only** available for `spurious_few_shot_hindsight`

### Custom dataset filenames (gotcha)

The eval pipeline derives the "original dataset name" by stripping the bias suffix from the filename stem (see `get_original_dataset_name` in `sycophancy_eval_inspect/eval_common.py`). So a file like `dataset_dumps/test/suggested_answer/mmlu_7000samples_suggested_answer.jsonl` is treated as dataset **`mmlu_7000samples`**, not `mmlu`. When referencing custom-dumped files:

- Use the full stem-minus-bias as the dataset name in `--datasets`, both for `generate_hash_file` and `run_tinker_evals`
- Custom-named files do not collide/group with the canonical ones (e.g. `mmlu_7000samples` is a separate dataset from `mmlu`)
- The visualizer will show the custom dataset name verbatim unless a display label exists; prefer adding labels/config through registry-backed config paths rather than editing plot code

**Important**: When running all 8 biases, include `hindsight_neglect` in `--datasets`:
```bash
--datasets hellaswag,logiqa,hindsight_neglect
```
Without it, `spurious_few_shot_hindsight` produces zero eval tasks (silently skipped).

## Key flags

| Flag | Description |
|------|-------------|
| `--checkpoint PATH` | Tinker checkpoint (tinker://...) |
| `--base-model MODEL` | Base model name |
| `--name NAME` | Model name for logging |
| `--bias-types TYPES` | Comma-separated bias types |
| `--datasets DATASETS` | Comma-separated dataset names |
| `--prompt-styles STYLES` | `cot`, `no_cot`, or `cot,no_cot` |
| `--limit N` | Limit samples per evaluation |
| `--max-tasks N` | Max parallel eval tasks (default 12) |
| `--max-connections N` | Max concurrent API connections |
| `--dry-run` | Print tasks without running |
| `--log-dir DIR` | Output log directory |
| `--skip-hash-filter` | Disable hash-based sample filtering |
| `--hash-file FILE` | Load pre-computed hash filter (**required** if `common_hashes.json` exists in log dir) |
| `--save-hash-file FILE` | Save computed hashes to a custom path (bypasses existence check) |

## Hash filtering (important)

The `common_hashes.json` file in each log directory contains curated hashes per dataset that ensure all evals compare the same questions.

**Workflow:**
1. Precompute the hash file once with `python -m sycophancy_eval_inspect.generate_hash_file` (see Step 1 above)
2. Pass `--hash-file path/to/common_hashes.json` on every eval run (step 2 above)
3. If you need to disable filtering entirely, use `--skip-hash-filter` (not recommended — breaks cross-bias BIR comparison)

`run_tinker_evals` does NOT auto-generate the hash file. If `--hash-file` points to a non-existent path, it errors out with a hint to run `generate_hash_file` first. This is intentional: precomputation lets parallel eval runs load the same file without races.

## After running: register for visualization

New model directories should be registered in `sycophancy_eval_inspect/model_registry.json`, preferably through `viz_registration` in the experiment config when using `scripts/tinker_training/run_experiment.py`.

If you skip this, the model's eval logs will be silently ignored by the visualizer.

## Naming convention

The `--name` flag sets the log subdirectory name (e.g., `llama-vft-mt-1675`). The visualizer strips the registered model prefix (`llama-`, `gpt-oss-20b-`, `gpt-oss-120b-`, `qwen3-`, etc.) and looks up the remainder in `model_registry.json` plus legacy mappings. So `llama-vft-mt-1675` -> strips `llama-` -> looks up `vft-mt-1675`.

When re-running evals with different sample counts or parameters, append a suffix to avoid collisions with existing directories (e.g., `gpt-bct-mti-4k-100samples`) and register that suffix in `model_registry.json` if it should appear in plots.

## Script location

`sycophancy_eval_inspect/run_tinker_evals.py` (run as module: `python -m sycophancy_eval_inspect.run_tinker_evals`)

## Dependencies: openai package version

⚠️ **This eval pipeline requires `openai>=2.8.0`** (via `inspect_ai`). If dataset regeneration (`scripts/dump_datasets_for_release.py`, which depends on `cot_transparency` and requires `openai<1.0`) was run recently, the environment will have `openai==0.28.1` and every eval will fail immediately with a `PrerequisiteError` saved into a tiny ~6KB `.eval` file.

Before running evals, verify and upgrade if needed:
```bash
python -c "import openai; print(openai.__version__)"   # must be >= 2.8.0
pip install 'openai>=2.8.0'                             # upgrade if needed
```

If a run already failed due to this, archive the tiny `.eval` files into an `_archive/` subtree before rerunning, otherwise the visualizer will try to load them. Do not use destructive recursive deletion.
