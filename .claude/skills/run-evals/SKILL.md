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

## Command template

**Always pass `--hash-file`** to ensure all models are evaluated on exactly the same questions:

### Single checkpoint
```bash
python -m sycophancy_eval_inspect.run_tinker_evals \
    --checkpoint "CHECKPOINT_PATH" \
    --base-model meta-llama/Llama-3.1-8B-Instruct \
    --name "MODEL_NAME" \
    --bias-types suggested_answer,distractor_argument,distractor_fact,wrong_few_shot,spurious_few_shot_squares \
    --datasets hellaswag,logiqa \
    --prompt-styles cot,no_cot \
    --hash-file sycophancy_eval_inspect/logs/cot_100samples/common_hashes.json \
    --log-dir sycophancy_eval_inspect/logs/EVAL_DIR
```

### Base model (no checkpoint)
```bash
python -m sycophancy_eval_inspect.run_tinker_evals \
    --base-model meta-llama/Llama-3.1-8B-Instruct \
    --name "llama-base" \
    --bias-types suggested_answer,distractor_argument,distractor_fact,wrong_few_shot,spurious_few_shot_squares \
    --datasets hellaswag,logiqa \
    --prompt-styles cot,no_cot \
    --hash-file sycophancy_eval_inspect/logs/cot_100samples/common_hashes.json \
    --log-dir sycophancy_eval_inspect/logs/EVAL_DIR
```

### Multiple checkpoints (run sequentially)
For multiple checkpoints, run separate commands for each, varying `--checkpoint`, `--name`, and the log subdirectory.

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

The `common_hashes.json` file in each log directory contains curated hashes per dataset. **Always pass `--hash-file`** to use it.

If you run without `--hash-file` and a `common_hashes.json` already exists in the log dir, the eval runner will error out with `FileExistsError` to prevent accidental overwrites. Options:
1. `--hash-file path/to/common_hashes.json` — load existing (normal case)
2. `--save-hash-file other_path.json` — compute new hashes and save elsewhere
3. `--skip-hash-filter` — disable filtering entirely (not recommended)

## After running: register for visualization

New model directories must be registered in `sycophancy_eval_inspect/visualize_results.py` to appear in plots and BIR tables. Update **all four** mappings:

1. `_DIR_TO_TRAINING_TYPE` — maps directory suffix to internal key (e.g., `"vft-mt-1675": "vft_mt_1675"`)
2. `COLORS` — color for plots (e.g., `"vft_mt_1675": "#9467bd"`)
3. `TRAINING_TYPE_ORDER` — append to list to control column ordering
4. `TRAINING_TYPE_NAMES` — display name for tables (e.g., `"vft_mt_1675": "VFT MT 1675"`)

If you skip this, the model's eval logs will be silently ignored by the visualizer.

## Naming convention

The `--name` flag sets the log subdirectory name (e.g., `llama-vft-mt-1675`). The visualizer strips the model prefix (`llama-`, `gpt-`) and looks up the remainder in `_DIR_TO_TRAINING_TYPE`. So `llama-vft-mt-1675` -> strips `llama-` -> looks up `vft-mt-1675`.

When re-running evals with different sample counts or parameters, append a suffix to avoid collisions with existing directories (e.g., `gpt-bct-mti-4k-100samples`). Register the suffixed name in `_DIR_TO_TRAINING_TYPE` too.

## Script location

`sycophancy_eval_inspect/run_tinker_evals.py` (run as module: `python -m sycophancy_eval_inspect.run_tinker_evals`)
