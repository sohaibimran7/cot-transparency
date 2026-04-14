---
name: generate-bct-data
description: Generate BCT training data for models using Tinker API. Use when user wants to sample completions from a model to create biased/unbiased training pairs.
argument-hint: [model] [prompts-dir]
---

# Generate BCT Training Data

Generate BCT (Bias Consistency Training) training data by sampling completions from a target model via Tinker API, then pairing those completions with both biased and unbiased prompts.

## Arguments

- `$0` — Model name (e.g., `meta-llama/Llama-3.1-8B-Instruct`, `openai/gpt-oss-120b`)
- `$1` — (Optional) Prompts source directory relative to `dataset_dumps/` (e.g., `logiqa`). If omitted, uses the root `control_seed_42/` and `train_seed_42/` directories.

## What this does

1. Takes unbiased prompts from `dataset_dumps/control_seed_42/`
2. Samples completions from the specified model via Tinker
3. Pairs each completion with BOTH the unbiased prompt (control) and the biased prompt (BCT train)
4. Saves to model-specific subdirectories

## Before running

**Confirm with the user:**
1. Which model to use (show available models if unsure: `python scripts/tinker_training/generate_bct_data.py --list-models` #the user might use a shorter model name eg. oss 120b instead of openai/gp-oss-120b so its important to check against the avaliable models)  
2. Which sample types: `--cot`, `--non-cot`, `--instruct` (reasoning models like gpt-oss-120b should skip `--cot`)
3. Which prompts directory if not default
4. Batch size (default 256) and save frequency (default every 5 batches)
5. Whether to do a small test run first (`--limit 5`)

**Explain the command** before executing, showing:
- The full command that will be run
- Expected output file locations
- Estimated number of samples

## Command template

```bash
python scripts/tinker_training/generate_bct_data.py \
    --model $0 \
    --cot --non-cot --instruct \
    --batch-size 256 \
    --save-every 5
```

With prompts directory:
```bash
python scripts/tinker_training/generate_bct_data.py \
    --model $0 \
    --source $1 \
    --cot --non-cot --instruct \
    --batch-size 256 \
    --save-every 5
```

## Output structure

```
dataset_dumps/control_seed_42/{model_name}/control_cot.jsonl
dataset_dumps/control_seed_42/{model_name}/control_non_cot.jsonl
dataset_dumps/control_seed_42/{model_name}/instruct_samples.jsonl
dataset_dumps/train_seed_42/{model_name}/bct_cot.jsonl
dataset_dumps/train_seed_42/{model_name}/bct_non_cot.jsonl
dataset_dumps/train_seed_42/{model_name}/instruct_samples.jsonl
```

## Key flags

| Flag | Description |
|------|-------------|
| `--list-models` | Query Tinker API for available models |
| `--cot` | Generate CoT samples |
| `--non-cot` | Generate non-CoT samples |
| `--instruct` | Generate instruction-following samples |
| `--source DIR` | Read prompts from a subdirectory |
| `--batch-size N` | Concurrent requests per batch (default 10, recommend 256) |
| `--save-every N` | Save checkpoint every N batches (default 10, recommend 5) |
| `--limit N` | Limit samples for testing |
| `--checkpoint PATH` | Use a specific model checkpoint |
| `--fresh` | Ignore existing checkpoints, start over |

## Alternative: Generate from test data

Use `generate_bct_from_test.py` when you want to generate BCT data from the **test set** (biased/unbiased question pairs) rather than from the training prompts. This is the preferred method for suggested_answer bias on mmlu/truthfulqa.

```bash
python scripts/tinker_training/generate_bct_from_test.py \
    --model openai/gpt-oss-120b \
    --datasets truthfulqa mmlu \
    --bias suggested_answer \
    --limits 817 1183 \
    --batch-size 64 \
    --non-cot \
    --output-dir dataset_dumps/train-from-test-mmlu-truthfulqa/suggested-answer/gpt_oss_120b
```

Key differences from `generate_bct_data.py`:
- Reads from `dataset_dumps/test/{bias_type}/{dataset}_{bias_type}.jsonl`
- Generates from test data with explicit bias/unbiased question pairs
- `--output-dir` puts both control and bct files in the same flat directory
- `--limits` controls how many samples per dataset (in same order as `--datasets`)
- `--non-cot` strips CoT instructions from prompts; outputs are saved as `control_non_cot.jsonl` / `bct_non_cot.jsonl`. **Default to `--non-cot` for reasoning models** (e.g. gpt-oss-120b) since they reason internally — asking for CoT in the prompt is redundant/harmful.

### Data format

Generated data is stored in tinker's native Message format. For models with internal reasoning (e.g. gpt-oss-120b), the assistant content is a structured list of `ThinkingPart`/`TextPart` dicts. `build_supervised_example` in finetune.py handles this natively — no cleaning needed.

To combine BCT + instruct samples:
```bash
python scripts/prepare_datasets.py \
    --input .../bct_cot.jsonl \
    --input dataset_dumps/train_seed_42/gpt-oss-120b/instruct_samples.jsonl:2000 \
    --output .../bct_cot_instruct.jsonl \
    --shuffle
```

## Script locations

- `scripts/tinker_training/generate_bct_data.py` — Generate from training prompts (control_seed_42/train_seed_42)
- `scripts/tinker_training/generate_bct_from_test.py` — Generate from test data (biased/unbiased pairs)
- `scripts/prepare_datasets.py` — Combine, shuffle, limit datasets