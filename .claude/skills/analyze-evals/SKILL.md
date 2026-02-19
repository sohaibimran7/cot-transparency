---
name: analyze-evals
description: Visualize and analyze sycophancy evaluation results. Generates plots, BIR tables, and summary statistics from eval logs.
argument-hint: [log-dir(s)]
---

# Analyze Evaluation Results

Generate plots, BIR (Bias Influence Rate) tables, and summary statistics from sycophancy eval logs.

## Arguments

- `$ARGUMENTS` — One or more log directories containing eval results (e.g., `sycophancy_eval_inspect/logs/cot_100samples`)

## Before running

**Confirm with the user:**
1. Which log directory/directories to analyze
2. What output to generate: plots, BIR tables, summary, or all
3. Any filters: model family, variant, prompt style, dataset, bias type
4. Output directory for plots (default: `sycophancy_eval_inspect/plots/`)

**Explain the command** before executing.

## Command templates

### Generate all plots
```bash
python -m sycophancy_eval_inspect.visualize_results \
    --log-dir LOG_DIR \
    --output-dir sycophancy_eval_inspect/plots/OUTPUT_NAME
```

### BIR tables (Bias Influence Rate)
```bash
python -m sycophancy_eval_inspect.visualize_results \
    --log-dir LOG_DIR \
    --bir
```

### Save BIR tables to file
```bash
python -m sycophancy_eval_inspect.visualize_results \
    --log-dir LOG_DIR \
    --bir --save-bir sycophancy_eval_inspect/plots/OUTPUT_NAME/bir_tables.md
```

### Summary table only
```bash
python -m sycophancy_eval_inspect.visualize_results \
    --log-dir LOG_DIR \
    --summary
```

### With filters
```bash
python -m sycophancy_eval_inspect.visualize_results \
    --log-dir LOG_DIR \
    --model llama \
    --variant biased \
    --prompt-style cot \
    --dataset mmlu \
    --bias-type suggested_answer
```

### Multiple log directories
```bash
python -m sycophancy_eval_inspect.visualize_results \
    --log-dir DIR1 --log-dir DIR2 \
    --output-dir sycophancy_eval_inspect/plots/combined
```

## Key flags

| Flag | Description |
|------|-------------|
| `--log-dir DIR` | Eval log directory (repeatable for multiple) |
| `--output-dir DIR` | Output directory for plots |
| `--model MODEL` | Filter by model family: `llama` or `gpt` (repeatable) |
| `--variant VAR` | Filter: `biased` or `unbiased` (repeatable) |
| `--prompt-style STYLE` | Filter: `cot` or `no_cot` (repeatable) |
| `--dataset DS` | Filter by dataset name (repeatable) |
| `--bias-type BIAS` | Filter by bias type (repeatable) |
| `--summary` | Print summary table only |
| `--bir` | Print BIR tables |
| `--save-bir FILE` | Save BIR tables to CSV/MD |
| `--no-n` | Hide sample counts in BIR tables |

## Registering new models

New model directories must be registered in `sycophancy_eval_inspect/visualize_results.py` before they appear in plots/tables. Update these four mappings:
- `_DIR_TO_TRAINING_TYPE` — directory suffix -> internal key
- `COLORS` — internal key -> hex color
- `TRAINING_TYPE_ORDER` — column order list
- `TRAINING_TYPE_NAMES` — internal key -> display name

The visualizer strips model prefixes (`llama-`, `gpt-`) from directory names and looks up the remainder in `_DIR_TO_TRAINING_TYPE`. Unregistered directories are silently skipped.

## Available metrics

- **correct** — Accuracy
- **matches_bias** — Bias Match Rate (sycophancy rate)
- **answer_parsed** — Parse success rate
- **options_considered** — Fraction of options considered in CoT
- **bias_acknowledged** — Whether model acknowledges the bias
- **target_answered** / **non_target_answered** — Few-shot confusion metrics

## Existing log directories

Check `sycophancy_eval_inspect/logs/` for available eval runs.

## Script location

`sycophancy_eval_inspect/visualize_results.py` (run as module: `python -m sycophancy_eval_inspect.visualize_results`)
