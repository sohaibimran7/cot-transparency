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
2. What output to generate: BIR tables, BA tables, plots, summary, or combination
3. Any filters: model family, variant, prompt style, dataset, bias type
4. Output directory for plots (default: `sycophancy_eval_inspect/plots/`)

**Explain the command** before executing.

## Command templates

### Print BIR tables (Bias Influence Rate)
```bash
python -m sycophancy_eval_inspect.visualize_results \
    --log-dir LOG_DIR \
    --bir
```

### Print BA tables (Bias Acknowledged)
```bash
python -m sycophancy_eval_inspect.visualize_results \
    --log-dir LOG_DIR \
    --ba
```

### Print both BIR and BA tables
```bash
python -m sycophancy_eval_inspect.visualize_results \
    --log-dir LOG_DIR \
    --bir --ba
```

### Generate plots (no tables)
```bash
python -m sycophancy_eval_inspect.visualize_results \
    --log-dir LOG_DIR \
    --bir --plot --no-tables \
    --output-dir sycophancy_eval_inspect/plots/OUTPUT_NAME
```

### Tables + plots together
```bash
python -m sycophancy_eval_inspect.visualize_results \
    --log-dir LOG_DIR \
    --bir --ba --plot \
    --output-dir sycophancy_eval_inspect/plots/OUTPUT_NAME
```

### Save tables to file
```bash
python -m sycophancy_eval_inspect.visualize_results \
    --log-dir LOG_DIR \
    --bir --save sycophancy_eval_inspect/plots/OUTPUT_NAME/bir_tables
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
    --bir \
    --model llama \
    --prompt-style cot \
    --dataset mmlu \
    --bias-type suggested_answer
```

### Multiple log directories
```bash
python -m sycophancy_eval_inspect.visualize_results \
    --log-dir DIR1 --log-dir DIR2 \
    --bir --ba
```

## Key flags

### Metric flags (what to compute)
| Flag | Description |
|------|-------------|
| `--bir` | Compute and print BIR (Bias Influence Rate) tables |
| `--ba` | Compute and print BA (Bias Acknowledged) tables |

### Output flags (how to present)
| Flag | Description |
|------|-------------|
| `--plot` | Generate plots (saved to `--output-dir`) |
| `--save PATH` | Save tables to CSV/MD file |
| `--no-tables` | Suppress printing tables to stdout (use with `--plot`) |
| `--output-dir DIR` | Output directory for plots (default: `plots`) |

### Filter flags
| Flag | Description |
|------|-------------|
| `--log-dir DIR` | Eval log directory (repeatable for multiple) |
| `--model MODEL` | Filter by model family: `llama` or `gpt` (repeatable) |
| `--variant VAR` | Filter: `biased` or `unbiased` (repeatable) |
| `--prompt-style STYLE` | Filter: `cot` or `no_cot` (repeatable) |
| `--dataset DS` | Filter by dataset name (repeatable) |
| `--bias-type BIAS` | Filter by bias type (repeatable) |

### Table configuration flags
| Flag | Description |
|------|-------------|
| `--no-n` | Hide sample counts in tables |
| `--bir-baseline TYPE` | Baseline for ratio computation (default: `base`) |
| `--bir-parser MODE` | BIR parser: `strict`, `lenient`, or `both` (default: `lenient`) |
| `--bir-verbalization MODE` | BA verbalization: `strict`, `lenient`, or `both` (default: `strict`) |

### Other flags
| Flag | Description |
|------|-------------|
| `--summary` | Print summary table only (uses sample-level data, not BIR) |
| `--no-splits` | Skip split-by-BA and split-by-BIR plot variants |

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
