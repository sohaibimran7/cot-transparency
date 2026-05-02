---
name: run-experiment
description: Run an end-to-end experiment pipeline (data generation, training, evaluation, analysis) from a YAML config.
argument-hint: [config.yaml] [--start-from STAGE] [--stages STAGE,...]
---

# Run Experiment Pipeline

Launch a full or partial experiment pipeline from a YAML config file.

## Arguments

- `$0` — Path to YAML experiment config (e.g., `scripts/tinker_training/experiment_configs/example_bct_sft.yaml`)
- `--start-from STAGE` — (Optional) Skip all stages before this one
- `--stages STAGE,...` — (Optional) Run only these specific stages
- `--checkpoint PATH` — (Optional) Override checkpoint for evaluation
- `--force` — Re-run already-completed tasks
- `--dry-run` — Print commands without executing
- `--max-parallel N` — (Optional) Cap concurrent subprocesses (also `parallelism: N` in YAML)

## Stage names (with aliases)

| Stage             | Aliases                    |
|-------------------|----------------------------|
| data_generation   | data_gen, datagen          |
| data_preparation  | data_prep, dataprep        |
| training          | train                      |
| evaluation        | eval                       |
| analysis          | analyze, viz               |

## Before running

**Confirm with the user:**
1. Which config file to use (or create one from a template)
2. Which stages to run (full pipeline vs partial)
3. For partial runs: that the required inputs exist (checkpoint, data files, eval logs)

**Show the user** the config contents and explain what each stage will do.

## Script location

```
python scripts/tinker_training/run_experiment.py CONFIG [OPTIONS]
```

## Config file format (YAML)

```yaml
name: experiment-name          # Used for state directory
model: meta-llama/Llama-3.1-8B-Instruct

data_generation:
  script: generate_bct_from_test   # or generate_bct_data
  args:
    bias: suggested_answer
    datasets: [mmlu, truthfulqa]
    limits: [1183, 817]
    batch_size: 64

training:
  method: sft                      # or rl
  seeds: [0, 42, 123]             # Optional: run multiple seeds in parallel
  include_control: true            # Also train control variant in parallel
  # control_only: true             # Only train control (skip main)
  args:
    experiment_name: bct_suggested_answer
    run_name: llama-bct-mti-4k
    data: auto                     # auto-resolved from data_gen output
    lr: 1.0e-4
    batch_size: 128
    lora_rank: 8
    save_every: 5

evaluation:
  include_base: true               # Also evaluate base model in parallel
  # base_only: true                # Only evaluate base model (no checkpoint needed)
  args:
    bias_types: "suggested_answer,distractor_argument,..."
    datasets: "hellaswag,logiqa"
    prompt_styles: "cot,no_cot"
    log_dir: sycophancy_eval_inspect/logs/tinker_evals
  base_args:                       # Optional: overrides merged on top of args for base eval only
    limit: 500                     # e.g. smaller limit for the base model

analysis:
  args:
    bir: true
    ba: true
    plot: true
    variance_across: seed          # Optional: generate variance plots split by column
```

## Multi-seed support (`training.seeds`)

Run multiple training seeds in parallel for statistical robustness:

```yaml
training:
  method: sft
  seeds: [0, 42, 123]
  include_control: true
  args:
    run_name: llama-bct-sa
```

- Each seed produces a separate run with `-s{N}` suffixed name (e.g., `llama-bct-sa-s0`, `llama-bct-sa-s42`)
- Seeds × control are crossed: 3 seeds + control = 6 parallel training runs
- Each checkpoint gets its own eval directory (also `-s{N}` suffixed)
- **Main plots** pool all seeds (more data, binomial SE) — identical to single-run behavior
- **Variance plots** (`analysis.args.variance_across: seed`) show per-seed means with cross-seed SE
- The `--variance-across` flag is generic: can also split by `dataset`, `prompt_style`, `bias_type`, etc.
- Not compatible with `data_mode: sequential`

## Control and base model support

### Control training (`include_control: true`)
- **SFT**: Trains a second model on control (unbiased) data in parallel with the main BCT model
- **RL**: Trains a second model with `--control` flag (unbiased perturbation for both ref and train)
- Control run name is auto-suffixed with `-ctrl` (e.g., `llama-bct-mti-4k-ctrl`)
- Control checkpoint is auto-passed to evaluation stage

### Base model evaluation (`include_base: true`)
- Evaluates the base model (no checkpoint) in parallel with finetuned checkpoints
- Base model name is auto-generated as `{model_prefix}-base` (e.g., `llama-base`)
- **Auto-skip when already covered**: before firing, the runner scans `{log_dir}/{base_name}/*.eval` (header-only reads) and computes the set of (bias_type, dataset) combos already present. If every combo from the config's `bias_types × datasets` is already covered, the `eval:base` task is skipped entirely. If only some are missing, the task fires as configured (re-running already-covered combos). Use `--force` to bypass the check and always re-run the base eval.

### Parallel execution (task DAG)
The pipeline is executed as a task DAG, **not** strict stages. Each subprocess (data-gen step, training run, eval run) is a node with explicit deps. As soon as a training task finishes, its eval task fires — independent of whether other training tasks (e.g., other seeds) are still running. `eval:base` (when `include_base: true`) has no deps and starts immediately. Cap concurrency with `--max-parallel N` or top-level `parallelism: N` in the YAML if you're hitting Tinker rate limits.

State is per-task in `experiments/{name}/state.json` under the `tasks` key. A failed task cancels only its transitive descendants — sibling branches keep running. Re-run the same command to skip completed tasks.

## Example configs

Templates in `scripts/tinker_training/experiment_configs/`:
- `example_bct_sft.yaml` — BCT via SFT with control + base eval
- `example_rlct.yaml` — RLCT via RL with control + base eval

## Common workflows

```bash
# Full BCT pipeline with control and base eval
python scripts/tinker_training/run_experiment.py scripts/tinker_training/experiment_configs/example_bct_sft.yaml

# Just eval + analysis on an existing checkpoint
python scripts/tinker_training/run_experiment.py config.yaml \
    --start-from eval --checkpoint "tinker://..."

# Base model eval only
python scripts/tinker_training/run_experiment.py config.yaml \
    --stages eval --force  # with base_only: true in config

# Re-run analysis with different settings
python scripts/tinker_training/run_experiment.py config.yaml \
    --stages analysis --force

# Dry run to preview all commands
python scripts/tinker_training/run_experiment.py config.yaml --dry-run
```

## Hash file precomputation

The evaluation stage requires a `common_hashes.json` file to ensure all checkpoints are evaluated on identical questions. The pipeline handles this automatically:

- Before dispatching any eval command, `build_eval_cmds` synchronously runs `python -m sycophancy_eval_inspect.generate_hash_file` with the `evaluation.args` config (datasets, bias_types, limit)
- The precomputed file defaults to `{log_dir}/common_hashes.json`; override with `evaluation.args.hash_file`
- If the file already exists, precomputation is a no-op
- Because the file is created before eval commands start, multi-seed and parallel checkpoint evals can all load it concurrently without races

Manual precomputation (e.g. if you want to pre-warm for a shared `log_dir`):
```bash
python -m sycophancy_eval_inspect.generate_hash_file \
    --datasets hellaswag,logiqa --bias-types suggested_answer,... \
    --limit 200 --output sycophancy_eval_inspect/logs/EVAL_DIR/common_hashes.json
```

### Custom dataset filenames (gotcha)

The eval pipeline derives the "original dataset name" by stripping the bias suffix from the filename stem (see `get_original_dataset_name` in `sycophancy_eval_inspect/eval_common.py`). So a file like `dataset_dumps/test/suggested_answer/mmlu_7000samples_suggested_answer.jsonl` is treated as dataset **`mmlu_7000samples`**, not `mmlu`. When your config references a custom-dumped file:

- Use the full stem-minus-bias in `evaluation.args.datasets` (e.g. `mmlu_7000samples`, not `mmlu`)
- Custom-named files do not collide/group with the canonical ones
- The hash file precompute step will use whatever name you pass; reuse the same name consistently across runs

## State and resumption

Pipeline state is saved to `experiments/{name}/state.json`. If a stage fails:
1. Fix the issue
2. Re-run the same command — completed stages auto-skip
3. Or use `--start-from {failed_stage}` to be explicit

Use `--force` to re-run already-completed stages.

## Multi-config (parallel experiments)

Pass multiple YAML configs to run experiments in parallel with combined analysis:

```bash
python scripts/tinker_training/run_experiment.py bct_config.yaml rlct_config.yaml
```

Each pipeline (data_gen → train → eval) runs in its own thread. Analysis runs once at the end, comparing all models in the same plots.

## Combining training methods (`data_mode`)

Support for combining data from multiple `data_generation` steps:

```yaml
data_generation:
  - script: generate_vft_data
    args: {datasets: [truthfulqa, mmlu], bias: suggested_answer, ...}
  - script: generate_bct_from_test
    args: {datasets: [mmlu, truthfulqa], bias: suggested_answer, ...}

training:
  method: sft
  data_mode: interleave   # or "sequential" or "concat"
```

- **`interleave`**: One training run, round-robin mixed data from all gen steps
- **`sequential`**: Train on each gen step's output in order, chaining checkpoints (VFT → BCT)
- **`concat`** (default): Concatenate all data files into one training run

Data generation steps with different scripts run in parallel. Args are passed through generically — any script works without runner changes.

## Model registry (IMPORTANT — always use for new models)

**Always include `viz_registration`** for a new training type. Otherwise plots/tables silently drop the run's eval logs (loader prints a warning naming the unrecognised directory).

The block becomes a `[training_types.<key>]` entry in the registry at viz import time. The model prefix (e.g. `qwen3-`) is read from `experiments.toml`'s `[models.*]` blocks; only training-type suffixes need declaring here.

```yaml
viz_registration:
  dir_suffix: bct-sa          # matches the dir name after the model prefix
  display_name: "BCT SA"      # legend label in plots
  color: "#4292c6"            # bar color
  control_color: "#9ecae1"    # if include_control: true; auto-spawns the ctrl variant
  hatch: ""                   # bar pattern (use "//" for controls; auto for ctrl variant)
  training_biases: [suggested_answer]  # biases this run trains on
```

For experiments with `include_control: true`, supply `control_color` — the registry auto-mints a `<dir_suffix>-ctrl` training_type with that color, hatch `//`, and `is_control: true`. No second block needed.

**Tip**: existing training types are listed in `sycophancy_eval_inspect/experiments.toml` under `[training_types.*]`. Check there for an existing match before inventing a new `dir_suffix`.
