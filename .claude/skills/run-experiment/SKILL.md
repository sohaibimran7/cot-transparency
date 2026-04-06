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
- `--force` — Re-run already-completed stages
- `--dry-run` — Print commands without executing

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

analysis:
  args:
    bir: true
    ba: true
    plot: true
```

## Control and base model support

### Control training (`include_control: true`)
- **SFT**: Trains a second model on control (unbiased) data in parallel with the main BCT model
- **RL**: Trains a second model with `--control` flag (unbiased perturbation for both ref and train)
- Control run name is auto-suffixed with `-ctrl` (e.g., `llama-bct-mti-4k-ctrl`)
- Control checkpoint is auto-passed to evaluation stage

### Base model evaluation (`include_base: true`)
- Evaluates the base model (no checkpoint) in parallel with finetuned checkpoints
- Base model name is auto-generated as `{model_prefix}-base` (e.g., `llama-base`)

### Parallel execution
When multiple training runs or evals are needed (main + control, base + checkpoints), they run in parallel as separate subprocesses with labeled output.

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

**Always include `viz_registration`** when using a new model or training type that hasn't been used before. Without it, the analysis stage will fail because `visualize_results.py` won't recognize the model directory names.

The config must register:
1. The **model prefix** (e.g., `qwen3-`) — derived automatically from the model name
2. Each **training type suffix** (e.g., `bct-sa`, `rlct-sa-ctrl`) — the part after the model prefix in the run name

```yaml
viz_registration:
  dir_suffix: bct-sa          # matches run_name suffix after model prefix
  display_name: "BCT SA"      # legend label in plots
  color: "#4292c6"             # bar color
  hatch: ""                    # bar pattern (use "//" for controls)
  training_biases: [suggested_answer]  # which biases this was trained on
```

For experiments with `include_control: true`, you need **two** registry entries (main + ctrl). Use the experiment runner's auto-registration or manually add to `sycophancy_eval_inspect/model_registry.json`.

**Tip**: Check existing entries in `model_registry.json` and `_DIR_TO_TRAINING_TYPE` in `visualize_results.py` before adding — the suffix may already be registered.
