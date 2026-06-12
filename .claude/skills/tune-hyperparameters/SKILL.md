---
name: tune-hyperparameters
description: Run a principled hyperparameter sweep for consistency training (BCT or RLCT). Builds 1D sweep configs over a single HP, shares eval infrastructure with prior sweeps, and pushes to the unified W&B catalog.
argument-hint: [hp-name] [model] [method]
---

# Tune Hyperparameters

Run a principled HP sweep for one consistency-training hyperparameter at a
time. Each run is expensive (training + eval on Tinker), so the goal is
methodologically clean comparisons with minimal wasted compute.

## Core principles

1. **One HP at a time (1D sweeps).** Combinatorial sweeps are unaffordable and
   results are noisy. Hold every other HP fixed at the canonical config for
   the (model × method) bucket; vary one axis with 3–5 values.

2. **Establish a canonical config first.** Before sweeping a new HP, fix a
   reference value for everything else by either (a) reusing an earlier
   sweep's best point, or (b) declaring sensible defaults. The canonical
   becomes the implicit "anchor" datapoint for future sweeps — see point 4.

3. **Share eval-suite directories across sweeps for the same
   (model, training-dataset, eval-dataset, prompt-style).** The base eval,
   `common_hashes.json`, and TQA question selection should match across all
   sweep points so comparisons are apples-to-apples. New sweeps reuse an
   existing `artifacts/eval_suites/<bucket>_truthfulqa/eval_logs/` and write
   plots to a sweep-specific subdir.

4. **Reuse anchor runs.** If a sweep point's HPs are identical to a prior
   run's (e.g. lr-sweep at lr=2.86e-4 and rollouts-sweep at r=32 both use the
   same canonical config), don't re-train — include the existing config in
   the analysis stage as the anchor. The catalog will tag it with both
   sweeps' axes anyway.

5. **Pro-BSR ratio is the primary metric.** `pro_bsr_ratio_training` (and
   `pro_bsr_ratio_held_out_avg` for generalization) — not `accuracy_unbiased`.
   See `analyze-evals` skill for the rationale.

## Repo conventions

- **Config dir**: `scripts/tinker_training/experiment_configs/<hp>_hparam/`
- **Config filename**: `<model_short>_<method>_<train_bias>_<hp_token>.yaml`
  - Examples: `gpt20b_rlct_da_g4_lr1e4.yaml`, `llama_bct_da_g4_r64.yaml`
- **Run name** (`training.args.run_name`): `<model_dir_prefix>-<method>-<bias>-<hp_token>`
  - e.g. `gpt-oss-20b-rlct-da-g4-r64`, `llama-bct-da-g4-lr5e4`
- **viz_registration.dir_suffix**: must be unique across all sweeps; for
  gpt-oss use a `20b`/`120b` infix to avoid collisions with Llama
  (e.g. `rlct-da-g4-20b-r64`).
- **Eval suite log_dir**: reuse the existing one for that
  (model, eval-dataset) pair, e.g. `artifacts/eval_suites/lr_hparam_truthfulqa/eval_logs`
- **Plot output_dir**: separate per sweep,
  e.g. `artifacts/eval_suites/<hp>_hparam_truthfulqa/plots`
- **COMMANDS.md**: each `_hparam/` directory carries a `COMMANDS.md` with the
  exact data-prep, base-eval, train-eval, analysis, and catalog-push
  invocations for that sweep. Always ship one.

## Workflow

### Step 1 — pick the axis and values

Decide on the HP and 3–5 candidate values. Common axes and reasonable spans:

| HP | Sweep range | Notes |
|----|-------------|-------|
| `lr` | 1e-4, 2.86e-4, 5e-4 | 2.86e-4 is the auto-LR for Llama-3.1-8B |
| `n_train_rollouts` (RL) | 32, 64, 128 | Linear compute scaling. Set all 4 rollout slots together |
| `kl_coef` (RL) | 0.01, 0.05, 0.2 | Stabilizes vs learning trade-off |
| `anchor_weight` (RL) | 0.0, 0.1, 0.5 | 0 disables anchor regularization |
| `batch_size` | 1, 2, 4, 8 | At fixed n_datapoints, fewer steps = faster but noisier |
| `n_datapoints` (RL) | 32, 64, 128, 256 | Per-epoch sample count |
| `lora_rank` | 4, 8, 16, 32 | Capacity vs overfit |

### Step 2 — fix the canonical

Read the latest accepted canonical for the (model × method) bucket. Until
documented in this skill, copy from an existing sweep's best point or use
the lr-sweep middle-value config as the template:

- Llama-3.1-8B + RLCT: `experiment_configs/lr_hparam/llama_rlct_da_g4_lr2p86e4.yaml`
- gpt-oss-20b + RLCT: `experiment_configs/lr_hparam/gpt20b_rlct_da_g4_lr2p86e4.yaml`
- Llama-3.1-8B + BCT: `experiment_configs/lr_hparam/llama_bct_da_g4_lr2p86e4.yaml`
- gpt-oss-20b + BCT: `experiment_configs/lr_hparam/gpt20b_bct_da_g4_lr2p86e4.yaml`

### Step 3 — generate the sweep configs

Copy the canonical YAML once per sweep value, change only the swept HP and
identity fields (`name`, `experiment_name`, `run_name`,
`viz_registration.dir_suffix`, `viz_registration.training_type`,
`viz_registration.display_name`, `viz_registration.color`). Leave
`training.args.bias_types`, `evaluation.args.*`, `analysis.args.*` alone.

For RL rollout sweeps, change all four slots together (`n_ref_rollouts`,
`n_train_rollouts`, `n_consistency_rollouts`, `n_anchor_rollouts`) — they
should always match unless you have a specific reason.

For BCT data-size sweeps, vary the `:N` cap on `data` and ensure the BCT
training data file at the parent path has at least N samples.

### Step 4 — show commands and get approval

**Always present the exact training and eval commands to the user before
running anything.** Per project rules (`.claude/CLAUDE.md`), training and
eval invocations require explicit approval. Use the `COMMANDS.md` template
below as your skeleton.

### Step 5 — execute

```bash
# Train + eval (sequential to respect Tinker rate limits; --max-parallel 1)
for cfg in <list of new YAMLs>; do
    /opt/anaconda3/envs/RLconsistencytraining/bin/python scripts/tinker_training/run_experiment.py "$cfg" \
        --stages training,evaluation --max-parallel 1 || exit 1
done

# Combined analysis — always include the anchor config (the reused canonical)
/opt/anaconda3/envs/RLconsistencytraining/bin/python scripts/tinker_training/run_experiment.py \
    <canonical anchor cfg> <new sweep cfgs...> \
    --stages analysis --max-parallel 1
```

### Step 6 — catalog and W&B

After every sweep, refresh the global catalog and re-push to W&B so the
parallel-coords / scatter views see the new datapoints:

```bash
/opt/anaconda3/envs/RLconsistencytraining/bin/python scripts/build_run_catalog.py
/opt/anaconda3/envs/RLconsistencytraining/bin/python scripts/push_catalog_to_wandb.py --only-with-metrics
```

Project: `consistency-training-catalog`. The catalog is checkpoint-URI
indexed (no name parsing), so configs with `viz_registration` in any of the
3 search locations get picked up automatically.

## Reading results

1. **Per-sweep plot**: `artifacts/eval_suites/<hp>_hparam_truthfulqa/plots/`
   contains a bar chart per (eval bias × dataset) with one bar per sweep
   value plus base-model baseline.
2. **Catalog (parquet)**: `artifacts/run_catalog.parquet` — every run is a
   row with all HPs as columns and pro-BSR ratio + bootstrap CI columns.
3. **W&B parallel coords**: filter to one model + method + sweep tag, then
   color by `summary:pro_bsr_ratio_training`. Identifies the best sweep
   value at a glance and surfaces interactions with prior sweeps.

## When to declare a new canonical

After a sweep, if the best value differs meaningfully from the previous
canonical (e.g. > 0.1 abs improvement in pro-BSR ratio, or outside the
overlap of bootstrap CIs), update the canonical config for that (model ×
method) bucket and note the change here. Future sweeps should pivot around
the new canonical.

## Common pitfalls

- **Re-running an existing canonical**: wastes compute. Search the catalog
  first (`pd.read_parquet('artifacts/run_catalog.parquet')`) for runs whose
  HPs match your "anchor" point.
- **dir_suffix collision**: `rlct-da-g4-r64` would clash if both Llama and
  gpt-oss use it. Always include the model size for non-Llama
  (`rlct-da-g4-20b-r64`).
- **Different hash file per sweep**: breaks cross-sweep comparison. Reuse
  the existing eval_suite log_dir + hash_file.
- **Forgetting `training_biases` in `analysis.args`**: the combined-analysis
  builder needs this to label plots correctly. Always set it.
- **TQA limit > 182**: TQA only has 182 questions common across the 5-bias
  suite. Use `evaluation.args.limit: 150` (or lower).
- **gpt-oss prompt style**: gpt-oss uses `no_cot` (it has internal CoT via
  channel tags). Llama uses `cot`.

## See also

- `run-experiment` — the underlying pipeline runner
- `analyze-evals` — pro-BSR ratio metric definitions
- `scripts/build_run_catalog.py` — checkpoint-URI-based catalog builder
- `scripts/push_catalog_to_wandb.py` — W&B push (config vs summary split)
