# n_rollouts hyperparameter sweep — gpt-oss-20b RLCT

1D sweep over `n_{ref,train,consistency,anchor}_rollouts ∈ {32, 64, 128}` with all
other HPs fixed at the LR-sweep canonical (lr=2.86e-4, n_datapoints=64,
batch_size=8, kl_coef=0.05, anchor_weight=0).

The r=32 datapoint reuses the existing LR-sweep run
`lr_hp_gpt20b_rlct_da_g4_lr2p86e4` (same config aside from rollouts). New runs
are r=64 and r=128 only.

Eval suite: shares `artifacts/eval_suites/lr_hparam_truthfulqa/eval_logs/`
(same hash file, same gpt-oss-20b base eval, same TQA 5-bias suite). Sweep
plots go to `artifacts/eval_suites/n_rollouts_hparam_truthfulqa/plots/`.

## Train + eval

```bash
for cfg in \
    scripts/tinker_training/experiment_configs/n_rollouts_hparam/gpt20b_rlct_da_g4_r64.yaml \
    scripts/tinker_training/experiment_configs/n_rollouts_hparam/gpt20b_rlct_da_g4_r128.yaml
do
    /opt/anaconda3/envs/RLconsistencytraining/bin/python scripts/tinker_training/run_experiment.py "$cfg" \
        --stages training,evaluation --max-parallel 1 || exit 1
done
```

## Analysis (3-point n_rollouts plot)

```bash
/opt/anaconda3/envs/RLconsistencytraining/bin/python scripts/tinker_training/run_experiment.py \
    scripts/tinker_training/experiment_configs/lr_hparam/gpt20b_rlct_da_g4_lr2p86e4.yaml \
    scripts/tinker_training/experiment_configs/n_rollouts_hparam/gpt20b_rlct_da_g4_r64.yaml \
    scripts/tinker_training/experiment_configs/n_rollouts_hparam/gpt20b_rlct_da_g4_r128.yaml \
    --stages analysis --max-parallel 1
```

## Push to catalog + W&B

```bash
/opt/anaconda3/envs/RLconsistencytraining/bin/python scripts/build_run_catalog.py
/opt/anaconda3/envs/RLconsistencytraining/bin/python scripts/push_catalog_to_wandb.py --only-with-metrics
```
