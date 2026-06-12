# Learning-rate hyperparameter sweep commands

Pure LR sweep — 3 LRs × 4 (model × method) buckets = 12 runs.

Protocol:

- Train on **MMLU `distractor_argument_g4`** (Llama: `cot`, gpt-oss-20b: `no_cot`).
- Evaluate on **TruthfulQA 5-bias suite**: `distractor_argument_g4`, `suggested_answer`, `distractor_fact`, `wrong_few_shot`, `spurious_few_shot_squares`.
- BCT n=2000 (capped via `--data PATH:2000` in train_sft); RLCT n_datapoints=64; rollouts=32 across all three rollout slots.
- Controls skipped; base evals included for plotting baseline.

## Data and hash setup

```bash
/opt/anaconda3/envs/RLconsistencytraining/bin/python scripts/tinker_training/generate_bct_from_test.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --bias distractor_argument_g4 \
    --datasets mmlu \
    --limits 2000 \
    --max-tokens 16384 \
    --temperature 1.0 \
    --batch-size 64 \
    --output-dir dataset_dumps/train-from-test-mmlu/da-g4/llama_3_1_8b_instruct

/opt/anaconda3/envs/RLconsistencytraining/bin/python scripts/tinker_training/generate_bct_from_test.py \
    --model openai/gpt-oss-20b \
    --bias distractor_argument_g4 \
    --datasets mmlu \
    --limits 2000 \
    --max-tokens 24576 \
    --temperature 1.0 \
    --batch-size 64 \
    --non-cot \
    --output-dir dataset_dumps/train-from-test-mmlu/da-g4/gpt_oss_20b

/opt/anaconda3/envs/RLconsistencytraining/bin/python -m sycophancy_eval_inspect.generate_hash_file \
    --datasets truthfulqa \
    --bias-types distractor_argument_g4,suggested_answer,distractor_fact,wrong_few_shot,spurious_few_shot_squares \
    --limit 200 \
    --output artifacts/eval_suites/lr_hparam_truthfulqa/eval_logs/common_hashes.json
```

## Base evals

```bash
/opt/anaconda3/envs/RLconsistencytraining/bin/python -m sycophancy_eval_inspect.run_tinker_evals \
    --base-model meta-llama/Llama-3.1-8B-Instruct \
    --name llama-base \
    --prompt-styles cot \
    --bias-types distractor_argument_g4,suggested_answer,distractor_fact,wrong_few_shot,spurious_few_shot_squares \
    --datasets truthfulqa \
    --log-dir artifacts/eval_suites/lr_hparam_truthfulqa/eval_logs \
    --max-tokens 16384 \
    --limit 200 \
    --max-tasks 50 \
    --hash-file artifacts/eval_suites/lr_hparam_truthfulqa/eval_logs/common_hashes.json

/opt/anaconda3/envs/RLconsistencytraining/bin/python -m sycophancy_eval_inspect.run_tinker_evals \
    --base-model openai/gpt-oss-20b \
    --name gpt-oss-20b-base \
    --prompt-styles no_cot \
    --bias-types distractor_argument_g4,suggested_answer,distractor_fact,wrong_few_shot,spurious_few_shot_squares \
    --datasets truthfulqa \
    --log-dir artifacts/eval_suites/lr_hparam_truthfulqa/eval_logs \
    --max-tokens 24576 \
    --limit 200 \
    --max-tasks 50 \
    --hash-file artifacts/eval_suites/lr_hparam_truthfulqa/eval_logs/common_hashes.json
```

## Training and checkpoint evals

```bash
for cfg in \
    scripts/tinker_training/experiment_configs/lr_hparam/llama_rlct_da_g4_lr1e4.yaml \
    scripts/tinker_training/experiment_configs/lr_hparam/llama_rlct_da_g4_lr2p86e4.yaml \
    scripts/tinker_training/experiment_configs/lr_hparam/llama_rlct_da_g4_lr5e4.yaml \
    scripts/tinker_training/experiment_configs/lr_hparam/llama_bct_da_g4_lr1e4.yaml \
    scripts/tinker_training/experiment_configs/lr_hparam/llama_bct_da_g4_lr2p86e4.yaml \
    scripts/tinker_training/experiment_configs/lr_hparam/llama_bct_da_g4_lr5e4.yaml \
    scripts/tinker_training/experiment_configs/lr_hparam/gpt20b_rlct_da_g4_lr1e4.yaml \
    scripts/tinker_training/experiment_configs/lr_hparam/gpt20b_rlct_da_g4_lr2p86e4.yaml \
    scripts/tinker_training/experiment_configs/lr_hparam/gpt20b_rlct_da_g4_lr5e4.yaml \
    scripts/tinker_training/experiment_configs/lr_hparam/gpt20b_bct_da_g4_lr1e4.yaml \
    scripts/tinker_training/experiment_configs/lr_hparam/gpt20b_bct_da_g4_lr2p86e4.yaml \
    scripts/tinker_training/experiment_configs/lr_hparam/gpt20b_bct_da_g4_lr5e4.yaml
do
    /opt/anaconda3/envs/RLconsistencytraining/bin/python scripts/tinker_training/run_experiment.py "$cfg" \
        --stages training,evaluation \
        --max-parallel 1 || exit 1
done
```

## Analysis

```bash
/opt/anaconda3/envs/RLconsistencytraining/bin/python scripts/tinker_training/run_experiment.py \
    scripts/tinker_training/experiment_configs/lr_hparam/llama_rlct_da_g4_lr1e4.yaml \
    scripts/tinker_training/experiment_configs/lr_hparam/llama_rlct_da_g4_lr2p86e4.yaml \
    scripts/tinker_training/experiment_configs/lr_hparam/llama_rlct_da_g4_lr5e4.yaml \
    scripts/tinker_training/experiment_configs/lr_hparam/llama_bct_da_g4_lr1e4.yaml \
    scripts/tinker_training/experiment_configs/lr_hparam/llama_bct_da_g4_lr2p86e4.yaml \
    scripts/tinker_training/experiment_configs/lr_hparam/llama_bct_da_g4_lr5e4.yaml \
    scripts/tinker_training/experiment_configs/lr_hparam/gpt20b_rlct_da_g4_lr1e4.yaml \
    scripts/tinker_training/experiment_configs/lr_hparam/gpt20b_rlct_da_g4_lr2p86e4.yaml \
    scripts/tinker_training/experiment_configs/lr_hparam/gpt20b_rlct_da_g4_lr5e4.yaml \
    scripts/tinker_training/experiment_configs/lr_hparam/gpt20b_bct_da_g4_lr1e4.yaml \
    scripts/tinker_training/experiment_configs/lr_hparam/gpt20b_bct_da_g4_lr2p86e4.yaml \
    scripts/tinker_training/experiment_configs/lr_hparam/gpt20b_bct_da_g4_lr5e4.yaml \
    --stages analysis \
    --max-parallel 1
```
