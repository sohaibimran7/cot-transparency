---
name: train-tinker
description: Launch SFT or RL consistency training on Tinker. Supports BCT (supervised) and RLCT (reinforcement learning) methods.
argument-hint: [sft|rl] [options]
---

# Train on Tinker

Launch supervised fine-tuning (SFT/BCT) or RL consistency training (RLCT) via the Tinker API.

## Arguments

- `$0` — Training type: `sft` or `rl`
- `$1` — (Optional) Additional context (e.g., data path, experiment name)

## Before running

**Confirm with the user:**
1. Training type: SFT (for BCT) or RL (for RLCT)
2. Training data file(s)
3. Model (default: `meta-llama/Llama-3.1-8B-Instruct`)
4. Experiment and run names
5. Key hyperparameters (see defaults below)
6. Checkpoint save frequency

**Explain the full training configuration** before executing, including:
- Data file and sample count
- All hyperparameters that differ from defaults
- Estimated training steps
- Checkpoint save schedule

## Known issues

### Learning rate for unknown models
`get_recommended_lr()` in `common.py` crashes with `AssertionError` for models not in `hyperparam_utils` (e.g., `openai/gpt-oss-120b`). **Always pass `--lr` explicitly** for non-Llama models:
```bash
--lr 1e-4   # Standard LoRA fallback
```

## Naming convention

Follow the pattern from `sycophancy_eval_inspect/logs/cot_100samples/`:
- `{model}-{method}-{details}` e.g. `gpt-bct-mti-4k`, `llama-rlct-s50`
- `mti` = mmlu+truthfulqa+instruct, `mt` = mmlu+truthfulqa only
- `4k` = ~4000 samples, `s50` = 50 situations per dataset
- Control runs: `gpt-control-mti-4k`, `gpt-rl-control-s50`

The `--run-name` for SFT and `--run_name` for RL should match the eval `--name` flag.

## SFT (BCT / VFT) Training

Used for Bias Consistency Training (BCT) or Verbalization Fine-Tuning (VFT) — both use SFT on different data formats.

### Preferred CLI: `train_sft.py`
Use the unified CLI script rather than writing custom Python scripts:
```bash
python scripts/tinker_training/train_sft.py \
    --model openai/gpt-oss-120b \
    --data path/to/data.jsonl \
    --experiment-name bct-suggested-answer \
    --run-name gpt-bct-mti-4k \
    --batch-size 128 --lora-rank 8 --save-every 8 --skip-near-final 21 --lr 1e-4 -y
```

### Selective checkpointing with `--skip-near-final`
`skip_near_final_steps` skips intermediate checkpoints within N steps of the final step.
Use it to keep only early + final checkpoints. Logic: `steps_remaining <= skip_near_final → skip`.

Example: 30 total steps, save_every=8 → checkpoints at 8, 16, 24, final.
To keep only step 8 + final: `--skip-near-final 21` (step 16 has 14 remaining ≤ 21 → skipped).

### Config defaults
```python
SFTConfig(
    model="meta-llama/Llama-3.1-8B-Instruct",
    lora=LoRAConfig(rank=8),                    # LoRA rank (8 or 32)
    optimizer=AdamConfig(lr_schedule="linear"),   # LR auto-detected from model
    batch_size=128,                               # or 16 for small datasets
    n_epochs=1,
    checkpoint=CheckpointConfig(
        save_every_n_steps=5,                     # Checkpoint frequency
        skip_near_final_steps=0,                  # Skip checkpoints near end
    ),
)
```

### Training data format (JSONL)
```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

### Command pattern
```python
# In a script:
from cot_transparency.apis.tinker.finetune import train_sft, SFTConfig
checkpoint = asyncio.run(train_sft(Path("data/train.jsonl"), config=config))
```

### Existing training scripts
- `scripts/tinker_training/train_sft.py` — Full-featured SFT (supports BCT and VFT data)
- `scripts/tinker_training/train_bct_suggested_answer.py` — BCT on suggested answer data
- `scripts/tinker_training/simple_sft_train.py` — Generic SFT training

### VFT-specific notes
- VFT training uses the same `train_sft.py` script with VFT data from `generate_vft_data.py`
- The final checkpoint is always saved with full state (`kind="both"`) for resuming
- To measure obfuscation: train BCT/RLCT on top of a VFT checkpoint using `--checkpoint`

## RL (RLCT) Training

RL Consistency Training — GRPO rate-matching to reduce sycophancy without labels.

### Config defaults
```python
RLConfig(
    model="meta-llama/Llama-3.1-8B-Instruct",
    lora=LoRAConfig(rank=8),
    optimizer=AdamConfig(lr_schedule="constant"),
    reference_rate=RateEstimationConfig(
        perturbation_indices=[0],    # Index 0 = unbiased
        n_samples=128,
    ),
    training=TrainingSamplingConfig(
        perturbation_indices=[1],    # Index 1 = biased
        n_samples_for_rate=128,
        n_samples_for_gradient=128,
    ),
    loop=TrainingLoopConfig(
        situations_per_group=1,
        gradient_accumulation_steps=1,
        refresh_policy_every_n_steps=1,
    ),
    generation=GenerationConfig(
        max_new_tokens=8192,
        temperature=1.0,
    ),
    checkpoint=CheckpointConfig(save_every_n_steps=50),
    kl_coef=0.05,
    loss_fn="ppo",
)
```

### RLCT requires
1. **Situations**: List of dicts with `unbiased_question`, `biased_question`, `biased_option`, `ground_truth`
2. **Perturbation functions**: Map situation -> prompt (unbiased and biased variants)
3. **Trait classifier**: Detect whether model follows biased suggestion

### Command pattern
```bash
python scripts/tinker_training/test_rl_training.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --n_samples 50 \
    --experiment_name rl_test \
    --run_name suggested_answer
```

Add `--control` for control run (both perturbations use unbiased prompts).

### Key RL flags
| Flag | Description |
|------|-------------|
| `--n_samples N` | Samples per dataset |
| `--model MODEL` | Base model |
| `--experiment_name NAME` | Experiment name |
| `--run_name NAME` | Run name |
| `--control` | Control run (unbiased for both ref and train) |
| `--ref_perturbations IDX` | Perturbation indices for reference rate |
| `--train_perturbations IDX` | Perturbation indices for training |
| `--dry_run` | Load data only, don't train |

## Resuming from a checkpoint (e.g., obfuscation: BCT/RLCT on VFT)

Both SFT and RL scripts support `--resume-from` / `--resume_from` to load weights from a previous checkpoint before training.

### How it works
- If the path contains `/weights/` (not `/sampler_weights/`), it uses `load_state_with_optimizer` (full resume with optimizer momentum)
- Otherwise (sampler weights or unknown), it uses `load_state` (weights only, optimizer resets)
- The Tinker API: `training_client.load_state(path)` or `training_client.load_state_with_optimizer(path)`

### SFT example (BCT on VFT)
```bash
# 1. Generate BCT data from VFT model
python scripts/tinker_training/generate_bct_from_test.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --checkpoint "tinker://...sampler_weights/vft-suggested-answer_train" \
    --datasets truthfulqa mmlu --bias suggested_answer \
    --output-name bct-on-vft

# 2. Train BCT starting from VFT weights
python scripts/tinker_training/train_sft.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --resume-from "tinker://...weights/vft-suggested-answer_train" \
    --data path/to/bct_cot.jsonl \
    --experiment-name bct-on-vft --run-name train -y
```

### RLCT example (RLCT on VFT)
```bash
python scripts/tinker_training/test_rl_training.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --resume_from "tinker://...weights/vft-suggested-answer_train" \
    --experiment_name rlct-on-vft --run_name suggested_answer
```

## Key modules

- `cot_transparency/apis/tinker/finetune.py` — SFT training (`train_sft`, `SFTConfig`)
- `cot_transparency/apis/tinker/rl_training.py` — RL training (`RLTrainer`, `RLConfig`)
- `cot_transparency/apis/tinker/common.py` — Shared configs (`LoRAConfig`, `AdamConfig`, `CheckpointConfig`)

## User preference

When creating new training scripts, **create a new file** rather than editing existing working scripts. This preserves working baselines.
