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

### Auto LR (Llama/Qwen only)
`get_recommended_lr()` uses the formula: `5e-5 × 10 × (2000/hidden_size)^exponent` where exponent is 0.781 (Llama) or 0.0775 (Qwen). For Llama-3.1-8B this gives **~0.000286**.

For models not in `hyperparam_utils` (e.g., `openai/gpt-oss-120b`), it crashes with `AssertionError`. **Always pass `--lr` explicitly** for non-Llama/Qwen models:
```bash
--lr 1e-4   # Standard LoRA fallback for GPT-OSS
```

## Naming convention

Follow the pattern from `sycophancy_eval_inspect/logs/cot_100samples/`:
- `{model}-{method}-{details}` e.g. `gpt-bct-mti-4k`, `llama-rlct-s50`
- `mti` = mmlu+truthfulqa+instruct, `mt` = mmlu+truthfulqa only
- `4k` = ~4000 samples, `s50` = 50 total situations
- Control runs: `gpt-control-mti-4k`, `gpt-rl-control-s50`

The `--run-name` should NOT include checkpoint step info (e.g., don't use `-s50`). Steps are appended automatically by `build_checkpoint_name()` → `{experiment_name}_{run_name}_step{N}`.

The eval `--name` flag should be `{model}-{experiment}_{run}_step{N}` to match the checkpoint label.

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

### Preferred CLI: `train_rl.py`
Use the unified RL CLI script rather than `test_rl_training.py`:
```bash
python scripts/tinker_training/train_rl.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --bias-types suggested_answer \
    --datasets mmlu,truthfulqa \
    --n-samples 50 \
    --experiment-name rl_suggested_answer \
    --run-name llama-rlct-sa \
    --lora-rank 8 --refresh-every 1 --checkpoint-every 50 -y
```
Note: No `--lr` for Llama — auto LR (~0.000286) is preferred. Only pass `--lr` explicitly for non-Llama/Qwen models.

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

### Multi-turn bias handling in RL

The RL pipeline handles multi-turn biases (are_you_sure, post_hoc) automatically:

- **are_you_sure**: Detected via `_is_on_policy_are_you_sure()`. Uses 3 on-policy generations:
  1. Teacher-forces the **ground truth** letter (not biased option) as the first answer
  2. Samples intermediate response to "Are you sure?" on-policy
  3. Samples final answer conditioned on the on-policy intermediate
  Only the final response is used for gradient computation.

- **post_hoc**: Standard single-generation with pre-filled assistant message (wrong answer from dataset). Model generates explanation after "Explain your reasoning."

### Reasoning model support (gpt-oss, Qwen3, DeepSeek-R1, etc.)

For reasoning models, the RL pipeline uses tinker-cookbook 0.2.1's native renderer to extract clean text:
- `renderers.get_text_content(parsed_msg)` in `_sample_from_client()` extracts only the final text, stripping thinking/reasoning parts generically across all model families
- This prevents thinking from leaking into conversation history in multi-turn flows (are_you_sure intermediate turns)
- `_format_prompt()` also calls `strip_thinking_from_previous_turns()` to remove the `thinking` dict key from prior assistant messages

### RLCT requires
1. **Situations**: List of dicts with `unbiased_question`, `biased_question`, `biased_option`, `ground_truth`
2. **Perturbation functions**: Map situation -> prompt (unbiased and biased variants)
3. **Trait classifier**: Detect whether model follows biased suggestion

### Key RL flags
| Flag | Description |
|------|-------------|
| `--model MODEL` | Base model |
| `--bias-types TYPES` | Comma-separated bias types (e.g. `suggested_answer,wrong_few_shot`) |
| `--datasets DATASETS` | Comma-separated datasets (default: `mmlu,truthfulqa`) |
| `--n-samples N` | Total situations (split evenly across dataset × bias_type combos, default: 100) |
| `--experiment-name NAME` | Experiment name |
| `--run-name NAME` | Run name (used in checkpoint path and eval `--name`) |
| `--lr RATE` | Learning rate (default: auto from Tinker's `get_recommended_lr`; pass explicitly for non-Llama models) |
| `--lr-schedule SCHED` | `constant`, `linear`, or `cosine` |
| `--lora-rank N` | LoRA rank (default: 8) |
| `--kl-coef FLOAT` | KL penalty coefficient (default: 0.05) |
| `--loss-fn FN` | `ppo` or `reinforce` |
| `--n-ref-samples N` | Samples for reference rate estimation (default: 128) |
| `--n-train-samples N` | Samples for training rate estimation (default: 128) |
| `--n-grad-samples N` | Samples for gradient (default: same as `--n-train-samples`) |
| `--temperature FLOAT` | Sampling temperature (default: 1.0) |
| `--max-new-tokens N` | Max generation tokens (default: 8192) |
| `--situations-per-group N` | Situations per gradient step (default: 1) |
| `--gradient-accumulation-steps N` | Gradient accumulation (default: 1) |
| `--refresh-every N` | Refresh sampling policy every N steps (default: 1). Higher values enable deeper prefetch pipelining but use slightly staler samples. |
| `--checkpoint-every N` | Save checkpoint every N steps (default: 50) |
| `--save-state` | Save full optimizer state for resuming |
| `--control` | Control run (unbiased for both ref and train) |
| `--resume-from PATH` | Tinker checkpoint to resume from |
| `--dry-run` | Load data and print config, don't train |
| `-y` | Skip confirmation prompt |

### Legacy script
`test_rl_training.py` is the older RL script with fewer CLI options. Use `train_rl.py` for new runs.

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
python scripts/tinker_training/train_rl.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --resume-from "tinker://...weights/vft-suggested-answer_train" \
    --bias-types suggested_answer \
    --experiment-name rlct-on-vft --run-name suggested_answer \
    --lr 1e-4 -y
```

## Key modules

- `cot_transparency/apis/tinker/finetune.py` — SFT training (`train_sft`, `SFTConfig`)
- `cot_transparency/apis/tinker/rl_training.py` — RL training (`RLTrainer`, `RLConfig`)
- `cot_transparency/apis/tinker/common.py` — Shared configs (`LoRAConfig`, `AdamConfig`, `CheckpointConfig`)

## User preference

When creating new training scripts, **create a new file** rather than editing existing working scripts. This preserves working baselines.
