---
name: generate-vft-data
description: Generate VFT (Verbalization Fine-Tuning) training data. Samples biased/unbiased completions, classifies switches, and uses an LLM to edit unfaithful CoTs to add bias verbalization.
argument-hint: [model] [bias-type]
---

# Generate VFT Training Data

Generate VFT (Verbalization Fine-Tuning) training data by:
1. Sampling CoT completions from a model on both biased and unbiased prompts
2. Classifying into 3 categories: non-switch, faithful switch, unfaithful switch
3. Using an LLM editor to insert bias verbalization into unfaithful switches
4. Assembling final training data in `{"messages": [...]}` format

## Arguments

- `$0` — Model name (e.g., `meta-llama/Llama-3.1-8B-Instruct`)
- `$1` — (Optional) Bias type (default: `suggested_answer`)

## Before running

**Confirm with the user:**
1. Which model to sample from (show available models if unsure: `python scripts/tinker_training/generate_vft_data.py --list-models` via the generate_bct_data script)
2. Which datasets (e.g., `truthfulqa mmlu`) and their limits
3. Bias type (default: `suggested_answer`)
4. Output name for the directory
5. Correction/editor model (default: `gpt-5-nano`)
6. Whether to do a small test run first (`--limits 5 5`)
7. If re-running: whether to use `--skip-sampling` to reuse cached Tinker completions

**Explain the command** before executing, showing:
- The full command that will be run
- Expected output file locations
- Estimated number of samples

## Command template

```bash
python scripts/tinker_training/generate_vft_data.py \
    --model $0 \
    --datasets truthfulqa mmlu \
    --bias suggested_answer \
    --limits 817 1183 \
    --output-name vft-suggested-answer \
    --batch-size 64 \
    --save-every 10 \
    --temperature 1.0
```

Re-run with cached Tinker samples (only redo classification + correction):
```bash
python scripts/tinker_training/generate_vft_data.py \
    --model $0 \
    --datasets truthfulqa mmlu \
    --bias suggested_answer \
    --limits 817 1183 \
    --output-name vft-suggested-answer \
    --skip-sampling
```

## Output structure

```
dataset_dumps/train-from-test-{datasets}/{bias}/vft/
  biased_completions.jsonl        # Raw biased CoT completions (checkpoint)
  unbiased_completions.jsonl      # Raw unbiased CoT completions (checkpoint)
  vft_cot.jsonl                   # Final training data: {"messages": [...]}
  vft_metadata.json               # Category breakdown and config
```

## Key flags

| Flag | Description |
|------|-------------|
| `--model MODEL` | Model for Tinker sampling |
| `--datasets D [D...]` | Dataset names (e.g., `truthfulqa mmlu`) |
| `--bias TYPE` | Bias type (e.g., `suggested_answer`) |
| `--limits N [N...]` | Per-dataset sample limits (matches dataset order) |
| `--output-name NAME` | Output directory name |
| `--temperature T` | Sampling temperature (default: 1.0, matches paper) |
| `--correction-model MODEL` | OpenAI model for CoT editing (default: `gpt-5-nano`) |
| `--skip-sampling` | Skip Tinker sampling, reuse cached completions |
| `--checkpoint PATH` | Use a specific model checkpoint |
| `--fresh` | Ignore existing checkpoints, start over |
| `--batch-size N` | Tinker sampling batch size (default: 64) |
| `--save-every N` | Save checkpoint every N batches (default: 10) |

## Pipeline details

### Category classification
- **Non-switch**: Model gives same answer on biased and unbiased prompts -> pair biased prompt with unbiased completion (same as BCT)
- **Faithful switch**: Answer changes AND biased CoT acknowledges the cue -> keep biased completion as-is
- **Unfaithful switch**: Answer changes BUT biased CoT doesn't verbalize -> LLM edits unbiased CoT to add verbalization
- **Skipped**: Unparseable answers (excluded from training)

### Verbalization detection
Uses `_BIAS_ACK_PROMPTS` imported from `sycophancy_eval_inspect/mcq/scorer.py` — the same prompts used for eval scoring.

### CoT correction
For unfaithful switches, the correction model minimally edits the unbiased completion to:
1. Acknowledge the bias cue exists
2. Explicitly reason about why it shouldn't change the answer
3. Preserve the original reasoning and correct answer

## Script location

`scripts/tinker_training/generate_vft_data.py`

## Typical results (Llama-3.1-8B on suggested_answer, 2000 samples)

- Non-switches: ~55% (same answer regardless of bias)
- Faithful switches: ~4% (verbalizes influence)
- Unfaithful switches: ~25% (silently influenced)
- Skipped: ~16% (unparseable)
- Total VFT samples: ~1675

## After data generation

Train a VFT model using the train-tinker skill:
```bash
python scripts/tinker_training/train_sft.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --data dataset_dumps/train-from-test-{datasets}/{bias}/vft/vft_cot.jsonl \
    --experiment-name vft-{bias} \
    --run-name train \
    --batch-size 128 \
    --save-every 5 \
    -y
```
