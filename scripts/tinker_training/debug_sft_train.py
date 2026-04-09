"""
Debug version of simple_sft_train.py to investigate mode collapse.

Run with:
    conda activate RLconsistencytraining
    python -m pdb scripts/tinker_training/debug_sft_train.py

Key breakpoints set at:
    1. Renderer selection - check model/renderer alignment
    2. Token rendering - inspect actual tokens and decoded text
    3. Weight/masking - verify loss masking is correct
    4. Datum construction - check proper format
    5. Forward/backward - inspect loss metrics
"""

import json
import random
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import tinker
from tinker import types
from tinker_cookbook import renderers, model_info
from tinker_cookbook.tokenizer_utils import get_tokenizer

# Set to True for interactive pdb debugging
INTERACTIVE = False


def load_sample_data(file_path: Path, n_samples: int = 3) -> list[dict]:
    """Load a few samples for debugging."""
    samples = []
    with open(file_path) as f:
        for i, line in enumerate(f):
            if i >= n_samples:
                break
            samples.append(json.loads(line))
    return samples


def debug_renderer_selection(model: str):
    """
    BREAKPOINT 1: Check renderer-model alignment

    Pitfall: Renderer-model mismatch corrupts token sequences
    """
    print("\n" + "="*60)
    print("BREAKPOINT 1: Renderer Selection")
    print("="*60)

    tokenizer = get_tokenizer(model)
    renderer_name = model_info.get_recommended_renderer_name(model)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    print(f"Model: {model}")
    print(f"Recommended renderer: {renderer_name}")
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    print(f"Tokenizer class: {type(tokenizer).__name__}")

    # Check special tokens
    print(f"\nSpecial tokens:")
    print(f"  BOS token: {tokenizer.bos_token} (id={tokenizer.bos_token_id})")
    print(f"  EOS token: {tokenizer.eos_token} (id={tokenizer.eos_token_id})")
    print(f"  PAD token: {tokenizer.pad_token} (id={tokenizer.pad_token_id})")

    if INTERACTIVE:
        import pdb; pdb.set_trace()  # BREAKPOINT 1

    return renderer, tokenizer


def debug_token_rendering(sample: dict, renderer, tokenizer):
    """
    BREAKPOINT 2: Inspect tokenization and rendered output

    Check: Are tokens being rendered correctly for the model?
    """
    print("\n" + "="*60)
    print("BREAKPOINT 2: Token Rendering")
    print("="*60)

    messages = sample["messages"]
    print(f"\nOriginal messages ({len(messages)} turns):")
    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")[:100]
        print(f"  [{i}] {role}: {content}...")

    # Render to tokens
    model_input, weights = renderer.build_supervised_example(messages)
    tokens = model_input.tolist()
    weights_list = weights.tolist()

    print(f"\nRendered output:")
    print(f"  Total tokens: {len(tokens)}")
    print(f"  Total weights: {len(weights_list)}")
    print(f"  Non-zero weights: {sum(1 for w in weights_list if w > 0)}")
    print(f"  Sum of weights: {sum(weights_list):.2f}")

    # Decode tokens to see what we're training on
    decoded = tokenizer.decode(tokens)
    print(f"\nDecoded text (first 500 chars):")
    print(decoded[:500])

    # Show token-by-token with weights for first 50 tokens
    print(f"\nToken-by-token (first 50):")
    print(f"{'Idx':<5} {'Token ID':<10} {'Weight':<8} {'Decoded':<30}")
    print("-" * 60)
    for i in range(min(50, len(tokens))):
        tok_id = tokens[i]
        weight = weights_list[i]
        decoded_tok = tokenizer.decode([tok_id])
        # Escape newlines and special chars for display
        decoded_tok = repr(decoded_tok)[1:-1][:25]
        print(f"{i:<5} {tok_id:<10} {weight:<8.2f} {decoded_tok:<30}")

    # Show where weights transition (assistant responses start)
    print(f"\nWeight transitions (where loss masking changes):")
    prev_weight = weights_list[0]
    for i, w in enumerate(weights_list):
        if w != prev_weight:
            context_start = max(0, i-5)
            context_tokens = tokens[context_start:i+5]
            context_decoded = tokenizer.decode(context_tokens)
            print(f"  Position {i}: weight {prev_weight} -> {w}")
            print(f"    Context: {repr(context_decoded)[:80]}")
            prev_weight = w

    if INTERACTIVE:
        import pdb; pdb.set_trace()  # BREAKPOINT 2

    return tokens, weights_list


def debug_datum_construction(tokens: list, weights: list, tokenizer):
    """
    BREAKPOINT 3: Check Datum construction

    CRITICAL PITFALL FOUND:
    - Cookbook uses tokens[:-1] as input, tokens[1:] as target (next-token prediction)
    - Our script used same tokens for both (WRONG!)
    """
    print("\n" + "="*60)
    print("BREAKPOINT 3: Datum Construction")
    print("="*60)

    import torch
    from tinker_cookbook.supervised.common import datum_from_tokens_weights

    # Convert to torch for cookbook helper
    tokens_tensor = torch.tensor(tokens)
    weights_tensor = torch.tensor(weights, dtype=torch.float32)

    # Our WRONG manual construction (what simple_sft_train.py does)
    print("\n--- OUR SCRIPT'S CONSTRUCTION (WRONG) ---")
    wrong_datum = types.Datum(
        model_input=types.ModelInput.from_ints(tokens=tokens),
        loss_fn_inputs=dict(target_tokens=tokens, weights=weights),
    )
    print(f"  Input tokens length: {len(tokens)}")
    print(f"  Target tokens length: {len(tokens)}")
    print(f"  Weights length: {len(weights)}")
    print(f"  Input[0:5]: {tokens[0:5]} -> {tokenizer.decode(tokens[0:5])!r}")
    print(f"  Target[0:5]: {tokens[0:5]} -> {tokenizer.decode(tokens[0:5])!r}")
    print(f"  ⚠️  INPUT == TARGET (model predicts token from itself!)")

    # Cookbook CORRECT construction
    print("\n--- COOKBOOK'S CONSTRUCTION (CORRECT) ---")
    correct_datum = datum_from_tokens_weights(tokens_tensor, weights_tensor, max_length=None)

    # Extract data from correct datum
    correct_input = correct_datum.model_input
    correct_target = correct_datum.loss_fn_inputs['target_tokens']
    correct_weights = correct_datum.loss_fn_inputs['weights']

    if hasattr(correct_target, 'data'):
        target_list = list(correct_target.data)
    else:
        target_list = correct_target

    # Get input length
    input_len = correct_input.length

    print(f"  Input tokens length: {input_len}")
    print(f"  Target tokens length: {len(target_list)}")
    print(f"  Weights shape: {correct_weights.shape}")

    print(f"\n  Input[0:5]:  {tokens[0:5]} -> {tokenizer.decode(tokens[0:5])!r}")
    print(f"  Target[0:5]: {target_list[0:5]} -> {tokenizer.decode(target_list[0:5])!r}")
    print(f"  ✓ TARGET = INPUT shifted by 1 (proper next-token prediction!)")

    # Show the shift visually
    print(f"\n  Token alignment:")
    print(f"    Position:  0    1    2    3    4")
    print(f"    Input:    [{tokens[0]:>4}][{tokens[1]:>4}][{tokens[2]:>4}][{tokens[3]:>4}][{tokens[4]:>4}] ...")
    print(f"    Target:   [{target_list[0]:>4}][{target_list[1]:>4}][{target_list[2]:>4}][{target_list[3]:>4}][{target_list[4]:>4}] ...")
    print(f"                ↑     ↑     ↑     ↑     ↑")
    print(f"             predict predict predict predict predict")

    print("\n" + "="*60)
    print("⚠️  CRITICAL BUG: Our script passes tokens directly without shifting!")
    print("   This breaks next-token prediction and causes garbage outputs.")
    print("="*60)

    if INTERACTIVE:
        import pdb; pdb.set_trace()  # BREAKPOINT 3

    return correct_datum


def debug_forward_backward(training_client, batch_data: list, batch_size: int):
    """
    BREAKPOINT 4: Inspect loss computation

    Check: Loss values and normalization
    """
    print("\n" + "="*60)
    print("BREAKPOINT 4: Forward/Backward Pass")
    print("="*60)

    print(f"Batch size: {len(batch_data)}")

    # Run forward/backward
    fwd_bwd_result = training_client.forward_backward(
        batch_data, loss_fn="cross_entropy"
    ).result()

    # Inspect all metrics
    print(f"\nAll metrics returned:")
    for key, value in fwd_bwd_result.metrics.items():
        print(f"  {key}: {value}")

    # Calculate different normalizations
    total_loss = fwd_bwd_result.metrics.get('loss:sum', 0.0)
    total_weighted_tokens = fwd_bwd_result.metrics.get('total_weighted_tokens', None)

    print(f"\nLoss calculations:")
    print(f"  loss:sum = {total_loss}")
    print(f"  total_weighted_tokens = {total_weighted_tokens}")

    per_sample_loss = total_loss / batch_size
    print(f"  Per-sample loss (loss:sum / batch_size) = {per_sample_loss:.4f}")

    if total_weighted_tokens and total_weighted_tokens > 0:
        per_token_loss = total_loss / total_weighted_tokens
        print(f"  Per-token loss (loss:sum / total_weighted_tokens) = {per_token_loss:.4f}")

    if INTERACTIVE:
        import pdb; pdb.set_trace()  # BREAKPOINT 4

    return fwd_bwd_result


def debug_fresh_lora_vs_base(model: str, batch_data: list):
    """
    BREAKPOINT 5: Compare fresh LoRA init vs base model

    Key question: Why does fresh LoRA have ~25x higher loss?
    """
    print("\n" + "="*60)
    print("BREAKPOINT 5: Fresh LoRA vs Base Model Comparison")
    print("="*60)

    service_client = tinker.ServiceClient()

    # Test 1: Fresh LoRA training client
    print("\nCreating fresh LoRA training client...")
    lora_client = service_client.create_lora_training_client(
        base_model=model,
        rank=32,
        train_mlp=True,
        train_attn=True,
        train_unembed=True,
    )

    lora_result = lora_client.forward_backward(
        batch_data, loss_fn="cross_entropy"
    ).result()

    lora_loss_sum = lora_result.metrics.get('loss:sum', 0.0)
    lora_weighted_tokens = lora_result.metrics.get('total_weighted_tokens', 1)
    lora_per_token = lora_loss_sum / lora_weighted_tokens if lora_weighted_tokens else 0

    print(f"\nFresh LoRA client:")
    print(f"  loss:sum = {lora_loss_sum:.4f}")
    print(f"  total_weighted_tokens = {lora_weighted_tokens}")
    print(f"  per-token loss = {lora_per_token:.4f}")

    # Test 2: Base model sampling client (for comparison)
    print("\nNote: Cannot easily compare to base model loss without LoRA")
    print(f"  Previous tests showed base model per-token loss ~0.74")

    # Calculate expected weighted tokens from batch data
    total_expected_weights = 0
    for d in batch_data:
        weights_data = d.loss_fn_inputs['weights']
        if hasattr(weights_data, 'tolist'):
            w_list = weights_data.tolist()
        elif hasattr(weights_data, 'data'):
            w_list = list(weights_data.data)
        else:
            w_list = weights_data
        total_expected_weights += sum(w_list)

    print(f"\nExpected total weighted tokens (from batch): {total_expected_weights}")
    print(f"Actual total_weighted_tokens from API: {lora_weighted_tokens}")

    # Per-token loss using expected weights
    if total_expected_weights > 0:
        corrected_per_token = lora_loss_sum / total_expected_weights
        print(f"\nCorrected per-token loss (using expected weights): {corrected_per_token:.4f}")
    else:
        corrected_per_token = 0

    print(f"\nComparison:")
    print(f"  Fresh LoRA per-token loss (API reported): {lora_per_token:.4f}")
    print(f"  Fresh LoRA per-token loss (corrected): {corrected_per_token:.4f}")
    print(f"  Expected base model loss: ~0.74")

    if corrected_per_token > 0:
        print(f"  Ratio (corrected): {corrected_per_token / 0.74:.1f}x higher with fresh LoRA")

    if INTERACTIVE:
        import pdb; pdb.set_trace()  # BREAKPOINT 5

    return lora_result


def main():
    # Config
    model = "meta-llama/Llama-3.1-8B-Instruct"
    data_path = Path("dataset_dumps/train_seed_42/llama-3-1-8b-instruct/mixed.jsonl")
    batch_size = 4  # Small batch for debugging

    print("="*60)
    print("DEBUG SESSION: Investigating Mode Collapse")
    print("="*60)
    print(f"\nModel: {model}")
    print(f"Data: {data_path}")
    print(f"Debug batch size: {batch_size}")

    # Load sample data
    samples = load_sample_data(data_path, n_samples=batch_size)
    print(f"Loaded {len(samples)} samples for debugging")

    # BREAKPOINT 1: Renderer selection
    renderer, tokenizer = debug_renderer_selection(model)

    # BREAKPOINT 2: Token rendering (on first sample)
    tokens, weights = debug_token_rendering(samples[0], renderer, tokenizer)

    # BREAKPOINT 3: Datum construction
    datum = debug_datum_construction(tokens, weights, tokenizer)

    # Create batch of data using CORRECT cookbook method
    from tinker_cookbook.supervised.common import datum_from_tokens_weights

    batch_data = []
    for sample in samples:
        model_input, weights = renderer.build_supervised_example(sample["messages"])
        d = datum_from_tokens_weights(model_input, weights, max_length=None)
        batch_data.append(d)

    print(f"\nCreated batch using cookbook's datum_from_tokens_weights (correct shifting)")

    # Create training client
    print("\n" + "="*60)
    print("Creating LoRA training client...")
    print("="*60)
    service_client = tinker.ServiceClient()
    training_client = service_client.create_lora_training_client(
        base_model=model,
        rank=32,
        train_mlp=True,
        train_attn=True,
        train_unembed=True,
    )

    # BREAKPOINT 4: Forward/backward
    fwd_bwd_result = debug_forward_backward(training_client, batch_data, batch_size)

    # BREAKPOINT 5: Compare fresh LoRA vs base
    # Create new batch for fair comparison
    debug_fresh_lora_vs_base(model, batch_data)

    print("\n" + "="*60)
    print("DEBUG SESSION COMPLETE")
    print("="*60)
    print("\nKey things to check at each breakpoint:")
    print("  1. Is renderer 'llama3' for Llama-3.1-8B-Instruct? (not 'role_colon')")
    print("  2. Are weights 0 for system/user, >0 for assistant?")
    print("  3. Does our Datum match cookbook helper?")
    print("  4. What's the per-token loss at step 0?")
    print("  5. Why is fresh LoRA loss so high?")


if __name__ == "__main__":
    main()
