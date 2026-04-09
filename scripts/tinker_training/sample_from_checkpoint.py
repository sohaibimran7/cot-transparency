"""
Sample from trained checkpoints to verify they work.

Usage:
    # Sample from a single checkpoint
    python scripts/tinker_training/sample_from_checkpoint.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --checkpoint "tinker://run-id/weights/checkpoint-name" \
        --data dataset_dumps/control_seed_42/control_cot.jsonl \
        --n-samples 2

    # Compare base model vs checkpoint
    python scripts/tinker_training/sample_from_checkpoint.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --checkpoint "tinker://run-id/weights/checkpoint-name" \
        --compare-base \
        --data dataset_dumps/control_seed_42/control_cot.jsonl

    # Sample from multiple checkpoints
    python scripts/tinker_training/sample_from_checkpoint.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --checkpoint "tinker://run1/weights/ckpt1" "tinker://run2/weights/ckpt2" \
        --data dataset_dumps/control_seed_42/control_cot.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path to allow direct imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import directly from modules to avoid broken __init__.py chains
# that try to import from old openai package
import importlib.util

def _import_module_directly(module_path: str, module_name: str):
    """Import a module directly from file path, bypassing __init__.py."""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Import data_models.messages first (no broken deps)
from cot_transparency.data_models.messages import StrictChatMessage, StrictMessageRole

# Import inference module directly to bypass tinker/__init__.py
_inference_path = Path(__file__).parent.parent.parent / "cot_transparency" / "apis" / "tinker" / "inference.py"
_inference = _import_module_directly(str(_inference_path), "tinker_inference_direct")
TinkerSamplingClient = _inference.TinkerSamplingClient
SamplingConfig = _inference.SamplingConfig


def load_prompts(file_path: Path, n: int, offset: int = 0) -> list[list[dict]]:
    """Load prompts from a jsonl file (just the user messages)."""
    prompts = []
    with open(file_path) as f:
        for i, line in enumerate(f):
            if i < offset:
                continue
            if len(prompts) >= n:
                break
            data = json.loads(line)
            # Extract just the user messages (prompt)
            prompt_msgs = [m for m in data["messages"] if m["role"] != "assistant"]
            prompts.append(prompt_msgs)
    return prompts


def dict_to_strict_messages(msg_dicts: list[dict]) -> list[StrictChatMessage]:
    """Convert dict messages to StrictChatMessage."""
    role_map = {
        "user": StrictMessageRole.user,
        "assistant": StrictMessageRole.assistant,
        "system": StrictMessageRole.system,
    }
    return [
        StrictChatMessage(role=role_map[m["role"]], content=m["content"])
        for m in msg_dicts
    ]


def sample_from_checkpoints(
    model: str,
    checkpoints: list[str | None],  # None means base model
    prompts: list[list[dict]],
    config: SamplingConfig,
) -> dict[str, list[str]]:
    """Sample from multiple checkpoints for the same prompts."""
    results = {}

    for checkpoint in checkpoints:
        name = checkpoint if checkpoint else "base_model"
        print(f"\n{'='*60}")
        print(f"Sampling from: {name}")
        print('='*60)

        client = TinkerSamplingClient(
            model=model,
            checkpoint=checkpoint,
            config=config,
        )
        client.setup()

        completions = []
        for i, prompt_msgs in enumerate(prompts):
            messages = dict_to_strict_messages(prompt_msgs)
            result = client.sample(messages, n_samples=1)
            completion = result[0].text if result else ""
            completions.append(completion)

            # Print prompt and completion
            print(f"\n--- Prompt {i+1} ---")
            for msg in prompt_msgs:
                print(f"[{msg['role']}]: {msg['content'][:200]}...")
            print(f"\n[completion]: {completion[:500]}{'...' if len(completion) > 500 else ''}")

        results[name] = completions

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Sample from trained checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Base model name (e.g., meta-llama/Llama-3.1-8B-Instruct)",
    )
    parser.add_argument(
        "--checkpoint",
        nargs="+",
        help="Checkpoint path(s) to sample from",
    )
    parser.add_argument(
        "--compare-base",
        action="store_true",
        help="Also sample from base model for comparison",
    )
    parser.add_argument(
        "--data",
        nargs="+",
        required=True,
        help="Data file(s) to load prompts from",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=2,
        help="Number of prompts to sample per file (default: 2)",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip first N samples in each file (default: 0)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Max tokens to generate (default: 256)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0 for deterministic)",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file path for saving results (JSONL format)",
    )

    args = parser.parse_args()

    # Collect checkpoints to sample from
    checkpoints: list[str | None] = []
    if args.compare_base:
        checkpoints.append(None)  # base model
    if args.checkpoint:
        checkpoints.extend(args.checkpoint)

    if not checkpoints:
        parser.error("Must specify --checkpoint and/or --compare-base")

    # Load prompts from each file
    all_prompts = []
    for data_file in args.data:
        path = Path(data_file)
        if not path.exists():
            parser.error(f"File not found: {path}")

        prompts = load_prompts(path, args.n_samples, args.offset)
        print(f"Loaded {len(prompts)} prompts from {path.name}")
        all_prompts.extend(prompts)

    print(f"Total prompts: {len(all_prompts)}")

    # Sample config
    config = SamplingConfig(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    # Sample from all checkpoints
    results = sample_from_checkpoints(
        model=args.model,
        checkpoints=checkpoints,
        prompts=all_prompts,
        config=config,
    )

    # Save results if output specified
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for checkpoint_name, completions in results.items():
                for prompt_msgs, completion in zip(all_prompts, completions):
                    # Extract user message content as prompt string
                    prompt_text = "\n".join(
                        f"{m['content']}" for m in prompt_msgs
                    )
                    record = {
                        "model": args.model,
                        "checkpoint": checkpoint_name,
                        "prompt": prompt_text,
                        "completion": completion,
                    }
                    f.write(json.dumps(record) + "\n")
        print(f"\nSaved {len(all_prompts) * len(results)} results to {output_path}")

    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print('='*60)
    for name, completions in results.items():
        avg_len = sum(len(c) for c in completions) / len(completions) if completions else 0
        print(f"{name}: {len(completions)} completions, avg length {avg_len:.0f} chars")


if __name__ == "__main__":
    main()
