"""
Generate BCT training data from test bias files using Tinker API.

Takes test files with unbiased/biased question pairs, samples completions
from the target model on unbiased questions, then pairs those completions
with biased questions to create BCT training data.

Usage:
    # Generate BCT data from truthfulqa and mmlu suggested_answer:
    python scripts/tinker_training/generate_bct_from_test.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --datasets truthfulqa mmlu \
        --bias suggested_answer \
        --limits 817 1183 \
        --output-name llama-suggested-answer
"""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")

import tinker  # noqa: E402
from tinker import types  # noqa: E402
from tqdm import tqdm  # noqa: E402

from cot_transparency.apis.tinker.inference import TinkerSamplingClient, SamplingConfig  # noqa: E402

MessageDict = dict[str, str]
SampleDict = dict[str, Any]


def load_jsonl(file_path: Path, limit: int | None = None) -> list[SampleDict]:
    """Load samples from a jsonl file."""
    samples = []
    with open(file_path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            samples.append(json.loads(line))
    return samples


def save_jsonl(samples: list[SampleDict], file_path: Path) -> None:
    """Save samples to a jsonl file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    print(f"Saved {len(samples)} samples to {file_path}")


def sample_batch(
    client: TinkerSamplingClient,
    prompts: list[list[MessageDict]],
) -> list[str]:
    """Sample completions for a batch of prompts concurrently."""
    completions: list[str] = []

    # Submit all requests in batch
    futures = []
    for prompt_msgs in prompts:
        msg_dicts = [{"role": m["role"], "content": m["content"]} for m in prompt_msgs]
        prompt_input = client.renderer.build_generation_prompt(msg_dicts)
        stop_sequences = client.renderer.get_stop_sequences()

        sampling_params = types.SamplingParams(
            max_tokens=client.config.max_tokens,
            temperature=client.config.temperature,
            top_p=client.config.top_p,
            stop=stop_sequences,
        )

        future = client.sampling_client.sample(
            prompt=prompt_input,
            sampling_params=sampling_params,
            num_samples=1,
        )
        futures.append(future)

    # Collect results
    for future in futures:
        try:
            result = future.result()
            if result.sequences:
                tokens = list(result.sequences[0].tokens)
                parsed_msg, _ = client.renderer.parse_response(tokens)
                text = parsed_msg.get("content", "") if parsed_msg else client.tokenizer.decode(tokens)
                completions.append(text)
            else:
                completions.append("")
        except Exception as e:
            error_str = str(e).lower()
            if "rate" in error_str or "limit" in error_str or "429" in error_str:
                print(f"\n⚠️  RATE LIMIT: {e}")
            else:
                print(f"\nError: {e}")
            completions.append("")

    return completions


def load_checkpoint(checkpoint_file: Path) -> tuple[int, list[str]]:
    """Load checkpoint if exists. Returns (start_index, completions_so_far)."""
    if not checkpoint_file.exists():
        return 0, []

    completions = []
    with open(checkpoint_file) as f:
        for line in f:
            data = json.loads(line)
            completions.append(data.get("completion", ""))

    print(f"Resuming from checkpoint: {len(completions)} completions already done")
    return len(completions), completions


def save_checkpoint(checkpoint_file: Path, completions: list[str], start_idx: int = 0) -> None:
    """Save completions to checkpoint file."""
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if start_idx > 0 else "w"
    with open(checkpoint_file, mode) as f:
        for completion in completions[start_idx:]:
            f.write(json.dumps({"completion": completion}) + "\n")


def sanitize_model_name(model: str) -> str:
    """Convert model name to filesystem-safe directory name."""
    name = model.split("/")[-1].lower()
    return name.replace(".", "-")


def main():
    parser = argparse.ArgumentParser(
        description="Generate BCT training data from test bias files"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name (e.g., meta-llama/Llama-3.1-8B-Instruct)"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="Dataset names (e.g., truthfulqa mmlu)"
    )
    parser.add_argument(
        "--bias",
        required=True,
        help="Bias type (e.g., suggested_answer, distractor_argument)"
    )
    parser.add_argument(
        "--limits",
        nargs="+",
        type=int,
        default=None,
        help="Limits per dataset (in same order as --datasets)"
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Output directory name under model dir (used with --base-dir)"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Flat output directory for both control and bct files"
    )
    parser.add_argument(
        "--base-dir",
        default="dataset_dumps",
        help="Base directory for dataset dumps"
    )
    parser.add_argument(
        "--seed",
        default="42",
        help="Seed identifier for the dataset (default: 42)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 for deterministic)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Number of concurrent requests per batch"
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Save checkpoint every N batches"
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Start fresh, ignoring any existing checkpoint"
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional model checkpoint to load"
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=42,
        help="Seed for shuffling combined samples"
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    test_dir = base_dir / "test" / args.bias

    # Validate limits
    if args.limits and len(args.limits) != len(args.datasets):
        parser.error(f"--limits must have same length as --datasets ({len(args.datasets)})")

    # Load samples from each dataset
    all_samples = []
    for i, dataset in enumerate(args.datasets):
        file_path = test_dir / f"{dataset}_{args.bias}.jsonl"
        limit = args.limits[i] if args.limits else None

        print(f"Loading {dataset} from {file_path}")
        samples = load_jsonl(file_path, limit=limit)
        print(f"  Loaded {len(samples)} samples")
        all_samples.extend(samples)

    print(f"\nTotal samples: {len(all_samples)}")

    # Shuffle combined samples
    random.seed(args.shuffle_seed)
    random.shuffle(all_samples)
    print(f"Shuffled with seed {args.shuffle_seed}")

    # Extract unbiased and biased prompts
    unbiased_prompts = [s["unbiased_question"] for s in all_samples]
    biased_prompts = [s["biased_question"] for s in all_samples]

    # Setup output paths
    model_name = sanitize_model_name(args.model)
    if args.output_dir:
        # Single directory for both control and bct files
        output_dir = Path(args.output_dir)
        control_output_dir = output_dir
        train_output_dir = output_dir
    else:
        control_dir = base_dir / f"control_seed_{args.seed}"
        train_dir = base_dir / f"train_seed_{args.seed}"
        control_output_dir = control_dir / model_name / args.output_name
        train_output_dir = train_dir / model_name / args.output_name

    control_output_dir.mkdir(parents=True, exist_ok=True)
    train_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nModel: {args.model}")
    print(f"Output dirs:")
    print(f"  Control: {control_output_dir}")
    print(f"  Train: {train_output_dir}")
    print()

    # Setup Tinker client
    client = TinkerSamplingClient(
        model=args.model,
        checkpoint=args.checkpoint,
        config=SamplingConfig(
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
    )
    print("Initializing Tinker client...")
    client.setup()
    print()

    # Checkpoint for resuming
    checkpoint_file = control_output_dir / ".checkpoint_completions.jsonl"

    if args.fresh and checkpoint_file.exists():
        checkpoint_file.unlink()
        print("Starting fresh - deleted existing checkpoint")

    start_idx, completions = load_checkpoint(checkpoint_file)

    total_prompts = len(unbiased_prompts)
    total_batches = (total_prompts + args.batch_size - 1) // args.batch_size
    start_batch = start_idx // args.batch_size

    print(f"Sampling {total_prompts} completions (batch_size={args.batch_size})")
    if start_idx > 0:
        print(f"Resuming from sample {start_idx} (batch {start_batch})")

    batch_times: list[float] = []
    last_save_idx = start_idx

    # Process in batches
    pbar = tqdm(range(start_batch, total_batches), desc="Generating", initial=start_batch, total=total_batches)
    for batch_idx in pbar:
        batch_start_time = time.time()
        batch_start = batch_idx * args.batch_size
        batch_end = min(batch_start + args.batch_size, total_prompts)
        batch_prompts = unbiased_prompts[batch_start:batch_end]

        # Sample this batch
        batch_completions = sample_batch(client, batch_prompts)
        completions.extend(batch_completions)

        batch_elapsed = time.time() - batch_start_time
        batch_times.append(batch_elapsed)

        # Update progress bar
        avg_time = sum(batch_times[-10:]) / min(len(batch_times), 10)
        pbar.set_postfix({"batch_time": f"{batch_elapsed:.1f}s", "avg": f"{avg_time:.1f}s"})

        # Save checkpoint periodically
        if (batch_idx + 1) % args.save_every == 0 or batch_idx == total_batches - 1:
            save_checkpoint(checkpoint_file, completions, last_save_idx)
            last_save_idx = len(completions)
            pbar.write(f"💾 Checkpoint saved: {len(completions)}/{total_prompts} samples")

    # Create final samples
    # Control: unbiased prompt + completion
    control_samples = [
        {"messages": prompt + [{"role": "assistant", "content": completion}]}
        for prompt, completion in zip(unbiased_prompts, completions)
    ]

    # BCT: biased prompt + completion (from unbiased)
    bct_samples = [
        {"messages": prompt + [{"role": "assistant", "content": completion}]}
        for prompt, completion in zip(biased_prompts, completions)
    ]

    # Save outputs
    save_jsonl(control_samples, control_output_dir / "control_cot.jsonl")
    save_jsonl(bct_samples, train_output_dir / "bct_cot.jsonl")

    # Clean up checkpoint
    if checkpoint_file.exists():
        checkpoint_file.unlink()

    # Summary
    if batch_times:
        avg_time = sum(batch_times) / len(batch_times)
        total_time = sum(batch_times)
        print(f"\nAverage batch time: {avg_time:.2f}s")
        print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")

    print(f"\nDone!")
    print(f"Control samples: {control_output_dir / 'control_cot.jsonl'}")
    print(f"BCT samples: {train_output_dir / 'bct_cot.jsonl'}")


if __name__ == "__main__":
    main()
