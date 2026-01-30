"""
Generate BCT training data for new models using Tinker API.

Takes control prompts (unbiased), samples completions from the target model,
then pairs those completions with the biased prompts to create BCT training data.

Output structure:
    dataset_dumps/control_seed_42/{model_name}/control_cot.jsonl
    dataset_dumps/control_seed_42/{model_name}/control_non_cot.jsonl
    dataset_dumps/train_seed_42/{model_name}/bct_cot.jsonl
    dataset_dumps/train_seed_42/{model_name}/bct_non_cot.jsonl

Usage:
    # List available models:
    python scripts/tinker_training/generate_bct_data.py --list-models

    # For Llama 3 8B (both CoT and non-CoT):
    python scripts/tinker_training/generate_bct_data.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --cot --non-cot

    # For GPT OSS 120B (non-CoT only, since it's a reasoning model):
    python scripts/tinker_training/generate_bct_data.py \
        --model gpt-oss-120b \
        --non-cot
"""

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import tinker
from tqdm import tqdm

from cot_transparency.apis.tinker.inference import TinkerSamplingClient, SamplingConfig

# Type alias for message dicts
MessageDict = dict[str, str]
SampleDict = dict[str, Any]


def list_available_models() -> list[str]:
    """Query Tinker API for available models."""
    service_client = tinker.ServiceClient()
    capabilities = service_client.get_server_capabilities()
    return [
        model.model_name
        for model in capabilities.supported_models
        if model.model_name is not None
    ]


def print_available_models() -> None:
    """Print available models from Tinker API."""
    print("Querying Tinker API for available models...")
    print()
    try:
        models = list_available_models()
        print("Available models:")
        print("-" * 50)
        for model in sorted(models):
            print(f"  {model}")
        print("-" * 50)
        print(f"Total: {len(models)} models")
    except Exception as e:
        print(f"Error querying models: {e}")
        print()
        print("Common models (may not all be available):")
        print("  meta-llama/Llama-3.1-8B-Instruct")
        print("  meta-llama/Llama-3.1-70B-Instruct")
        print("  Qwen/Qwen3-8B")
        print("  Qwen/Qwen3-30B-A3B")


def sanitize_model_name(model: str) -> str:
    """Convert model name to filesystem-safe directory name."""
    # e.g., "meta-llama/Llama-3.1-8B-Instruct" -> "llama-3.1-8b-instruct"
    name = model.split("/")[-1].lower()
    return name.replace(".", "-")


def load_jsonl(file_path: Path) -> list[SampleDict]:
    """Load all records from a jsonl file."""
    records: list[SampleDict] = []
    with open(file_path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def save_jsonl(records: list[SampleDict], file_path: Path) -> None:
    """Save records to a jsonl file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    print(f"Saved {len(records)} samples to {file_path}")


def extract_prompt_messages(sample: SampleDict) -> list[MessageDict]:
    """Extract only the user message(s) from a sample (exclude assistant completion)."""
    return [m for m in sample["messages"] if m["role"] != "assistant"]


def sample_batch(
    client: TinkerSamplingClient,
    prompts: list[list[MessageDict]],
) -> list[str]:
    """Sample completions for a single batch of prompts concurrently."""
    from tinker import types

    completions: list[str] = []

    # Submit all requests in batch (non-blocking)
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
    """Save completions to checkpoint file (append mode for incremental saves)."""
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if start_idx > 0 else "w"
    with open(checkpoint_file, mode) as f:
        for completion in completions[start_idx:]:
            f.write(json.dumps({"completion": completion}) + "\n")


def create_sample_with_completion(prompt_msgs: list[MessageDict], completion: str) -> SampleDict:
    """Create a training sample by combining prompt messages with completion."""
    return {
        "messages": prompt_msgs + [{"role": "assistant", "content": completion}]
    }


def generate_samples(
    client: TinkerSamplingClient,
    control_file: Path,
    bct_file: Path,
    control_output: Path,
    bct_output: Path,
    limit: Optional[int] = None,
    batch_size: int = 10,
    save_every: int = 10,
    fresh: bool = False,
) -> None:
    """
    Generate control and BCT samples for a given file pair with incremental saving.

    Args:
        client: Tinker sampling client
        control_file: Path to control samples (unbiased prompts + GPT completions)
        bct_file: Path to BCT samples (biased prompts + GPT completions)
        control_output: Where to save new control samples
        bct_output: Where to save new BCT samples
        limit: Optional limit on number of samples to process
        batch_size: Number of concurrent requests per batch
        save_every: Save checkpoint every N batches
    """
    import time

    # Load original samples
    control_samples = load_jsonl(control_file)
    bct_samples = load_jsonl(bct_file)

    if limit:
        control_samples = control_samples[:limit]
        bct_samples = bct_samples[:limit]

    assert len(control_samples) == len(bct_samples), (
        f"Mismatch: {len(control_samples)} control vs {len(bct_samples)} BCT samples"
    )

    # Extract prompts (user messages only)
    control_prompts = [extract_prompt_messages(s) for s in control_samples]
    bct_prompts = [extract_prompt_messages(s) for s in bct_samples]

    # Checkpoint file for resuming
    checkpoint_file = control_output.parent / f".checkpoint_{control_output.name}"

    # Load existing progress (or start fresh)
    if fresh and checkpoint_file.exists():
        checkpoint_file.unlink()
        print("Starting fresh - deleted existing checkpoint")
    start_idx, completions = load_checkpoint(checkpoint_file)

    total_prompts = len(control_prompts)
    total_batches = (total_prompts + batch_size - 1) // batch_size
    start_batch = start_idx // batch_size

    print(f"Sampling {total_prompts} completions (batch_size={batch_size}, save_every={save_every})")
    if start_idx > 0:
        print(f"Resuming from sample {start_idx} (batch {start_batch})")

    batch_times: list[float] = []
    last_save_idx = start_idx

    # Process in batches
    pbar = tqdm(range(start_batch, total_batches), desc="Generating", initial=start_batch, total=total_batches)
    for batch_idx in pbar:
        batch_start_time = time.time()
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, total_prompts)
        batch_prompts = control_prompts[batch_start:batch_end]

        # Sample this batch
        batch_completions = sample_batch(client, batch_prompts)
        completions.extend(batch_completions)

        batch_elapsed = time.time() - batch_start_time
        batch_times.append(batch_elapsed)

        # Update progress bar
        avg_time = sum(batch_times[-10:]) / min(len(batch_times), 10)
        pbar.set_postfix({"batch_time": f"{batch_elapsed:.1f}s", "avg": f"{avg_time:.1f}s"})

        # Save checkpoint periodically
        if (batch_idx + 1) % save_every == 0 or batch_idx == total_batches - 1:
            save_checkpoint(checkpoint_file, completions, last_save_idx)
            last_save_idx = len(completions)
            pbar.write(f"💾 Checkpoint saved: {len(completions)}/{total_prompts} samples")

    # Create final samples
    new_control_samples = [
        create_sample_with_completion(prompt, completion)
        for prompt, completion in zip(control_prompts, completions)
    ]
    new_bct_samples = [
        create_sample_with_completion(prompt, completion)
        for prompt, completion in zip(bct_prompts, completions)
    ]

    # Save final outputs
    save_jsonl(new_control_samples, control_output)
    save_jsonl(new_bct_samples, bct_output)

    # Clean up checkpoint
    if checkpoint_file.exists():
        checkpoint_file.unlink()

    # Summary
    if batch_times:
        avg_time = sum(batch_times) / len(batch_times)
        print(f"Average batch time: {avg_time:.2f}s")


def generate_instruct_samples(
    client: TinkerSamplingClient,
    instruct_file: Path,
    control_output: Path,
    train_output: Path,
    limit: Optional[int] = None,
    batch_size: int = 10,
    save_every: int = 10,
    fresh: bool = False,
) -> None:
    """
    Generate instruction-following samples by sampling from the model with incremental saving.

    Args:
        client: Tinker sampling client
        instruct_file: Path to original instruction samples
        control_output: Where to save control instruction samples
        train_output: Where to save train instruction samples
        limit: Optional limit on number of samples to process
        batch_size: Number of concurrent requests per batch
        save_every: Save checkpoint every N batches
    """
    import time

    # Load original samples
    instruct_samples = load_jsonl(instruct_file)

    if limit:
        instruct_samples = instruct_samples[:limit]

    # Extract prompts (user messages only)
    prompts = [extract_prompt_messages(s) for s in instruct_samples]

    # Checkpoint file for resuming
    checkpoint_file = control_output.parent / f".checkpoint_{control_output.name}"

    # Load existing progress (or start fresh)
    if fresh and checkpoint_file.exists():
        checkpoint_file.unlink()
        print("Starting fresh - deleted existing checkpoint")
    start_idx, completions = load_checkpoint(checkpoint_file)

    total_prompts = len(prompts)
    total_batches = (total_prompts + batch_size - 1) // batch_size
    start_batch = start_idx // batch_size

    print(f"Sampling {total_prompts} instruction completions (batch_size={batch_size}, save_every={save_every})")
    if start_idx > 0:
        print(f"Resuming from sample {start_idx} (batch {start_batch})")

    batch_times: list[float] = []
    last_save_idx = start_idx

    # Process in batches
    pbar = tqdm(range(start_batch, total_batches), desc="Instruct", initial=start_batch, total=total_batches)
    for batch_idx in pbar:
        batch_start_time = time.time()
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, total_prompts)
        batch_prompts = prompts[batch_start:batch_end]

        # Sample this batch
        batch_completions = sample_batch(client, batch_prompts)
        completions.extend(batch_completions)

        batch_elapsed = time.time() - batch_start_time
        batch_times.append(batch_elapsed)

        # Update progress bar
        avg_time = sum(batch_times[-10:]) / min(len(batch_times), 10)
        pbar.set_postfix({"batch_time": f"{batch_elapsed:.1f}s", "avg": f"{avg_time:.1f}s"})

        # Save checkpoint periodically
        if (batch_idx + 1) % save_every == 0 or batch_idx == total_batches - 1:
            save_checkpoint(checkpoint_file, completions, last_save_idx)
            last_save_idx = len(completions)
            pbar.write(f"💾 Checkpoint saved: {len(completions)}/{total_prompts} samples")

    # Create new samples with model completions
    new_samples = [
        create_sample_with_completion(prompt, completion)
        for prompt, completion in zip(prompts, completions)
    ]

    # Save to both control and train directories
    save_jsonl(new_samples, control_output)
    save_jsonl(new_samples, train_output)

    # Clean up checkpoint
    if checkpoint_file.exists():
        checkpoint_file.unlink()

    # Summary
    if batch_times:
        avg_time = sum(batch_times) / len(batch_times)
        print(f"Average batch time: {avg_time:.2f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Generate BCT training data for new models using Tinker API"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models from Tinker API and exit"
    )
    parser.add_argument(
        "--model",
        help="Model name (e.g., meta-llama/Llama-3.1-8B-Instruct, gpt-oss-120b)"
    )
    parser.add_argument(
        "--cot",
        action="store_true",
        help="Generate CoT samples"
    )
    parser.add_argument(
        "--non-cot",
        action="store_true",
        help="Generate non-CoT samples"
    )
    parser.add_argument(
        "--instruct",
        action="store_true",
        help="Generate instruction-following samples"
    )
    parser.add_argument(
        "--instruct-limit",
        type=int,
        default=None,
        help="Separate limit for instruction samples (uses --limit if not specified)"
    )
    parser.add_argument(
        "--base-dir",
        default="dataset_dumps",
        help="Base directory for dataset dumps"
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Source subdirectory for input files (e.g., 'logiqa' to read from control_seed_42/logiqa/)"
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
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples to process (for testing)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of concurrent requests per batch (default: 10)"
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Save checkpoint every N batches (default: 10)"
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional checkpoint to load for the model"
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Start fresh, ignoring any existing checkpoint files"
    )
    args = parser.parse_args()

    # Handle --list-models
    if args.list_models:
        print_available_models()
        return

    # Validate required args for generation
    if not args.model:
        parser.error("--model is required unless using --list-models")

    if not args.cot and not args.non_cot and not args.instruct:
        parser.error("At least one of --cot, --non-cot, or --instruct must be specified")

    # Setup paths
    base_dir = Path(args.base_dir)
    control_dir = base_dir / f"control_seed_{args.seed}"
    train_dir = base_dir / f"train_seed_{args.seed}"

    # Source subdirectory for input files (e.g., 'logiqa')
    if args.source:
        source_control_dir = control_dir / args.source
        source_train_dir = train_dir / args.source
    else:
        source_control_dir = control_dir
        source_train_dir = train_dir

    model_name = sanitize_model_name(args.model)
    # Output to source subdirectory within model dir when --source is specified
    if args.source:
        control_output_dir = control_dir / model_name / args.source
        train_output_dir = train_dir / model_name / args.source
    else:
        control_output_dir = control_dir / model_name
        train_output_dir = train_dir / model_name

    print(f"Model: {args.model}")
    print(f"Model dir name: {model_name}")
    if args.source:
        print(f"Source: {args.source}")
    print(f"Input control dir: {source_control_dir}")
    print(f"Input train dir: {source_train_dir}")
    print(f"Output control dir: {control_output_dir}")
    print(f"Output train dir: {train_output_dir}")
    print(f"Max tokens: {args.max_tokens}, Temperature: {args.temperature}")
    if args.limit:
        print(f"Limit: {args.limit} samples")
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

    # Generate CoT samples
    if args.cot:
        print("=" * 60)
        print("Generating CoT samples...")
        print("=" * 60)
        generate_samples(
            client=client,
            control_file=source_control_dir / "control_cot.jsonl",
            bct_file=source_train_dir / "bct_cot.jsonl",
            control_output=control_output_dir / "control_cot.jsonl",
            bct_output=train_output_dir / "bct_cot.jsonl",
            limit=args.limit,
            batch_size=args.batch_size,
            save_every=args.save_every,
            fresh=args.fresh,
        )
        print()

    # Generate non-CoT samples
    if args.non_cot:
        print("=" * 60)
        print("Generating non-CoT samples...")
        print("=" * 60)
        generate_samples(
            client=client,
            control_file=source_control_dir / "control_non_cot.jsonl",
            bct_file=source_train_dir / "bct_non_cot.jsonl",
            control_output=control_output_dir / "control_non_cot.jsonl",
            bct_output=train_output_dir / "bct_non_cot.jsonl",
            limit=args.limit,
            batch_size=args.batch_size,
            save_every=args.save_every,
            fresh=args.fresh,
        )
        print()

    # Generate instruction samples
    if args.instruct:
        print("=" * 60)
        print("Generating instruction samples...")
        print("=" * 60)
        instruct_limit = args.instruct_limit if args.instruct_limit is not None else args.limit
        generate_instruct_samples(
            client=client,
            instruct_file=source_control_dir / "instruct_samples.jsonl",
            control_output=control_output_dir / "instruct_samples.jsonl",
            train_output=train_output_dir / "instruct_samples.jsonl",
            limit=instruct_limit,
            batch_size=args.batch_size,
            save_every=args.save_every,
            fresh=args.fresh,
        )
        print()

    print("Done!")
    print(f"Control samples saved to: {control_output_dir}")
    print(f"BCT train samples saved to: {train_output_dir}")


if __name__ == "__main__":
    main()
