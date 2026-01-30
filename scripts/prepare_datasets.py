#!/usr/bin/env python3
"""
Prepare training datasets: load, clean, combine, shuffle, and save.

Combines the functionality of the former clean_gpt_datasets.py,
combine_logiqa_datasets.py, and create_mixed_datasets.py scripts.

Examples:
    # Clean channel tags from a dataset:
    python scripts/prepare_datasets.py \
        --input dataset_dumps/train_seed_42/gpt-oss-120b/gpt-bct-train.jsonl \
        --output dataset_dumps/train_seed_42/gpt-oss-120b/gpt-bct-train_cleaned.jsonl \
        --clean

    # Combine BCT + limited instruct samples, clean and shuffle:
    python scripts/prepare_datasets.py \
        --input dataset_dumps/train_seed_42/gpt-oss-120b/logiqa/bct_cot.jsonl \
        --input dataset_dumps/train_seed_42/gpt-oss-120b/instruct_samples.jsonl:500 \
        --output dataset_dumps/train_seed_42/gpt-oss-120b/logiqa/gpt-logiqa-train.jsonl \
        --clean --shuffle

    # Mix non_cot + half of instruct:
    python scripts/prepare_datasets.py \
        --input dataset_dumps/train_seed_42/gpt-oss-120b/bct_non_cot.jsonl \
        --input dataset_dumps/train_seed_42/gpt-oss-120b/instruct_samples.jsonl:half \
        --output dataset_dumps/train_seed_42/gpt-oss-120b/mixed.jsonl \
        --shuffle
"""

import argparse
import json
import random
import re
from pathlib import Path


def load_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    samples = []
    with open(path) as f:
        for line in f:
            samples.append(json.loads(line))
            if limit and len(samples) >= limit:
                break
    return samples


def save_jsonl(samples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")


def clean_channel_tags(sample: dict) -> dict | None:
    """Extract thinking/content from channel-tagged assistant messages.

    Returns None if unparseable channel tags remain after extraction.
    """
    analysis_re = re.compile(r'<\|channel\|>analysis<\|message\|>(.*?)<\|end\|>', re.DOTALL)
    final_re = re.compile(r'<\|channel\|>final<\|message\|>(.*?)(?:<\|end\|>)?$', re.DOTALL)
    tag_re = re.compile(r'<\|(?:channel|message|end)\|>')

    new_messages = []
    for msg in sample.get("messages", []):
        content = msg.get("content", "")

        if msg.get("role") != "assistant" or "<|channel|>" not in content:
            new_messages.append(msg)
            continue

        # Extract thinking and final content
        analysis = analysis_re.search(content)
        final = final_re.search(content)

        final_content = final.group(1).strip() if final else content.strip()

        # Reject if tags remain
        if tag_re.search(final_content):
            return None

        new_msg = {"role": "assistant", "content": final_content}
        if analysis:
            thinking = analysis.group(1).strip()
            if thinking and not tag_re.search(thinking):
                new_msg["thinking"] = thinking
        new_messages.append(new_msg)

    return {**sample, "messages": new_messages}


def parse_input_spec(spec: str) -> tuple[Path, int | None]:
    """Parse 'path/to/file.jsonl:limit' into (path, limit).

    limit can be an integer or 'half'. No suffix means no limit.
    """
    if ":" in spec:
        path_str, limit_str = spec.rsplit(":", 1)
        if limit_str == "half":
            return Path(path_str), -1  # Sentinel for half
        return Path(path_str), int(limit_str)
    return Path(spec), None


def main():
    parser = argparse.ArgumentParser(
        description="Prepare training datasets: load, clean, combine, shuffle, save.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Input limit syntax:
  --input file.jsonl          Use all samples
  --input file.jsonl:500      Use first 500 samples
  --input file.jsonl:half     Use a random half of samples
        """,
    )
    parser.add_argument(
        "--input", action="append", required=True, dest="inputs",
        help="Input JSONL file, optionally with :limit (e.g. file.jsonl:500 or file.jsonl:half)",
    )
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--clean", action="store_true", help="Extract/strip channel tags from assistant messages")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle the combined output")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffle and half-sampling (default: 42)")
    args = parser.parse_args()

    random.seed(args.seed)

    # Load all inputs
    all_samples: list[dict] = []
    for spec in args.inputs:
        path, limit = parse_input_spec(spec)
        samples = load_jsonl(path)
        if limit == -1:  # half
            random.shuffle(samples)
            samples = samples[:len(samples) // 2]
        elif limit is not None:
            samples = samples[:limit]
        print(f"Loaded {len(samples)} samples from {path}")
        all_samples.extend(samples)

    print(f"Total: {len(all_samples)} samples")

    # Clean
    if args.clean:
        cleaned = []
        for s in all_samples:
            result = clean_channel_tags(s)
            if result is not None:
                cleaned.append(result)
        removed = len(all_samples) - len(cleaned)
        if removed:
            print(f"Cleaned: kept {len(cleaned)}, removed {removed} with unparseable channel tags")
        all_samples = cleaned

    # Shuffle
    if args.shuffle:
        random.shuffle(all_samples)

    # Save
    save_jsonl(all_samples, Path(args.output))
    print(f"Saved {len(all_samples)} samples to {args.output}")


if __name__ == "__main__":
    main()
