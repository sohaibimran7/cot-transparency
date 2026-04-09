"""
Generate base control and BCT prompts from a dataset.

This creates the prompt files that are then used by generate_bct_data.py
to sample completions from a target model.

Output structure:
    dataset_dumps/control_seed_{seed}/{source_name}/control_cot.jsonl
    dataset_dumps/train_seed_{seed}/{source_name}/bct_cot.jsonl

Usage:
    # Generate prompts from LogiQA train data:
    python scripts/generate_base_prompts.py --task logiqa_train --seed 42

    # Generate prompts from a specific dataset with custom output:
    python scripts/generate_base_prompts.py --task logiqa_train --seed 42 --source-name logiqa
"""

import argparse
import json
from pathlib import Path
from typing import Any

from slist import Slist

from cot_transparency.data_models.data import get_list_of_examples, TASK_LIST
from cot_transparency.data_models.example_base import DataExampleBase
from cot_transparency.formatters.core.unbiased import ZeroShotCOTUnbiasedFormatter, ZeroShotUnbiasedFormatter
from cot_transparency.formatters.more_biases.random_bias_formatter import RandomBiasedFormatter, RandomBiasedNoCOTFormatter

MessageDict = dict[str, str]
SampleDict = dict[str, Any]


def save_jsonl(records: list[SampleDict], file_path: Path) -> None:
    """Save records to a jsonl file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    print(f"Saved {len(records)} samples to {file_path}")


def format_messages_to_dict(messages: list) -> list[MessageDict]:
    """Convert ChatMessage objects to dict format."""
    return [{"role": m.role, "content": m.content} for m in messages]


def create_prompt_sample(messages: list[MessageDict]) -> SampleDict:
    """Create a prompt-only sample (no assistant completion)."""
    user_messages = [m for m in messages if m["role"] not in ("assistant_if_completion", "assistant_preferred", "assistant")]
    return {"messages": user_messages}


def generate_base_prompts(
    task: str,
    seed: int = 42,
    base_dir: str = "dataset_dumps",
    source_name: str | None = None,
    example_cap: int | None = None,
    model: str = "gpt-4",
    cot: bool = True,
    non_cot: bool = False,
) -> None:
    """Generate base control and BCT prompt files from a dataset."""
    print(f"Loading data for task: {task}")
    data: Slist[DataExampleBase] = get_list_of_examples(task)
    data = data.shuffle(str(seed))

    if example_cap:
        data = data.take(example_cap)

    print(f"Loaded {len(data)} examples")

    source = source_name or task.replace("_train", "").replace("_test", "")
    control_dir = Path(base_dir) / f"control_seed_{seed}" / source
    train_dir = Path(base_dir) / f"train_seed_{seed}" / source

    if cot:
        print("\nGenerating CoT prompts...")
        control_samples = []
        bct_samples = []

        for example in data:
            unbiased_msgs = ZeroShotCOTUnbiasedFormatter.format_example(question=example, model=model)
            control_samples.append(create_prompt_sample(format_messages_to_dict(unbiased_msgs)))

            biased_msgs = RandomBiasedFormatter.format_example(question=example, model=model)
            bct_samples.append(create_prompt_sample(format_messages_to_dict(biased_msgs)))

        save_jsonl(control_samples, control_dir / "control_cot.jsonl")
        save_jsonl(bct_samples, train_dir / "bct_cot.jsonl")

    if non_cot:
        print("\nGenerating non-CoT prompts...")
        control_samples = []
        bct_samples = []

        for example in data:
            unbiased_msgs = ZeroShotUnbiasedFormatter.format_example(question=example, model=model)
            control_samples.append(create_prompt_sample(format_messages_to_dict(unbiased_msgs)))

            biased_msgs = RandomBiasedNoCOTFormatter.format_example(question=example, model=model)
            bct_samples.append(create_prompt_sample(format_messages_to_dict(biased_msgs)))

        save_jsonl(control_samples, control_dir / "control_non_cot.jsonl")
        save_jsonl(bct_samples, train_dir / "bct_non_cot.jsonl")

    print("\nDone!")
    print(f"Control prompts: {control_dir}")
    print(f"BCT prompts: {train_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate base control and BCT prompts from a dataset")
    parser.add_argument("--task", required=True, help="Task name (e.g., logiqa_train)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    parser.add_argument("--base-dir", default="dataset_dumps", help="Base directory for output")
    parser.add_argument("--source-name", default=None, help="Name for output directory")
    parser.add_argument("--example-cap", type=int, default=None, help="Limit number of examples")
    parser.add_argument("--cot", action="store_true", default=True, help="Generate CoT prompts")
    parser.add_argument("--no-cot", action="store_true", help="Skip CoT prompts")
    parser.add_argument("--non-cot", action="store_true", help="Generate non-CoT prompts")
    parser.add_argument("--list-tasks", action="store_true", help="List available tasks and exit")
    args = parser.parse_args()

    if args.list_tasks:
        print("Available task collections:")
        for name, tasks in TASK_LIST.items():
            print(f"  {name}: {tasks[:3]}..." if len(tasks) > 3 else f"  {name}: {tasks}")
        return

    generate_base_prompts(
        task=args.task,
        seed=args.seed,
        base_dir=args.base_dir,
        source_name=args.source_name,
        example_cap=args.example_cap,
        cot=not args.no_cot,
        non_cot=args.non_cot,
    )


if __name__ == "__main__":
    main()
