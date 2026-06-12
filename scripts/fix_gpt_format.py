#!/usr/bin/env python3
"""Fix GPT dataset format by extracting thinking/content from channel tags."""

import json
import re
from pathlib import Path
from typing import Optional, Tuple


def extract_thinking_and_content(text: str) -> Tuple[Optional[str], str]:
    """Extract thinking and content from channel-tagged text.

    Input format:
    <|channel|>analysis<|message|>...<|end|><|start|>assistant<|channel|>final<|message|>...

    Output: (thinking, content)
    """
    # Pattern to extract analysis (thinking) content
    analysis_pattern = r'<\|channel\|>analysis<\|message\|>(.*?)<\|end\|>'
    analysis_match = re.search(analysis_pattern, text, re.DOTALL)

    # Pattern to extract final content
    final_pattern = r'<\|channel\|>final<\|message\|>(.*?)(?:<\|end\|>)?$'
    final_match = re.search(final_pattern, text, re.DOTALL)

    thinking = analysis_match.group(1).strip() if analysis_match else None
    content = final_match.group(1).strip() if final_match else text.strip()

    return thinking, content


def fix_message(message: dict) -> dict:
    """Fix a single message, extracting thinking if present."""
    if message.get("role") != "assistant":
        return message

    content = message.get("content", "")

    # Check if it has the channel tags
    if "<|channel|>" not in content:
        return message

    thinking, final_content = extract_thinking_and_content(content)

    new_message = {"role": "assistant", "content": final_content}
    if thinking:
        new_message["thinking"] = thinking

    return new_message


def fix_dataset(input_path: Path, output_path: Path) -> None:
    """Fix all messages in a dataset."""
    fixed_count = 0
    total_count = 0

    with open(input_path) as f_in, open(output_path, "w") as f_out:
        for line in f_in:
            item = json.loads(line)
            total_count += 1

            messages = item.get("messages", [])
            new_messages = []

            for msg in messages:
                new_msg = fix_message(msg)
                if new_msg != msg:
                    fixed_count += 1
                new_messages.append(new_msg)

            item["messages"] = new_messages
            f_out.write(json.dumps(item) + "\n")

    print(f"Processed {total_count} samples, fixed {fixed_count} messages")
    print(f"Saved to {output_path}")


def main():
    base = Path("/Users/work/consistency-training-methods/dataset_dumps")

    datasets = [
        (base / "train_seed_42/gpt-oss-120b/gpt-train-mixed.jsonl",
         base / "train_seed_42/gpt-oss-120b/gpt-bct-train.jsonl"),
        (base / "control_seed_42/gpt-oss-120b/gpt-control-mixed.jsonl",
         base / "control_seed_42/gpt-oss-120b/gpt-bct-control.jsonl"),
    ]

    for input_path, output_path in datasets:
        if input_path.exists():
            print(f"\nProcessing {input_path.name}...")
            fix_dataset(input_path, output_path)
        else:
            print(f"File not found: {input_path}")


if __name__ == "__main__":
    main()
