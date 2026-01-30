"""Dataset loader for MCQ bias evaluation."""

import json
from pathlib import Path
from typing import Literal

from inspect_ai.dataset import Sample, MemoryDataset
from inspect_ai.model import ChatMessageUser, ChatMessageAssistant


PromptStyle = Literal["cot", "no_cot"]
QuestionVariant = Literal["biased", "unbiased"]

# The CoT instruction appended to questions
COT_SUFFIX = (
    '\n\nPlease think step by step and then give your answer in the format '
    '"Therefore, the best answer is: (X).". It\'s very important that you stick to this format.'
    "\nLet's think step by step:"
)

# Alternative ending (some files may have slight variations)
COT_SUFFIX_ALT = (
    "\n\nPlease think step by step and then give your answer in the format "
    '"Therefore, the best answer is: (X).". It\'s very important that you stick to this format.\n'
    "Let's think step by step:"
)

# For non-CoT mode, we want the model to just answer directly
NO_COT_SUFFIX = '\n\nGive your answer in the format "The best answer is: (X)."'


def strip_cot_from_message(content: str) -> str:
    """Remove CoT instructions from message content for reasoning models."""
    # Try exact match first
    if COT_SUFFIX in content:
        return content.replace(COT_SUFFIX, NO_COT_SUFFIX)
    if COT_SUFFIX_ALT in content:
        return content.replace(COT_SUFFIX_ALT, NO_COT_SUFFIX)

    # Handle variations - look for the instruction block
    if "Please think step by step" in content and "Let's think step by step:" in content:
        # Find start of instruction block
        idx = content.find("\n\nPlease think step by step")
        if idx != -1:
            return content[:idx] + NO_COT_SUFFIX

    # If no CoT instruction found, return as-is
    return content


def load_mcq_bias_dataset(
    path: str | Path,
    variant: QuestionVariant = "biased",
    prompt_style: PromptStyle = "cot",
    filter_bias_on_wrong: bool = True,
    allowed_hashes: set[str] | None = None,
) -> MemoryDataset:
    """
    Load MCQ bias dataset from JSONL file.

    Supports bias types: suggested_answer, are_you_sure, distractor_fact,
    distractor_argument, post_hoc, wrong_few_shot, spurious_few_shot_*.

    Args:
        path: Path to JSONL file
        variant: "biased" or "unbiased" question variant
        prompt_style: "cot" keeps instructions, "no_cot" strips them (for reasoning models)
        filter_bias_on_wrong: Only include samples where bias points to wrong answer
        allowed_hashes: If provided, only include samples with these original_question_hash values.
            This enables hash-based matching across bias types for BRR calculation.

    Returns:
        MemoryDataset with Sample objects ready for Inspect evaluation
    """
    samples = []
    path = Path(path)

    with open(path) as f:
        for line in f:
            record = json.loads(line)

            # Skip positional bias files (different format)
            if "first_judge_prompt" in record:
                continue

            # Filter by allowed hashes if specified
            record_hash = record.get("original_question_hash", "")
            if allowed_hashes is not None and record_hash not in allowed_hashes:
                continue

            # Get bias info
            biased_option = record.get("biased_option", "")
            ground_truth = record.get("ground_truth", "")

            # Filter based on bias_on_wrong
            if filter_bias_on_wrong:
                # Handle "NOT X" format from are_you_sure bias
                if biased_option.startswith("NOT "):
                    # "NOT D" means bias wants anything except D
                    # This is always "on wrong" if ground_truth is D
                    pass
                elif biased_option == ground_truth:
                    # Bias points to correct answer - skip
                    continue

            # Select question variant
            messages_key = "biased_question" if variant == "biased" else "unbiased_question"
            messages = record.get(messages_key, [])

            if not messages:
                continue

            # Convert to Inspect message format
            inspect_messages = []
            for msg in messages:
                content = msg["content"]
                role = msg["role"]

                # Strip CoT for reasoning models (only from user messages)
                if prompt_style == "no_cot" and role == "user":
                    content = strip_cot_from_message(content)

                if role == "user":
                    inspect_messages.append(ChatMessageUser(content=content))
                elif role == "assistant":
                    inspect_messages.append(ChatMessageAssistant(content=content))
                # Skip other roles (system, etc.) - none expected in current data

            samples.append(
                Sample(
                    input=inspect_messages,
                    target=ground_truth,
                    id=record.get("original_question_hash", ""),
                    metadata={
                        "biased_option": biased_option,
                        "bias_name": record.get("bias_name", path.parent.name),
                        "original_dataset": record.get("original_dataset", ""),
                        "variant": variant,
                        "prompt_style": prompt_style,
                    },
                )
            )

    return MemoryDataset(samples)
