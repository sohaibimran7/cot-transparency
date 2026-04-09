"""
Humanity's Last Exam (HLE) dataset loader for MCQ questions.

Dataset: https://huggingface.co/datasets/cais/hle
Paper: https://arxiv.org/abs/2501.14249

HLE contains ~2,500 expert-level questions, with ~20% being MCQ format.
This module filters to only MCQ questions (text-only, no images).

Note: HLE is only used for testing, not training.
"""

import re
from functools import lru_cache
from string import ascii_uppercase
from typing import Optional

from datasets import load_dataset
from slist import Slist

from cot_transparency.data_models.example_base import (
    DataExampleBase,
    MultipleChoiceAnswer,
)


class HLEExample(DataExampleBase):
    """HLE multiple choice question example."""

    question_text: str
    options: list[str]
    correct_ans_letter: MultipleChoiceAnswer
    hle_id: str
    category: str
    raw_subject: str

    def _get_options(self) -> list[str]:
        return self.options

    def _get_question(self) -> str:
        return self.question_text

    @property
    def _ground_truth(self) -> MultipleChoiceAnswer:
        return self.correct_ans_letter


def _parse_mcq_options(question: str) -> tuple[str, list[str], Optional[str]]:
    """
    Parse MCQ options from a question string.

    HLE MCQ questions have options embedded in the question text, typically formatted as:
    - (A) option1 (B) option2 ...
    - A. option1 B. option2 ...
    - A) option1 B) option2 ...

    Returns:
        tuple of (question_text, options_list, None if parsing failed)
    """
    # Try different MCQ patterns
    # Pattern 1: (A) option (B) option ...
    pattern1 = r'\(([A-Z])\)\s*([^(]+?)(?=\s*\([A-Z]\)|$)'
    # Pattern 2: A. option B. option ...
    pattern2 = r'([A-Z])\.\s*([^A-Z\n]+?)(?=\s*[A-Z]\.|$)'
    # Pattern 3: A) option B) option ...
    pattern3 = r'([A-Z])\)\s*([^A-Z\n]+?)(?=\s*[A-Z]\)|$)'

    for pattern in [pattern1, pattern2, pattern3]:
        matches = re.findall(pattern, question)
        if len(matches) >= 2:  # At least 2 options to be a valid MCQ
            # Extract question text (everything before first option)
            first_match_pos = question.find(matches[0][1].strip())
            if first_match_pos > 0:
                # Find where the options section starts
                if pattern == pattern1:
                    q_text = re.split(r'\s*\([A-Z]\)', question)[0].strip()
                elif pattern == pattern2:
                    q_text = re.split(r'\s*[A-Z]\.', question)[0].strip()
                else:
                    q_text = re.split(r'\s*[A-Z]\)', question)[0].strip()

                options = [opt.strip() for _, opt in matches]
                return q_text, options, None

    return question, [], "Could not parse options"


def _process_hle_example(example: dict) -> Optional[HLEExample]:
    """
    Process a single HLE example into an HLEExample object.

    Filters for MCQ questions and parses the options.
    """
    # Filter for MCQ questions only (answer_type indicates question type)
    answer_type = example.get("answer_type", "")
    if answer_type not in ["multiple_choice", "MCQ", "mc"]:
        # Also check if it looks like an MCQ based on answer format
        answer = example.get("answer", "")
        if not (len(answer) == 1 and answer.upper() in ascii_uppercase[:10]):
            return None

    # Skip questions with images (multimodal)
    if example.get("image") and example["image"] != "":
        return None

    question = example.get("question", "")
    answer = example.get("answer", "").upper()

    # Parse the question to extract options
    question_text, options, error = _parse_mcq_options(question)

    if error or len(options) < 2:
        # Try alternative: maybe options are separate lines
        lines = question.split("\n")
        option_lines = []
        q_lines = []
        for line in lines:
            stripped = line.strip()
            if re.match(r'^[A-Z][\.\)]\s', stripped) or re.match(r'^\([A-Z]\)\s', stripped):
                # Clean the option prefix
                cleaned = re.sub(r'^[A-Z][\.\)]\s*', '', stripped)
                cleaned = re.sub(r'^\([A-Z]\)\s*', '', cleaned)
                option_lines.append(cleaned)
            else:
                q_lines.append(line)

        if len(option_lines) >= 2:
            question_text = "\n".join(q_lines).strip()
            options = option_lines
        else:
            return None  # Cannot parse options

    # Clean up question text by removing common answer choice headers
    # These will be added back by the formatter
    question_text = re.sub(r'\n*\s*(Answer Choices|Options|Choices|Select one):?\s*:?\s*$', '', question_text, flags=re.IGNORECASE).strip()

    # Validate answer is within range
    if answer not in ascii_uppercase[:len(options)]:
        return None

    # Filter out questions with too many options (codebase supports max 15: A-O)
    if len(options) > 15:
        return None

    return HLEExample(
        question_text=question_text,
        options=options,
        correct_ans_letter=answer,  # type: ignore
        hle_id=example.get("id", ""),
        category=example.get("category", ""),
        raw_subject=example.get("raw_subject", ""),
    )


@lru_cache(maxsize=1)
def _load_hle_dataset() -> Slist[HLEExample]:
    """Load and cache the HLE dataset from Hugging Face."""
    dataset = load_dataset("cais/hle", split="test")

    examples = []
    for item in dataset:
        example = _process_hle_example(item)
        if example is not None:
            examples.append(example)

    return Slist(examples)


def test() -> Slist[HLEExample]:
    """Get HLE MCQ test examples (for evaluation only)."""
    return _load_hle_dataset()
