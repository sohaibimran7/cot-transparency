"""
GPQA (main subset) loader for MCQ questions.

Dataset: https://huggingface.co/datasets/Idavidrein/gpqa
Paper: https://arxiv.org/abs/2311.12022

GPQA main contains 448 expert-written graduate-level multiple-choice questions
across physics, chemistry, and biology. Each question has 1 correct and 3
incorrect answers; we shuffle deterministically per-question so the correct
option is not always in the same position.
"""

import hashlib
from functools import lru_cache
from random import Random
from string import ascii_uppercase

from datasets import load_dataset
from slist import Slist

from cot_transparency.data_models.example_base import (
    DataExampleBase,
    MultipleChoiceAnswer,
)


class GPQAExample(DataExampleBase):
    question_text: str
    options: list[str]
    correct_ans_letter: MultipleChoiceAnswer
    gpqa_id: str
    high_level_domain: str
    subdomain: str

    def _get_options(self) -> list[str]:
        return self.options

    def _get_question(self) -> str:
        return self.question_text

    @property
    def _ground_truth(self) -> MultipleChoiceAnswer:
        return self.correct_ans_letter


def _process_gpqa_example(example: dict) -> GPQAExample | None:
    question = (example.get("Question") or "").strip()
    correct = (example.get("Correct Answer") or "").strip()
    incorrect = [
        (example.get("Incorrect Answer 1") or "").strip(),
        (example.get("Incorrect Answer 2") or "").strip(),
        (example.get("Incorrect Answer 3") or "").strip(),
    ]
    if not question or not correct or any(not o for o in incorrect):
        return None

    record_id = example.get("Record ID") or hashlib.sha1(question.encode()).hexdigest()
    # Deterministic per-question shuffle so correct answer position varies but is reproducible.
    rng = Random(f"gpqa:{record_id}")
    options = [correct, *incorrect]
    indices = list(range(len(options)))
    rng.shuffle(indices)
    shuffled = [options[i] for i in indices]
    correct_pos = indices.index(0)
    correct_letter = ascii_uppercase[correct_pos]

    return GPQAExample(
        question_text=question,
        options=shuffled,
        correct_ans_letter=correct_letter,  # type: ignore
        gpqa_id=str(record_id),
        high_level_domain=str(example.get("High-level domain") or ""),
        subdomain=str(example.get("Subdomain") or ""),
    )


@lru_cache(maxsize=1)
def _load_gpqa_main() -> Slist[GPQAExample]:
    dataset = load_dataset("Idavidrein/gpqa", "gpqa_main", split="train")
    examples: list[GPQAExample] = []
    for item in dataset:
        ex = _process_gpqa_example(item)
        if ex is not None:
            examples.append(ex)
    return Slist(examples)


def test() -> Slist[GPQAExample]:
    """Get GPQA main MCQ examples (for evaluation only)."""
    return _load_gpqa_main()
