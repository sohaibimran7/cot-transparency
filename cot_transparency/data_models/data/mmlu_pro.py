"""
MMLU-Pro loader for MCQ questions.

Dataset: https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro
Paper: https://arxiv.org/abs/2406.01574

MMLU-Pro contains 12,032 reasoning-focused MCQ questions across 14 categories,
each with up to 10 answer options (A-J). Loader returns the `test` split;
questions where options/answer parsing fails are skipped.
"""

from functools import lru_cache
from string import ascii_uppercase

from datasets import load_dataset
from slist import Slist

from cot_transparency.data_models.example_base import (
    DataExampleBase,
    MultipleChoiceAnswer,
)


class MMLUProExample(DataExampleBase):
    question_text: str
    options: list[str]
    correct_ans_letter: MultipleChoiceAnswer
    question_id: str
    category: str

    def _get_options(self) -> list[str]:
        return self.options

    def _get_question(self) -> str:
        return self.question_text

    @property
    def _ground_truth(self) -> MultipleChoiceAnswer:
        return self.correct_ans_letter


def _process_mmlu_pro_example(example: dict) -> MMLUProExample | None:
    question = (example.get("question") or "").strip()
    options = example.get("options") or []
    answer = (example.get("answer") or "").strip().upper()
    answer_index = example.get("answer_index")

    if not question or not options:
        return None
    options = [str(o).strip() for o in options if o is not None and str(o).strip()]
    if len(options) < 2:
        return None
    # MMLU-Pro caps at 10 (A-J); codebase supports up to 15 (A-O).
    if len(options) > 15:
        return None

    if answer and answer in ascii_uppercase[: len(options)]:
        correct_letter = answer
    elif isinstance(answer_index, int) and 0 <= answer_index < len(options):
        correct_letter = ascii_uppercase[answer_index]
    else:
        return None

    return MMLUProExample(
        question_text=question,
        options=options,
        correct_ans_letter=correct_letter,  # type: ignore
        question_id=str(example.get("question_id") or ""),
        category=str(example.get("category") or ""),
    )


@lru_cache(maxsize=1)
def _load_mmlu_pro_test() -> Slist[MMLUProExample]:
    dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    examples: list[MMLUProExample] = []
    for item in dataset:
        ex = _process_mmlu_pro_example(item)
        if ex is not None:
            examples.append(ex)
    return Slist(examples)


def test() -> Slist[MMLUProExample]:
    """Get MMLU-Pro test examples (for evaluation only)."""
    return _load_mmlu_pro_test()
