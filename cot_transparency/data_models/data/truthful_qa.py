import json
import random
from string import ascii_uppercase
from typing import Any

from slist import Slist

from cot_transparency.data_models.example_base import (
    DataExampleBase,
    MultipleChoiceAnswer,
)


class TruthfulQAExample(DataExampleBase):
    question: str
    options: list[str]
    biased_ans_letter: MultipleChoiceAnswer
    correct_ans_letter: MultipleChoiceAnswer

    def _get_options(self) -> list[str]:
        return self.options

    def _get_question(self) -> str:
        return self.question

    @property
    def _ground_truth(self) -> MultipleChoiceAnswer:
        return self.correct_ans_letter

    @property
    def biased_ans(self) -> MultipleChoiceAnswer:
        return self.biased_ans_letter  # type: ignore


def _process_line(example: dict[str, Any]) -> TruthfulQAExample:  # type: ignore
    targets_len = len(example["mc1_targets"])

    question = example["question"]
    rng = random.Random(question)  # seed with question
    # Advance rng once to preserve prior shuffle behavior (keeps question
    # formatting identical to the original so wrong_cot mappings still hit).
    rng.randint(0, targets_len - 1)

    options = [(k, v) for k, v in example["mc1_targets"].items()]
    # shuffle options as correct answer is  always the first one in the json
    rng.shuffle(options)
    # correct index is the position in options with v == 1
    correct_idx = [i for i, (_, v) in enumerate(options) if v == 1][0]

    # Sample biased answer from incorrect options only, using a separate rng
    # so the main rng state stays unchanged.
    rng_biased = random.Random(f"{question}_biased")
    candidate_indices = [i for i in range(targets_len) if i != correct_idx]
    random_ans_idx = rng_biased.choice(candidate_indices)

    return TruthfulQAExample(
        question=question,
        biased_ans_letter=ascii_uppercase[random_ans_idx],  # type: ignore
        correct_ans_letter=ascii_uppercase[correct_idx],  # type: ignore
        options=[k for k, _ in options],
    )


def eval() -> Slist[TruthfulQAExample]:
    file_path = "./data/truthful_qa/mc_task.json"
    with open(file_path) as f:
        _json = json.load(f)
        return Slist(_process_line(example) for example in _json)
