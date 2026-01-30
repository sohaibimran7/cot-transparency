"""Answer parser for multiple choice questions.

Ported from dataset_dumps/sample_biased_reasoning_calculation.py
"""

import re
from string import ascii_uppercase


BREAK_WORDS: list[str] = [
    "answer is (",
    "answer is  (",
    "answer is: (",
    "answer is:(",
    "answer is:  (",
    "answer is:\n(",
    "answer is: \n(",
    "answer is:\n\n(",
    "answer is: ",
    "answer is ",
    "answer is $\\boxed{\\text{(",
    "answer is: $\\boxed{\\text{(",
    "choices is: " r"is: $\boxed{\textbf{(",
    "answer: ",
    "answer is ",
    r"answer is: \[\boxed{\text{",
    r"is: $\boxed{\textbf{(",
    "choices is: ",
    r"is: $\boxed{\textbf{(",
    r"is $\boxed{\textbf{(",
    r"is $\boxed{\text{(",
    r"is: \boxed{\text{(",
    r"is: $\boxed{\text{(",
    r"is: (\boxed{\text{(",
    "accurate answer would be",
    "is: $\\boxed{\\textbf{(",
]


def cot_answer_parser(model_answer: str) -> str | None:
    """
    Parse a multiple choice answer from model output.

    Looks for patterns like "the answer is (A)" or "answer is: B" and extracts
    the letter choice.

    Args:
        model_answer: The raw model response text

    Returns:
        The parsed answer letter (A-J) or None if no answer found
    """
    for break_word in BREAK_WORDS:
        if break_word not in model_answer:
            continue
        tmp = model_answer.split(break_word)
        # Sometimes there is a space in front of the answer
        last_item = tmp[-1].lstrip()

        if not last_item:
            continue

        # also add lowercase variants
        possible_indicators = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        possible_indicators_lower = [indicator.lower() for indicator in possible_indicators]
        possible_indicators_re = "|".join(possible_indicators + possible_indicators_lower)

        pattern = rf"^(?:[Oo]ption |[Ss]tatement )?\(?({possible_indicators_re})\)?(\s|\)|\.|\,|$)+.*$"

        match = re.search(pattern, last_item)
        if match:
            candidate_ans = match.group(1)
            if candidate_ans in possible_indicators:
                idx = possible_indicators.index(candidate_ans)
                return ascii_uppercase[idx]
            elif candidate_ans in possible_indicators_lower:
                idx = possible_indicators_lower.index(candidate_ans)
                return ascii_uppercase[idx]

        return None

    return None
