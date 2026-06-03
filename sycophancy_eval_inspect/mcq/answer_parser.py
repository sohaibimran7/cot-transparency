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


_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
# Uppercase, word-boundaried option letter — avoids grabbing lowercase letters
# from prose inside the tag (e.g. the "e" in "The answer is C").
_TAG_LETTER_RE = re.compile(r"\b([A-J])\b")


def fallback_answer_parser(model_answer: str) -> str | None:
    """
    Lenient fallback parser that strips formatting (markdown bold, latex
    boxing) and then runs the standard parser on the cleaned text.

    If the response contains an explicit ``<answer>X</answer>`` tag, that takes
    precedence (we read the LAST such tag, since models sometimes restate). This
    keeps backward compatibility: responses without the tag fall through to the
    legacy "the answer is (X)" parsing unchanged.

    Kept separate from cot_answer_parser because a match here means the
    model did NOT follow the expected format — useful as a signal.

    Args:
        model_answer: The raw model response text

    Returns:
        The parsed answer letter (A-J) or None if no answer found
    """
    for content in reversed(_ANSWER_TAG_RE.findall(model_answer)):
        m = _TAG_LETTER_RE.search(content.strip())
        if m:
            return m.group(1).upper()

    cleaned = model_answer
    # **X** -> X, **(X)** -> (X)
    cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)
    # $\boxed{X}$ variants: \boxed{C}, \boxed{\text{C}}, \boxed{\textbf{(C)}}
    cleaned = re.sub(
        r"\$?\\boxed\{(?:\\text(?:bf)?\{)?([^}]*)\}?\}\$?", r"\1", cleaned
    )

    return cot_answer_parser(cleaned)


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

        match = re.search(pattern, last_item, re.MULTILINE)
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
