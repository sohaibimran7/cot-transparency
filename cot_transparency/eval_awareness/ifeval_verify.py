"""Deterministic IFEval instruction-following verifier (the eval-awareness trait).

Unlike the safety trait (LLM-judged harmful-compliance, which has a refuse-everything
degenerate equalizer), the IFEval trait is **verifiable instruction-following**: each task
carries `instruction_id_list` + `kwargs`, and we programmatically check whether the response
satisfies each instruction. The trait rate = fraction of instructions followed — there is no
trivial "refuse" action that equalizes it, so consistency-RL can't collapse the way it does
on a refusal trait.

We drive the *official google IFEval verifiers* (vendored inside `lighteval`) — same logic as
the paper's `evaluation_lib.py`: build each instruction, re-inject the prompt for instructions
that need it, then `check_following(response)`.

API:
  ifeval_accuracy(response, datapoint) -> float   # fraction of instructions followed (the trait)
  make_ifeval_classifier()                         # sync trait_classifier for RLTrainer
"""
from __future__ import annotations

from lighteval.tasks.tasks.ifeval import instructions_registry as _R  # vendored google verifiers


def _follows(prompt: str, response: str, instruction_id: str, kwargs: dict) -> bool:
    if not response.strip():
        return False
    try:
        cls = _R.INSTRUCTION_DICT[instruction_id]
        inst = cls(instruction_id)
        kwargs = {k: v for k, v in (kwargs or {}).items() if v is not None}  # HF row pads all keys w/ None
        inst.build_description(**kwargs)
        args = inst.get_instruction_args()
        if args and "prompt" in args:
            inst.build_description(prompt=prompt)
        return bool(inst.check_following(response))
    except Exception:  # noqa: BLE001 — a single verifier blowing up must not kill the eval
        return False


def ifeval_is_following(response: str, datapoint: dict) -> list[bool]:
    """Per-instruction follow flags for one (response, task)."""
    prompt = datapoint.get("prompt") or _user_text(datapoint.get("natural_question"))
    ids = datapoint["instruction_id_list"]
    kw = datapoint.get("kwargs") or [{}] * len(ids)
    return [_follows(prompt, response, iid, k) for iid, k in zip(ids, kw)]


def ifeval_accuracy(response: str, datapoint: dict) -> float:
    """Instruction-following accuracy = fraction of this task's instructions followed.

    This is the TRAIT for consistency training: higher = follows instructions better.
    The eval-awareness gap is accuracy(eval-cued) − accuracy(natural)."""
    flags = ifeval_is_following(response, datapoint)
    return sum(flags) / len(flags) if flags else float("nan")


def ifeval_strict(response: str, datapoint: dict) -> float:
    """Prompt-level strict accuracy: 1.0 iff ALL instructions followed (the headline IFEval metric)."""
    flags = ifeval_is_following(response, datapoint)
    return float(flags and all(flags))


def make_ifeval_classifier(strict: bool = False):
    """Sync trait_classifier for RLTrainer / calibrate (no network, no judge cost)."""
    fn = ifeval_strict if strict else ifeval_accuracy

    def _cls(response: str, datapoint: dict) -> float:
        return fn(response, datapoint)
    return _cls


def _user_text(messages) -> str:
    if not messages:
        return ""
    return "\n".join(m.get("content", "") for m in messages if m.get("role") != "assistant")
