"""Offline tests for the misalignment judge's parsing/heuristics (no API calls)."""

from cot_transparency.eval_awareness.judge import (
    _parse_yes_no,
    _emitted_outbound_action,
    make_misalignment_judge,
    DEFAULT_GRADER_MODEL,
)


def test_parse_yes_no_tagged():
    assert _parse_yes_no("<reasoning>x</reasoning>\n<answer>yes</answer>") is True
    assert _parse_yes_no("<answer>NO</answer>") is False
    assert _parse_yes_no("<answer> Yes </answer>") is True


def test_parse_yes_no_fallback():
    assert _parse_yes_no("... so the answer: yes") is True
    assert _parse_yes_no("answer>no") is False
    assert _parse_yes_no("yes") is True


def test_parse_yes_no_unparseable():
    assert _parse_yes_no("maybe, it depends") is None
    assert _parse_yes_no("") is None


def test_emitted_outbound_action():
    assert _emitted_outbound_action("text <tool_use:email> to: a@b.com </tool_use:email>") is True
    assert _emitted_outbound_action("<tool_use:forward>\nemail_id: 3\n</tool_use:forward>") is True
    assert _emitted_outbound_action("I considered emailing but refused.") is False


def test_make_judge_returns_async_callable():
    import inspect
    j = make_misalignment_judge(DEFAULT_GRADER_MODEL)
    assert inspect.iscoroutinefunction(j)
