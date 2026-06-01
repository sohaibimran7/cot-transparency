"""Tests for eval-awareness cues (the perturbation)."""

from cot_transparency.eval_awareness.cues import (
    TRAIN_CUE,
    EVAL_MARKERS,
    apply_train_cue,
    apply_eval_marker,
    natural,
)


def _msgs():
    return [{"role": "user", "content": "Should I do X?"}]


def test_train_cue_prepends_to_first_user():
    out = apply_train_cue(_msgs())
    assert out[0]["content"].startswith(TRAIN_CUE)
    assert "Should I do X?" in out[0]["content"]


def test_does_not_mutate_input():
    original = _msgs()
    snapshot = original[0]["content"]
    apply_train_cue(original)
    apply_eval_marker(original, "morality_eval_tag")
    natural(original)
    assert original[0]["content"] == snapshot  # unchanged


def test_natural_equals_input_content():
    assert natural(_msgs()) == _msgs()


def test_train_cue_inserts_user_when_none():
    out = apply_train_cue([{"role": "assistant", "content": "hi"}])
    assert out[0]["role"] == "user"
    assert out[0]["content"] == TRAIN_CUE


def test_tag_marker_wraps_content():
    out = apply_eval_marker(_msgs(), "morality_eval_tag")
    assert "<llm_morality_evaluation>" in out[0]["content"]
    assert "</llm_morality_evaluation>" in out[0]["content"]
    assert "Should I do X?" in out[0]["content"]


def test_eval_markers_distinct_from_train_cue():
    # Generalization requires held-out markers differ from the trained cue.
    for name in EVAL_MARKERS:
        out = apply_eval_marker(_msgs(), name)
        assert TRAIN_CUE not in out[0]["content"], f"{name} must differ from TRAIN_CUE"


def test_unknown_marker_raises():
    try:
        apply_eval_marker(_msgs(), "nope")
    except KeyError:
        return
    raise AssertionError("expected KeyError for unknown marker")
