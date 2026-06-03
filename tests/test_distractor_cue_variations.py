"""Offline unit tests for the distractor-cue variation family.

Run: .venv/bin/python -m pytest tests/test_distractor_cue_variations.py -v
"""

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent

from cot_transparency.formatters.more_biases.distractor_cue_variations import (
    DISTRACTOR_CUES,
    HOLDOUT_CUES,
    apply_distractor_cue,
    cue_keys,
    extract_wrong_cot,
    split_cues,
)

QUESTION = "What is the capital of France?\nA) Paris\nB) Lyon\nPlease answer with A or B."
ARGUMENT = "It must be Lyon because Lyon is the largest and most central city."


def _messages():
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": QUESTION},
    ]


class TestRegistry:
    def test_has_many_distinct_cues(self):
        assert len(DISTRACTOR_CUES) >= 24
        # keys unique and match declaration-order helper
        assert len(set(DISTRACTOR_CUES)) == len(DISTRACTOR_CUES)
        assert set(cue_keys()) == set(DISTRACTOR_CUES)

    def test_templates_distinct_outputs(self):
        outs = {key: fn(QUESTION, ARGUMENT) for key, fn in DISTRACTOR_CUES.items()}
        # No two framings produce identical text (genuinely distinct paraphrases).
        assert len(set(outs.values())) == len(outs)


class TestTemplateContent:
    @pytest.mark.parametrize("cue", list(DISTRACTOR_CUES))
    def test_embeds_question_and_argument(self, cue):
        out = DISTRACTOR_CUES[cue](QUESTION, ARGUMENT)
        assert QUESTION.strip() in out, f"{cue} dropped the question"
        assert ARGUMENT.strip() in out, f"{cue} dropped the argument"
        assert "<argument>" in out and "<question>" in out

    def test_ordering_varies(self):
        # At least one framing puts the question before the argument and vice-versa.
        first_arg, first_q = [], []
        for fn in DISTRACTOR_CUES.values():
            out = fn(QUESTION, ARGUMENT)
            (first_arg if out.index("<argument>") < out.index("<question>") else first_q).append(out)
        assert first_arg and first_q


class TestApply:
    def test_replaces_first_user_preserves_system(self):
        msgs = _messages()
        out = apply_distractor_cue(msgs, ARGUMENT, "read_somewhere_dontbias")
        assert out[0] == {"role": "system", "content": "You are a helpful assistant."}
        assert out[1]["role"] == "user"
        assert ARGUMENT in out[1]["content"]
        assert QUESTION.strip() in out[1]["content"]

    def test_does_not_mutate_input(self):
        msgs = _messages()
        before = [dict(m) for m in msgs]
        apply_distractor_cue(msgs, ARGUMENT, "colleague")
        assert msgs == before

    def test_unknown_cue_raises(self):
        with pytest.raises(KeyError, match="Unknown distractor cue"):
            apply_distractor_cue(_messages(), ARGUMENT, "does_not_exist")

    def test_no_user_message_raises(self):
        with pytest.raises(ValueError, match="No user message"):
            apply_distractor_cue([{"role": "system", "content": "sys"}], ARGUMENT, "colleague")


class TestExtractWrongCot:
    def test_extracts_from_argument_tags(self):
        bq = [
            {"role": "user", "content": f"prefix\n<argument>\n{ARGUMENT}\n</argument>\nsuffix"},
            {"role": "assistant", "content": "..."},
        ]
        assert extract_wrong_cot(bq) == ARGUMENT

    def test_returns_none_when_absent(self):
        assert extract_wrong_cot([{"role": "user", "content": "no tags here"}]) is None
        assert extract_wrong_cot([]) is None

    def test_round_trip_with_canonical_wrapper(self):
        # Build a distractor prompt, then recover the argument from it.
        built = DISTRACTOR_CUES["read_somewhere_dontbias"](QUESTION, ARGUMENT)
        assert extract_wrong_cot([{"role": "user", "content": built}]) == ARGUMENT


class TestSplit:
    def test_train_holdout_disjoint_and_complete(self):
        train, hold = split_cues()
        assert set(hold) == set(HOLDOUT_CUES)
        assert set(train).isdisjoint(set(hold))
        assert set(train) | set(hold) == set(DISTRACTOR_CUES)

    def test_unknown_holdout_raises(self):
        with pytest.raises(KeyError, match="Unknown holdout"):
            split_cues(holdout=("nope",))
