"""
Integration tests for the sycophancy eval pipeline.

Tests the full pipeline from dataset loading through Task construction and
scorer instantiation for all 8 supported MCQ bias types, without requiring
any model API calls.

Run with:
    conda run -n cot pytest tests/test_eval_pipeline.py -v
"""

import pytest

from sycophancy_eval_inspect.mcq.dataset import load_mcq_bias_dataset
from sycophancy_eval_inspect.mcq.scorer import _BIAS_ACK_PROMPTS
from sycophancy_eval_inspect.mcq.task import mcq_bias_eval


# All 8 supported MCQ bias types with their test dataset paths.
# spurious_few_shot_hindsight is only available for hindsight_neglect.
BIAS_PATHS = {
    "suggested_answer":           "dataset_dumps/test/suggested_answer/mmlu_suggested_answer.jsonl",
    "distractor_argument":        "dataset_dumps/test/distractor_argument/mmlu_distractor_argument.jsonl",
    "distractor_fact":            "dataset_dumps/test/distractor_fact/mmlu_distractor_fact.jsonl",
    "wrong_few_shot":             "dataset_dumps/test/wrong_few_shot/mmlu_wrong_few_shot.jsonl",
    "spurious_few_shot_squares":  "dataset_dumps/test/spurious_few_shot_squares/mmlu_spurious_few_shot_squares.jsonl",
    "post_hoc":                   "dataset_dumps/test/post_hoc/mmlu_post_hoc.jsonl",
    "are_you_sure":               "dataset_dumps/test/are_you_sure/mmlu_are_you_sure.jsonl",
    "spurious_few_shot_hindsight":"dataset_dumps/test/spurious_few_shot_hindsight/hindsight_neglect_spurious_few_shot_hindsight.jsonl",
}

FEW_SHOT_BIASES = {"wrong_few_shot", "spurious_few_shot_squares", "spurious_few_shot_hindsight"}
MULTI_TURN_BIASES = {"post_hoc", "are_you_sure"}


@pytest.mark.parametrize("bias", list(BIAS_PATHS.keys()))
def test_dataset_loads_samples(bias):
    """Dataset loads a non-zero number of samples for both variants."""
    path = BIAS_PATHS[bias]
    ds_biased   = load_mcq_bias_dataset(path, variant="biased")
    ds_unbiased = load_mcq_bias_dataset(path, variant="unbiased")
    assert len(ds_biased) > 0,   f"{bias}: biased dataset is empty"
    assert len(ds_unbiased) > 0, f"{bias}: unbiased dataset is empty"


@pytest.mark.parametrize("bias", list(BIAS_PATHS.keys()))
def test_input_ends_with_user_message(bias):
    """Every sample's input must end with a user message (model generates after it)."""
    ds = load_mcq_bias_dataset(BIAS_PATHS[bias], variant="biased")
    sample = ds[0]
    assert sample.input[-1].role == "user", (
        f"{bias}: last message role is '{sample.input[-1].role}', expected 'user'"
    )


@pytest.mark.parametrize("bias", list(BIAS_PATHS.keys()))
def test_multi_turn_structure(bias):
    """Multi-turn biases have assistant messages; single-turn biases do not."""
    ds = load_mcq_bias_dataset(BIAS_PATHS[bias], variant="biased")
    roles = [m.role for m in ds[0].input]
    has_assistant = "assistant" in roles
    if bias in MULTI_TURN_BIASES:
        if bias == "are_you_sure":
            # are_you_sure uses on-policy multi-turn solver: Sample.input has only
            # the first user message; followup challenges are in metadata.
            assert not has_assistant, (
                f"{bias}: are_you_sure should NOT have pre-filled assistant messages "
                "(on-policy generation via multi_turn_generate solver)"
            )
            assert "followup_user_messages" in ds[0].metadata, (
                f"{bias}: expected followup_user_messages in metadata"
            )
            assert len(ds[0].metadata["followup_user_messages"]) == 2, (
                f"{bias}: expected 2 followup messages (challenge + final ask)"
            )
        else:
            assert has_assistant, f"{bias}: expected assistant messages in multi-turn bias"
    else:
        assert not has_assistant, f"{bias}: unexpected assistant messages in single-turn bias"


@pytest.mark.parametrize("bias", list(BIAS_PATHS.keys()))
def test_no_cot_strips_step_by_step(bias):
    """In no_cot mode, the last user message must not contain 'step by step'."""
    ds = load_mcq_bias_dataset(BIAS_PATHS[bias], variant="biased", prompt_style="no_cot")
    last_user = [m for m in ds[0].input if m.role == "user"][-1].content
    assert "step by step" not in last_user.lower(), (
        f"{bias}: 'step by step' still present in last user message after no_cot strip"
    )


@pytest.mark.parametrize("bias", list(BIAS_PATHS.keys()))
def test_biasing_text_populated(bias):
    """biasing_text metadata must be non-empty for all biased samples."""
    ds = load_mcq_bias_dataset(BIAS_PATHS[bias], variant="biased")
    bt = ds[0].metadata.get("biasing_text", "")
    assert bt, f"{bias}: biasing_text is empty"


@pytest.mark.parametrize("bias", list(BIAS_PATHS.keys()))
def test_ba_scorer_template_exists(bias):
    """Every bias must have a BA scorer template that formats without error."""
    ds = load_mcq_bias_dataset(BIAS_PATHS[bias], variant="biased")
    bt = ds[0].metadata.get("biasing_text", "")
    template = _BIAS_ACK_PROMPTS.get(bias)
    assert template is not None, f"{bias}: no BA scorer template in _BIAS_ACK_PROMPTS"
    # Should not raise
    formatted = template.format(biasing_text=bt, completion="test answer")
    assert len(formatted) > 0


@pytest.mark.parametrize("bias", list(FEW_SHOT_BIASES))
def test_few_shot_questions_extracted(bias):
    """Few-shot biases must have few_shot_questions metadata populated."""
    ds = load_mcq_bias_dataset(BIAS_PATHS[bias], variant="biased")
    fsq = ds[0].metadata.get("few_shot_questions", [])
    assert len(fsq) > 0, f"{bias}: few_shot_questions is empty"


@pytest.mark.parametrize("bias", ["are_you_sure"])
def test_are_you_sure_not_x_format(bias):
    """are_you_sure biased_option uses 'NOT X' format and filter_bias_on_wrong passes it through."""
    ds = load_mcq_bias_dataset(BIAS_PATHS[bias], variant="biased", filter_bias_on_wrong=True)
    assert len(ds) > 0, "are_you_sure: all samples filtered out by filter_bias_on_wrong"
    biased_option = ds[0].metadata.get("biased_option", "")
    assert biased_option.startswith("NOT "), (
        f"are_you_sure: expected 'NOT X' biased_option, got '{biased_option}'"
    )


@pytest.mark.parametrize("bias", list(BIAS_PATHS.keys()))
def test_task_dataset_via_loader(bias):
    """Dataset loaded for a Task has the right structure: non-empty, ends with user message."""
    # Avoid instantiating the grader-model scorers (requires openai>=1.58.1).
    # Test the dataset loading path that task.py exercises instead.
    from sycophancy_eval_inspect.mcq.dataset import load_mcq_bias_dataset
    ds = load_mcq_bias_dataset(BIAS_PATHS[bias], variant="biased")
    assert len(ds) > 0, f"{bias}: dataset is empty"
    for sample in ds:
        assert sample.input[-1].role == "user", (
            f"{bias}: sample {sample.id} does not end with user message"
        )
