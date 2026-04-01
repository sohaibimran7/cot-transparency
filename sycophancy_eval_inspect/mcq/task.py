"""MCQ bias evaluation task."""

import json
from pathlib import Path
from typing import Any

from inspect_ai import Task, task
from inspect_ai.model import ChatMessageAssistant, ChatMessageUser
from inspect_ai.solver import Generate, Solver, TaskState, solver

from sycophancy_eval_inspect.mcq.dataset import (
    PromptStyle,
    QuestionVariant,
    load_mcq_bias_dataset,
)
from sycophancy_eval_inspect.mcq.scorer import (
    bias_acknowledged_scorer,
    few_shot_confusion_scorer,
    mcq_bias_scorer,
    options_considered_scorer,
)


@solver
def multi_turn_generate() -> Solver:
    """Solver that handles multi-turn biases with on-policy generation.

    For samples with ``followup_user_messages`` in metadata (e.g. are_you_sure),
    generates the model's response at each assistant turn rather than using
    pre-filled canned responses.  For single-turn samples, behaves identically
    to ``generate()``.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        followups = state.metadata.get("followup_user_messages")
        if not followups:
            # Single-turn (or post_hoc with pre-filled assistant turn) — one generation.
            return await generate(state)

        # Multi-turn (are_you_sure): force correct answer → inject challenge → generate.
        # The first assistant turn is teacher-forced with the correct ground truth
        # letter (no CoT).  The dataset confirms all first answers are single
        # letters.  Generating on-policy and filtering to a correct single letter
        # is equivalent but wastes compute.
        first_answer = state.target.text
        state.messages.append(ChatMessageAssistant(content=first_answer))

        # Inject challenge messages and generate responses.
        # reasoning_history="none" strips all reasoning from prior assistant
        # turns so thinking tokens don't leak into subsequent prompts.
        for followup in followups:
            state.messages.append(ChatMessageUser(content=followup))
            state = await generate(state, reasoning_history="none")

        return state

    return solve


def load_hashes_from_file(path: str | Path) -> set[str]:
    """Load allowed hashes from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    # Support both flat list and dict with original_dataset keys
    if isinstance(data, list):
        return set(data)
    elif isinstance(data, dict):
        # Flatten all hashes from all datasets
        all_hashes = set()
        for hashes in data.values():
            all_hashes.update(hashes)
        return all_hashes
    else:
        raise ValueError(f"Invalid hash file format: expected list or dict, got {type(data)}")


@task
def mcq_bias_eval(
    dataset_path: str,
    variant: QuestionVariant = "biased",
    prompt_style: PromptStyle = "cot",
    limit: int | None = None,
    filter_bias_on_wrong: bool = True,
    allowed_hashes: set[str] | None = None,
    hash_filter_file: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Task:
    """
    Evaluate model on a single MCQ bias dataset file.

    Args:
        dataset_path: Path to JSONL file (e.g., dataset_dumps/test/suggested_answer/mmlu_suggested_answer.jsonl)
        variant: "biased" or "unbiased" question variant
        prompt_style: "cot" for standard models, "no_cot" for reasoning models (o1, o3, etc.)
        limit: Max samples to evaluate (None for all). Note: when allowed_hashes is provided,
            limit is applied AFTER hash filtering (i.e., limit samples from the filtered set).
        filter_bias_on_wrong: Only include samples where bias points to wrong answer
        allowed_hashes: If provided, only include samples with these original_question_hash values.
            This enables hash-based matching across bias types for BRR calculation.
            When using programmatically, compute common hashes using hash_utils.compute_common_hashes_for_eval().
        hash_filter_file: Path to JSON file containing allowed hashes (for CLI usage).
        metadata: Additional metadata to include in the eval log (e.g., checkpoint_path, base_model).
            Can be a list of hashes or a dict mapping original dataset names to hash lists.
            If both allowed_hashes and hash_filter_file are provided, allowed_hashes takes precedence.

    Example CLI usage:
        inspect eval sycophancy_eval_inspect/mcq/task.py@mcq_bias_eval \\
            -T dataset_path=dataset_dumps/test/suggested_answer/mmlu_suggested_answer.jsonl \\
            -T variant=biased \\
            --model openai/gpt-4o

        # With hash filtering (for proper BRR calculation):
        inspect eval sycophancy_eval_inspect/mcq/task.py@mcq_bias_eval \\
            -T dataset_path=dataset_dumps/test/suggested_answer/mmlu_suggested_answer.jsonl \\
            -T variant=biased \\
            -T hash_filter_file=common_hashes.json \\
            --model openai/gpt-4o

        # For reasoning models (no CoT instructions):
        inspect eval sycophancy_eval_inspect/mcq/task.py@mcq_bias_eval \\
            -T dataset_path=dataset_dumps/test/suggested_answer/mmlu_suggested_answer.jsonl \\
            -T variant=biased \\
            -T prompt_style=no_cot \\
            --model openai/o1

    Note: For hash-based matching across bias types, use the runner scripts
    (run_tinker_evals.py, run_suite.py) which handle hash computation automatically.
    """
    # Resolve allowed_hashes from file if provided and not already set
    effective_hashes = allowed_hashes
    if effective_hashes is None and hash_filter_file is not None:
        effective_hashes = load_hashes_from_file(hash_filter_file)

    dataset = load_mcq_bias_dataset(
        dataset_path,
        variant=variant,
        prompt_style=prompt_style,
        filter_bias_on_wrong=filter_bias_on_wrong,
        allowed_hashes=effective_hashes,
    )

    if limit:
        dataset = dataset[:limit]

    return Task(
        dataset=dataset,
        solver=[multi_turn_generate()],
        scorer=[
            mcq_bias_scorer(),
            options_considered_scorer(),
            bias_acknowledged_scorer(),
            few_shot_confusion_scorer(),
        ],
        metadata=metadata,
    )
