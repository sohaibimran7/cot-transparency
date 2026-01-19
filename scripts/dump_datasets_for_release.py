from pathlib import Path
from typing import Literal, Optional, Sequence

import fire
from grugstream import Observable
from pydantic import BaseModel
from slist import Group, Slist
from cot_transparency.apis import UniversalCaller
from cot_transparency.apis.openai.formatting import append_assistant_preferred_to_last_user
from cot_transparency.data_models.data.inverse_scaling import InverseScalingTask
from cot_transparency.data_models.example_base import MultipleChoiceAnswer
from cot_transparency.data_models.messages import StrictChatMessage
from cot_transparency.data_models.models import TaskSpec
from cot_transparency.formatters.core.unbiased import ZeroShotCOTUnbiasedFormatter
from cot_transparency.formatters.inverse_scaling.no_few_shot import (
    ClearFewShotsThinkStepByStepCOT,
)
from cot_transparency.formatters.more_biases.anchor_initial_wrong import PostHocNoPlease
from cot_transparency.formatters.more_biases.distractor_fact import FirstLetterDistractor
from cot_transparency.formatters.more_biases.random_bias_formatter import RandomBiasedFormatter
from cot_transparency.formatters.more_biases.user_wrong_cot import (
    ImprovedDistractorArgument,
)
from cot_transparency.formatters.more_biases.wrong_few_shot import WrongFewShotMoreClearlyLabelledAtBottom
from cot_transparency.formatters.verbalize.formatters import BlackSquareBiasedFormatter
from cot_transparency.json_utils.read_write import write_jsonl_file_from_basemodel
from cot_transparency.util import assert_not_none
from scripts.are_you_sure.eval_are_you_sure_second_cot import (
    OutputWithAreYouSure,
    run_are_you_sure_multi_model_second_round_cot,
)
from scripts.evaluate_judge_consistency.judge_consistency import (
    BothJudgements,
    ComparisonGenerationJudged,
    many_judge_obs,
)

from stage_one import create_stage_one_task_specs


FORMATTERS_TO_DUMP = {
    RandomBiasedFormatter.name(): "suggested_answer",
    PostHocNoPlease.name(): "post_hoc",
    WrongFewShotMoreClearlyLabelledAtBottom.name(): "wrong_few_shot",
    BlackSquareBiasedFormatter.name(): "spurious_few_shot_squares",
    FirstLetterDistractor.name(): "distractor_fact",
    ImprovedDistractorArgument.name(): "distractor_argument",
    # DistractorAnswerWithoutInfluence.name(): "distractor_argument_2",
    # DistractorArgumentCorrectOrWrong.name(): "distractor_argument_3",
    # DistractorArgumentImportant.name(): "distractor_argument_4",
    # DistractorArgumentNotsure.name(): "distractor_argument_5",
    # DistractorArgumentNoTruthfullyAnswer.name(): "distractor_argument_6",
}

formatters_list = [formatter for formatter in FORMATTERS_TO_DUMP]


def rename_dataset_name(dataset_name: str) -> str:
    match dataset_name:
        case "truthful_qa":
            return "truthfulqa"
        case "mmlu_test":
            return "mmlu"
        case _:
            return dataset_name


class PositionalBiasDump(BaseModel):
    original_instruction: list[StrictChatMessage]
    first_model: str
    first_model_response: str
    second_model: str
    second_model_response: str
    first_judge_prompt: list[StrictChatMessage]
    # Same as the first_judge_prompt, but flipped the order of choices
    second_judge_prompt: list[StrictChatMessage]
    original_dataset: Literal["alpaca_cleaned"] = "alpaca_cleaned"
    bias_name: Literal["positional_bias"] = "positional_bias"

    @staticmethod
    def from_positional_bias(
        output: BothJudgements,
        first_model: str,
        second_model: str,
    ) -> "PositionalBiasDump":
        first_judgement: ComparisonGenerationJudged = assert_not_none(output.first_judgement)
        second_judgement: ComparisonGenerationJudged = assert_not_none(output.second_judgement)
        original_instruction = output.original_instruction_seq
        first_model_response = second_judgement.generation.a_response
        second_model_response = first_judgement.generation.b_response
        return PositionalBiasDump(
            original_instruction=[instruction.to_strict() for instruction in original_instruction],
            first_model=first_model,
            first_model_response=first_model_response,
            second_model=second_model,
            second_model_response=second_model_response,
            first_judge_prompt=[prompt.to_strict() for prompt in first_judgement.judge_prompt],
            second_judge_prompt=[prompt.to_strict() for prompt in second_judgement.judge_prompt],
        )


class StandardDatasetDump(BaseModel):
    original_question: str
    original_question_hash: str
    original_dataset: str
    unbiased_question: list[StrictChatMessage]
    biased_question: list[StrictChatMessage]
    bias_name: str
    ground_truth: MultipleChoiceAnswer
    biased_option: str

    @staticmethod
    def from_task_spec(task_spec: TaskSpec) -> "StandardDatasetDump":
        bias_messages = task_spec.messages
        assistant_on_user_side: list[StrictChatMessage] = append_assistant_preferred_to_last_user(bias_messages)

        unbiased_question: list[StrictChatMessage] = append_assistant_preferred_to_last_user(
            ZeroShotCOTUnbiasedFormatter.format_example(question=task_spec.get_data_example_obj())
        )
        biased_question = assistant_on_user_side
        bias_name: str = FORMATTERS_TO_DUMP.get(task_spec.formatter_name, task_spec.formatter_name)
        biased_option = task_spec.biased_ans
        assert biased_option is not None, "Biased option should not be None"

        return StandardDatasetDump(
            original_question=task_spec.get_data_example_obj().get_parsed_input(),
            original_question_hash=task_spec.task_hash,
            original_dataset=rename_dataset_name(task_spec.task_name),
            unbiased_question=unbiased_question,
            biased_question=biased_question,
            bias_name=bias_name,
            ground_truth=task_spec.ground_truth,  # type: ignore
            biased_option=biased_option,
        )

    @staticmethod
    def from_are_you_sure(output: OutputWithAreYouSure) -> "StandardDatasetDump":
        task_spec = output.task_spec
        bias_messages = task_spec.messages
        assistant_on_user_side: list[StrictChatMessage] = append_assistant_preferred_to_last_user(bias_messages)

        unbiased_question: list[StrictChatMessage] = append_assistant_preferred_to_last_user(
            ZeroShotCOTUnbiasedFormatter.format_example(question=task_spec.get_data_example_obj())
        )
        biased_question = assistant_on_user_side
        biased_option = f"NOT {output.first_round_inference.parsed_response}"
        assert biased_option is not None, "Biased option should not be None"

        return StandardDatasetDump(
            original_question=task_spec.get_data_example_obj().get_parsed_input(),
            original_question_hash=task_spec.task_hash,
            original_dataset=rename_dataset_name(task_spec.task_name),
            unbiased_question=unbiased_question,
            biased_question=biased_question,
            bias_name="are_you_sure",
            ground_truth=task_spec.ground_truth,  # type: ignore
            biased_option=biased_option,
        )


async def dump_data(
    dataset: str = "cot_testing",
    models: Sequence[str] = ("gpt-3.5-turbo-0613",),
    formatters: Optional[Sequence[str]] = None,
    example_cap: int = 2000,
    output_dir: str = "dataset_dumps/test",
    include_hindsight: bool = True,
    include_are_you_sure: bool = True,
    include_positional_bias: bool = True,
    hindsight_example_cap: int = 1000,
    are_you_sure_example_cap: int = 1000,
    positional_bias_samples: int = 800,
    positional_bias_first_model: str = "gpt-4",
    positional_bias_second_model: str = "gpt-3.5-turbo-0613",
    cache_dir: str = "experiments/grid_exp",
):
    """
    Dump test datasets for bias evaluation.

    Args:
        dataset: Dataset to use (e.g., "cot_testing", "mmlu_test", "truthful_qa")
        models: List of models to use for formatting
        formatters: List of formatter names to use. If None, uses all default formatters.
                   Available: suggested_answer, post_hoc, wrong_few_shot,
                   spurious_few_shot_squares, distractor_fact, distractor_argument
        example_cap: Maximum examples per dataset
        output_dir: Output directory for dumped files
        include_hindsight: Whether to include hindsight neglect bias
        include_are_you_sure: Whether to include "are you sure" bias
        include_positional_bias: Whether to include positional bias
        hindsight_example_cap: Max examples for hindsight neglect
        are_you_sure_example_cap: Max examples for "are you sure"
        positional_bias_samples: Number of samples for positional bias
        positional_bias_first_model: First model for positional bias comparison
        positional_bias_second_model: Second model for positional bias comparison
        cache_dir: Directory for caching API calls
    """
    # Convert to list if needed
    models = list(models)

    # Use provided formatters or default to all
    if formatters is None:
        selected_formatters = formatters_list
    else:
        formatters = list(formatters)
        # Map friendly names back to formatter names
        name_to_formatter = {v: k for k, v in FORMATTERS_TO_DUMP.items()}
        selected_formatters = []
        for f in formatters:
            if f in name_to_formatter:
                selected_formatters.append(name_to_formatter[f])
            elif f in FORMATTERS_TO_DUMP:
                selected_formatters.append(f)
            else:
                raise ValueError(f"Unknown formatter: {f}. Available: {list(FORMATTERS_TO_DUMP.values())}")

    print(f"Dumping data with:")
    print(f"  Dataset: {dataset}")
    print(f"  Models: {models}")
    print(f"  Formatters: {[FORMATTERS_TO_DUMP.get(f, f) for f in selected_formatters]}")
    print(f"  Example cap: {example_cap}")
    print(f"  Output dir: {output_dir}")

    tasks_to_run: Slist[TaskSpec] = Slist(
        create_stage_one_task_specs(
            dataset=dataset,
            models=models,
            formatters=selected_formatters,
            example_cap=example_cap,
            temperature=0,
            raise_after_retries=False,
            max_tokens=1000,
            n_responses_per_request=1,
        )
    )

    standard_with_hindsight = tasks_to_run

    if include_hindsight:
        hindsight_neglect = Slist(
            create_stage_one_task_specs(
                tasks=[InverseScalingTask.hindsight_neglect],
                models=models,
                formatters=[
                    ClearFewShotsThinkStepByStepCOT().name(),
                ],
                example_cap=hindsight_example_cap,
                temperature=0,
                raise_after_retries=False,
                max_tokens=1000,
                n_responses_per_request=1,
            )
        ).map(
            # rename formatter to inverse_scaling
            lambda x: x.copy_update(formatter_name="spurious_few_shot_hindsight")
        )
        standard_with_hindsight = tasks_to_run + hindsight_neglect

    stage_one_path = Path(cache_dir)
    stage_one_caller = UniversalCaller().with_model_specific_file_cache(stage_one_path, write_every_n=600)

    standard_dumps = standard_with_hindsight.map(StandardDatasetDump.from_task_spec)

    dumps = standard_dumps

    if include_are_you_sure:
        # Are you sure actually depends on the model being used
        _are_you_sure_second_round_cot: Slist[OutputWithAreYouSure] = await run_are_you_sure_multi_model_second_round_cot(
            models=models, caller=stage_one_caller, example_cap=are_you_sure_example_cap
        )
        are_you_sure_dump: Slist[StandardDatasetDump] = _are_you_sure_second_round_cot.map(
            StandardDatasetDump.from_are_you_sure
        )
        dumps = standard_dumps + are_you_sure_dump

    # Group and write files
    dumps_grouped: Slist[Group[str, Slist[StandardDatasetDump]]] = dumps.group_by(
        lambda x: f"{x.bias_name}/{x.original_dataset}_{x.bias_name}.jsonl"
    )
    for group in dumps_grouped:
        output_path = f"{output_dir}/{group.key}"
        print(f"Writing {len(group.values)} samples to {output_path}")
        write_jsonl_file_from_basemodel(output_path, group.values)

    if include_positional_bias:
        print(f"Positional bias config:")
        print(f"  First model: {positional_bias_first_model}")
        print(f"  Second model: {positional_bias_second_model}")
        print(f"  Judge models: {models}")

        pipeline: Observable[BothJudgements] = many_judge_obs(
            judge_models=models,
            caller=stage_one_caller,
            samples_to_judge=positional_bias_samples,
            first_model=positional_bias_first_model,
            second_model=positional_bias_second_model,
        ).filter(lambda x: x.first_judgement is not None and x.second_judgement is not None)
        results: Slist[BothJudgements] = await pipeline.to_slist()
        positional_bias_dumps: Slist[PositionalBiasDump] = results.map(
            lambda x: PositionalBiasDump.from_positional_bias(
                x, positional_bias_first_model, positional_bias_second_model
            )
        )
        positional_output = f"{output_dir}/positional_bias/alpaca_positional_bias.jsonl"
        print(f"Writing {len(positional_bias_dumps)} samples to {positional_output}")
        write_jsonl_file_from_basemodel(positional_output, positional_bias_dumps)

    print("Done!")


def test_parse_one_file(path: str = "dataset_dumps/test/distractor_fact/mmlu_distractor_fact.jsonl"):
    """Parse and print samples from a dumped file."""
    with open(path, "r") as f:
        for line in f.readlines():
            parsed = StandardDatasetDump.model_validate_json(line)
            print(parsed.biased_question)


def main(
    dataset: str = "cot_testing",
    models: Sequence[str] = ("gpt-3.5-turbo-0613",),
    formatters: Optional[Sequence[str]] = None,
    example_cap: int = 2000,
    output_dir: str = "dataset_dumps/test",
    include_hindsight: bool = True,
    include_are_you_sure: bool = True,
    include_positional_bias: bool = True,
    hindsight_example_cap: int = 1000,
    are_you_sure_example_cap: int = 1000,
    positional_bias_samples: int = 800,
    positional_bias_first_model: str = "gpt-4",
    positional_bias_second_model: str = "gpt-3.5-turbo-0613",
    cache_dir: str = "experiments/grid_exp",
    test_parse: bool = False,
):
    """
    Dump test datasets for bias evaluation.

    Note: For list parameters (models, formatters), use Python list syntax with quotes:
        --models='["llama-3"]'
        --formatters='["suggested_answer", "distractor_fact"]'

    Examples:
        # Dump all biases with default settings
        python scripts/dump_datasets_for_release.py

        # Dump only suggested_answer and distractor_fact biases
        python scripts/dump_datasets_for_release.py --formatters='["suggested_answer", "distractor_fact"]'

        # Use a different model
        python scripts/dump_datasets_for_release.py --models='["llama-3"]'

        # Multiple models
        python scripts/dump_datasets_for_release.py --models='["gpt-3.5-turbo-0613", "gpt-4"]'

        # Dump to a custom directory with fewer examples
        python scripts/dump_datasets_for_release.py --output_dir=my_output --example_cap=100

        # Skip special biases (hindsight, are_you_sure, positional)
        python scripts/dump_datasets_for_release.py --include_hindsight=False --include_are_you_sure=False --include_positional_bias=False

        # Configure positional bias models
        python scripts/dump_datasets_for_release.py --positional_bias_first_model=llama-3 --positional_bias_second_model=mistral

    Args:
        dataset: Dataset to use (e.g., "cot_testing", "mmlu_test", "truthful_qa")
        models: List of models to use for formatting and as judges for positional bias.
                Use list syntax: --models='["model1", "model2"]'
        formatters: List of formatter names to use. If None, uses all default formatters.
                   Use list syntax: --formatters='["formatter1", "formatter2"]'
                   Available: suggested_answer, post_hoc, wrong_few_shot,
                   spurious_few_shot_squares, distractor_fact, distractor_argument
        example_cap: Maximum examples per dataset
        output_dir: Output directory for dumped files
        include_hindsight: Whether to include hindsight neglect bias
        include_are_you_sure: Whether to include "are you sure" bias
        include_positional_bias: Whether to include positional bias
        hindsight_example_cap: Max examples for hindsight neglect
        are_you_sure_example_cap: Max examples for "are you sure"
        positional_bias_samples: Number of samples for positional bias
        positional_bias_first_model: First model for positional bias comparison
        positional_bias_second_model: Second model for positional bias comparison
        cache_dir: Directory for caching API calls
        test_parse: If True, only run test_parse_one_file on the output
    """
    import asyncio

    if test_parse:
        test_parse_one_file(f"{output_dir}/distractor_fact/mmlu_distractor_fact.jsonl")
        return

    asyncio.run(
        dump_data(
            dataset=dataset,
            models=models,
            formatters=formatters,
            example_cap=example_cap,
            output_dir=output_dir,
            include_hindsight=include_hindsight,
            include_are_you_sure=include_are_you_sure,
            include_positional_bias=include_positional_bias,
            hindsight_example_cap=hindsight_example_cap,
            are_you_sure_example_cap=are_you_sure_example_cap,
            positional_bias_samples=positional_bias_samples,
            positional_bias_first_model=positional_bias_first_model,
            positional_bias_second_model=positional_bias_second_model,
            cache_dir=cache_dir,
        )
    )


if __name__ == "__main__":
    fire.Fire(main)
