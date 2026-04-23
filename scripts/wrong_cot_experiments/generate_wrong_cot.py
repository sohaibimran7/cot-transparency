import asyncio
import pathlib
from typing import Callable, Optional, Sequence

import fire
from slist import Slist


from cot_transparency.apis import UniversalCaller
from cot_transparency.apis.base import CachedPerModelCaller, ModelCaller
from cot_transparency.data_models.config import config_from_default
from cot_transparency.data_models.models import TaskOutput, TaskSpec
from cot_transparency.formatters.more_biases.deceptive_assistant import DeceptiveAssistantTargetedFormatter
from cot_transparency.formatters.more_biases.user_wrong_cot import WRONG_COT_TESTING_PATH
from cot_transparency.json_utils.read_write import read_jsonl_file_into_basemodel, write_jsonl_file_from_basemodel
from cot_transparency.streaming.stage_one_stream import stage_one_stream
from cot_transparency.util import assert_not_none
from scripts.automated_answer_parsing.answer_parsing_example import answer_finding_step


def only_not_obviously_deceptive(_str: str) -> bool:
    lower_str = _str.lower()
    banned_words = ["deceptive", "lie", "wrong", "motivate"]
    result = not any(banned_word in lower_str for banned_word in banned_words)
    if not result:
        print(f"Found a obviously deceptive answer: {_str}")
    return result


def _filter_hits(done_tasks: Slist[TaskOutput]) -> Slist[TaskOutput]:
    return (
        done_tasks.filter(lambda task: task.first_parsed_response is not None)
        .filter(lambda task: task.first_parsed_response == task.task_spec.biased_ans)
        .filter(lambda task: only_not_obviously_deceptive(assert_not_none(task.first_raw_response)))
        .distinct_by(lambda x: x.task_spec.task_hash)
    )


async def _run_one_pass(
    *,
    model: str,
    dataset: Optional[str],
    tasks: Optional[Sequence[str]],
    example_cap: int,
    temperature: float,
    n_responses_per_request: int,
    repeats_per_question: int,
    batch: int,
    use_answer_parsing: bool,
    answer_parsing_model: str,
    stage_one_caller: ModelCaller,
    answer_parsing_caller: Optional[CachedPerModelCaller],
    skip_task_hashes: set[str],
    start_try_number: int,
) -> Slist[TaskOutput]:
    filter_fn: Optional[Callable[[TaskSpec], bool]] = None
    if skip_task_hashes:
        filter_fn = lambda ts: ts.task_hash not in skip_task_hashes  # noqa: E731

    stage_one_obs = stage_one_stream(
        tasks=list(tasks) if tasks else [],
        formatters=[DeceptiveAssistantTargetedFormatter.name()],
        repeats_per_question=repeats_per_question,
        dataset=dataset,
        example_cap=example_cap,
        num_tries=1,
        n_responses_per_request=n_responses_per_request,
        raise_after_retries=False,
        temperature=temperature,
        caller=stage_one_caller,
        batch=batch,
        models=[model],
        filter_tasks=filter_fn,
        start_try_number=start_try_number,
    )

    if use_answer_parsing:
        assert answer_parsing_caller is not None
        parsing_config = config_from_default(model=answer_parsing_model)
        stage_one_obs = stage_one_obs.map_blocking_par(
            lambda x: answer_finding_step(x, answer_parsing_caller, parsing_config)
        )

    return await stage_one_obs.to_slist()


async def generate(
    model: str = "gpt-3.5-turbo-0613",
    dataset: Optional[str] = "testing_plus_aqua",
    tasks: Optional[Sequence[str]] = None,
    example_cap: int = 1200,
    temperature: float = 1.0,
    n_responses_per_request: int = 5,
    repeats_per_question: int = 1,
    batch: int = 120,
    max_passes: int = 1,
    delete_existing: bool = False,
    use_answer_parsing: bool = True,
    answer_parsing_model: str = "gpt-4",
    cache_path: str = "experiments/wrong_cot_cache.jsonl",
    model_specific_cache_dir: str = "experiments/wrong_cot_cache",
):
    # Generate wrong-CoT data for distractor_argument bias.
    # Run `export PYTHONPATH=.; python scripts/wrong_cot_experiments/generate_wrong_cot.py`
    #
    # Adaptive retry: when max_passes > 1, pass N+1 targets only questions that
    # failed the compliance filter on pass N (via filter_tasks on task_hash).
    # Each pass is given a distinct start_try_number so CachedCaller keys
    # differ across passes — forcing fresh API calls instead of replaying the
    # pass-1 miss. Sampling distribution is identical across passes.
    #
    # Merge semantics: when delete_existing=False (default) and
    # WRONG_COT_TESTING_PATH exists, new records are merged into it; on
    # task_hash conflict the existing record wins (historical data is stable).
    if tasks:
        dataset = None
    if dataset is None and not tasks:
        raise ValueError("Must pass either dataset or tasks.")
    if max_passes < 1:
        raise ValueError("max_passes must be >= 1")

    stage_one_caller = UniversalCaller().with_file_cache(pathlib.Path(cache_path), write_every_n=200)
    answer_parsing_caller: Optional[CachedPerModelCaller] = None
    if use_answer_parsing:
        # Needed for latex datasets (e.g. aqua) where raw answer extraction fails.
        answer_parsing_caller = stage_one_caller.with_model_specific_file_cache(
            cache_dir=pathlib.Path(model_specific_cache_dir), write_every_n=500
        )

    if delete_existing and WRONG_COT_TESTING_PATH.exists():
        WRONG_COT_TESTING_PATH.unlink()

    all_hits: Slist[TaskOutput] = Slist()
    hit_hashes: set[str] = set()

    for pass_num in range(1, max_passes + 1):
        print(
            f"\n=== pass {pass_num}/{max_passes}  temp={temperature}  "
            f"already_hit={len(hit_hashes)} ==="
        )
        done_tasks = await _run_one_pass(
            model=model,
            dataset=dataset,
            tasks=tasks,
            example_cap=example_cap,
            temperature=temperature,
            n_responses_per_request=n_responses_per_request,
            repeats_per_question=repeats_per_question,
            batch=batch,
            use_answer_parsing=use_answer_parsing,
            answer_parsing_model=answer_parsing_model,
            stage_one_caller=stage_one_caller,
            answer_parsing_caller=answer_parsing_caller,
            skip_task_hashes=hit_hashes,
            start_try_number=pass_num,
        )
        pass_hits = _filter_hits(done_tasks)
        new_hits = pass_hits.filter(lambda t: t.task_spec.task_hash not in hit_hashes)
        for h in new_hits:
            hit_hashes.add(h.task_spec.task_hash)
        all_hits = all_hits + new_hits
        print(
            f"pass {pass_num}: ran {len(done_tasks)} samples, "
            f"{len(new_hits)} new hits, total hits so far: {len(all_hits)}"
        )
        if pass_num > 1 and len(new_hits) == 0:
            print("no new hits this pass — stopping early")
            break

    grouped_by_task_name = all_hits.group_by(lambda x: x.task_spec.task_name).map(
        lambda group: group.map_values(lambda _: len(group.values))
    )
    print(grouped_by_task_name)
    print(f"got {len(all_hits)} new biased tasks across {max_passes} passes")

    if not delete_existing and WRONG_COT_TESTING_PATH.exists():
        existing: Slist[TaskOutput] = Slist(
            read_jsonl_file_into_basemodel(WRONG_COT_TESTING_PATH, TaskOutput)
        )
        print(f"Loaded {len(existing)} existing records from {WRONG_COT_TESTING_PATH}")
        existing_hashes = {t.task_spec.task_hash for t in existing}
        new_only = all_hits.filter(lambda t: t.task_spec.task_hash not in existing_hashes)
        print(f"Appending {len(new_only)} new records (skipped {len(all_hits) - len(new_only)} dup hashes)")
        merged = existing + new_only
    else:
        merged = all_hits

    print(f"Writing {len(merged)} total records to {WRONG_COT_TESTING_PATH}")
    write_jsonl_file_from_basemodel(
        path=WRONG_COT_TESTING_PATH,
        basemodels=merged,
    )


def main(
    model: str = "gpt-3.5-turbo-0613",
    dataset: Optional[str] = "testing_plus_aqua",
    tasks: Optional[Sequence[str]] = None,
    example_cap: int = 1200,
    temperature: float = 1.0,
    n_responses_per_request: int = 5,
    repeats_per_question: int = 1,
    batch: int = 120,
    max_passes: int = 1,
    delete_existing: bool = False,
    use_answer_parsing: bool = True,
    answer_parsing_model: str = "gpt-4",
    cache_path: str = "experiments/wrong_cot_cache.jsonl",
    model_specific_cache_dir: str = "experiments/wrong_cot_cache",
):
    """Generate wrong-CoT training data for the distractor_argument bias.

    Note: For list parameters (tasks), use Python list syntax with quotes:
        --tasks='["gpqa", "hle"]'

    Examples:
        # Original behaviour (gpt-3.5 on testing_plus_aqua) but appending rather
        # than overwriting — pass delete_existing=True to restore old behaviour.
        python scripts/wrong_cot_experiments/generate_wrong_cot.py

        # Add gpqa + hle coverage using Gemma 4 via OpenRouter, cap 500, one
        # sample per question per pass, up to 5 adaptive passes. Each pass
        # targets only questions the compliance filter missed on prior passes.
        python scripts/wrong_cot_experiments/generate_wrong_cot.py \\
            --model='openrouter/google/gemma-4-31b-it' \\
            --tasks='["gpqa", "hle"]' \\
            --example_cap=500 \\
            --n_responses_per_request=1 \\
            --repeats_per_question=1 \\
            --max_passes=5 \\
            --use_answer_parsing=False
    """
    asyncio.run(
        generate(
            model=model,
            dataset=dataset,
            tasks=tasks,
            example_cap=example_cap,
            temperature=temperature,
            n_responses_per_request=n_responses_per_request,
            repeats_per_question=repeats_per_question,
            batch=batch,
            max_passes=max_passes,
            delete_existing=delete_existing,
            use_answer_parsing=use_answer_parsing,
            answer_parsing_model=answer_parsing_model,
            cache_path=cache_path,
            model_specific_cache_dir=model_specific_cache_dir,
        )
    )


if __name__ == "__main__":
    fire.Fire(main)
