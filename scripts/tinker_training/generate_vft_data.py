"""
Generate VFT (Verbalization Fine-Tuning) training data from test bias files.

Extends the BCT data generation pipeline with an LLM editing step:
1. Sample completions on BOTH unbiased and biased prompts
2. Classify each sample into 3 categories:
   - Unfaithful switch: answer changes, biased CoT doesn't verbalize -> LLM edits unbiased CoT
   - Faithful switch: answer changes, biased CoT verbalizes -> keep biased completion as-is
   - Non-switch: same answer -> use unbiased completion with biased prompt (same as BCT)
3. For unfaithful switches, call Claude Sonnet to minimally edit the unbiased CoT
   to insert acknowledgment of the bias cue.

Usage:
    python scripts/tinker_training/generate_vft_data.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --datasets truthfulqa mmlu \
        --bias suggested_answer \
        --limits 817 1183 \
        --output-name vft-suggested-answer
"""

import argparse
import json
import random
from pathlib import Path
from typing import Any

import sys

# Add project root and eval suite to path for imports
_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "sycophancy_eval_inspect"))

from dotenv import load_dotenv

load_dotenv(_PROJECT_ROOT / ".env")

import asyncio  # noqa: E402

from openai import AsyncOpenAI  # noqa: E402
from tqdm import tqdm  # noqa: E402

from cot_transparency.apis.tinker.inference import TinkerSamplingClient, SamplingConfig  # noqa: E402
from mcq.answer_parser import cot_answer_parser, fallback_answer_parser
from mcq.scorer import _BIAS_ACK_PROMPTS

MessageDict = dict[str, str]
SampleDict = dict[str, Any]


# ── CoT correction prompt ───────────────────────────────────────────────────

_COT_CORRECTION_PROMPT = """\
You are editing a model's chain-of-thought response to a multiple-choice question.

The question contained the following biasing cue:
"{biasing_text}"

This cue suggests answer ({biased_option}).

Below is the model's reasoning when it answered the question WITHOUT the biasing cue present. \
The model arrived at answer ({unbiased_answer}).

<unbiased_reasoning>
{unbiased_completion}
</unbiased_reasoning>

Your task: Make MINIMAL edits to this reasoning so that it:
1. Acknowledges the biasing cue naturally (e.g., references the suggestion, hint, or prompt feature)
2. Arrives at answer ({biased_option}) instead of ({unbiased_answer})
3. Explains why the cue influenced the answer choice

Guidelines:
- Use the unbiased reasoning as a template. Make minimal edits.
- Attribute giving the biased answer to the cue/suggestion.
- Use natural language to refer to the cue — words like "suggestion", "hint", "the note", \
"the comment" are fine. Avoid the word "bias".
- Keep the same format, ending with "Therefore, the best answer is: ({biased_option})."
- Do NOT add meta-commentary about editing or the task itself.

Output ONLY the edited reasoning, nothing else."""


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_jsonl(file_path: Path, limit: int | None = None) -> list[SampleDict]:
    """Load samples from a jsonl file."""
    samples = []
    with open(file_path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            samples.append(json.loads(line))
    return samples


def save_jsonl(samples: list[SampleDict], file_path: Path) -> None:
    """Save samples to a jsonl file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    print(f"Saved {len(samples)} samples to {file_path}")


def sanitize_model_name(model: str) -> str:
    """Convert model name to filesystem-safe directory name."""
    name = model.split("/")[-1].lower()
    return name.replace(".", "-")


def parse_answer(completion: str) -> str | None:
    """Parse answer from completion, trying strict then fallback."""
    ans = cot_answer_parser(completion)
    if ans is None:
        ans = fallback_answer_parser(completion)
    return ans


def extract_biasing_text(unbiased_content: str, biased_content: str) -> str:
    """Extract the biasing text by diffing unbiased and biased question content."""
    # The biasing text is inserted before the instruction paragraph.
    # Find the common prefix and suffix to isolate the inserted text.
    # For suggested_answer, it's like "I keep coming back to C.\n"
    # Simple approach: find where they diverge
    min_len = min(len(unbiased_content), len(biased_content))
    prefix_end = 0
    for i in range(min_len):
        if unbiased_content[i] != biased_content[i]:
            prefix_end = i
            break

    # Find common suffix
    suffix_start_b = len(biased_content)
    for i in range(1, min_len + 1):
        if unbiased_content[-i] != biased_content[-i]:
            suffix_start_b = len(biased_content) - i + 1
            break

    biasing_text = biased_content[prefix_end:suffix_start_b].strip()
    return biasing_text


# ── Tinker batch sampling ────────────────────────────────────────────────────

def sample_batch(
    client: TinkerSamplingClient,
    prompts: list[list[MessageDict]],
) -> list[str]:
    """Sample completions for a batch of prompts concurrently."""
    from tinker import types

    completions: list[str] = []
    futures = []
    for prompt_msgs in prompts:
        msg_dicts = [{"role": m["role"], "content": m["content"]} for m in prompt_msgs]
        prompt_input = client.renderer.build_generation_prompt(msg_dicts)
        stop_sequences = client.renderer.get_stop_sequences()

        sampling_params = types.SamplingParams(
            max_tokens=client.config.max_tokens,
            temperature=client.config.temperature,
            top_p=client.config.top_p,
            stop=stop_sequences,
        )

        future = client.sampling_client.sample(
            prompt=prompt_input,
            sampling_params=sampling_params,
            num_samples=1,
        )
        futures.append(future)

    for future in futures:
        try:
            result = future.result()
            if result.sequences:
                tokens = list(result.sequences[0].tokens)
                parsed_msg, _ = client.renderer.parse_response(tokens)
                text = parsed_msg.get("content", "") if parsed_msg else client.tokenizer.decode(tokens)
                completions.append(text)
            else:
                completions.append("")
        except Exception as e:
            print(f"\nError: {e}")
            completions.append("")

    return completions


def load_checkpoint(checkpoint_file: Path) -> tuple[int, list[dict]]:
    """Load checkpoint if exists. Returns (start_index, records_so_far)."""
    if not checkpoint_file.exists():
        return 0, []
    records = []
    with open(checkpoint_file) as f:
        for line in f:
            records.append(json.loads(line))
    print(f"Resuming from checkpoint: {len(records)} records already done")
    return len(records), records


def save_checkpoint(checkpoint_file: Path, records: list[dict], start_idx: int = 0) -> None:
    """Save records to checkpoint file."""
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if start_idx > 0 else "w"
    with open(checkpoint_file, mode) as f:
        for record in records[start_idx:]:
            f.write(json.dumps(record) + "\n")


# ── Verbalization judge ──────────────────────────────────────────────────────

async def _judge_one(
    client: AsyncOpenAI,
    completion: str,
    biasing_text: str,
    bias_type: str,
    model: str,
    semaphore: asyncio.Semaphore,
) -> bool:
    if not completion or not biasing_text:
        return False
    template = _BIAS_ACK_PROMPTS.get(bias_type, _BIAS_ACK_PROMPTS["suggested_answer"])
    prompt = template.format(biasing_text=biasing_text, completion=completion)
    async with semaphore:
        try:
            resp = await client.chat.completions.create(
                model=model,
                max_completion_tokens=8192,
                messages=[{"role": "user", "content": prompt}],
            )
            answer = (resp.choices[0].message.content or "").strip().upper()
            return answer.startswith("YES")
        except Exception as e:
            print(f"  Judge error: {e}")
            return False


def judge_verbalization_batch(
    client: AsyncOpenAI,
    completions: list[str],
    biasing_texts: list[str],
    bias_type: str = "suggested_answer",
    model: str = "gpt-5-nano",
    max_concurrent: int = 50,
) -> list[bool]:
    """Judge whether each completion verbalizes the bias cue (concurrent)."""
    async def _run_all():
        semaphore = asyncio.Semaphore(max_concurrent)
        tasks = [
            _judge_one(client, comp, bt, bias_type, model, semaphore)
            for comp, bt in zip(completions, biasing_texts)
        ]
        return await asyncio.gather(*tasks)
    return asyncio.run(_run_all())


# ── CoT correction via LLM ──────────────────────────────────────────────────

async def _correct_one(
    client: AsyncOpenAI,
    item: dict,
    model: str,
    semaphore: asyncio.Semaphore,
) -> str:
    prompt = _COT_CORRECTION_PROMPT.format(**item)
    async with semaphore:
        try:
            resp = await client.chat.completions.create(
                model=model,
                max_completion_tokens=8192,
                messages=[{"role": "user", "content": prompt}],
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            print(f"  Correction error: {e}")
            return item["unbiased_completion"]


def correct_cot_batch(
    client: AsyncOpenAI,
    items: list[dict],
    model: str = "gpt-5-nano",
    max_concurrent: int = 50,
) -> list[str]:
    """Edit unbiased CoTs to add verbalization (concurrent)."""
    async def _run_all():
        semaphore = asyncio.Semaphore(max_concurrent)
        tasks = [_correct_one(client, item, model, semaphore) for item in items]
        return await asyncio.gather(*tasks)
    return asyncio.run(_run_all())


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate VFT training data from test bias files"
    )
    parser.add_argument("--model", required=True, help="Model name for Tinker sampling")
    parser.add_argument("--datasets", nargs="+", required=True, help="Dataset names")
    parser.add_argument("--bias", required=True, help="Bias type (e.g., suggested_answer)")
    parser.add_argument("--limits", nargs="+", type=int, default=None)
    parser.add_argument("--output-name", required=True, help="Output directory name")
    parser.add_argument("--base-dir", default="dataset_dumps")
    parser.add_argument("--seed", default="42", help="Seed identifier for dataset")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature for both unbiased and biased (paper uses 1.0)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--checkpoint", default=None, help="Model checkpoint to load")
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument("--correction-model", default="gpt-5-nano",
                        help="LLM for verbalization judge and CoT correction")
    parser.add_argument("--skip-sampling", action="store_true",
                        help="Skip Tinker sampling, use existing checkpoint")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    test_dir = base_dir / "test" / args.bias

    if args.limits and len(args.limits) != len(args.datasets):
        parser.error("--limits must have same length as --datasets")

    # Load samples
    all_samples = []
    for i, dataset in enumerate(args.datasets):
        file_path = test_dir / f"{dataset}_{args.bias}.jsonl"
        limit = args.limits[i] if args.limits else None
        print(f"Loading {dataset} from {file_path}")
        samples = load_jsonl(file_path, limit=limit)
        print(f"  Loaded {len(samples)} samples")
        all_samples.extend(samples)

    print(f"\nTotal samples: {len(all_samples)}")

    random.seed(args.shuffle_seed)
    random.shuffle(all_samples)

    unbiased_prompts = [s["unbiased_question"] for s in all_samples]
    biased_prompts = [s["biased_question"] for s in all_samples]
    biased_options = [s["biased_option"] for s in all_samples]

    # Extract biasing text for each sample
    biasing_texts = [
        extract_biasing_text(
            s["unbiased_question"][0]["content"],
            s["biased_question"][0]["content"],
        )
        for s in all_samples
    ]

    # Setup output
    bias_dir_name = args.bias.replace("_", "-")
    train_dir = base_dir / f"train-from-test-{'-'.join(args.datasets)}" / bias_dir_name / "vft"
    train_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nModel: {args.model}")
    print(f"Output dir: {train_dir}")

    # ── Phase 1: Sample from Tinker on unbiased AND biased prompts ──────────

    checkpoint_dir = train_dir / ".checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    unbiased_ckpt = checkpoint_dir / "unbiased_completions.jsonl"
    biased_ckpt = checkpoint_dir / "biased_completions.jsonl"

    total = len(all_samples)

    if args.skip_sampling:
        print("\nSkipping Tinker sampling, loading from checkpoints...")
        _, unbiased_records = load_checkpoint(unbiased_ckpt)
        _, biased_records = load_checkpoint(biased_ckpt)
        unbiased_completions = [r["completion"] for r in unbiased_records]
        biased_completions = [r["completion"] for r in biased_records]
        assert len(unbiased_completions) == total, (
            f"Expected {total} unbiased completions, got {len(unbiased_completions)}"
        )
        assert len(biased_completions) == total, (
            f"Expected {total} biased completions, got {len(biased_completions)}"
        )
    else:
        client = TinkerSamplingClient(
            model=args.model,
            checkpoint=args.checkpoint,
            config=SamplingConfig(
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
        )
        print("\nInitializing Tinker client...")
        client.setup()

        # Sample unbiased completions
        start_idx, unbiased_records = (0, []) if args.fresh else load_checkpoint(unbiased_ckpt)
        unbiased_completions = [r["completion"] for r in unbiased_records]

        if start_idx < total:
            print(f"\n── Sampling unbiased completions ({start_idx}/{total} done) ──")
            total_batches = (total + args.batch_size - 1) // args.batch_size
            start_batch = start_idx // args.batch_size

            pbar = tqdm(range(start_batch, total_batches), desc="Unbiased",
                        initial=start_batch, total=total_batches)
            for batch_idx in pbar:
                bs = batch_idx * args.batch_size
                be = min(bs + args.batch_size, total)
                batch = sample_batch(client, unbiased_prompts[bs:be])
                unbiased_completions.extend(batch)
                unbiased_records.extend([{"completion": c} for c in batch])

                if (batch_idx + 1) % args.save_every == 0 or batch_idx == total_batches - 1:
                    save_checkpoint(unbiased_ckpt, unbiased_records)
                    pbar.write(f"Checkpoint: {len(unbiased_completions)}/{total}")

        # Sample biased completions (same temperature)
        start_idx, biased_records = (0, []) if args.fresh else load_checkpoint(biased_ckpt)
        biased_completions = [r["completion"] for r in biased_records]

        if start_idx < total:
            print(f"\n── Sampling biased completions ({start_idx}/{total} done) ──")
            total_batches = (total + args.batch_size - 1) // args.batch_size
            start_batch = start_idx // args.batch_size

            pbar = tqdm(range(start_batch, total_batches), desc="Biased",
                        initial=start_batch, total=total_batches)
            for batch_idx in pbar:
                bs = batch_idx * args.batch_size
                be = min(bs + args.batch_size, total)
                batch = sample_batch(client, biased_prompts[bs:be])
                biased_completions.extend(batch)
                biased_records.extend([{"completion": c} for c in batch])

                if (batch_idx + 1) % args.save_every == 0 or batch_idx == total_batches - 1:
                    save_checkpoint(biased_ckpt, biased_records)
                    pbar.write(f"Checkpoint: {len(biased_completions)}/{total}")

    # ── Phase 2: Classify samples ───────────────────────────────────────────

    print(f"\n── Classifying {total} samples ──")

    # Parse answers
    unbiased_answers = [parse_answer(c) for c in unbiased_completions]
    biased_answers = [parse_answer(c) for c in biased_completions]

    # Classify: switch vs non-switch
    switches = []  # indices where answer changed
    non_switches = []  # indices where answer stayed the same
    unparseable = []  # indices where parsing failed

    for i in range(total):
        ua, ba = unbiased_answers[i], biased_answers[i]
        if ua is None or ba is None:
            unparseable.append(i)
        elif ua != ba:
            switches.append(i)
        else:
            non_switches.append(i)

    print(f"  Switches: {len(switches)}")
    print(f"  Non-switches: {len(non_switches)}")
    print(f"  Unparseable: {len(unparseable)}")

    # Judge verbalization for switches using Claude
    async_client = AsyncOpenAI()

    faithful_switches = []
    unfaithful_switches = []

    if switches:
        print(f"\n── Judging verbalization for {len(switches)} switches ──")
        switch_completions = [biased_completions[i] for i in switches]
        switch_biasing = [biasing_texts[i] for i in switches]

        verbalizes = judge_verbalization_batch(
            async_client, switch_completions, switch_biasing,
            bias_type=args.bias, model=args.correction_model,
        )

        for idx, verb in zip(switches, verbalizes):
            if verb:
                faithful_switches.append(idx)
            else:
                unfaithful_switches.append(idx)

    print(f"\n  Faithful switches (verbalized): {len(faithful_switches)}")
    print(f"  Unfaithful switches (needs editing): {len(unfaithful_switches)}")

    # ── Phase 3: Correct unfaithful switches via Claude ─────────────────────

    corrected_completions: dict[int, str] = {}

    if unfaithful_switches:
        print(f"\n── Correcting {len(unfaithful_switches)} unfaithful CoTs ──")
        items = [
            {
                "biasing_text": biasing_texts[i],
                "biased_option": biased_options[i],
                "unbiased_answer": unbiased_answers[i],
                "unbiased_completion": unbiased_completions[i],
            }
            for i in unfaithful_switches
        ]

        corrected = correct_cot_batch(
            async_client, items, model=args.correction_model,
        )

        for idx, corrected_text in zip(unfaithful_switches, corrected):
            corrected_completions[idx] = corrected_text

    # ── Phase 4: Assemble VFT training data ─────────────────────────────────

    print(f"\n── Assembling VFT training data ──")

    vft_samples = []
    categories = {"non_switch": 0, "faithful_switch": 0, "unfaithful_switch": 0, "skipped": 0}

    for i in range(total):
        if i in unparseable:
            categories["skipped"] += 1
            continue

        if i in non_switches:
            # Non-switch: biased prompt + unbiased completion (same as BCT)
            completion = unbiased_completions[i]
            category = "non_switch"
        elif i in faithful_switches:
            # Faithful switch: biased prompt + biased completion (already verbalizes)
            completion = biased_completions[i]
            category = "faithful_switch"
        elif i in unfaithful_switches:
            # Unfaithful switch: biased prompt + corrected completion
            completion = corrected_completions.get(i, unbiased_completions[i])
            category = "unfaithful_switch"
        else:
            categories["skipped"] += 1
            continue

        if not completion:
            categories["skipped"] += 1
            continue

        categories[category] += 1
        sample = {
            "messages": biased_prompts[i] + [{"role": "assistant", "content": completion}],
        }
        vft_samples.append(sample)

    print(f"\n  Category breakdown:")
    for cat, count in categories.items():
        print(f"    {cat}: {count}")
    print(f"  Total VFT samples: {len(vft_samples)}")

    # Save
    save_jsonl(vft_samples, train_dir / "vft_cot.jsonl")

    # Also save a metadata summary
    summary = {
        "model": args.model,
        "datasets": args.datasets,
        "bias": args.bias,
        "total_input": total,
        "categories": categories,
        "total_vft_samples": len(vft_samples),
        "temperature": args.temperature,
        "correction_model": args.correction_model,
    }
    with open(train_dir / "vft_metadata.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved metadata to {train_dir / 'vft_metadata.json'}")

    # Clean up checkpoints
    print("\nDone!")


if __name__ == "__main__":
    main()
