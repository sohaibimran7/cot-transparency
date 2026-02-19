"""
Test script for RL consistency training.

Uses suggested_answer prompts from MMLU and TruthfulQA test sets.

Usage:
    python scripts/tinker_training/test_rl_training.py
    python scripts/tinker_training/test_rl_training.py --n_samples 10 --dry_run
"""

import asyncio
import argparse
import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from cot_transparency.apis.tinker.rl_training import (
    RLConfig,
    RLTrainer,
    RateEstimationConfig,
    TrainingSamplingConfig,
    TrainingLoopConfig,
    GenerationConfig,
)
from cot_transparency.apis.tinker.common import CheckpointConfig, AdamConfig, LoRAConfig
from sycophancy_eval_inspect.mcq.answer_parser import cot_answer_parser


def load_samples(file_path: Path, n_samples: int) -> list[dict]:
    """Load samples from JSONL file."""
    samples = []
    with open(file_path) as f:
        for line in f:
            samples.append(json.loads(line))
            if len(samples) >= n_samples:
                break
    return samples


def unbiased_perturbation(situation: dict) -> dict:
    """Neutral prompt without suggested answer."""
    return {"messages": situation["unbiased_question"]}


def biased_perturbation(situation: dict) -> dict:
    """Biased prompt with suggested answer."""
    return {"messages": situation["biased_question"]}


def trait_classifier(response: str, situation: dict) -> float:
    """
    Returns 1.0 if model follows the biased suggestion, 0.0 otherwise.

    For consistency training, we want to reduce the rate of following suggestions
    to match the unbiased baseline.
    """
    answer = cot_answer_parser(response)
    biased_option = situation.get("biased_option", "")
    return 1.0 if answer == biased_option else 0.0


def main():
    parser = argparse.ArgumentParser(description="Test RL consistency training")
    parser.add_argument("--n_samples", type=int, default=50, help="Samples per dataset")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--dry_run", action="store_true", help="Just load data, don't train")
    parser.add_argument("--experiment_name", type=str, default="rl_test")
    parser.add_argument("--run_name", type=str, default="suggested_answer")
    parser.add_argument("--ref_perturbations", type=int, nargs="+", default=[0], help="Perturbation indices for reference rate")
    parser.add_argument("--train_perturbations", type=int, nargs="+", default=[1], help="Perturbation indices for training")
    parser.add_argument("--control", action="store_true", help="Control run: use unbiased perturbation for both ref and train")
    parser.add_argument("--resume_from", type=str, default=None, help="Tinker checkpoint path to load before training (tinker://...)")
    args = parser.parse_args()

    # Load samples from both datasets
    mmlu_path = Path("dataset_dumps/test/suggested_answer/mmlu_suggested_answer.jsonl")
    tqa_path = Path("dataset_dumps/test/suggested_answer/truthfulqa_suggested_answer.jsonl")

    situations = []
    if mmlu_path.exists():
        situations.extend(load_samples(mmlu_path, args.n_samples))
        print(f"Loaded {min(args.n_samples, len(situations))} MMLU samples")
    else:
        print(f"Warning: {mmlu_path} not found")

    if tqa_path.exists():
        tqa_samples = load_samples(tqa_path, args.n_samples)
        situations.extend(tqa_samples)
        print(f"Loaded {len(tqa_samples)} TruthfulQA samples")
    else:
        print(f"Warning: {tqa_path} not found")

    print(f"Total situations: {len(situations)}")

    if args.dry_run:
        print("\n=== Dry run: checking data format ===")
        sample = situations[0]
        print(f"Keys: {sample.keys()}")
        print(f"Unbiased prompt: {sample['unbiased_question'][0]['content'][:200]}...")
        print(f"Biased prompt: {sample['biased_question'][0]['content'][:200]}...")
        print(f"Biased option: {sample['biased_option']}")
        print(f"Ground truth: {sample['ground_truth']}")

        # Test perturbation functions
        print("\n=== Testing perturbation functions ===")
        unbiased = unbiased_perturbation(sample)
        biased = biased_perturbation(sample)
        print(f"Unbiased messages: {len(unbiased['messages'])} message(s)")
        print(f"Biased messages: {len(biased['messages'])} message(s)")

        # Test classifier
        print("\n=== Testing trait classifier ===")
        test_responses = [
            f"Therefore, the best answer is: ({sample['biased_option']}).",
            f"Therefore, the best answer is: ({sample['ground_truth']}).",
            "I'm not sure about the answer.",
        ]
        for resp in test_responses:
            score = trait_classifier(resp, sample)
            print(f"  '{resp[:50]}...' -> {score}")

        return

    # Configure training
    config = RLConfig(
        experiment_name=args.experiment_name,
        run_name=args.run_name,
        model=args.model,
        lora=LoRAConfig(rank=8),  # Lower rank like BCT
        optimizer=AdamConfig(
            learning_rate=None,  # Use recommended LR
            lr_schedule="constant",
        ),
        reference_rate=RateEstimationConfig(
            perturbation_indices=args.ref_perturbations,
            n_samples=128,
        ),
        training=TrainingSamplingConfig(
            perturbation_indices=args.train_perturbations,
            n_samples_for_rate=128,
            n_samples_for_gradient=128,
        ),
        loop=TrainingLoopConfig(
            situations_per_group=1,
            gradient_accumulation_steps=1,
            refresh_policy_every_n_steps=1,  # Refresh every step
            n_epochs=1,
        ),
        generation=GenerationConfig(
            max_new_tokens=8192,  # High limit - model stops naturally at EOS, no truncation risk
            temperature=1.0,  # Higher temperature for more exploration
        ),
        checkpoint=CheckpointConfig(
            save_every_n_steps=50,
            save_state=False,
        ),
        kl_coef=0.05,
        loss_fn="ppo",
        log_base_dir="logs",
    )

    # Set up perturbation functions
    if args.control:
        # Control: both perturbations are unbiased (but separate samples)
        perturbation_fns = [unbiased_perturbation, unbiased_perturbation]
        pert_desc = "unbiased (ref), unbiased (train) [CONTROL]"
    else:
        perturbation_fns = [unbiased_perturbation, biased_perturbation]
        pert_desc = "unbiased (ref), biased (train)"

    print(f"\n=== Starting RL Training ===")
    print(f"Model: {config.model}")
    print(f"Experiment: {config.experiment_name}/{config.run_name}")
    print(f"Situations: {len(situations)}")
    print(f"Perturbations: {pert_desc}")

    trainer = RLTrainer(config=config, resume_from=args.resume_from)
    trainer.setup()

    final_checkpoint = asyncio.run(
        trainer.train(
            situations=situations,
            perturbation_fns=perturbation_fns,
            trait_classifier=trait_classifier,
            answer_parser=cot_answer_parser,  # Track parse rate
        )
    )

    print(f"\n=== Training Complete ===")
    print(f"Final checkpoint: {final_checkpoint}")


if __name__ == "__main__":
    main()
