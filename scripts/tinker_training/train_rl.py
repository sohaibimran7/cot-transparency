"""
RL Consistency Training CLI.

Launch RLCT runs with flexible bias type, dataset, and hyperparameter configuration.
Supports single-bias, multi-bias, and control runs.

Usage:
    # Single bias, 100 total situations (50 per dataset)
    python scripts/tinker_training/train_rl.py \\
        --bias-types suggested_answer \\
        --experiment-name rl_test \\
        --run-name llama-rlct-sa-s100

    # Multi-bias, 200 total situations (50 per dataset x bias_type combo)
    python scripts/tinker_training/train_rl.py \\
        --bias-types distractor_argument,wrong_few_shot \\
        --n-samples 200 \\
        --experiment-name rl-da-wfs \\
        --run-name gpt-rlct-da-wfs-s200

    # Control run
    python scripts/tinker_training/train_rl.py \\
        --bias-types distractor_argument \\
        --experiment-name rl-distractor-argument \\
        --run-name gpt-rl-control-da-s100 --control

    # Explicit LR (default: auto from Tinker's get_recommended_lr)
    python scripts/tinker_training/train_rl.py \\
        --bias-types distractor_argument \\
        --experiment-name rl-distractor-argument \\
        --run-name gpt-rlct-da-s100 \\
        --lr 1e-4
"""

import asyncio
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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


DATASET_ALIASES = {
    "mmlu": "mmlu",
    "truthfulqa": "truthfulqa",
    "hellaswag": "hellaswag",
    "logiqa": "logiqa",
}


def load_situations(bias_types: list[str], datasets: list[str], n_samples: int) -> list[dict]:
    """Load and concatenate situations from all bias_type x dataset combinations.

    Args:
        n_samples: Total number of situations to load, split evenly across
            all bias_type x dataset combinations.
    """
    n_combos = len(bias_types) * len(datasets)
    per_combo = n_samples // n_combos if n_combos > 0 else n_samples
    situations = []
    for bias_type in bias_types:
        for dataset in datasets:
            path = PROJECT_ROOT / "dataset_dumps" / "test" / bias_type / f"{dataset}_{bias_type}.jsonl"
            if not path.exists():
                print(f"Warning: {path} not found, skipping")
                continue
            loaded = []
            with open(path) as f:
                for line in f:
                    loaded.append(json.loads(line))
                    if len(loaded) >= per_combo:
                        break
            situations.extend(loaded)
            print(f"  Loaded {len(loaded)} situations from {path.name}")
    return situations


def unbiased_perturbation(situation: dict) -> dict:
    return {"messages": situation["unbiased_question"]}


def biased_perturbation(situation: dict) -> dict:
    return {"messages": situation["biased_question"]}


def trait_classifier(response: str, situation: dict) -> float:
    answer = cot_answer_parser(response)
    biased_option = situation.get("biased_option", "")
    return 1.0 if answer == biased_option else 0.0


def main():
    parser = argparse.ArgumentParser(
        description="RL Consistency Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # === Model & data ===
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct", help="Base model name")
    parser.add_argument("--bias-types", required=True, help="Comma-separated bias types (e.g. distractor_argument,wrong_few_shot)")
    parser.add_argument("--datasets", default="mmlu,truthfulqa", help="Comma-separated datasets")
    parser.add_argument("--n-samples", type=int, default=100, help="Total number of situations (split evenly across dataset x bias_type combinations)")
    parser.add_argument("--data-dir", default=None, help="Override default dataset_dumps/test directory")

    # === Naming ===
    parser.add_argument("--experiment-name", required=True, help="Experiment name")
    parser.add_argument("--run-name", required=True, help="Run name (used in checkpoint path)")

    # === Optimiser ===
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (default: auto from Tinker's get_recommended_lr)")
    parser.add_argument("--lr-schedule", default="constant", choices=["constant", "linear", "cosine"])
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--kl-coef", type=float, default=0.05)
    parser.add_argument("--loss-fn", default="ppo", choices=["ppo", "reinforce"])

    # === Sampling ===
    parser.add_argument("--n-ref-samples", type=int, default=128, help="Samples for reference rate estimation")
    parser.add_argument("--n-train-samples", type=int, default=128, help="Samples for training rate estimation")
    parser.add_argument("--n-grad-samples", type=int, default=None, help="Samples for gradient (default: same as --n-train-samples)")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=8192)

    # === Training loop ===
    parser.add_argument("--n-epochs", type=int, default=1)
    parser.add_argument("--situations-per-group", type=int, default=1, help="Situations processed per gradient step (group size)")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--refresh-every", type=int, default=1, help="Refresh policy every N steps")

    # === Checkpointing ===
    parser.add_argument("--checkpoint-every", type=int, default=50, help="Save checkpoint every N steps")
    parser.add_argument("--save-state", action="store_true", help="Save full optimizer state (for resuming)")

    # === Run modes ===
    parser.add_argument("--control", action="store_true", help="Control: use unbiased perturbation for both ref and train")
    parser.add_argument("--resume-from", default=None, help="Tinker checkpoint path to resume from")
    parser.add_argument("--dry-run", action="store_true", help="Load data and print config, don't train")
    parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompt")

    args = parser.parse_args()

    bias_types = [b.strip() for b in args.bias_types.split(",")]
    datasets = [d.strip() for d in args.datasets.split(",")]
    n_grad = args.n_grad_samples or args.n_train_samples

    # Load situations — n_samples is the TOTAL, split evenly across combinations
    data_dir = Path(args.data_dir) if args.data_dir else PROJECT_ROOT / "dataset_dumps" / "test"
    n_combos = len(bias_types) * len(datasets)
    per_combo = args.n_samples // n_combos if n_combos > 0 else args.n_samples
    print(f"\nLoading situations: {args.n_samples} total across {n_combos} combos ({per_combo} per combo)")

    situations = []
    for bias_type in bias_types:
        for dataset in datasets:
            path = data_dir / bias_type / f"{dataset}_{bias_type}.jsonl"
            if not path.exists():
                print(f"  Warning: {path} not found, skipping")
                continue
            loaded = []
            with open(path) as f:
                for line in f:
                    loaded.append(json.loads(line))
                    if len(loaded) >= per_combo:
                        break
            situations.extend(loaded)
            print(f"  Loaded {len(loaded)} from {path.name}")

    if not situations:
        print("Error: no situations loaded. Check --bias-types and --datasets.")
        sys.exit(1)

    n_groups = len(situations) // args.situations_per_group
    total_steps = n_groups * args.n_epochs
    pert_desc = "unbiased (ref) + unbiased (train) [CONTROL]" if args.control else "unbiased (ref) + biased (train)"

    config = RLConfig(
        experiment_name=args.experiment_name,
        run_name=args.run_name,
        model=args.model,
        lora=LoRAConfig(rank=args.lora_rank),
        optimizer=AdamConfig(
            learning_rate=args.lr,
            lr_schedule=args.lr_schedule,
        ),
        reference_rate=RateEstimationConfig(
            perturbation_indices=[0],
            n_samples=args.n_ref_samples,
        ),
        training=TrainingSamplingConfig(
            perturbation_indices=[1],
            n_samples_for_rate=args.n_train_samples,
            n_samples_for_gradient=n_grad,
        ),
        loop=TrainingLoopConfig(
            situations_per_group=args.situations_per_group,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            refresh_policy_every_n_steps=args.refresh_every,
            n_epochs=args.n_epochs,
        ),
        generation=GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        ),
        checkpoint=CheckpointConfig(
            save_every_n_steps=args.checkpoint_every,
            save_state=args.save_state,
        ),
        kl_coef=args.kl_coef,
        loss_fn=args.loss_fn,
        log_base_dir="logs",
    )

    print(f"\n{'='*60}")
    print(f"RL Training Configuration")
    print(f"{'='*60}")
    print(f"  Model:              {args.model}")
    print(f"  Experiment:         {args.experiment_name}/{args.run_name}")
    print(f"  Bias types:         {bias_types}")
    print(f"  Datasets:           {datasets}")
    print(f"  Total situations:   {len(situations)}")
    print(f"  Perturbations:      {pert_desc}")
    print(f"  LR:                 {args.lr} ({args.lr_schedule})")
    print(f"  LoRA rank:          {args.lora_rank}")
    print(f"  Situations/group:   {args.situations_per_group}")
    print(f"  Grad accum steps:   {args.gradient_accumulation_steps}")
    print(f"  N epochs:           {args.n_epochs}")
    print(f"  Estimated steps:    {total_steps}")
    print(f"  Checkpoint every:   {args.checkpoint_every} steps")
    print(f"  n_ref_samples:      {args.n_ref_samples}")
    print(f"  n_train_samples:    {args.n_train_samples}")
    print(f"  n_grad_samples:     {n_grad}")
    print(f"  KL coef:            {args.kl_coef}")
    print(f"  Loss fn:            {args.loss_fn}")
    if args.resume_from:
        print(f"  Resume from:        {args.resume_from}")
    print(f"{'='*60}")

    if args.dry_run:
        print("\nDry run complete.")
        if situations:
            s = situations[0]
            print(f"\nSample situation keys: {list(s.keys())}")
            print(f"  biased_option: {s.get('biased_option')}")
            print(f"  ground_truth:  {s.get('ground_truth')}")
            print(f"  bias_name:     {s.get('bias_name', 'n/a')}")
        return

    if not args.yes:
        response = input("\nProceed with training? [y/N] ").strip().lower()
        if response != "y":
            print("Aborted.")
            sys.exit(0)

    if args.control:
        perturbation_fns = [unbiased_perturbation, unbiased_perturbation]
    else:
        perturbation_fns = [unbiased_perturbation, biased_perturbation]

    trainer = RLTrainer(config=config, resume_from=args.resume_from)
    trainer.setup()

    final_checkpoint = asyncio.run(
        trainer.train(
            situations=situations,
            perturbation_fns=perturbation_fns,
            trait_classifier=trait_classifier,
            answer_parser=cot_answer_parser,
        )
    )

    print(f"\n{'='*60}")
    print(f"Training Complete")
    print(f"Final checkpoint: {final_checkpoint}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
