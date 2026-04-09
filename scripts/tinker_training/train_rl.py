"""
RL Consistency Training CLI.

Launch RLCT runs with flexible bias type, dataset, and hyperparameter configuration.
Supports single-bias, multi-bias, and control runs.

Usage:
    # Single bias, 100 total datapoints (50 per dataset)
    python scripts/tinker_training/train_rl.py \\
        --bias-types suggested_answer \\
        --experiment-name rl_test \\
        --run-name llama-rlct-sa-s100

    # Multi-bias, 200 total datapoints (50 per dataset x bias_type combo)
    python scripts/tinker_training/train_rl.py \\
        --bias-types distractor_argument,wrong_few_shot \\
        --n-datapoints 200 \\
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


def load_datapoints(bias_types: list[str], datasets: list[str], n_datapoints: int, data_dir: Path) -> list[dict]:
    """Load and concatenate datapoints from all bias_type x dataset combinations.

    Args:
        n_datapoints: Total number of datapoints to load, split evenly across
            all bias_type x dataset combinations.
    """
    n_combos = len(bias_types) * len(datasets)
    per_combo = n_datapoints // n_combos if n_combos > 0 else n_datapoints
    datapoints = []
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
            datapoints.extend(loaded)
            print(f"  Loaded {len(loaded)} from {path.name}")
    return datapoints


def unbiased_perturbation(datapoint: dict) -> dict:
    return {"messages": datapoint["unbiased_question"]}


def biased_perturbation(datapoint: dict) -> dict:
    return {"messages": datapoint["biased_question"]}


def trait_classifier(response: str, datapoint: dict) -> float:
    answer = cot_answer_parser(response)
    biased_option = datapoint.get("biased_option", "")
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
    parser.add_argument("--n-datapoints", type=int, default=100, help="Total number of datapoints (split evenly across dataset x bias_type combinations)")
    parser.add_argument("--data-dir", default=None, help="Override default dataset_dumps/test directory")

    # === Naming ===
    parser.add_argument("--experiment-name", required=True, help="Experiment name")
    parser.add_argument("--run-name", required=True, help="Run name (used in checkpoint path)")

    # === Optimiser ===
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (default: auto from Tinker's get_recommended_lr)")
    parser.add_argument("--lr-schedule", default="constant", choices=["constant", "linear", "cosine"])
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--kl-coef", type=float, default=0.05)
    parser.add_argument("--anchor-weight", type=float, default=0.5, help="Anchor weight (alpha): 0=pure consistency, 1=pure anchor, 0.5=equal")
    parser.add_argument("--anchor-model", default="base", choices=["base", "initial_policy"], help="Model for anchor reference rate: 'base' (frozen base) or 'initial_policy' (policy at init, incl. resumed ckpt)")
    parser.add_argument("--loss-fn", default="ppo", choices=["ppo", "reinforce"])

    # === Sampling ===
    parser.add_argument("--n-ref-rollouts", type=int, default=128, help="Rollouts for reference rate estimation")
    parser.add_argument("--n-train-rollouts", type=int, default=128, help="Rollouts for training rate estimation")
    parser.add_argument("--n-consistency-rollouts", type=int, default=None, help="Consistency gradient rollouts (default: same as --n-train-rollouts)")
    parser.add_argument("--n-anchor-rollouts", type=int, default=None, help="Anchor gradient rollouts (default: all parsed ref rollouts)")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=8192)

    # === Training loop ===
    parser.add_argument("--n-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1, help="Datapoints per gradient step")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--refresh-every", type=int, default=1, help="Refresh policy every N steps")

    # === Checkpointing ===
    parser.add_argument("--checkpoint-every", type=int, default=50, help="Save checkpoint every N steps")
    parser.add_argument("--save-state", action="store_true", help="Save full optimizer state (for resuming)")

    # === Run modes ===
    parser.add_argument("--control", action="store_true", help="Control: use unbiased perturbation for both ref and train")
    parser.add_argument("--resume-from", default=None, help="Tinker checkpoint path to resume from")
    parser.add_argument("--resume-with-optimizer", action="store_true", help="Also restore optimizer state when resuming (for exact continuation)")
    parser.add_argument("--dry-run", action="store_true", help="Load data and print config, don't train")
    parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompt")

    args = parser.parse_args()

    bias_types = [b.strip() for b in args.bias_types.split(",")]
    datasets = [d.strip() for d in args.datasets.split(",")]
    n_consistency = args.n_consistency_rollouts or args.n_train_rollouts

    data_dir = Path(args.data_dir) if args.data_dir else PROJECT_ROOT / "dataset_dumps" / "test"
    n_combos = len(bias_types) * len(datasets)
    per_combo = args.n_datapoints // n_combos if n_combos > 0 else args.n_datapoints
    print(f"\nLoading datapoints: {args.n_datapoints} total across {n_combos} combos ({per_combo} per combo)")

    datapoints = load_datapoints(bias_types, datasets, args.n_datapoints, data_dir)

    if not datapoints:
        print("Error: no datapoints loaded. Check --bias-types and --datasets.")
        sys.exit(1)

    n_steps = len(datapoints) // args.batch_size
    total_steps = n_steps * args.n_epochs
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
            n_rollouts=args.n_ref_rollouts,
        ),
        training=TrainingSamplingConfig(
            perturbation_indices=[1],
            n_rollouts_for_rate=args.n_train_rollouts,
            n_rollouts_for_consistency=n_consistency,
            n_rollouts_for_anchor=args.n_anchor_rollouts,
        ),
        loop=TrainingLoopConfig(
            batch_size=args.batch_size,
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
        anchor_weight=args.anchor_weight,
        anchor_model=args.anchor_model,
        log_base_dir="logs",
    )

    print(f"\n{'='*60}")
    print(f"RL Training Configuration")
    print(f"{'='*60}")
    print(f"  Model:              {args.model}")
    print(f"  Experiment:         {args.experiment_name}/{args.run_name}")
    print(f"  Bias types:         {bias_types}")
    print(f"  Datasets:           {datasets}")
    print(f"  Total datapoints:   {len(datapoints)}")
    print(f"  Perturbations:      {pert_desc}")
    print(f"  LR:                 {args.lr} ({args.lr_schedule})")
    print(f"  LoRA rank:          {args.lora_rank}")
    print(f"  Batch size:         {args.batch_size}")
    print(f"  Grad accum steps:   {args.gradient_accumulation_steps}")
    print(f"  N epochs:           {args.n_epochs}")
    print(f"  Estimated steps:    {total_steps}")
    print(f"  Checkpoint every:   {args.checkpoint_every} steps")
    print(f"  n_ref_rollouts:     {args.n_ref_rollouts}")
    print(f"  n_train_rollouts:   {args.n_train_rollouts}")
    print(f"  n_consistency_rollouts: {n_consistency}")
    print(f"  n_anchor_rollouts:  {args.n_anchor_rollouts}")
    print(f"  KL coef:            {args.kl_coef}")
    print(f"  Anchor weight:      {args.anchor_weight}")
    print(f"  Anchor model:       {args.anchor_model}")
    print(f"  Loss fn:            {args.loss_fn}")
    if args.resume_from:
        print(f"  Resume from:        {args.resume_from}")
        print(f"  With optimizer:     {args.resume_with_optimizer}")
    print(f"{'='*60}")

    if args.dry_run:
        print("\nDry run complete.")
        if datapoints:
            dp = datapoints[0]
            print(f"\nSample datapoint keys: {list(dp.keys())}")
            print(f"  biased_option: {dp.get('biased_option')}")
            print(f"  ground_truth:  {dp.get('ground_truth')}")
            print(f"  bias_name:     {dp.get('bias_name', 'n/a')}")
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

    trainer = RLTrainer(config=config, resume_from=args.resume_from, resume_with_optimizer=args.resume_with_optimizer)
    trainer.setup()

    final_checkpoint = asyncio.run(
        trainer.train(
            datapoints=datapoints,
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
