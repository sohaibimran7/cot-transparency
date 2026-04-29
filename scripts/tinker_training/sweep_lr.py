"""
Learning rate sweep for RLCT.

Launches sequential training runs with different learning rates,
then prints a summary of all checkpoints for evaluation.

Usage:
    # Default sweep: 5e-5, 1e-4, 3e-4, 5e-4
    python scripts/tinker_training/sweep_lr.py \
        --bias-types suggested_answer \
        --experiment-name lr-sweep-sa

    # Custom LRs
    python scripts/tinker_training/sweep_lr.py \
        --bias-types suggested_answer \
        --experiment-name lr-sweep-sa \
        --lrs 1e-5 5e-5 1e-4 3e-4

    # Dry run to preview all commands
    python scripts/tinker_training/sweep_lr.py \
        --bias-types suggested_answer \
        --experiment-name lr-sweep-sa \
        --dry-run
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

DEFAULT_LRS = [5e-5, 1e-4, 3e-4, 5e-4]


def build_train_cmd(args: argparse.Namespace, lr: float) -> list[str]:
    lr_tag = f"{lr:.0e}".replace("+", "").replace("-", "m")  # e.g. 1e-04 -> 1em04
    run_name = f"{args.run_name_prefix}-lr{lr_tag}"

    cmd = [
        sys.executable, str(PROJECT_ROOT / "scripts" / "tinker_training" / "train_rl.py"),
        "--bias-types", args.bias_types,
        "--datasets", args.datasets,
        "--n-samples", str(args.n_samples),
        "--experiment-name", args.experiment_name,
        "--run-name", run_name,
        "--model", args.model,
        "--lr", str(lr),
        "--lr-schedule", args.lr_schedule,
        "--lora-rank", str(args.lora_rank),
        "--kl-coef", str(args.kl_coef),
        "--loss-fn", args.loss_fn,
        "--n-ref-samples", str(args.n_ref_samples),
        "--n-train-samples", str(args.n_train_samples),
        "--n-grad-samples", str(args.n_grad_samples),
        "--temperature", str(args.temperature),
        "--n-epochs", str(args.n_epochs),
        "--situations-per-group", str(args.situations_per_group),
        "--gradient-accumulation-steps", str(args.gradient_accumulation_steps),
        "--refresh-every", str(args.refresh_every),
        "--checkpoint-every", str(args.checkpoint_every),
        "-y",  # skip confirmation
    ]
    return cmd, run_name


def main():
    parser = argparse.ArgumentParser(
        description="LR sweep for RLCT",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Sweep-specific
    parser.add_argument("--lrs", type=float, nargs="+", default=DEFAULT_LRS,
                        help="Learning rates to sweep")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--run-name-prefix", default="rlct",
                        help="Prefix for run names (LR tag is appended)")

    # Passed through to train_rl.py
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--bias-types", required=True)
    parser.add_argument("--datasets", default="mmlu,truthfulqa")
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--lr-schedule", default="constant", choices=["constant", "linear", "cosine"])
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--kl-coef", type=float, default=0.05)
    parser.add_argument("--loss-fn", default="ppo", choices=["ppo", "reinforce"])
    parser.add_argument("--n-ref-samples", type=int, default=128)
    parser.add_argument("--n-train-samples", type=int, default=128)
    parser.add_argument("--n-grad-samples", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--n-epochs", type=int, default=1)
    parser.add_argument("--situations-per-group", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--refresh-every", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=50)

    args = parser.parse_args()

    print(f"=== LR Sweep: {len(args.lrs)} runs ===")
    print(f"LRs: {args.lrs}")
    print(f"Experiment: {args.experiment_name}")
    print()

    results = []

    for i, lr in enumerate(args.lrs):
        cmd, run_name = build_train_cmd(args, lr)
        print(f"[{i+1}/{len(args.lrs)}] LR={lr}  run_name={run_name}")

        if args.dry_run:
            print(f"  CMD: {' '.join(cmd)}\n")
            continue

        print(f"  Running...")
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
        results.append((lr, run_name, status))
        print(f"  {status}\n")

    if not args.dry_run and results:
        print(f"\n{'='*60}")
        print("Sweep Summary")
        print(f"{'='*60}")
        for lr, run_name, status in results:
            print(f"  LR={lr:<10}  {run_name:<40}  {status}")
        print(f"\nCheckpoints saved under: logs/{args.experiment_name}/")
        print(f"Evaluate with:")
        print(f"  python sycophancy_eval_inspect/run_tinker_evals.py --checkpoint <checkpoint_path>")


if __name__ == "__main__":
    main()
