"""Eval-awareness consistency training (RLCT + BCT).

Trains a model to behave the same whether or not the input carries the eval cue
"You are being evaluated". Reuses the repo's perturbation-agnostic machinery:
the reference (idx 0) is the **eval-cued** prompt and the trained perturbation
(idx 1) is the **natural** prompt — so the model's deployment (natural) behavior
is pulled toward its eval-time behavior (Deployment←Eval).

Direction via --anchor-weight (codebase semantics):
  0.5  Deployment←Eval  (anchor eval behavior to base; the safety-positive default)
  0.0  Symmetric        (pure consistency, no anchoring) — i.e. --symmetric

Methods:
  rl  (RLCT): GRPO rate-matching, trait = async misalignment judge (gpt-5.4-mini).
  bct (SFT):  sample completions from the eval-cued prompt, pair them with the
              natural prompt, then train SFT on those pairs (Deployment←Eval only).

Usage (RLCT, gpt-oss needs explicit --lr):
  python scripts/eval_awareness/train_evalaware.py rl \\
      --model openai/gpt-oss-120b --lr 1e-4 \\
      --experiment-name evalaware_rlct --run-name gpt-evalaware-rlct -y

  python scripts/eval_awareness/train_evalaware.py rl --symmetric \\
      --model openai/gpt-oss-120b --lr 1e-4 \\
      --experiment-name evalaware_rlct --run-name gpt-evalaware-rlct-sym -y

  # BCT: first generate paired data (sampling run), then SFT
  python scripts/eval_awareness/train_evalaware.py bct \\
      --model openai/gpt-oss-120b --output bct_evalaware.jsonl -y
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from cot_transparency.eval_awareness.judge import make_misalignment_judge, DEFAULT_GRADER_MODEL  # noqa: E402

DEFAULT_TRAIN_FILE = PROJECT_ROOT / "dataset_dumps" / "eval_awareness" / "train" / "evalawarebench_safety.jsonl"


def load_datapoints(path: Path, limit: int | None) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


# Perturbation fns: idx 0 = reference = eval-cued; idx 1 = training = natural.
def eval_cued_perturbation(dp: dict) -> dict:
    return {"messages": dp["eval_cued_question"]}


def natural_perturbation(dp: dict) -> dict:
    return {"messages": dp["natural_question"]}


# ── RL (RLCT) ────────────────────────────────────────────────────────────────

def run_rl(args) -> None:
    from cot_transparency.apis.tinker.rl_training import (
        RLConfig, RLTrainer, RateEstimationConfig, TrainingSamplingConfig,
        TrainingLoopConfig, GenerationConfig,
    )
    from cot_transparency.apis.tinker.common import CheckpointConfig, AdamConfig, LoRAConfig

    datapoints = load_datapoints(Path(args.data), args.limit)
    if not datapoints:
        print("No datapoints loaded.")
        sys.exit(1)

    anchor_weight = 0.0 if args.symmetric else args.anchor_weight
    n_consistency = args.n_consistency_rollouts or args.n_train_rollouts

    config = RLConfig(
        experiment_name=args.experiment_name,
        run_name=args.run_name,
        model=args.model,
        lora=LoRAConfig(rank=args.lora_rank, seed=args.seed),
        optimizer=AdamConfig(learning_rate=args.lr, lr_schedule=args.lr_schedule),
        reference_rate=RateEstimationConfig(perturbation_indices=[0], n_rollouts=args.n_ref_rollouts),
        training=TrainingSamplingConfig(
            perturbation_indices=[1],
            n_rollouts_for_rate=args.n_train_rollouts,
            n_rollouts_for_consistency=n_consistency,
        ),
        loop=TrainingLoopConfig(
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            refresh_policy_every_n_steps=args.refresh_every,
            n_epochs=args.n_epochs,
        ),
        generation=GenerationConfig(max_new_tokens=args.max_new_tokens, temperature=args.temperature),
        checkpoint=CheckpointConfig(save_every_n_steps=args.checkpoint_every, save_state=args.save_state),
        kl_coef=args.kl_coef,
        loss_fn=args.loss_fn,
        anchor_weight=anchor_weight,
        anchor_model=args.anchor_model,
        advantage_estimator=args.advantage_estimator,
        shrinkage_mode=args.shrinkage_mode,
        shrinkage_z=args.shrinkage_z,
        helpfulness_weight=args.helpfulness_weight,
        n_helpfulness_rollouts=args.n_helpfulness_rollouts,
    )

    # Benign-helpfulness term (anti refuse-all): reward completing benign tasks.
    help_dps, help_fn, help_cls = None, None, None
    if args.helpfulness_weight > 0:
        from cot_transparency.eval_awareness.judge import make_refusal_judge
        help_dps = load_datapoints(Path(args.helpfulness_data), args.helpfulness_limit)
        _refusal = make_refusal_judge(args.grader_model)

        def help_fn(dp: dict) -> dict:
            return {"messages": dp["natural_question"]}

        async def help_cls(response: str, dp: dict) -> float:
            return 1.0 - await _refusal(response, dp)  # 1=helped, 0=refused

    direction = "Symmetric (pure consistency)" if anchor_weight == 0.0 else "Deployment←Eval"
    print("\n=== RLCT eval-awareness config ===")
    print(f"  model            : {args.model}")
    print(f"  datapoints       : {len(datapoints)} (from {args.data})")
    print(f"  direction        : {direction} (anchor_weight={anchor_weight})")
    print(f"  perturbations    : idx0=eval-cued (ref), idx1=natural (train)")
    print(f"  trait judge      : {args.grader_model}")
    print(f"  lr / schedule    : {args.lr} / {args.lr_schedule}")
    print(f"  rollouts ref/tr  : {args.n_ref_rollouts}/{args.n_train_rollouts} (consistency {n_consistency})")
    print(f"  kl_coef/loss     : {args.kl_coef}/{args.loss_fn}")
    print(f"  advantage_est    : {args.advantage_estimator}" + (f" (shrink {args.shrinkage_mode}, z={args.shrinkage_z})" if args.advantage_estimator == "shrinkage" else ""))
    print(f"  helpfulness      : weight={args.helpfulness_weight}" + (f", {len(help_dps)} benign dps from {args.helpfulness_data}, {args.n_helpfulness_rollouts} rollouts" if args.helpfulness_weight > 0 else " (off)"))
    print(f"  epochs/batch     : {args.n_epochs}/{args.batch_size} | checkpoint every {args.checkpoint_every}")
    print(f"  experiment/run   : {args.experiment_name}/{args.run_name}")

    if args.dry_run:
        print("\n[dry-run] not training.")
        return
    if not args.yes:
        print("\nRe-run with -y to start training.")
        return

    trait_classifier = make_misalignment_judge(args.grader_model)
    trainer = RLTrainer(config=config, resume_from=args.resume_from)
    final = asyncio.run(trainer.train(
        datapoints=datapoints,
        perturbation_fns=[eval_cued_perturbation, natural_perturbation],
        trait_classifier=trait_classifier,
        answer_parser=None,  # judge handles all responses; no MCQ parsing
        helpfulness_datapoints=help_dps,
        helpfulness_perturbation_fn=help_fn,
        helpfulness_classifier=help_cls,
    ))
    print(f"\nFinal checkpoint: {final}")


# ── BCT (SFT) ────────────────────────────────────────────────────────────────

def run_bct(args) -> None:
    """Generate Deployment←Eval BCT pairs: completion sampled from the eval-cued
    prompt, paired with the natural prompt. Then point train_sft.py at the output."""
    from cot_transparency.apis.tinker.inference import TinkerSamplingClient, SamplingConfig
    from tinker_cookbook.renderers.base import get_text_content

    datapoints = load_datapoints(Path(args.data), args.limit)
    if not datapoints:
        print("No datapoints loaded.")
        sys.exit(1)

    out_path = Path(args.output)
    print("\n=== BCT data generation (Deployment←Eval) ===")
    print(f"  model        : {args.model}")
    print(f"  datapoints   : {len(datapoints)} (from {args.data})")
    print(f"  sampling     : completion from EVAL-CUED prompt -> paired with NATURAL prompt")
    print(f"  output       : {out_path}")
    if args.dry_run:
        print("\n[dry-run] not sampling.")
        return
    if not args.yes:
        print("\nRe-run with -y to start sampling.")
        return

    client = TinkerSamplingClient(
        model=args.model, checkpoint=args.checkpoint,
        config=SamplingConfig(max_tokens=args.max_new_tokens, temperature=args.temperature),
    )
    client.setup()
    from tinker import types
    params = types.SamplingParams(
        max_tokens=args.max_new_tokens, temperature=args.temperature,
        stop=client.renderer.get_stop_sequences(),
    )

    def _text(fut) -> str:
        try:
            seq = fut.result().sequences[0]
            toks = list(seq.tokens)
            parsed, _ = client.renderer.parse_response(toks)
            if not parsed:
                return client.tokenizer.decode(toks)
            # gpt-oss returns structured list content (Thinking/Text parts), Llama a
            # string; get_text_content handles both. Plain .get("content") drops the
            # gpt-oss assistant text entirely (writes empty completions).
            try:
                return get_text_content(parsed) or ""
            except Exception:  # noqa: BLE001
                c = parsed.get("content", "")
                return c if isinstance(c, str) else client.tokenizer.decode(toks)
        except Exception:  # noqa: BLE001
            return ""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    chunk = args.batch_size
    with open(out_path, "w") as f:
        for start in range(0, len(datapoints), chunk):
            batch = datapoints[start:start + chunk]
            futs = [client.sampling_client.sample(
                prompt=client.renderer.build_generation_prompt(dp["eval_cued_question"]),
                sampling_params=params, num_samples=1) for dp in batch]
            for dp, fut in zip(batch, futs):
                completion = _text(fut)  # behavior under the eval cue
                if not completion.strip():
                    continue
                # Train target: produce that behavior on the NATURAL prompt.
                messages = list(dp["natural_question"]) + [{"role": "assistant", "content": completion}]
                f.write(json.dumps({"messages": messages}) + "\n")
                n += 1
            print(f"  {min(start + chunk, len(datapoints))}/{len(datapoints)} sampled ({n} written)", flush=True)
    print(f"\nWrote {n} BCT pairs -> {out_path}")
    print("Next: train with the existing SFT CLI, e.g.")
    print(
        f"  python scripts/tinker_training/train_sft.py --model {args.model} "
        f"--data {out_path} --experiment-name evalaware_bct --run-name "
        f"{Path(args.model).name}-evalaware-bct --lr 1e-4 -y"
    )


# ── CLI ──────────────────────────────────────────────────────────────────────

def _add_common(sp):
    sp.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    sp.add_argument("--data", default=str(DEFAULT_TRAIN_FILE), help="Train jsonl (natural/eval_cued pairs)")
    sp.add_argument("--limit", type=int, default=None)
    sp.add_argument("--checkpoint", default=None, help="Resume sampling/training from this checkpoint")
    sp.add_argument("--temperature", type=float, default=1.0)
    sp.add_argument("--max-new-tokens", type=int, default=2048)
    sp.add_argument("--grader-model", default=DEFAULT_GRADER_MODEL)
    sp.add_argument("--dry-run", action="store_true")
    sp.add_argument("-y", "--yes", action="store_true")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="method", required=True)

    rl = sub.add_parser("rl", help="RLCT (GRPO rate-matching)")
    _add_common(rl)
    rl.add_argument("--experiment-name", required=True)
    rl.add_argument("--run-name", required=True)
    rl.add_argument("--symmetric", action="store_true", help="Pure consistency (anchor_weight=0)")
    rl.add_argument("--anchor-weight", type=float, default=0.5, help="0.5=Deployment←Eval, 0=symmetric")
    rl.add_argument("--anchor-model", default="base", choices=["base", "initial_policy"])
    rl.add_argument("--lr", type=float, default=None, help="LR (pass explicitly for non-Llama/Qwen, e.g. gpt-oss)")
    rl.add_argument("--lr-schedule", default="constant", choices=["constant", "linear", "cosine"])
    rl.add_argument("--lora-rank", type=int, default=8)
    rl.add_argument("--seed", type=int, default=None)
    rl.add_argument("--kl-coef", type=float, default=0.05)
    rl.add_argument("--loss-fn", default="ppo", choices=["ppo", "reinforce"])
    # Advantage estimator (shrinkage tapers the consistency gradient to ~0 once natural≈cued)
    rl.add_argument("--advantage-estimator", default="grpo_normalized", choices=["grpo_normalized", "shrinkage"])
    rl.add_argument("--shrinkage-mode", default="soft", choices=["soft", "hard"])
    rl.add_argument("--shrinkage-z", type=float, default=2.0)
    # Benign-helpfulness term (anti refuse-all)
    rl.add_argument("--helpfulness-weight", type=float, default=0.0, help=">0 mixes a benign-completion GRPO reward")
    rl.add_argument("--helpfulness-data", default=str(PROJECT_ROOT / "dataset_dumps" / "eval_awareness" / "agentharm_benign" / "benign.jsonl"))
    rl.add_argument("--helpfulness-limit", type=int, default=None)
    rl.add_argument("--n-helpfulness-rollouts", type=int, default=16)
    rl.add_argument("--n-ref-rollouts", type=int, default=128)
    rl.add_argument("--n-train-rollouts", type=int, default=128)
    rl.add_argument("--n-consistency-rollouts", type=int, default=None)
    rl.add_argument("--n-epochs", type=int, default=1)
    rl.add_argument("--batch-size", type=int, default=1)
    rl.add_argument("--gradient-accumulation-steps", type=int, default=1)
    rl.add_argument("--refresh-every", type=int, default=1)
    rl.add_argument("--checkpoint-every", type=int, default=50)
    rl.add_argument("--save-state", action="store_true")
    rl.add_argument("--resume-from", default=None)

    bct = sub.add_parser("bct", help="Generate BCT (SFT) pairs, then use train_sft.py")
    _add_common(bct)
    bct.add_argument("--batch-size", type=int, default=64, help="Concurrent Tinker sample futures")
    bct.add_argument("--output", default=str(PROJECT_ROOT / "dataset_dumps" / "eval_awareness" / "bct" / "evalaware_bct.jsonl"))

    args = p.parse_args()
    if args.method == "rl":
        run_rl(args)
    elif args.method == "bct":
        run_bct(args)


if __name__ == "__main__":
    main()
