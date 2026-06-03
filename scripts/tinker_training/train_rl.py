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
from cot_transparency.formatters.more_biases.distractor_cue_variations import (
    DISTRACTOR_CUES,
    apply_distractor_cue,
    cue_keys as all_distractor_cue_keys,
    extract_wrong_cot,
    split_cues,
)
from sycophancy_eval_inspect.mcq.answer_parser import fallback_answer_parser
from sycophancy_eval_inspect.mcq.dataset import strip_cot_from_message


def load_datapoints(bias_types: list[str], datasets: list[str], n_datapoints: int, data_dir: Path) -> list[dict]:
    """Load and concatenate datapoints from all bias_type x dataset combinations.

    Args:
        n_datapoints: Total number of datapoints to load, split evenly across
            all bias_type x dataset combinations.
    """
    n_combos = len(bias_types) * len(datasets)
    per_combo = n_datapoints // n_combos if n_combos > 0 else n_datapoints
    datapoints = []
    missing, short = [], []
    for bias_type in bias_types:
        for dataset in datasets:
            path = data_dir / bias_type / f"{dataset}_{bias_type}.jsonl"
            if not path.exists():
                print(f"  Warning: {path} not found, skipping")
                missing.append(f"{dataset}/{bias_type}")
                continue
            loaded = []
            with open(path) as f:
                for line in f:
                    loaded.append(json.loads(line))
                    if len(loaded) >= per_combo:
                        break
            datapoints.extend(loaded)
            if len(loaded) < per_combo:
                short.append(f"{dataset}/{bias_type} ({len(loaded)}/{per_combo})")
            print(f"  Loaded {len(loaded)} from {path.name}")
    # Surface silent dataset skew: floored per_combo, missing files, and short combos all
    # change the trained mix vs --bias-types/--datasets intent without an error otherwise.
    if missing or short or len(datapoints) != n_datapoints:
        print(f"  ⚠️  Loaded {len(datapoints)}/{n_datapoints} requested datapoints across {n_combos} combo(s).")
        if missing:
            print(f"      Missing files (combo skipped entirely): {', '.join(missing)}")
        if short:
            print(f"      Short combos (fewer rows than per-combo target {per_combo}): {', '.join(short)}")
        print("      The trained mix may differ from intent; pass matching --n-datapoints/--datasets.")
    return datapoints


def _apply_prompt_style(messages: list[dict], prompt_style: str) -> list[dict]:
    """Strip CoT instructions from user messages if prompt_style is 'no_cot'."""
    if prompt_style != "no_cot":
        return messages
    out = []
    for m in messages:
        if m.get("role") == "user":
            out.append({**m, "content": strip_cot_from_message(m["content"])})
        else:
            out.append(m)
    return out


def make_perturbation_fns(prompt_style: str):
    def unbiased_perturbation(datapoint: dict) -> dict:
        return {"messages": _apply_prompt_style(datapoint["unbiased_question"], prompt_style)}
    def biased_perturbation(datapoint: dict) -> dict:
        return {"messages": _apply_prompt_style(datapoint["biased_question"], prompt_style)}
    return unbiased_perturbation, biased_perturbation


def resolve_distractor_cues(spec: str | None) -> list[str]:
    """Resolve --distractor-cues into a list of cue keys.

    'none'/'' -> [] (single-cue mode); 'all'/'train'/'holdout' -> registry splits;
    otherwise a comma-separated list of explicit cue keys.
    """
    if spec in (None, "", "none"):
        return []
    if spec == "all":
        return all_distractor_cue_keys()
    if spec == "train":
        return split_cues()[0]
    if spec == "holdout":
        return split_cues()[1]
    keys = [k.strip() for k in spec.split(",") if k.strip()]
    unknown = [k for k in keys if k not in DISTRACTOR_CUES]
    if unknown:
        raise SystemExit(f"Unknown distractor cue(s): {unknown}. Known: {all_distractor_cue_keys()}")
    return keys


def make_distractor_cue_perturbations(cues: list[str], prompt_style: str):
    """[unbiased (ref, idx 0)] + one perturbation per distractor cue (idx 1..N).

    Each cue re-frames the datapoint's pre-extracted wrong argument (dp['_wrong_cot'])
    around its unbiased question, so the same argument is presented many ways.
    """
    def unbiased(dp: dict) -> dict:
        return {"messages": _apply_prompt_style(dp["unbiased_question"], prompt_style)}

    def make_cue(key: str):
        def cue_pert(dp: dict) -> dict:
            # Strip CoT on the bare question FIRST, then wrap. Transforming the wrapped
            # blob would truncate at the CoT phrase now sitting inside <question>,
            # deleting </question> and (for question-first cues) the entire <argument> block.
            base = _apply_prompt_style(dp["unbiased_question"], prompt_style)
            return {"messages": apply_distractor_cue(base, dp["_wrong_cot"], key)}
        return cue_pert

    return [unbiased] + [make_cue(k) for k in cues]


def attach_wrong_cots(datapoints: list[dict]) -> list[dict]:
    """Keep only datapoints whose biased_question carries an extractable wrong argument,
    stashing it on dp['_wrong_cot'] for the cue perturbations."""
    kept = []
    for dp in datapoints:
        wc = extract_wrong_cot(dp.get("biased_question") or [])
        if wc:
            dp["_wrong_cot"] = wc
            kept.append(dp)
    return kept


def trait_classifier(response: str, datapoint: dict) -> float:
    """Trait = 1 iff the answer equals the datapoint's fixed designated `biased_option`.

    Note: for distractor_argument, `biased_option` is a deterministic-random WRONG option
    (seeded by the question), NOT the argument's actual conclusion — the wrapped argument
    only argues for `biased_option` ~1/3 of the time. So the trait measures "rate of
    selecting a fixed designated option" (consistent with the eval BIR scorer), not "rate
    of being swayed to the argument's conclusion". The consistency objective is still
    well-defined; just don't over-read the matched_pair rationale.
    """
    answer = fallback_answer_parser(response)
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
    parser.add_argument("--prompt-style", choices=["cot", "no_cot"], default="cot", help="Strip CoT instructions for reasoning models (e.g. gpt-oss)")

    # === Naming ===
    parser.add_argument("--experiment-name", required=True, help="Experiment name")
    parser.add_argument("--run-name", required=True, help="Run name (used in checkpoint path)")

    # === Optimiser ===
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (default: auto from Tinker's get_recommended_lr)")
    parser.add_argument("--lr-schedule", default="linear", choices=["constant", "linear", "cosine"],
                        help="LR schedule (shared SFT+RL default: linear). RL now honors this per optim step.")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for LoRA init, epoch shuffle, and gradient-rollout "
                             "subsampling (stochastic temperature sampling is NOT seeded)")
    parser.add_argument("--kl-coef", type=float, default=0.05)
    parser.add_argument("--anchor-weight", type=float, default=0.5, help="Anchor weight (alpha): 0=pure consistency, 1=pure anchor, 0.5=equal")
    parser.add_argument("--anchor-model", default="base", choices=["base", "initial_policy"], help="Model for anchor reference rate: 'base' (frozen base) or 'initial_policy' (policy at init, incl. resumed ckpt)")
    parser.add_argument("--loss-fn", default="ppo", choices=["ppo", "importance_sampling"])
    parser.add_argument("--advantage-estimator", default="grpo_normalized", choices=["grpo_normalized", "snr_scaling", "matched_pair"],
                        help="Advantage construction: 'grpo_normalized' (std-normalize; drops gap magnitude, keeps only its sign), "
                             "'snr_scaling' (still GRPO; keep gap magnitude, shrunk toward 0 by its sampling SNR), or "
                             "'matched_pair' (pool the cued rate across the cue family into one gap vs the neutral control; "
                             "use with --distractor-cues and a small --n-train-rollouts)")
    parser.add_argument("--snr-mode", default="soft", choices=["soft", "hard"],
                        help="SNR-scaling shape (advantage-estimator=snr_scaling only): 'soft' smooth taper, 'hard' significance gate")
    parser.add_argument("--snr-z", type=float, default=2.0,
                        help="SNR scale in SEs: half-weight (soft) / cutoff (hard) at |gap| = z·SE; z=0 = no floor (full faithful gap)")
    parser.add_argument("--snr-normalizer", default="trait_std", choices=["trait_std", "none"],
                        help="SNR-scaling advantage scaling: 'trait_std' (divide by sqrt(p(1-p)+floor)) or 'none' (bare A=-gap*(T-p))")
    parser.add_argument("--unparsed-handling", default="discard", choices=["discard", "resample"],
                        help="Unparsed/hedged rollouts: 'discard' (drop from rate denominator + gradient, "
                             "default) or 'resample' (re-sample until a usable answer, up to "
                             "--max-resample-attempts; logs resample amplification + give-up so hedging stays visible)")
    parser.add_argument("--max-resample-attempts", type=int, default=4,
                        help="Max resample rounds per slot when --unparsed-handling=resample")

    # === Sampling ===
    parser.add_argument("--n-ref-rollouts", type=int, default=128, help="Rollouts for reference rate estimation")
    parser.add_argument("--n-train-rollouts", type=int, default=128, help="Rollouts for training rate estimation")
    parser.add_argument("--n-consistency-rollouts", type=int, default=None, help="Consistency gradient rollouts (default: same as --n-train-rollouts)")
    parser.add_argument("--n-anchor-rollouts", type=int, default=None, help="Anchor gradient rollouts (default: all parsed ref rollouts)")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=16384)

    # === Training loop ===
    parser.add_argument("--n-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1, help="Datapoints per gradient step")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--refresh-every", type=int, default=1, help="Refresh policy every N steps")

    # === Checkpointing ===
    parser.add_argument("--checkpoint-every", type=int, default=50, help="Save checkpoint every N steps")
    parser.add_argument("--save-state", action="store_true", help="Save full optimizer state (for resuming)")

    # === Distractor cue family (matched-pair / RLOO) ===
    parser.add_argument("--distractor-cues", default="none",
                        help="Cue family for matched-pair training: 'none' (single biased prompt), "
                             "'all'/'train'/'holdout' (registry splits), or a comma-list of cue keys. "
                             "When set, the cued side becomes N re-framings of each item's wrong argument "
                             "(idx 1..N); pair with --advantage-estimator matched_pair and a small --n-train-rollouts.")

    # === Run modes ===
    parser.add_argument("--control", action="store_true", help="Control: use unbiased perturbation for both ref and train")
    parser.add_argument("--resume-from", default=None, help="Tinker checkpoint path to resume from")
    parser.add_argument("--resume-with-optimizer", action="store_true", help="Also restore optimizer state when resuming (for exact continuation)")
    parser.add_argument("--dry-run", action="store_true", help="Load data and print config, don't train")
    parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompt")

    args = parser.parse_args()

    bias_types = [b.strip() for b in args.bias_types.split(",")]
    datasets = [d.strip() for d in args.datasets.split(",")]
    # `is not None` (not `or`) so an explicit --n-consistency-rollouts 0 isn't silently
    # overridden to n_train_rollouts.
    n_consistency = args.n_consistency_rollouts if args.n_consistency_rollouts is not None else args.n_train_rollouts

    data_dir = Path(args.data_dir) if args.data_dir else PROJECT_ROOT / "dataset_dumps" / "test"
    n_combos = len(bias_types) * len(datasets)
    per_combo = args.n_datapoints // n_combos if n_combos > 0 else args.n_datapoints
    print(f"\nLoading datapoints: {args.n_datapoints} total across {n_combos} combos ({per_combo} per combo)")

    datapoints = load_datapoints(bias_types, datasets, args.n_datapoints, data_dir)

    if not datapoints:
        print("Error: no datapoints loaded. Check --bias-types and --datasets.")
        sys.exit(1)

    distractor_cues = resolve_distractor_cues(args.distractor_cues)
    if distractor_cues and args.control:
        print("Error: --distractor-cues is incompatible with --control.")
        sys.exit(1)
    if distractor_cues:
        from collections import Counter
        n_before = len(datapoints)
        before_by_bias = Counter(dp.get("bias_name", "?") for dp in datapoints)
        datapoints = attach_wrong_cots(datapoints)
        after_by_bias = Counter(dp.get("bias_name", "?") for dp in datapoints)
        print(f"  Distractor cue family: {len(distractor_cues)} cues; kept "
              f"{len(datapoints)}/{n_before} datapoints with an extractable <argument>")
        # The cue family only supports <argument>-bearing data (distractor_argument).
        # Other bias types (distractor_fact=<fun_fact>, etc.) would be silently dropped,
        # collapsing a mixed --bias-types run to a different composition than requested.
        for bias, n0 in before_by_bias.items():
            n1 = after_by_bias.get(bias, 0)
            if n1 == 0:
                print(f"  ⚠️  WARNING: bias type {bias!r} has NO <argument> blocks (0/{n0} kept) "
                      f"and is being DROPPED ENTIRELY — the cue family only supports "
                      f"distractor_argument. Your trained mix no longer matches --bias-types.")
            elif n1 < n0:
                print(f"     {bias}: kept {n1}/{n0}")
        if not datapoints:
            print("Error: no datapoints have an extractable <argument>. The distractor-cue "
                  "family requires distractor_argument data.")
            sys.exit(1)
        train_indices = list(range(1, len(distractor_cues) + 1))
    else:
        train_indices = [1]

    n_steps = len(datapoints) // args.batch_size
    total_steps = n_steps * args.n_epochs
    if args.control:
        pert_desc = "unbiased (ref) + unbiased (train) [CONTROL]"
    elif distractor_cues:
        pert_desc = f"unbiased (ref) + {len(distractor_cues)} distractor cues (train)"
    else:
        pert_desc = "unbiased (ref) + biased (train)"

    config = RLConfig(
        experiment_name=args.experiment_name,
        run_name=args.run_name,
        model=args.model,
        lora=LoRAConfig(rank=args.lora_rank, seed=args.seed),
        optimizer=AdamConfig(
            learning_rate=args.lr,
            lr_schedule=args.lr_schedule,
        ),
        reference_rate=RateEstimationConfig(
            perturbation_indices=[0],
            n_rollouts=args.n_ref_rollouts,
        ),
        training=TrainingSamplingConfig(
            perturbation_indices=train_indices,
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
        advantage_estimator=args.advantage_estimator,
        snr_mode=args.snr_mode,
        snr_z=args.snr_z,
        snr_normalizer=args.snr_normalizer,
        unparsed_handling=args.unparsed_handling,
        max_resample_attempts=args.max_resample_attempts,
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
    if distractor_cues:
        print(f"  Distractor cues:    {', '.join(distractor_cues)}")
    print(f"  LR:                 {args.lr} ({args.lr_schedule})")
    print(f"  LoRA rank:          {args.lora_rank}")
    if args.seed is not None:
        print(f"  Seed:               {args.seed}")
    print(f"  Batch size:         {args.batch_size}")
    print(f"  Grad accum steps:   {args.gradient_accumulation_steps}")
    print(f"  N epochs:           {args.n_epochs}")
    print(f"  Estimated steps:    {total_steps}")
    print(f"  Checkpoint every:   {args.checkpoint_every} steps")
    print(f"  n_ref_rollouts:     {args.n_ref_rollouts}")
    print(f"  n_train_rollouts:   {args.n_train_rollouts}")
    print(f"  n_consistency_rollouts: {n_consistency}")
    print(f"  n_anchor_rollouts:  {args.n_anchor_rollouts}")
    # Surface the per-datapoint sampling cost: each training perturbation (cue) samples
    # n_train_rollouts. With the full cue family this multiplies fast.
    n_train_perts = len(train_indices)
    eff_rollouts = args.n_ref_rollouts + n_train_perts * args.n_train_rollouts
    print(f"  Rollouts/datapoint: {eff_rollouts} (= {args.n_ref_rollouts} ref + {n_train_perts}×{args.n_train_rollouts} cued)")
    if distractor_cues and args.n_train_rollouts > 8:
        print(f"  ⚠️  WARNING: {n_train_perts} cues × {args.n_train_rollouts} rollouts/cue is a large "
              f"per-datapoint sampling cost (×{eff_rollouts // (args.n_ref_rollouts + args.n_train_rollouts)} "
              f"vs single-cue). matched_pair targets ~1-2 rollouts/cue — consider --n-train-rollouts 2.")
    print(f"  KL coef:            {args.kl_coef}")
    print(f"  Anchor weight:      {args.anchor_weight}")
    print(f"  Anchor model:       {args.anchor_model}")
    if args.advantage_estimator == "snr_scaling":
        _adv_desc = f"snr_scaling ({args.snr_mode}, z={args.snr_z}, norm={args.snr_normalizer})"
    elif args.advantage_estimator == "matched_pair":
        _adv_desc = f"matched_pair (pooled gap, z={args.snr_z}, norm={args.snr_normalizer}, {len(distractor_cues) or 1} cue(s))"
    else:
        _adv_desc = args.advantage_estimator
    print(f"  Advantage est.:     {_adv_desc}")
    if distractor_cues and args.advantage_estimator != "matched_pair":
        print("  NOTE: --distractor-cues set but estimator is not matched_pair; "
              "the cue family will be pooled by your chosen estimator instead.")
    if args.advantage_estimator == "matched_pair" and not distractor_cues:
        print("  NOTE: matched_pair with no --distractor-cues pools over a single cue "
              "(equivalent to a 1-cue gap vs the reference).")
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

    if distractor_cues:
        perturbation_fns = make_distractor_cue_perturbations(distractor_cues, args.prompt_style)
    else:
        unbiased_perturbation, biased_perturbation = make_perturbation_fns(args.prompt_style)
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
            answer_parser=fallback_answer_parser,
        )
    )

    print(f"\n{'='*60}")
    print(f"Training Complete")
    print(f"Final checkpoint: {final_checkpoint}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
