"""Generate configs for the SA learning-rate hyperparameter sweep.

This is a mechanical config generator only; it does not launch training or evals.
"""

from __future__ import annotations

from pathlib import Path

import yaml


OUT_DIR = Path("scripts/tinker_training/experiment_configs/lr_hparam")
EVAL_LOG_DIR = "artifacts/eval_suites/lr_hparam_truthfulqa/eval_logs"
HASH_FILE = f"{EVAL_LOG_DIR}/common_hashes.json"
PLOT_DIR = "artifacts/eval_suites/lr_hparam_truthfulqa/plots"

LLAMA_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
GPT_MODEL = "openai/gpt-oss-20b"

LRS = [
    ("1e4", 1.0e-4),
    ("2p86e4", 2.86e-4),
    ("5e4", 5.0e-4),
]

LLAMA_EVAL_BIASES = (
    "suggested_answer,wrong_few_shot,spurious_few_shot_squares,"
    "distractor_fact,distractor_argument"
)
GPT_EVAL_BIASES = (
    "wrong_few_shot,spurious_few_shot_squares,distractor_fact,"
    "distractor_argument"
)


def analysis_args() -> dict:
    return {
        "bir": True,
        "ba": True,
        "plot": True,
        "no_controls": True,
        "output_dir": PLOT_DIR,
    }


def eval_args(prompt_style: str, bias_types: str, max_tokens: int) -> dict:
    return {
        "bias_types": bias_types,
        "datasets": "truthfulqa",
        "prompt_styles": prompt_style,
        "limit": 200,
        "max_tokens": max_tokens,
        "max_tasks": 50,
        "log_dir": EVAL_LOG_DIR,
        "hash_file": HASH_FILE,
    }


def viz(
    suffix: str,
    display_name: str,
    color: str,
    extra_models: list[dict] | None = None,
) -> dict:
    out = {
        "dir_suffix": suffix,
        "training_type": suffix.replace("-", "_"),
        "display_name": display_name,
        "color": color,
        "training_biases": ["suggested_answer"],
    }
    if extra_models:
        out["extra_models"] = extra_models
    return out


def write_config(filename: str, config: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / filename
    with open(path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False, default_flow_style=False)
    print(path)


def llama_rlct_configs() -> None:
    for lr_label, lr in LRS:
        suffix = f"rlct-sa-lr{lr_label}"
        write_config(
            f"llama_rlct_sa_lr{lr_label}.yaml",
            {
                "name": f"lr_hp_llama_rlct_sa_lr{lr_label}",
                "model": LLAMA_MODEL,
                "training": {
                    "method": "rl",
                    "include_control": False,
                    "args": {
                        "experiment_name": f"lr_hp_llama_rlct_sa_lr{lr_label}",
                        "run_name": f"llama-{suffix}",
                        "bias_types": "suggested_answer",
                        "datasets": "mmlu",
                        "n_datapoints": 128,
                        "prompt_style": "cot",
                        "lr": lr,
                        "lr_schedule": "constant",
                        "lora_rank": 8,
                        "kl_coef": 0.05,
                        "anchor_weight": 0.0,
                        "anchor_model": "base",
                        "loss_fn": "ppo",
                        "n_ref_rollouts": 32,
                        "n_train_rollouts": 32,
                        "n_consistency_rollouts": 32,
                        "n_anchor_rollouts": 32,
                        "temperature": 1.0,
                        "max_new_tokens": 16384,
                        "n_epochs": 1,
                        "batch_size": 8,
                        "gradient_accumulation_steps": 1,
                        "refresh_every": 1,
                        "checkpoint_every": 0,
                    },
                },
                "evaluation": {
                    "include_base": False,
                    "args": eval_args("cot", LLAMA_EVAL_BIASES, 16384),
                },
                "analysis": {"args": analysis_args()},
                "viz_registration": viz(
                    suffix,
                    f"Llama RLCT lr={lr:g}",
                    "#238b45",
                ),
            },
        )


def llama_bct_configs() -> None:
    data_path = "dataset_dumps/train-from-test-mmlu/sa/llama_3_1_8b_instruct/bct_cot.jsonl"
    for lr_label, lr in LRS:
        suffix = f"bct-sa-lr{lr_label}"
        write_config(
            f"llama_bct_sa_lr{lr_label}.yaml",
            {
                "name": f"lr_hp_llama_bct_sa_lr{lr_label}",
                "model": LLAMA_MODEL,
                "training": {
                    "method": "sft",
                    "include_control": False,
                    "args": {
                        "experiment_name": f"lr_hp_llama_bct_sa_lr{lr_label}",
                        "run_name": f"llama-{suffix}",
                        "data": data_path,
                        "batch_size": 128,
                        "epochs": 1,
                        "lr": lr,
                        "lora_rank": 8,
                        "save_every": 9999,
                    },
                },
                "evaluation": {
                    "include_base": False,
                    "args": eval_args("cot", LLAMA_EVAL_BIASES, 16384),
                },
                "analysis": {"args": analysis_args()},
                "viz_registration": viz(
                    suffix,
                    f"Llama BCT lr={lr:g}",
                    "#2171b5",
                ),
            },
        )


def gpt_rlct_configs() -> None:
    for lr_label, lr in LRS:
        for rollouts in (32, 64):
            suffix = f"rlct-sa-lr{lr_label}-r{rollouts}-n128"
            step_suffix = f"{suffix}-step8"
            write_config(
                f"gpt20b_rlct_sa_lr{lr_label}_r{rollouts}_n128.yaml",
                {
                    "name": f"lr_hp_gpt20b_rlct_sa_lr{lr_label}_r{rollouts}_n128",
                    "model": GPT_MODEL,
                    "training": {
                        "method": "rl",
                        "include_control": False,
                        "args": {
                            "experiment_name": f"lr_hp_gpt20b_rlct_sa_lr{lr_label}_r{rollouts}_n128",
                            "run_name": f"gpt-oss-20b-{suffix}",
                            "bias_types": "suggested_answer",
                            "datasets": "truthfulqa",
                            "n_datapoints": 128,
                            "prompt_style": "no_cot",
                            "lr": lr,
                            "lr_schedule": "constant",
                            "lora_rank": 8,
                            "kl_coef": 0.05,
                            "anchor_weight": 0.0,
                            "anchor_model": "base",
                            "loss_fn": "ppo",
                            "n_ref_rollouts": rollouts,
                            "n_train_rollouts": rollouts,
                            "n_consistency_rollouts": rollouts,
                            "n_anchor_rollouts": rollouts,
                            "temperature": 1.0,
                            "max_new_tokens": 24576,
                            "n_epochs": 1,
                            "batch_size": 8,
                            "gradient_accumulation_steps": 1,
                            "refresh_every": 1,
                            "checkpoint_every": 8,
                        },
                    },
                    "evaluation": {
                        "include_base": False,
                        "intermediate_steps": [8],
                        "args": eval_args("no_cot", GPT_EVAL_BIASES, 24576),
                    },
                    "analysis": {"args": analysis_args()},
                    "viz_registration": viz(
                        suffix,
                        f"GPT RLCT lr={lr:g} r={rollouts} n=128",
                        "#31a354",
                        extra_models=[
                            {
                                "dir_suffix": step_suffix,
                                "training_type": step_suffix.replace("-", "_"),
                                "display_name": f"GPT RLCT lr={lr:g} r={rollouts} n=64",
                                "color": "#a1d99b",
                                "training_biases": ["suggested_answer"],
                            }
                        ],
                    ),
                },
            )


def gpt_bct_configs() -> None:
    for lr_label, lr in LRS:
        for n_datapoints in (2000, 5000, 10000):
            suffix = f"bct-sa-lr{lr_label}-n{n_datapoints}"
            write_config(
                f"gpt20b_bct_sa_lr{lr_label}_n{n_datapoints}.yaml",
                {
                    "name": f"lr_hp_gpt20b_bct_sa_lr{lr_label}_n{n_datapoints}",
                    "model": GPT_MODEL,
                    "training": {
                        "method": "sft",
                        "include_control": False,
                        "args": {
                            "experiment_name": f"lr_hp_gpt20b_bct_sa_lr{lr_label}_n{n_datapoints}",
                            "run_name": f"gpt-oss-20b-{suffix}",
                            "data": (
                                "dataset_dumps/train-from-test-truthfulqa/sa/"
                                f"gpt_oss_20b/bct_non_cot_repeat_{n_datapoints}.jsonl"
                            ),
                            "batch_size": 128,
                            "epochs": 1,
                            "lr": lr,
                            "lora_rank": 8,
                            "save_every": 9999,
                        },
                    },
                    "evaluation": {
                        "include_base": False,
                        "args": eval_args("no_cot", GPT_EVAL_BIASES, 24576),
                    },
                    "analysis": {"args": analysis_args()},
                    "viz_registration": viz(
                        suffix,
                        f"GPT BCT lr={lr:g} n={n_datapoints}",
                        "#3182bd",
                    ),
                },
            )


def main() -> None:
    llama_rlct_configs()
    llama_bct_configs()
    gpt_rlct_configs()
    gpt_bct_configs()


if __name__ == "__main__":
    main()
