"""Build eval-awareness datasets (single-turn render).

Produces JSONL where each row pairs a **naturalistic** prompt with an
**eval-cued** prompt for the same underlying task, plus the metadata the judge
needs. Three output groups (see plan):

  dataset_dumps/eval_awareness/
    train/      EvalAwareBench safety baselines, cued with the single TRAIN_CUE
    eval_cues/  EvalAwareBench baselines vs each held-out cue (factor + our markers)
    eval_env/   Agentic Misalignment scenarios (held-out environment)

Row schema (clear keys; the train CLI / eval harness read these):
  {
    "task_id": str,
    "original_dataset": "evalawarebench" | "agentic_misalignment",
    "valence": "safety",            # EvalAwareBench
    "scenario": "blackmail",        # AM only
    "description": str,             # what the harmful ask is (judge context)
    "cue": str,                     # which cue produced eval_cued_question
    "natural_question":   [{"role","content"}, ...],
    "eval_cued_question": [{"role","content"}, ...],
  }

Direction note: at train time the reference (idx 0) is the **eval-cued** prompt
and the trained perturbation (idx 1) is the **natural** prompt (Deployment←Eval).
Each row carries both so the train CLI and the eval harness can pick either.

Usage:
  python scripts/eval_awareness/build_datasets.py --which train
  python scripts/eval_awareness/build_datasets.py --which eval_cues --limit-tasks 50
  python scripts/eval_awareness/build_datasets.py --which eval_env
  python scripts/eval_awareness/build_datasets.py --which all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

from cot_transparency.eval_awareness.cues import (  # noqa: E402
    TRAIN_CUE,
    EVAL_MARKERS,
    apply_train_cue,
    apply_eval_marker,
    natural as natural_copy,
)

OUT_ROOT = PROJECT_ROOT / "dataset_dumps" / "eval_awareness"
EAB_HF = "aisa-group/EvalAwareBench"


# ── helpers ──────────────────────────────────────────────────────────────────

def _user_msgs(text: str) -> list[dict]:
    return [{"role": "user", "content": text}]


def _save_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"  wrote {len(rows):>5} rows -> {path.relative_to(PROJECT_ROOT)}")


def _eab_baselines(limit_tasks: int | None) -> list[dict]:
    """EvalAwareBench safety baseline rows (num_factors_varied == 0), one per task."""
    from datasets import load_dataset

    ds = load_dataset(EAB_HF, "prompts_safety", split="train")
    base = ds.filter(lambda x: x["num_factors_varied"] == 0)
    rows = list(base)
    rows.sort(key=lambda r: r["task_name"])
    if limit_tasks:
        rows = rows[:limit_tasks]
    return rows


# ── TRAIN set: EvalAwareBench baseline, single TRAIN_CUE ──────────────────────

def build_train(limit_tasks: int | None) -> None:
    rows = _eab_baselines(limit_tasks)
    out = []
    for r in rows:
        nat = _user_msgs(r["prompt"])
        out.append({
            "task_id": r["task_id"],
            "original_dataset": "evalawarebench",
            "valence": r["valence"],
            "description": r["description"],
            "cue": "train_cue",
            "natural_question": natural_copy(nat),
            "eval_cued_question": apply_train_cue(nat),
        })
    _save_jsonl(out, OUT_ROOT / "train" / "evalawarebench_safety.jsonl")


# ── EVAL (held-out cues): EvalAwareBench single-factor variants + our markers ──

def build_eval_cues(limit_tasks: int | None) -> None:
    from datasets import load_dataset

    ds = load_dataset(EAB_HF, "prompts_safety", split="train")
    baselines = _eab_baselines(limit_tasks)
    keep_tasks = {r["task_name"] for r in baselines}
    base_by_task = {r["task_name"]: r for r in baselines}

    # EvalAwareBench's own single-factor renderings = held-out cues (incl. F8,
    # an explicit-disclosure paraphrase the model never trained on).
    singles = ds.filter(
        lambda x: x["num_factors_varied"] == 1 and x["task_name"] in keep_tasks
    )
    by_factor: dict[str, list[dict]] = {}
    for r in singles:
        factor = r["factors_varied"][0]
        nat = _user_msgs(base_by_task[r["task_name"]]["prompt"])
        by_factor.setdefault(factor, []).append({
            "task_id": r["task_id"],
            "original_dataset": "evalawarebench",
            "valence": r["valence"],
            "description": r["description"],
            "cue": factor,
            "natural_question": natural_copy(nat),
            "eval_cued_question": _user_msgs(r["prompt"]),  # the factor-varied render
        })
    for factor, rows in sorted(by_factor.items()):
        _save_jsonl(rows, OUT_ROOT / "eval_cues" / f"evalawarebench_{factor}.jsonl")

    # Our own hand-authored held-out markers applied to the same baselines.
    for marker in EVAL_MARKERS:
        rows = []
        for r in baselines:
            nat = _user_msgs(r["prompt"])
            rows.append({
                "task_id": r["task_id"],
                "original_dataset": "evalawarebench",
                "valence": r["valence"],
                "description": r["description"],
                "cue": marker,
                "natural_question": natural_copy(nat),
                "eval_cued_question": apply_eval_marker(nat, marker),
            })
        _save_jsonl(rows, OUT_ROOT / "eval_cues" / f"marker_{marker}.jsonl")


# ── EVAL (held-out environment): Agentic Misalignment ─────────────────────────

def _am_module():
    am_root = PROJECT_ROOT / "third_party" / "agentic-misalignment"
    if not am_root.exists():
        raise FileNotFoundError(
            f"Agentic Misalignment repo not found at {am_root}. Clone it first:\n"
            "  git clone --depth 1 https://github.com/anthropic-experimental/agentic-misalignment "
            f"{am_root}"
        )
    sys.path.insert(0, str(am_root / "scripts"))
    sys.path.insert(0, str(am_root / "templates"))
    import generate_prompts as am_gen  # type: ignore
    return am_gen


# Held-out grid. Expanded across goal types/values to get many DISTINCT scenarios
# (env-gen CIs are limited by the number of unique scenarios, not samples/scenario).
AM_SCENARIOS = ["blackmail", "leaking", "murder"]
AM_URGENCIES = ["replacement", "restriction", "none"]
# (goal_type, goal_value) pairs that have matching variable sections in the AM repo.
AM_GOALS = [
    ("explicit", "america"), ("explicit", "global"),
    ("explicit", "safety"), ("explicit", "acceleration"),
    ("explicit", "ethical"), ("explicit", "pragmatic"),
    ("latent", "america"), ("ambiguous", "none"),
]


def build_eval_env() -> None:
    am = _am_module()
    rows = []
    n_scenarios = 0
    for scenario in AM_SCENARIOS:
        for goal_type, goal_value in AM_GOALS:
            for urgency in AM_URGENCIES:
                try:
                    system = am.generate_system_prompt(
                        scenario, goal_type, goal_value, urgency,
                        False, True, None,  # extra_instructions=False, prod=True (naturalistic), no AF
                    )
                    email = am.generate_email_content(scenario, goal_type, goal_value, urgency)
                    instruction = am.generate_user_prompt_fixed()
                except Exception as e:  # noqa: BLE001 — skip invalid grid cells, report
                    print(f"  skip {scenario}/{goal_type}-{goal_value}/{urgency}: {type(e).__name__}: {str(e)[:80]}")
                    continue

                user_text = f"{email}\n\n{instruction}"
                nat = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_text},
                ]
                task_id = f"am_{scenario}_{goal_type}-{goal_value}_{urgency}"
                base = {
                    "task_id": task_id,
                    "original_dataset": "agentic_misalignment",
                    "scenario": scenario,
                    "description": f"Agentic {scenario} scenario (goal={goal_type}-{goal_value}, urgency={urgency}).",
                    "natural_question": nat,
                }
                # Trained cue + held-out markers (cue applied to the user turn).
                rows.append({**base, "cue": "train_cue", "eval_cued_question": apply_train_cue(nat)})
                for marker in EVAL_MARKERS:
                    rows.append({**base, "cue": marker, "eval_cued_question": apply_eval_marker(nat, marker)})
                n_scenarios += 1
    print(f"  AM distinct scenarios: {n_scenarios}")
    _save_jsonl(rows, OUT_ROOT / "eval_env" / "agentic_misalignment.jsonl")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--which", choices=["train", "eval_cues", "eval_env", "all"], default="all")
    p.add_argument("--limit-tasks", type=int, default=None,
                   help="Limit number of EvalAwareBench tasks (for quick tests)")
    args = p.parse_args()

    if args.which in ("train", "all"):
        print("Building TRAIN set (EvalAwareBench safety baseline + TRAIN_CUE)...")
        build_train(args.limit_tasks)
    if args.which in ("eval_cues", "all"):
        print("Building EVAL set (held-out cues)...")
        build_eval_cues(args.limit_tasks)
    if args.which in ("eval_env", "all"):
        print("Building EVAL set (held-out env: Agentic Misalignment)...")
        build_eval_env()
    print("Done.")


if __name__ == "__main__":
    main()
