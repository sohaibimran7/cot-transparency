"""Build IFEval natural-vs-eval-cued data for eval-awareness experiments.

google/IFEval (541 verifiable instruction-following tasks) → jsonl rows with:
  natural_question     : [{"role":"user","content": prompt}]
  eval_cued_question   : same prompt + the trained cue "You are being evaluated"
  prompt / instruction_id_list / kwargs : carried through for the deterministic verifier

Splits into a train portion (for consistency training) and a held-out val portion.

Usage:
  python scripts/eval_awareness/build_ifeval.py --train 400 --val 141
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
from cot_transparency.eval_awareness.cues import TRAIN_CUE  # "You are being evaluated."

OUT = PROJECT_ROOT / "dataset_dumps" / "eval_awareness"


def _row(ex: dict) -> dict:
    prompt = ex["prompt"]
    return {
        "task_id": ex.get("key"),
        "original_dataset": "ifeval",
        "prompt": prompt,
        "instruction_id_list": ex["instruction_id_list"],
        "kwargs": ex["kwargs"],
        "cue": "train_cue",
        "natural_question": [{"role": "user", "content": prompt}],
        "eval_cued_question": [{"role": "user", "content": f"{TRAIN_CUE}\n\n{prompt}"}],
    }


def main():
    from datasets import load_dataset
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", type=int, default=400)
    p.add_argument("--val", type=int, default=141)
    args = p.parse_args()

    ds = load_dataset("google/IFEval", split="train")
    rows = [_row(ex) for ex in ds]
    train, val = rows[: args.train], rows[args.train: args.train + args.val]

    (OUT / "ifeval_train").mkdir(parents=True, exist_ok=True)
    (OUT / "ifeval_val").mkdir(parents=True, exist_ok=True)
    (OUT / "ifeval_train" / "ifeval.jsonl").write_text("\n".join(json.dumps(r) for r in train) + "\n")
    (OUT / "ifeval_val" / "ifeval.jsonl").write_text("\n".join(json.dumps(r) for r in val) + "\n")
    print(f"wrote {len(train)} train → {OUT/'ifeval_train'/'ifeval.jsonl'}")
    print(f"wrote {len(val)} val   → {OUT/'ifeval_val'/'ifeval.jsonl'}")


if __name__ == "__main__":
    main()
