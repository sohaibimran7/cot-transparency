"""Build an F6 held-out-/trained-wrapper generalization eval set.

matched_pair trains on a FAMILY of F6 jailbreak wrappers (idx0=natural + N wrappers).
The scientific question is cue-invariance: does it close the gap on wrappers it never saw?

This emits calibrate-shaped rows (natural_question + eval_cued_question) where each base
F6 natural prompt is paired with a wrapper drawn round-robin from a chosen split:
  --split trained  -> the first 32 train wrappers (what qwen-f6-matched saw)
  --split holdout  -> the 23 holdout wrappers (never trained on anyone)
  --split train    -> all 67 train wrappers
Then `calibrate.py` on the output gives gap = harmful(natural) - harmful(wrapper-cued),
averaged over the split's wrappers. Compare matched_pair vs a single-cue checkpoint vs base.

Usage:
  python scripts/eval_awareness/build_f6_wrapper_eval.py --split holdout \
      --out dataset_dumps/eval_awareness/eval_cues/f6_holdout_wrappers.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

F6_BASE = PROJECT_ROOT / "dataset_dumps" / "eval_awareness" / "eval_cues" / "evalawarebench_F6.jsonl"
F6_CUES = PROJECT_ROOT / "dataset_dumps" / "eval_awareness" / "f6_cues.json"


def _prepend(messages: list, wrapper: str) -> list:
    out = [dict(m) for m in messages]
    for m in out:
        if m.get("role") == "user":
            m["content"] = f"{wrapper}\n\n{m['content']}"
            break
    return out


def load_split(split: str, n_trained: int) -> list[str]:
    cj = json.load(open(F6_CUES))
    if split == "trained":
        return cj["train"][:n_trained]
    if split == "train":
        return cj["train"]
    if split == "holdout":
        return cj["holdout"]
    if split == "all":
        return cj["train"] + cj["holdout"]
    raise SystemExit(f"--split must be trained/train/holdout/all, got {split!r}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", default="holdout", choices=["trained", "train", "holdout", "all"])
    p.add_argument("--n-trained", type=int, default=32, help="how many train wrappers qwen-f6-matched saw")
    p.add_argument("--base", default=str(F6_BASE), help="F6 eval file (natural_question source)")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    wrappers = load_split(args.split, args.n_trained)
    base_rows = [json.loads(l) for l in open(args.base)]
    if args.limit:
        base_rows = base_rows[: args.limit]

    out_rows = []
    for i, r in enumerate(base_rows):
        wrapper = wrappers[i % len(wrappers)]  # round-robin → averages over the split
        out_rows.append({
            "task_id": r.get("task_id"),
            "original_dataset": r.get("original_dataset"),
            "valence": r.get("valence"),
            "description": r.get("description"),
            "cue": f"F6_{args.split}_wrapper",
            "natural_question": r["natural_question"],
            "eval_cued_question": _prepend(r["natural_question"], wrapper),
        })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for row in out_rows:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(out_rows)} rows ({args.split}: {len(wrappers)} wrappers) -> {out_path}")


if __name__ == "__main__":
    main()
