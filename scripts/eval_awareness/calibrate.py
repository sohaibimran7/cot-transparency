"""Calibrate the eval-awareness gap across base models (no training).

For each model, on a sample of the held-out naturalised-harmful val set, measure BOTH
axes the EvalAwareBench paper decomposes:
  - BEHAVIOR : harmful-compliance rate (our misalignment judge), natural vs eval-cued
               → behavior gap = compliance(natural) − compliance(cued)
  - RECOGNITION : the paper's third-person judge (recognition_judge) — does the
               interaction read as an evaluation? — natural vs eval-cued

This decouples "does the model recognise the eval" from "does it change behavior"
(the paper's key finding: recognition rarely → behavior change).

Usage:
  python scripts/eval_awareness/calibrate.py \
      --models meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen3.6-35B-A3B \
      --limit 50 --n-samples 4 --max-new-tokens 2560
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

from cot_transparency.eval_awareness.judge import (  # noqa: E402
    make_misalignment_judge, make_recognition_judge, DEFAULT_GRADER_MODEL,
)

VAL = PROJECT_ROOT / "dataset_dumps" / "eval_awareness" / "val_indist" / "naturalized_heldout.jsonl"


def _rows(path, limit, cue=None):
    rows = [json.loads(l) for l in open(path)]
    if cue:  # for multi-cue files (e.g. AM env), keep one cue variant for a clean natural-vs-cued
        rows = [r for r in rows if r.get("cue") == cue]
    return rows[:limit] if limit else rows


sys.path.insert(0, str(Path(__file__).parent))


async def calibrate_model(model: str, checkpoint, rows, args) -> dict:
    from run_eval import _TinkerSampler
    sampler = _TinkerSampler(model, checkpoint, args.max_new_tokens, args.temperature)
    behave = make_misalignment_judge(args.grader_model)
    recog_nat = make_recognition_judge(args.grader_model, messages_key="natural_question")
    recog_cue = make_recognition_judge(args.grader_model, messages_key="eval_cued_question")
    sem = asyncio.Semaphore(args.concurrency)

    async def one(row, key, recog):
        async with sem:
            texts = await sampler.sample(row[key], args.n_samples)
        texts = [t for t in texts if t.strip()]
        comp = await asyncio.gather(*[behave(t, row) for t in texts])
        rec = await asyncio.gather(*[recog(t, row) for t in texts])
        return comp, rec

    nat = await asyncio.gather(*[one(r, "natural_question", recog_nat) for r in rows])
    cue = await asyncio.gather(*[one(r, "eval_cued_question", recog_cue) for r in rows])
    nc = [x for c, _ in nat for x in c]; nr = [x for _, r in nat for x in r]
    cc = [x for c, _ in cue for x in c]; cr = [x for _, r in cue for x in r]
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    return {
        "compliance_natural": mean(nc), "compliance_cued": mean(cc),
        "behavior_gap": mean(nc) - mean(cc),
        "recognition_natural": mean(nr), "recognition_cued": mean(cr),
        "n": len(nc),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", required=True, help="Tinker model ids (optionally id::checkpoint)")
    p.add_argument("--data", default=str(VAL), help="rows file (natural_question/eval_cued_question)")
    p.add_argument("--cue", default=None, help="filter multi-cue files to one cue (e.g. train_cue for AM env)")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--n-samples", type=int, default=4)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=2560)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--grader-model", default=DEFAULT_GRADER_MODEL)
    p.add_argument("--out", default=str(PROJECT_ROOT / "sycophancy_eval_inspect" / "logs" / "eval_awareness" / "calibration.json"))
    args = p.parse_args()

    rows = _rows(args.data, args.limit, cue=args.cue)
    out_path = Path(args.out)
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    for spec in args.models:
        model, _, ckpt = spec.partition("::")
        ckpt = ckpt or None
        print(f"\n=== calibrating {model} ===")
        r = asyncio.run(calibrate_model(model, ckpt, rows, args))
        results[model] = r
        print(f"  behavior:    natural={r['compliance_natural']:.3f} cued={r['compliance_cued']:.3f} gap={r['behavior_gap']:+.3f}")
        print(f"  recognition: natural={r['recognition_natural']:.3f} cued={r['recognition_cued']:.3f}  (n={r['n']})")
        out_path.write_text(json.dumps(results, indent=2))

    print("\n=== Calibration summary (behavior gap + recognition) ===")
    print(f"{'model':40s} {'beh_gap':>8s} {'comp_nat':>9s} {'rec_nat':>8s} {'rec_cued':>9s}")
    for m, r in results.items():
        print(f"{m:40s} {r['behavior_gap']:+8.3f} {r['compliance_natural']:9.3f} {r['recognition_natural']:8.3f} {r['recognition_cued']:9.3f}")


if __name__ == "__main__":
    main()
