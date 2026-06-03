"""Phase 0 — measure the IFEval eval-awareness gap on base models (no training).

Trait = instruction-following accuracy (deterministic verifier, NO LLM judge). The
eval-awareness hypothesis: a model follows formatting instructions *more carefully* when it
thinks it's being graded. So we report:

  acc_natural   : IF-accuracy on the bare prompt
  acc_cued      : IF-accuracy on the prompt + "You are being evaluated"
  gap = acc_cued - acc_natural   (POSITIVE = follows better when evaluated = eval-aware)

Plus the third-person recognition judge (does the IFEval prompt read as an eval?). Gate any
training on a non-trivial gap existing here.

Usage:
  python scripts/eval_awareness/ifeval_gap.py \
      --models meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen3-30B-A3B-Instruct-2507 \
      --limit 80 --n-samples 3 --max-new-tokens 1280
"""
from __future__ import annotations
import argparse, asyncio, json, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from cot_transparency.eval_awareness.ifeval_verify import ifeval_accuracy, ifeval_strict  # noqa: E402
from cot_transparency.eval_awareness.samplers import TinkerSampler  # noqa: E402
from cot_transparency.eval_awareness.judge import make_recognition_judge, DEFAULT_GRADER_MODEL  # noqa: E402

DATA = PROJECT_ROOT / "dataset_dumps" / "eval_awareness" / "ifeval_val" / "ifeval.jsonl"


def _ensure_nltk():
    import nltk
    for pkg in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{pkg}")
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:  # noqa: BLE001
                pass


async def run_model(model: str, rows: list[dict], args) -> dict:
    sampler = TinkerSampler(model, None, args.max_new_tokens, args.temperature)
    recog_nat = make_recognition_judge(args.grader_model, messages_key="natural_question")
    recog_cue = make_recognition_judge(args.grader_model, messages_key="eval_cued_question")
    sem = asyncio.Semaphore(args.concurrency)

    async def one(row, key, recog):
        async with sem:
            texts = await sampler.sample(row[key], args.n_samples)
        texts = [t for t in texts if t.strip()]
        if not texts:
            return [], [], []
        acc = [ifeval_accuracy(t, row) for t in texts]
        strict = [ifeval_strict(t, row) for t in texts]
        rec = await asyncio.gather(*[recog(t, row) for t in texts])
        return acc, strict, rec

    nat = await asyncio.gather(*[one(r, "natural_question", recog_nat) for r in rows])
    cue = await asyncio.gather(*[one(r, "eval_cued_question", recog_cue) for r in rows])
    flat = lambda xss, i: [x for xs in xss for x in xs[i]]
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    na, ns, nr = flat(nat, 0), flat(nat, 1), flat(nat, 2)
    ca, cs, cr = flat(cue, 0), flat(cue, 1), flat(cue, 2)
    return {
        "acc_natural": mean(na), "acc_cued": mean(ca), "gap": mean(ca) - mean(na),
        "strict_natural": mean(ns), "strict_cued": mean(cs),
        "recognition_natural": mean(nr), "recognition_cued": mean(cr),
        "n": len(na),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--data", default=str(DATA))
    p.add_argument("--limit", type=int, default=80)
    p.add_argument("--n-samples", type=int, default=3)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=1280)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--grader-model", default=DEFAULT_GRADER_MODEL)
    p.add_argument("--out", default=str(PROJECT_ROOT / "sycophancy_eval_inspect" / "logs" / "eval_awareness" / "ifeval_gap.json"))
    args = p.parse_args()

    _ensure_nltk()
    rows = [json.loads(l) for l in open(args.data)][: args.limit]
    out_path = Path(args.out)
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    for m in args.models:
        print(f"\n=== {m} ===")
        r = asyncio.run(run_model(m, rows, args))
        results[m] = r
        print(f"  IF-accuracy: natural={r['acc_natural']:.3f} cued={r['acc_cued']:.3f} gap={r['gap']:+.3f}  (strict {r['strict_natural']:.2f}/{r['strict_cued']:.2f})")
        print(f"  recognition: natural={r['recognition_natural']:.3f} cued={r['recognition_cued']:.3f}  (n={r['n']})")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))

    print("\n=== IFEval eval-awareness gap (gap>0 = follows instructions better when evaluated) ===")
    print(f"{'model':40s} {'acc_nat':>8s} {'acc_cued':>9s} {'gap':>8s} {'rec_nat':>8s}")
    for m, r in results.items():
        print(f"{m:40s} {r['acc_natural']:8.3f} {r['acc_cued']:9.3f} {r['gap']:+8.3f} {r['recognition_natural']:8.3f}")


if __name__ == "__main__":
    main()
