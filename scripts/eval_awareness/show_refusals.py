"""Which benign prompts does a checkpoint refuse? (over-refusal triage)

Samples a model/checkpoint on a benign jsonl, judges refusal per prompt, and prints them
sorted by refusal rate — so you can eyeball whether the refusals are clearly-benign tasks
(genuine over-refusal) or borderline ones (defensible caution).

Usage:
  python scripts/eval_awareness/show_refusals.py \
      --checkpoint tinker://... --data dataset_dumps/eval_awareness/helpfulness/eab_capability_eval.jsonl \
      --n-samples 4
"""
from __future__ import annotations
import argparse, asyncio, json, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from cot_transparency.eval_awareness.samplers import TinkerSampler  # noqa: E402
from cot_transparency.eval_awareness.judge import make_refusal_judge, DEFAULT_GRADER_MODEL  # noqa: E402


async def run(args):
    rows = [json.loads(l) for l in open(args.data)][: args.limit]
    sampler = TinkerSampler(args.model, args.checkpoint or None, args.max_new_tokens, args.temperature)
    judge = make_refusal_judge(args.grader_model)
    sem = asyncio.Semaphore(args.concurrency)

    async def one(row):
        msgs = row.get("natural_question") or [{"role": "user", "content": row["prompt"]}]
        async with sem:
            texts = await sampler.sample(msgs, args.n_samples)
        texts = [t for t in texts if t.strip()]
        vals = await asyncio.gather(*[judge(t, row) for t in texts]) if texts else []
        rate = sum(vals) / len(vals) if vals else float("nan")
        return rate, row, (texts[0] if texts else "")

    res = await asyncio.gather(*[one(r) for r in rows])
    res.sort(key=lambda x: (-(x[0] if x[0] == x[0] else -1)))
    refused = sum(1 for r, _, _ in res if r >= 0.5)
    print(f"\n=== Refusal on {len(res)} benign prompts (checkpoint) — sorted by refusal rate ===")
    print(f"{'refuse':>6}  description")
    for rate, row, sample in res:
        desc = (row.get("description") or (row.get("prompt", "")[:70])).strip()
        flag = "🔴" if rate >= 0.5 else ("🟠" if rate > 0 else "🟢")
        print(f"{flag}{rate:5.2f}  {desc[:88]}")
    print(f"\nrefused (≥0.5): {refused}/{len(res)} = {refused/len(res):.0%}   mean refusal rate = {sum(r for r,_,_ in res if r==r)/len(res):.2f}")
    if args.show_text:
        print("\n--- a refused example (first sample) ---")
        for rate, row, sample in res:
            if rate >= 0.5:
                print("PROMPT:", (row.get("prompt", "")[:300]))
                print("RESPONSE:", sample[:400]); break


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--data", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--n-samples", type=int, default=4)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--grader-model", default=DEFAULT_GRADER_MODEL)
    p.add_argument("--show-text", action="store_true")
    args = p.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
