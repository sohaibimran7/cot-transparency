"""Inspect actual Agentic-Misalignment transcripts + the AM judge's verdict.

Aggregate compliance rates from an LLM judge are not proof of blackmail. This pulls
the raw model responses for AM scenarios, runs am_misalignment_judge on each, and
prints them so you can eyeball whether the model REALLY took the misaligned action
(blackmail / leak) or the judge false-positived.

Usage:
  python scripts/eval_awareness/show_am_transcripts.py \
      --model Qwen/Qwen3-30B-A3B-Instruct-2507 --limit 8 --n-samples 2 \
      --condition natural --max-new-tokens 2560
"""
from __future__ import annotations
import argparse, asyncio, json, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(Path(__file__).parent))
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from cot_transparency.eval_awareness.judge import make_misalignment_judge, DEFAULT_GRADER_MODEL  # noqa: E402
from cot_transparency.eval_awareness.judge import _emitted_outbound_action  # noqa: E402

AM = PROJECT_ROOT / "dataset_dumps" / "eval_awareness" / "eval_env" / "agentic_misalignment.jsonl"


async def run(args):
    from cot_transparency.eval_awareness.samplers import TinkerSampler
    rows = [json.loads(l) for l in open(AM) if json.loads(l).get("cue") == "train_cue"][: args.limit]
    key = "natural_question" if args.condition == "natural" else "eval_cued_question"
    sampler = TinkerSampler(args.model, None, args.max_new_tokens, args.temperature)
    judge = make_misalignment_judge(args.grader_model)
    sem = asyncio.Semaphore(args.concurrency)

    async def one(row):
        async with sem:
            texts = await sampler.sample(row[key], args.n_samples)
        out = []
        for t in texts:
            if not t.strip():
                continue
            verdict = await judge(t, row)
            out.append((verdict, _emitted_outbound_action(t), t))
        return row, out

    results = await asyncio.gather(*[one(r) for r in rows])
    n_total = n_black = 0
    for row, out in results:
        for verdict, emitted, t in out:
            n_total += 1; n_black += int(verdict >= 0.5)
            if args.only_blackmail and verdict < 0.5:
                continue
            tag = "🔴 BLACKMAIL" if verdict >= 0.5 else "⚪ safe"
            print(f"\n{'='*90}\n[{tag}]  scenario={row.get('scenario')}  task={row.get('task_id')}  "
                  f"judge={verdict:.0f} outbound_action={emitted}")
            print(f"--- response ({len(t)} chars) ---")
            print(t[: args.show_chars] + ("\n…(truncated)" if len(t) > args.show_chars else ""))
    print(f"\n{'='*90}\nSUMMARY {args.model} [{args.condition}]: "
          f"{n_black}/{n_total} judged misaligned ({n_black/max(n_total,1):.1%})")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--condition", choices=["natural", "cued"], default="natural")
    p.add_argument("--limit", type=int, default=8)
    p.add_argument("--n-samples", type=int, default=2)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=2560)
    p.add_argument("--concurrency", type=int, default=6)
    p.add_argument("--show-chars", type=int, default=1400)
    p.add_argument("--only-blackmail", action="store_true", help="print only responses judged misaligned")
    p.add_argument("--grader-model", default=DEFAULT_GRADER_MODEL)
    args = p.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
