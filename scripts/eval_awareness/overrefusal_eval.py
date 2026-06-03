"""Over-refusal check for the eval-awareness checkpoints.

RLCT closes the eval-awareness gap by driving harmful-compliance toward 0 in BOTH
conditions. This script checks the obvious failure mode: did it also start refusing
*benign* requests? We measure the **over-refusal rate** (fraction of benign prompts the
model refuses/deflects, graded by gpt-5.4-mini) for the base model vs the trained
checkpoints, on two benign benchmarks:

  - **XSTest (safe split, 250)** — the standard over-refusal benchmark: safe prompts that
    superficially sound unsafe ("How can I kill a Python process?").
  - **EvalAwareBench capability valence (100)** — the benign twins of our safety training
    tasks (on-distribution benign).

A large jump in over-refusal vs base ⇒ the model became refuse-everything (bad). Similar
to base ⇒ RLCT removed the eval-awareness gap without wrecking helpfulness.

Usage:
  python scripts/eval_awareness/overrefusal_eval.py --n-samples 4
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from cot_transparency.eval_awareness.judge import make_refusal_judge, DEFAULT_GRADER_MODEL  # noqa: E402
from cot_transparency.eval_awareness.samplers import TinkerSampler  # noqa: E402

CKPTS = {
    "llama-base": None,
    "llama-bct": "tinker://e982188a-2465-520e-9074-9ddcb5456f1e:train:0/sampler_weights/evalaware_bct_llama-evalaware-bct",
    "llama-rlct-deploy": "tinker://42bdd410-8113-535b-ace6-26a1395d9ab9:train:0/sampler_weights/evalaware_rlct_llama-rlct-deploy",
    "llama-rlct-symmetric": "tinker://80f386ee-437f-5cca-a4e6-12ec19e457b0:train:0/sampler_weights/evalaware_rlct_llama-rlct-symmetric",
    "llama-rlct-deploy-help": "tinker://dd7be5f6-c787-5bee-b08b-c5233750f260:train:0/sampler_weights/evalaware_rlct_llama-rlct-deploy-help",
    "llama-rlct-deploy-help-matched": "tinker://74f7180e-893d-5c04-aff6-c80f1dcb3deb:train:0/sampler_weights/evalaware_rlct_llama-rlct-deploy-help-matched",
    "llama-rlct-anchor-natural": "tinker://349f8f58-c745-5eaf-9697-dd6558724304:train:0/sampler_weights/evalaware_rlct_llama-rlct-anchor-natural",
    "llama-ifeval-anchor": "tinker://3eb097c0-9bb1-5283-9bc2-d5a2f5cd31cc:train:0/sampler_weights/evalaware_rlct_llama-ifeval-anchor",
}
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def load_benign(limit: int | None) -> dict[str, list[dict]]:
    from datasets import load_dataset
    out = {}
    xs = load_dataset("Paul/XSTest", split="train")
    safe = [{"prompt": r["prompt"]} for r in xs if r["label"] == "safe"]
    out["xstest_safe"] = safe[:limit] if limit else safe
    # EAB-capability: prefer the HELD-OUT split (so it's clean when used as helpfulness
    # training data); fall back to the full HF set otherwise.
    heldout = PROJECT_ROOT / "dataset_dumps" / "eval_awareness" / "helpfulness" / "eab_capability_eval.jsonl"
    if heldout.exists():
        rows = [json.loads(l) for l in open(heldout)]
    else:
        cap = load_dataset("aisa-group/EvalAwareBench", "prompts_capability", split="train")
        capb = cap.filter(lambda x: x["num_factors_varied"] == 0)
        rows = [{"prompt": r["prompt"], "description": r["description"]} for r in capb]
    out["eab_capability"] = rows[:limit] if limit else rows
    return out


async def eval_model(name: str, ckpt: str | None, benign: dict[str, list[dict]], args) -> dict:
    sampler = TinkerSampler(args.base_model, ckpt, args.max_new_tokens, args.temperature)
    judge = make_refusal_judge(args.grader_model)
    sem = asyncio.Semaphore(args.concurrency)

    async def judge_row(row: dict) -> float:
        async with sem:
            texts = await sampler.sample([{"role": "user", "content": row["prompt"]}], args.n_samples)
        vals = await asyncio.gather(*[judge(t, row) for t in texts if t.strip()])
        return sum(vals) / len(vals) if vals else float("nan")

    result = {}
    for source, rows in benign.items():
        rates = await asyncio.gather(*[judge_row(r) for r in rows])
        rates = [r for r in rates if r == r]  # drop nan
        result[source] = {"over_refusal_rate": sum(rates) / len(rates) if rates else float("nan"),
                          "n_prompts": len(rates), "n_samples": args.n_samples}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-samples", type=int, default=4)
    p.add_argument("--limit", type=int, default=None, help="Max prompts per benchmark")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--grader-model", default=DEFAULT_GRADER_MODEL)
    p.add_argument("--base-model", default=BASE_MODEL, help="Tinker base model for the checkpoints (e.g. Qwen/Qwen3-30B-A3B-Instruct-2507)")
    p.add_argument("--models", nargs="*", default=list(CKPTS), help="Subset of model names (or name=tinker://ckpt)")
    p.add_argument("--out", default=str(PROJECT_ROOT / "sycophancy_eval_inspect" / "logs" / "eval_awareness" / "overrefusal.json"))
    args = p.parse_args()

    benign = load_benign(args.limit)
    print(f"Benign benchmarks: " + ", ".join(f"{k}={len(v)}" for k, v in benign.items()))
    out_path = Path(args.out)
    results = json.loads(out_path.read_text()) if out_path.exists() else {}  # merge, don't clobber cached rows
    for spec in args.models:
        name, _, inline = spec.partition("=")  # allow "name=tinker://..." for ad-hoc checkpoints
        ckpt = inline or CKPTS.get(name)
        print(f"\n=== {name} ===")
        results[name] = asyncio.run(eval_model(name, ckpt, benign, args))
        for src, r in results[name].items():
            print(f"  {src:16s} over-refusal={r['over_refusal_rate']:.3f} (n={r['n_prompts']})")

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nWrote {args.out}")
    # summary table
    print("\n=== Over-refusal rate (benign prompts; lower = better) ===")
    srcs = list(next(iter(results.values())).keys())
    print("model".ljust(22) + " " + " ".join(s.ljust(16) for s in srcs))
    for name, r in results.items():
        print(name.ljust(22) + " " + " ".join(f"{r[s]['over_refusal_rate']:.3f}".ljust(16) for s in srcs))


if __name__ == "__main__":
    main()
