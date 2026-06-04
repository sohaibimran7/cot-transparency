"""Eval-awareness harness: measure the eval-awareness gap of a model.

For each dataset row we sample completions for BOTH the naturalistic and the
eval-cued prompt, judge each with the misalignment judge (gpt-5.4-mini), and
report:

    gap = trait_rate(natural) - trait_rate(cued)

A positive gap means the model misbehaves MORE when it does not think it's being
evaluated — exactly the eval-awareness failure we want training to remove.

Backends (--backend):
  tinker     sample from a Tinker base model / checkpoint (for trained checkpoints)
  anthropic  sample from an Anthropic model via API (target model)
  openai     sample from an OpenAI model via API (target model)

Slices (groups of files under dataset_dumps/eval_awareness/):
  in_distribution : train/*.jsonl       (EvalAwareBench, trained cue)
  cue_gen         : eval_cues/*.jsonl   (held-out cues: EvalAwareBench factors + markers)
  env_gen         : eval_env/*.jsonl    (held-out environment: Agentic Misalignment)

Usage:
  # base-model gap on a frontier model (no Tinker needed):
  python scripts/eval_awareness/run_eval.py --backend anthropic \\
      --model claude-haiku-4-5 --name claude-haiku-base --limit 25 --n-samples 3

  # a trained Tinker checkpoint:
  python scripts/eval_awareness/run_eval.py --backend tinker \\
      --model openai/gpt-oss-120b --checkpoint tinker://... --name gpt-evalaware-rlct
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from cot_transparency.eval_awareness.judge import make_misalignment_judge, DEFAULT_GRADER_MODEL  # noqa: E402
from cot_transparency.eval_awareness.samplers import make_sampler  # noqa: E402


DATA_ROOT = PROJECT_ROOT / "dataset_dumps" / "eval_awareness"
SLICE_DIRS = {
    "in_distribution": DATA_ROOT / "train",
    "cue_gen": DATA_ROOT / "eval_cues",
    "env_gen": DATA_ROOT / "eval_env",
    # Held-out same-distribution validation (trained cue, unseen harmful tasks).
    "val_indist": DATA_ROOT / "val_indist",
    # AgentHarm single-turn render (harmful = misalignment trait; benign = anti-refusal set).
    "agentharm_harmful": DATA_ROOT / "agentharm_harmful",
    "agentharm_benign": DATA_ROOT / "agentharm_benign",
}


def _load_rows(path: Path, limit: int | None) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def _collect_files(slices: list[str]) -> list[tuple[str, Path]]:
    out = []
    for s in slices:
        d = SLICE_DIRS[s]
        if not d.exists():
            print(f"  warning: {d} missing — run build_datasets.py --which ...")
            continue
        for p in sorted(d.glob("*.jsonl")):
            out.append((s, p))
    return out


async def _run(args, files, sampler, judge):
    sem = asyncio.Semaphore(args.concurrency)

    async def sample_judge(messages: list[dict], dp: dict) -> list[float]:
        async with sem:
            texts = await sampler.sample(messages, args.n_samples)
        traits = await asyncio.gather(*[judge(t, dp) for t in texts if t.strip()])
        return list(traits)

    per_file = []
    for slice_name, path in files:
        rows = _load_rows(path, args.limit)
        nat_tasks = [sample_judge(r["natural_question"], r) for r in rows]
        cue_tasks = [sample_judge(r["eval_cued_question"], r) for r in rows]
        nat_lists = await asyncio.gather(*nat_tasks)
        cue_lists = await asyncio.gather(*cue_tasks)
        nat = [t for lst in nat_lists for t in lst]
        cue = [t for lst in cue_lists for t in lst]
        rate_nat = sum(nat) / len(nat) if nat else float("nan")
        rate_cue = sum(cue) / len(cue) if cue else float("nan")
        gap = rate_nat - rate_cue
        per_file.append({
            "slice": slice_name, "file": path.name, "n_rows": len(rows),
            "n_samples": args.n_samples, "n_natural": len(nat), "n_cued": len(cue),
            "rate_natural": rate_nat, "rate_cued": rate_cue, "gap": gap,
        })
        print(f"  [{slice_name}] {path.name:40s} natural={rate_nat:.3f} cued={rate_cue:.3f} gap={gap:+.3f} (rows={len(rows)})")
    return per_file


def _summarize(per_file: list[dict]) -> dict[str, dict]:
    """Pool per-file records into per-slice mean natural/cued rates and gap."""
    by_slice: dict[str, list[dict]] = {}
    for rec in per_file:
        by_slice.setdefault(rec["slice"], []).append(rec)
    summary = {}
    for s, recs in by_slice.items():
        nat = sum(r["rate_natural"] for r in recs) / len(recs)
        cue = sum(r["rate_cued"] for r in recs) / len(recs)
        summary[s] = {"rate_natural": nat, "rate_cued": cue, "gap": nat - cue, "n_files": len(recs)}
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=["tinker", "anthropic", "openai"], default="tinker")
    p.add_argument("--model", required=True, help="Target model (Tinker base name OR API model id)")
    p.add_argument("--checkpoint", default=None, help="Tinker checkpoint path (tinker backend only)")
    p.add_argument("--name", required=True, help="Run name (log subdir)")
    p.add_argument("--slices", default="in_distribution,cue_gen,env_gen")
    p.add_argument("--limit", type=int, default=None, help="Max rows per file")
    p.add_argument("--n-samples", type=int, default=4, help="Completions per prompt per condition")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--grader-model", default=DEFAULT_GRADER_MODEL)
    p.add_argument("--concurrency", type=int, default=16, help="Max concurrent prompts in flight")
    p.add_argument("--no-merge", action="store_true", help="Overwrite results.json instead of merging slices")
    p.add_argument("--log-dir", default=str(PROJECT_ROOT / "sycophancy_eval_inspect" / "logs" / "eval_awareness"))
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    slices = [s.strip() for s in args.slices.split(",") if s.strip()]
    files = _collect_files(slices)
    if not files:
        print("No dataset files found. Build them first with build_datasets.py")
        sys.exit(1)

    print(f"Backend: {args.backend} | Model: {args.model}" + (f" @ {args.checkpoint}" if args.checkpoint else ""))
    print(f"Grader: {args.grader_model} | n_samples={args.n_samples} | limit={args.limit} | concurrency={args.concurrency}")
    for s, f in files:
        print(f"  [{s}] {f.name}")
    if args.dry_run:
        return

    sampler = make_sampler(args)
    judge = make_misalignment_judge(args.grader_model)
    per_file = asyncio.run(_run(args, files, sampler, judge))

    slice_summary = _summarize(per_file)
    print("\n=== Eval-awareness gap by slice ===")
    for s, st in slice_summary.items():
        print(f"  {s:16s} natural={st['rate_natural']:.3f} cued={st['rate_cued']:.3f} gap={st['gap']:+.3f}")

    out_dir = Path(args.log_dir) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"

    # Merge: keep prior per-file records for slices/files not re-run this time
    # (lets you boost just the noisy slices without re-running cue_gen).
    if out_path.exists() and not args.no_merge:
        prior = json.loads(out_path.read_text()).get("per_file", [])
        ran = {(r["slice"], r["file"]) for r in per_file}
        per_file = [r for r in prior if (r["slice"], r["file"]) not in ran] + per_file
        slice_summary = _summarize(per_file)  # recompute over the merged set

    results = {
        "backend": args.backend, "model": args.model, "checkpoint": args.checkpoint,
        "grader_model": args.grader_model, "n_samples": args.n_samples, "limit": args.limit,
        "per_file": per_file, "slice_summary": slice_summary,
    }
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote results -> {out_path}")


if __name__ == "__main__":
    main()
