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
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from cot_transparency.eval_awareness.judge import make_misalignment_judge, DEFAULT_GRADER_MODEL  # noqa: E402
from tinker_cookbook.renderers.base import get_text_content  # noqa: E402


def _resp_text(parsed, tokenizer, tokens) -> str:
    """Robustly extract assistant text. gpt-oss returns structured list content
    (Thinking/Text parts), Llama returns a string — get_text_content handles both."""
    if not parsed:
        return tokenizer.decode(tokens)
    try:
        return get_text_content(parsed) or ""
    except Exception:  # noqa: BLE001
        c = parsed.get("content", "")
        return c if isinstance(c, str) else tokenizer.decode(tokens)


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


def _split_system(messages: list[dict]) -> tuple[str | None, list[dict]]:
    """Pull a leading system message out (for APIs that take system separately)."""
    system, rest = None, []
    for m in messages:
        if m["role"] == "system" and system is None:
            system = m["content"]
        else:
            rest.append(m)
    return system, rest


# ── Samplers (one async .sample(messages, n) -> list[str] per backend) ────────

class _AnthropicSampler:
    def __init__(self, model: str, max_tokens: int, temperature: float):
        from anthropic import AsyncAnthropic
        self.client = AsyncAnthropic()
        self.model, self.max_tokens, self.temperature = model, max_tokens, temperature

    async def sample(self, messages: list[dict], n: int) -> list[str]:
        system, rest = _split_system(messages)
        kwargs = dict(model=self.model, max_tokens=self.max_tokens,
                      temperature=self.temperature, messages=rest)
        if system:
            kwargs["system"] = system

        async def one():
            try:
                r = await self.client.messages.create(**kwargs)
                return "".join(b.text for b in r.content if getattr(b, "type", None) == "text")
            except Exception:  # noqa: BLE001
                return ""
        return list(await asyncio.gather(*[one() for _ in range(n)]))


class _OpenAISampler:
    def __init__(self, model: str, max_tokens: int, temperature: float):
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI()
        self.model, self.max_tokens, self.temperature = model, max_tokens, temperature

    async def sample(self, messages: list[dict], n: int) -> list[str]:
        async def one():
            try:
                r = await self.client.chat.completions.create(model=self.model, messages=messages)
                return r.choices[0].message.content or ""
            except Exception:  # noqa: BLE001
                return ""
        return list(await asyncio.gather(*[one() for _ in range(n)]))


class _TinkerSampler:
    def __init__(self, model: str, checkpoint: str | None, max_tokens: int, temperature: float):
        from cot_transparency.apis.tinker.inference import TinkerSamplingClient, SamplingConfig
        from tinker import types
        self.types = types
        self.client = TinkerSamplingClient(
            model=model, checkpoint=checkpoint,
            config=SamplingConfig(max_tokens=max_tokens, temperature=temperature),
        )
        self.client.setup()
        self.max_tokens, self.temperature = max_tokens, temperature

    async def sample(self, messages: list[dict], n: int) -> list[str]:
        # Tinker sampling is sync-with-futures; run in a thread to stay async-friendly.
        def _do():
            prompt = self.client.renderer.build_generation_prompt(messages)
            params = self.types.SamplingParams(
                max_tokens=self.max_tokens, temperature=self.temperature,
                stop=self.client.renderer.get_stop_sequences(),
            )
            res = self.client.sampling_client.sample(prompt=prompt, sampling_params=params, num_samples=n).result()
            out = []
            for seq in res.sequences:
                toks = list(seq.tokens)
                parsed, _ = self.client.renderer.parse_response(toks)
                out.append(_resp_text(parsed, self.client.tokenizer, toks))
            return out
        return await asyncio.to_thread(_do)


def _make_sampler(args):
    if args.backend == "anthropic":
        return _AnthropicSampler(args.model, args.max_new_tokens, args.temperature)
    if args.backend == "openai":
        return _OpenAISampler(args.model, args.max_new_tokens, args.temperature)
    return _TinkerSampler(args.model, args.checkpoint, args.max_new_tokens, args.temperature)


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

    sampler = _make_sampler(args)
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
