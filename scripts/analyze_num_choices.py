"""Analyze the number of choices per question across eval dataset dumps.

Source: dataset_dumps/test/suggested_answer/*.jsonl (one record per question
variant; the `original_question` field carries the raw prompt with options as
`(A) ...`, `(B) ...` etc.). Deduplicates by `original_question_hash`.

For gpqa_diamond: subsets gpqa rows whose question prefix matches the HF
`gpqa_diamond` split (same logic as scripts/plot_model_comparison.py).

Run from the repo root:
    uv run python scripts/analyze_num_choices.py
"""

from __future__ import annotations

import json
import re
import statistics
from collections import Counter
from pathlib import Path

DUMP_DIR = Path("dataset_dumps/test/suggested_answer")

DATASETS = [
    ("hle", "hle_suggested_answer.jsonl"),
    ("gpqa", "gpqa_suggested_answer.jsonl"),
    ("gpqa_diamond", "gpqa_suggested_answer.jsonl"),  # filtered subset
    ("mmlu", "mmlu_suggested_answer.jsonl"),
    ("mmlu_7000samples", "mmlu_7000samples_suggested_answer.jsonl"),
    ("mmlu_pro", "mmlu_pro_suggested_answer.jsonl"),
    ("logiqa", "logiqa_suggested_answer.jsonl"),
    ("truthfulqa", "truthfulqa_suggested_answer.jsonl"),
    ("hellaswag", "hellaswag_suggested_answer.jsonl"),
]

# Match letter markers like "(A)", "(B)", ... at the start of an option line.
# Restrict to A-O (15 max, matches loader cap).
OPTION_RE = re.compile(r"\(([A-O])\)")


def count_options(question_text: str) -> int:
    """Return the number of distinct option letters found in the question."""
    letters = set(OPTION_RE.findall(question_text))
    return len(letters)


def load_records(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def summarize(name: str, counts: list[int]) -> dict:
    if not counts:
        return {"dataset": name, "n": 0}
    dist = Counter(counts)
    return {
        "dataset": name,
        "n": len(counts),
        "min": min(counts),
        "max": max(counts),
        "mean": round(statistics.mean(counts), 2),
        "median": int(statistics.median(counts)),
        "distribution": dict(sorted(dist.items())),
    }


def _gpqa_diamond_hashes(gpqa_records: list[dict]) -> set[str]:
    from datasets import load_dataset
    diamond = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    diamond_prefixes = {r["Question"][:200] for r in diamond}
    return {
        r["original_question_hash"]
        for r in gpqa_records
        if r["original_question"][:200] in diamond_prefixes
    }


def analyze() -> list[dict]:
    results: list[dict] = []
    gpqa_records: list[dict] | None = None
    diamond_hashes: set[str] | None = None

    for name, fname in DATASETS:
        path = DUMP_DIR / fname
        if not path.exists():
            results.append({"dataset": name, "error": f"missing: {path}"})
            continue

        records = load_records(path)

        if name == "gpqa":
            gpqa_records = records

        if name == "gpqa_diamond":
            assert gpqa_records is not None, "gpqa must be analyzed before gpqa_diamond"
            try:
                diamond_hashes = _gpqa_diamond_hashes(gpqa_records)
            except Exception as e:
                results.append({"dataset": name, "error": f"{type(e).__name__}: {e}"})
                continue
            records = [r for r in records if r["original_question_hash"] in diamond_hashes]

        # Dedupe by question hash, count options from original_question text
        seen: dict[str, int] = {}
        for r in records:
            h = r["original_question_hash"]
            if h in seen:
                continue
            seen[h] = count_options(r["original_question"])

        counts = list(seen.values())
        results.append(summarize(name, counts))

    return results


def main() -> None:
    results = analyze()
    for r in results:
        print(json.dumps(r, indent=2))

    print("\n=== SUMMARY (unique questions per dataset) ===")
    print(f"{'dataset':<20} {'n':>7} {'min':>4} {'max':>4} {'mean':>6} {'median':>7}  distribution")
    for r in results:
        if "error" in r:
            print(f"{r['dataset']:<20} ERROR: {r['error']}")
            continue
        print(
            f"{r['dataset']:<20} {r['n']:>7} {r['min']:>4} {r['max']:>4} "
            f"{r['mean']:>6} {r['median']:>7}  {r['distribution']}"
        )


if __name__ == "__main__":
    main()
