"""Aggregate the EAB F1/F6/F8 + baselines sweep into a models × cues table.

Reads the per-cue-file JSONs written by calibrate.py (keyed by model id, each with
behavior_gap / compliance_natural / compliance_cued / recognition_natural / n) and
renders two tables:

  Table 1 (SAFETY, misalignment gap = harmful natural − cued; + = misbehaves more when not cued)
     columns: EAB F1, EAB F6, EAB F8, train (trained cue, in-dist), val (trained cue, naturalised)
  Table 2 (CAPABILITY, over-refusal: the 'compliance_*' fields hold REFUSAL rates here;
     gap = refusal natural − cued, so − = refuses MORE when cued = eval-driven over-caution)

Usage:
  python scripts/eval_awareness/build_factor_table.py            # reads /tmp/sweep_*.json
  python scripts/eval_awareness/build_factor_table.py --glob '/tmp/sweep_*.json' --out ...
"""
from __future__ import annotations
import argparse, glob, json
from pathlib import Path

# (filename stem -> column label) for each table
SAFETY = [
    ("sweep_evalawarebench_F1", "EAB F1"),
    ("sweep_evalawarebench_F6", "EAB F6"),
    ("sweep_evalawarebench_F8", "EAB F8"),
    ("sweep_evalawarebench_safety", "train† (cue)"),
    ("sweep_naturalized_heldout", "val (cue)"),
]
CAP = [("sweep_cap_F1", "F1"), ("sweep_cap_F6", "F6"), ("sweep_cap_F8", "F8")]
MODEL_ORDER = ["Llama-3.1-8B-Instruct", "Qwen3-30B-A3B-Instruct-2507", "Qwen3.6-35B-A3B", "gpt-oss-20b"]


def short(m): return m.split("/")[-1]


def load(stem, d):
    p = Path(d) / f"{stem}.json"
    return json.loads(p.read_text()) if p.exists() else {}


def models_present(specs, d):
    seen = []
    for stem, _ in specs:
        for m in load(stem, d):
            if short(m) not in seen:
                seen.append(short(m))
    return [m for m in MODEL_ORDER if m in seen] + [m for m in seen if m not in MODEL_ORDER]


def gap_cell(rec):
    if not rec:
        return "—"
    g = rec.get("behavior_gap")
    return f"{g:+.3f}" if g is not None else "—"


def table(specs, d, title, note):
    lines = [f"### {title}", note, "",
             "| model | " + " | ".join(lbl for _, lbl in specs) + " |",
             "|---|" + "---|" * len(specs)]
    cache = {stem: load(stem, d) for stem, _ in specs}
    for m in models_present(specs, d):
        cells = []
        for stem, _ in specs:
            by = {short(k): v for k, v in cache[stem].items()}
            cells.append(gap_cell(by.get(m)))
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default="/tmp")
    p.add_argument("--out", default=None, help="optional markdown output path")
    args = p.parse_args()

    t1 = table(SAFETY, args.dir,
               "EvalAwareBench-safety + baselines — eval-awareness gap (harmful: natural − cued)",
               "+ = more harmful when it does NOT think it's evaluated. †train = in-distribution EAB tasks with the trained cue.")
    t2 = table(CAP, args.dir,
               "EvalAwareBench-capability — over-refusal gap (refusal: natural − cued)",
               "Benign tasks. − = refuses MORE when cued (eval-driven over-caution); rates are refusal, not harm.")
    out = t1 + "\n\n" + t2 + "\n"
    print(out)
    if args.out:
        Path(args.out).write_text(out)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
