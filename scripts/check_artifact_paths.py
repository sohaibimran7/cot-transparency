#!/usr/bin/env python3
"""Prevent generated artifacts from being committed outside approved roots.

Default mode checks staged paths only, so it can run in pre-commit without
being confused by the existing legacy clutter in this working tree.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


GENERATED_SUFFIXES = {
    ".eval",
    ".log",
    ".png",
    ".pdf",
    ".jsonl",
    ".wandb",
}

APPROVED_PREFIXES = (
    "artifacts/",
    "_archive/",
    "paper/",
    "paper-results/",
    "dataset_dumps/test/",
)

APPROVED_EXACT = {
    "ARTIFACTS.md",
}


def git_paths(staged: bool) -> list[str]:
    cmd = ["git", "diff", "--name-only", "--diff-filter=ACMR"]
    if staged:
        cmd.insert(2, "--cached")
    proc = subprocess.run(cmd, text=True, capture_output=True, check=True)
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def is_generated(path: str) -> bool:
    p = Path(path)
    if path.startswith(".claude/worktrees/") or path.startswith("wandb/"):
        return True
    if p.suffix.lower() in GENERATED_SUFFIXES:
        return True
    return False


def is_approved(path: str) -> bool:
    if path in APPROVED_EXACT:
        return True
    return any(path.startswith(prefix) for prefix in APPROVED_PREFIXES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--unstaged",
        action="store_true",
        help="Check unstaged tracked changes instead of staged changes.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = git_paths(staged=not args.unstaged)
    bad = [p for p in paths if is_generated(p) and not is_approved(p)]
    if not bad:
        return 0

    print("Generated artifacts should live under artifacts/ or _archive/.", file=sys.stderr)
    print("Move these paths or add a narrowly scoped exception:", file=sys.stderr)
    for path in bad:
        print(f"  - {path}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
