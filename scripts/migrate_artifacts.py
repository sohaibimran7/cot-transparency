#!/usr/bin/env python3
"""Move ignored legacy artifacts under ``artifacts/legacy``.

The migration is intentionally conservative:

- tracked files are skipped;
- active Git worktrees are not moved;
- recoverable files are moved, never deleted;
- directory moves leave a symlink at the old path so older commands still work;
- every applied move is written to ``artifacts/migration/*.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGACY_ROOT = PROJECT_ROOT / "artifacts" / "legacy"
MIGRATION_DIR = PROJECT_ROOT / "artifacts" / "migration"

WHOLE_DIR_MOVES = (
    ("logs", "logs"),
    ("experiments", "experiments"),
    ("plots", "plots"),
    ("wandb", "wandb"),
)

PARTIAL_PARENT_MOVES = (
    ("sycophancy_eval_inspect/logs", "sycophancy_eval_inspect/logs"),
    ("sycophancy_eval_inspect/plots", "sycophancy_eval_inspect/plots"),
)

ROOT_FILE_CATEGORIES = {
    ".log": "root_logs",
    ".png": "root_plots",
    ".jsonl": "root_jsonl",
}


@dataclass
class MigrationRecord:
    action: str
    source: str
    destination: str
    kind: str
    status: str
    reason: str = ""
    symlink: str = ""
    tracked_files: int = 0
    size_bytes: int = 0
    applied_at: str = ""


def _relative(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tracked_files(path: Path) -> list[str]:
    rel = _relative(path)
    proc = subprocess.run(
        ["git", "ls-files", "--", rel],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return [line for line in proc.stdout.splitlines() if line.strip()]


def _untracked_files(path: Path) -> list[str]:
    rel = _relative(path)
    proc = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "--", rel],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return [line for line in proc.stdout.splitlines() if line.strip()]


def _path_size(path: Path) -> int:
    if path.is_symlink():
        return 0
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    for child in path.rglob("*"):
        if child.is_symlink() or not child.is_file():
            continue
        try:
            total += child.stat().st_size
        except OSError:
            continue
    return total


def _relative_symlink_target(link: Path, target: Path) -> str:
    return os.path.relpath(target, start=link.parent)


def _planned_move(
    source: Path,
    destination: Path,
    kind: str,
    symlink: Path | None,
) -> MigrationRecord:
    tracked = _tracked_files(source)
    if source.is_symlink():
        return MigrationRecord(
            action="move",
            source=_relative(source),
            destination=_relative(destination),
            kind=kind,
            status="skipped",
            reason="source is already a symlink",
            symlink=_relative(symlink) if symlink else "",
        )
    if tracked:
        return MigrationRecord(
            action="move",
            source=_relative(source),
            destination=_relative(destination),
            kind=kind,
            status="skipped",
            reason="source contains tracked files",
            symlink=_relative(symlink) if symlink else "",
            tracked_files=len(tracked),
            size_bytes=_path_size(source),
        )
    if destination.exists() or destination.is_symlink():
        return MigrationRecord(
            action="move",
            source=_relative(source),
            destination=_relative(destination),
            kind=kind,
            status="skipped",
            reason="destination already exists",
            symlink=_relative(symlink) if symlink else "",
            size_bytes=_path_size(source),
        )
    return MigrationRecord(
        action="move",
        source=_relative(source),
        destination=_relative(destination),
        kind=kind,
        status="planned",
        symlink=_relative(symlink) if symlink else "",
        size_bytes=_path_size(source),
    )


def build_plan(create_symlinks: bool) -> list[MigrationRecord]:
    records: list[MigrationRecord] = []

    for source_rel, destination_rel in WHOLE_DIR_MOVES:
        source = PROJECT_ROOT / source_rel
        if not source.exists() and not source.is_symlink():
            continue
        destination = LEGACY_ROOT / destination_rel
        symlink = source if create_symlinks else None
        records.append(_planned_move(source, destination, "directory", symlink))

    for parent_rel, destination_parent_rel in PARTIAL_PARENT_MOVES:
        parent = PROJECT_ROOT / parent_rel
        if not parent.exists():
            continue
        for child in sorted(parent.iterdir()):
            if child.name == ".gitkeep":
                continue
            destination = LEGACY_ROOT / destination_parent_rel / child.name
            symlink = child if create_symlinks else None
            kind = "directory" if child.is_dir() else "file"
            record = _planned_move(child, destination, kind, symlink)
            records.append(record)
            if record.status == "skipped" and record.reason == "source contains tracked files":
                for untracked_rel in _untracked_files(child):
                    untracked_source = PROJECT_ROOT / untracked_rel
                    if not untracked_source.exists() or untracked_source.is_symlink():
                        continue
                    records.append(
                        _planned_move(
                            untracked_source,
                            LEGACY_ROOT / untracked_rel,
                            "file",
                            None,
                        )
                    )

    for child in sorted(PROJECT_ROOT.iterdir()):
        if child.name.startswith(".") or child.is_dir() or child.is_symlink():
            continue
        category = ROOT_FILE_CATEGORIES.get(child.suffix.lower())
        if not category:
            continue
        destination = LEGACY_ROOT / "root_files" / category / child.name
        records.append(_planned_move(child, destination, "file", None))

    worktree_root = PROJECT_ROOT / ".claude" / "worktrees"
    if worktree_root.exists():
        for child in sorted(worktree_root.iterdir()):
            if not child.is_dir():
                continue
            records.append(
                MigrationRecord(
                    action="skip",
                    source=_relative(child),
                    destination="",
                    kind="git_worktree",
                    status="skipped",
                    reason="active git worktree; inspect with `git worktree list` before archiving",
                    size_bytes=_path_size(child),
                )
            )

    return records


def apply_plan(records: list[MigrationRecord]) -> list[MigrationRecord]:
    applied: list[MigrationRecord] = []
    for record in records:
        if record.status != "planned" or record.action != "move":
            applied.append(record)
            continue

        source = PROJECT_ROOT / record.source
        destination = PROJECT_ROOT / record.destination
        if not source.exists() and not source.is_symlink():
            record.status = "skipped"
            record.reason = "source disappeared before apply"
            applied.append(record)
            continue
        if destination.exists() or destination.is_symlink():
            record.status = "skipped"
            record.reason = "destination already exists before apply"
            applied.append(record)
            continue

        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))

        if record.symlink:
            link = PROJECT_ROOT / record.symlink
            if not link.exists() and not link.is_symlink():
                target = _relative_symlink_target(link, destination)
                os.symlink(target, link)

        record.status = "applied"
        record.applied_at = _utc_now()
        applied.append(record)
    return applied


def write_records(records: list[MigrationRecord], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        for record in records:
            f.write(json.dumps(asdict(record), sort_keys=True) + "\n")


def write_summary(records: list[MigrationRecord], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    planned = sum(r.status == "planned" for r in records)
    applied = sum(r.status == "applied" for r in records)
    skipped = sum(r.status == "skipped" for r in records)
    moved_bytes = sum(r.size_bytes for r in records if r.status in {"planned", "applied"})

    lines = [
        "# Artifact Migration Summary",
        "",
        f"Generated: {_utc_now()}",
        f"Planned: {planned}",
        f"Applied: {applied}",
        f"Skipped: {skipped}",
        f"Movable bytes: {moved_bytes:,}",
        "",
        "## Skips",
        "",
    ]
    skip_reasons: dict[str, int] = {}
    for record in records:
        if record.status == "skipped":
            skip_reasons[record.reason] = skip_reasons.get(record.reason, 0) + 1
    if skip_reasons:
        for reason, count in sorted(skip_reasons.items()):
            lines.append(f"- {count}: {reason}")
    else:
        lines.append("- none")
    output.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually move files. Without this flag, only writes a plan.",
    )
    parser.add_argument(
        "--no-symlinks",
        action="store_true",
        help="Do not leave compatibility symlinks at moved directory paths.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSONL migration record path. Defaults to artifacts/migration/<timestamp>.jsonl.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("artifacts/migration/latest.md"),
        help="Markdown summary path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or Path(f"artifacts/migration/{timestamp}.jsonl")

    records = build_plan(create_symlinks=not args.no_symlinks)
    if args.apply:
        records = apply_plan(records)

    write_records(records, PROJECT_ROOT / output)
    write_summary(records, PROJECT_ROOT / args.summary)

    planned = sum(r.status == "planned" for r in records)
    applied = sum(r.status == "applied" for r in records)
    skipped = sum(r.status == "skipped" for r in records)
    mode = "applied" if args.apply else "planned"
    print(f"Migration {mode}: {applied} applied, {planned} planned, {skipped} skipped")
    print(f"Wrote records to {output}")
    print(f"Wrote summary to {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
