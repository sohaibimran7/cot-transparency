#!/usr/bin/env python3
"""Catalog generated experiment artifacts without moving or deleting them.

The catalog is intentionally append-friendly and path-preserving: every record
keeps the current location so legacy experiments remain findable while future
runs move to the canonical ``artifacts/runs/...`` and
``artifacts/eval_suites/...`` layouts.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_ROOTS = [
    "artifacts",
    "experiments",
    "logs",
    "plots",
    "results",
    "wandb",
    "dataset_dumps",
    "paper",
    "paper-results",
    "sycophancy_eval_inspect/logs",
    "sycophancy_eval_inspect/plots",
    ".claude/worktrees",
]

MIGRATION_DIR = PROJECT_ROOT / "artifacts" / "migration"

PRUNE_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
}


@dataclass(frozen=True)
class ArtifactRecord:
    artifact_type: str
    path: str
    legacy_path: str
    root: str
    size_bytes: int
    modified_at: str
    experiment_hint: str
    run_hint: str
    suffix: str

    def to_json(self) -> str:
        return json.dumps(self.__dict__, sort_keys=True)


def _iso_timestamp(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat()


def _relative(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _parts(path: Path) -> tuple[str, ...]:
    return Path(_relative(path)).parts


def load_migration_prefixes() -> list[tuple[str, str]]:
    """Return ``(current_prefix, legacy_prefix)`` pairs from migration logs."""
    prefixes: list[tuple[str, str]] = []
    if not MIGRATION_DIR.exists():
        return prefixes

    for path in sorted(MIGRATION_DIR.glob("*.jsonl")):
        try:
            lines = path.read_text().splitlines()
        except OSError:
            continue
        for line in lines:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("action") != "move" or record.get("status") != "applied":
                continue
            source = str(record.get("source", "")).rstrip("/")
            destination = str(record.get("destination", "")).rstrip("/")
            if source and destination:
                prefixes.append((destination, source))

    # Prefer the most specific migrated path when records overlap.
    return sorted(prefixes, key=lambda pair: len(pair[0]), reverse=True)


def legacy_path_for(rel: str, migration_prefixes: list[tuple[str, str]]) -> str:
    for current_prefix, legacy_prefix in migration_prefixes:
        if rel == current_prefix:
            return legacy_prefix
        if rel.startswith(current_prefix + "/"):
            return legacy_prefix + rel[len(current_prefix):]
    return rel


def classify(path: Path) -> str:
    rel = _relative(path)
    suffix = path.suffix.lower()
    parts = _parts(path)
    name = path.name

    if parts[:2] == ("artifacts", "legacy") and len(parts) > 2:
        return f"migrated_{classify(Path(*parts[2:]))}"
    if parts[:2] == ("artifacts", "migration"):
        return "artifact_migration_index"
    if rel.startswith(".claude/worktrees/"):
        if suffix in {".eval", ".log", ".png", ".json", ".jsonl", ".csv", ".md", ".yaml", ".yml"}:
            return f"worktree_{classify(Path(*parts[3:]))}"
        return "worktree_file"
    if parts[:2] == ("sycophancy_eval_inspect", "logs"):
        if name.startswith("common_hashes") and suffix == ".json":
            return "eval_hash_file"
        if suffix == ".eval":
            return "eval_log"
        if suffix == ".json":
            return "eval_metadata"
        if suffix == ".log":
            return "eval_run_log"
        return "eval_artifact"
    if parts[:2] == ("sycophancy_eval_inspect", "plots") or parts[:1] == ("plots",):
        if suffix in {".png", ".pdf", ".svg"}:
            return "plot"
        if suffix in {".csv", ".md", ".json"}:
            return "plot_table"
        return "plot_artifact"
    if parts[:1] == ("logs",):
        if suffix == ".jsonl":
            return "training_metric_log"
        if suffix == ".log":
            return "training_log"
        if suffix == ".json":
            return "training_metadata"
        return "training_artifact"
    if parts[:1] == ("experiments",):
        if name == "state.json":
            return "experiment_state"
        if suffix in {".yaml", ".yml"}:
            return "experiment_config_copy"
        if suffix == ".log":
            return "stage_log"
        return "experiment_artifact"
    if parts[:1] == ("artifacts",):
        if suffix == ".json" and name == "manifest.json":
            return "artifact_manifest"
        if suffix == ".eval":
            return "canonical_eval_log"
        if suffix == ".png":
            return "canonical_plot"
        if suffix == ".log":
            return "canonical_stage_log"
        return "canonical_artifact"
    if parts[:1] == ("wandb",):
        return "wandb_artifact"
    if parts[:1] == ("dataset_dumps",):
        return "dataset_dump"
    if suffix == ".log":
        return "root_log"
    if suffix in {".png", ".pdf"}:
        return "root_figure"
    return "other"


def hints(path: Path, root: Path) -> tuple[str, str]:
    rel_parts = _parts(path)
    root_parts = _parts(root)
    if rel_parts[:2] == ("artifacts", "legacy") and len(rel_parts) > 2:
        rel_parts = rel_parts[2:]
        root_parts = ()
    after_root = rel_parts[len(root_parts):]

    experiment_hint = after_root[0] if after_root else ""
    run_hint = after_root[1] if len(after_root) > 1 else ""

    if rel_parts[:3] == ("artifacts", "runs") and len(rel_parts) > 2:
        experiment_hint = rel_parts[2]
        run_hint = rel_parts[3] if len(rel_parts) > 3 else ""
    elif rel_parts[:1] == ("logs",) and len(rel_parts) > 2:
        experiment_hint = rel_parts[1]
        run_hint = rel_parts[2]
    elif rel_parts[:2] == ("sycophancy_eval_inspect", "logs") and len(rel_parts) > 3:
        experiment_hint = rel_parts[2]
        run_hint = rel_parts[3]
    elif rel_parts[:2] == ("sycophancy_eval_inspect", "plots") and len(rel_parts) > 2:
        experiment_hint = rel_parts[2]
        run_hint = rel_parts[3] if len(rel_parts) > 3 else ""
    elif rel_parts[:2] == (".claude", "worktrees") and len(rel_parts) > 2:
        experiment_hint = rel_parts[2]
        run_hint = "/".join(rel_parts[3:6])

    return experiment_hint, run_hint


def iter_files(root: Path, max_depth: int | None) -> Iterable[Path]:
    if not root.exists():
        return
    if root.is_symlink():
        return

    root_depth = len(root.parts)
    for current, dirnames, filenames in os.walk(root):
        current_path = Path(current)
        depth = len(current_path.parts) - root_depth
        dirnames[:] = [
            d for d in dirnames
            if (
                d not in PRUNE_DIRS
                and not d.endswith(".egg-info")
                and not (current_path / d).is_symlink()
            )
        ]
        if max_depth is not None and depth >= max_depth:
            dirnames[:] = []
        for filename in filenames:
            yield current_path / filename


def build_records(roots: list[Path], max_depth: int | None) -> list[ArtifactRecord]:
    records: list[ArtifactRecord] = []
    migration_prefixes = load_migration_prefixes()
    for root in roots:
        if not root.exists():
            continue
        for path in iter_files(root, max_depth):
            try:
                stat = path.stat()
            except OSError:
                continue
            experiment_hint, run_hint = hints(path, root)
            rel = _relative(path)
            records.append(
                ArtifactRecord(
                    artifact_type=classify(path),
                    path=rel,
                    legacy_path=legacy_path_for(rel, migration_prefixes),
                    root=_relative(root),
                    size_bytes=stat.st_size,
                    modified_at=_iso_timestamp(stat.st_mtime),
                    experiment_hint=experiment_hint,
                    run_hint=run_hint,
                    suffix=path.suffix.lower(),
                )
            )
    return records


def write_jsonl(records: list[ArtifactRecord], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        for record in records:
            f.write(record.to_json() + "\n")


def write_summary(records: list[ArtifactRecord], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    by_type = Counter(r.artifact_type for r in records)
    by_root: dict[str, list[ArtifactRecord]] = defaultdict(list)
    for record in records:
        by_root[record.root].append(record)

    lines = [
        "# Artifact Catalog Summary",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Records: {len(records)}",
        "",
        "## By Type",
        "",
        "| Type | Count | Size |",
        "| --- | ---: | ---: |",
    ]
    for artifact_type, count in by_type.most_common():
        total = sum(r.size_bytes for r in records if r.artifact_type == artifact_type)
        lines.append(f"| `{artifact_type}` | {count} | {total:,} |")

    lines.extend(["", "## By Root", "", "| Root | Count | Size |", "| --- | ---: | ---: |"])
    for root, root_records in sorted(by_root.items()):
        total = sum(r.size_bytes for r in root_records)
        lines.append(f"| `{root}` | {len(root_records)} | {total:,} |")

    output.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        action="append",
        dest="roots",
        help="Artifact root to scan. Repeatable. Defaults to known project roots.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/catalog.jsonl"),
        help="JSONL catalog path.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("artifacts/catalog.md"),
        help="Markdown summary path.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Optional maximum recursion depth below each root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_names = args.roots or DEFAULT_ROOTS
    roots = [PROJECT_ROOT / root for root in root_names]
    records = build_records(roots, args.max_depth)
    write_jsonl(records, PROJECT_ROOT / args.output)
    write_summary(records, PROJECT_ROOT / args.summary)
    print(f"Wrote {len(records)} records to {args.output}")
    print(f"Wrote summary to {args.summary}")


if __name__ == "__main__":
    main()
