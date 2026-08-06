"""Scan artifacts/runs/*/metadata/manifest.json into a flat roster of
queryable models — one entry per (run, seed, main/control) plus base-model
entries for every distinct base model seen.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RUNS_DIR = REPO_ROOT / "artifacts" / "runs"
# Committed snapshot of the publicly-published checkpoints. Used when no local
# run manifests are present (e.g. a fresh clone) so the app is standalone.
STATIC_ROSTER = Path(__file__).resolve().parent / "published_models.json"


@dataclass(frozen=True)
class ModelEntry:
    label: str           # what shows up in the dropdown
    model: str           # base model name (e.g. "openai/gpt-oss-20b")
    checkpoint: str | None  # tinker:// URL, or None for base model
    run: str | None = None
    seed: int | None = None
    is_control: bool = False


def _short_model(model: str) -> str:
    return model.split("/")[-1]


def _parse_task_label(label: str) -> tuple[int | None, bool]:
    """`main-s42` -> (42, False), `control-s42` -> (42, True)."""
    is_control = label.startswith("control")
    m = re.search(r"s(\d+)$", label)
    seed = int(m.group(1)) if m else None
    return seed, is_control


def _collect_from_manifest(path: Path) -> list[ModelEntry]:
    try:
        data = json.loads(path.read_text())
    except Exception:
        return []

    run = data.get("experiment") or path.parent.parent.name
    model = data.get("model")
    if not model:
        return []

    entries: list[ModelEntry] = []
    tasks = data.get("tasks", {}) or {}

    # Prefer per-task enumeration (gives one entry per seed + main/control).
    for task_name, task in tasks.items():
        if not task_name.startswith("train:"):
            continue
        if task.get("status") != "completed":
            continue
        ckpt = (task.get("outputs") or {}).get("checkpoint")
        if not ckpt:
            continue
        label_part = task.get("label") or task_name.split(":", 1)[1]
        seed, is_control = _parse_task_label(label_part)
        suffix = " [ctrl]" if is_control else ""
        seed_str = f" · s{seed}" if seed is not None else ""
        label = f"{_short_model(model)} · {run}{seed_str}{suffix}"
        entries.append(
            ModelEntry(
                label=label,
                model=model,
                checkpoint=ckpt,
                run=run,
                seed=seed,
                is_control=is_control,
            )
        )

    # Fallback: no per-task entries, use stage-level checkpoint.
    if not entries:
        stage_outs = (data.get("stages", {}).get("training", {}).get("outputs") or {})
        ckpt = stage_outs.get("checkpoint")
        if ckpt:
            entries.append(
                ModelEntry(
                    label=f"{_short_model(model)} · {run}",
                    model=model,
                    checkpoint=ckpt,
                    run=run,
                )
            )

    return entries


def _load_static_roster(path: Path = STATIC_ROSTER) -> list[ModelEntry]:
    """Load the committed roster of published checkpoints (empty if missing)."""
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except Exception:
        return []
    return [ModelEntry(**entry) for entry in data]


def build_registry(runs_dir: Path = RUNS_DIR) -> list[ModelEntry]:
    """Return the queryable model roster.

    Prefers a live scan of local run manifests (full set, for development).
    Falls back to the committed ``published_models.json`` snapshot when no
    manifests are present, so a fresh clone still exposes the published models.
    """
    trained: list[ModelEntry] = []
    seen_ckpts: set[str] = set()

    manifests = sorted(runs_dir.glob("*/metadata/manifest.json"))
    if not manifests:
        return _load_static_roster()

    for manifest in manifests:
        for entry in _collect_from_manifest(manifest):
            if entry.checkpoint is None or entry.checkpoint in seen_ckpts:
                continue
            seen_ckpts.add(entry.checkpoint)
            trained.append(entry)

    base_entries = [
        ModelEntry(label=f"{_short_model(m)} · base", model=m, checkpoint=None)
        for m in sorted({e.model for e in trained})
    ]

    # Bases first, then trained sorted by label.
    return base_entries + sorted(trained, key=lambda e: e.label)


if __name__ == "__main__":
    reg = build_registry()
    print(f"{len(reg)} entries\n")
    for e in reg:
        ckpt_disp = "base" if e.checkpoint is None else e.checkpoint[:80] + "…"
        print(f"  {e.label}\n    {ckpt_disp}")
