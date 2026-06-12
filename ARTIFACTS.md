# Artifact Layout

This repo has legacy artifacts in several places. Keep them findable by
cataloging them before moving anything, then write new runs to the canonical
layout below.

## Canonical Layout

New experiment runs should write under:

```text
artifacts/
  runs/
    <experiment-name>/
      train_logs/
      eval_logs/
      plots/
      tables/
      metadata/
        config.yaml
        state.json
        manifest.json
        stage_logs/
  eval_suites/
    <suite-name>/
      eval_logs/
      plots/
  legacy/
    ...
  migration/
    <timestamp>.jsonl
```

The runner writes `metadata/manifest.json` with canonical paths, legacy state
paths, config hash, stage status, and task outputs. This is the first place to
look when answering "where did that run go?"

Shared eval suites that intentionally compare many experiments can live under
`artifacts/eval_suites/<suite-name>/`. One-off/manual runs should stay under
`artifacts/runs/<name>/`.

## Legacy Locations

These roots are historical locations. Ignored/generated payloads should be
moved under `artifacts/legacy/` with `scripts/migrate_artifacts.py`; compatibility
symlinks may remain at old directory paths so older commands still resolve.
Tracked reference artifacts are not moved by the migration script.

| Root | Contents |
| --- | --- |
| `logs/` | Older Tinker training logs, metrics, and checkpoint metadata |
| `experiments/` | Older pipeline state files and stage logs |
| `sycophancy_eval_inspect/logs/` | Inspect eval logs and common hash files |
| `sycophancy_eval_inspect/plots/` | Older eval plots |
| `plots/` | Older visualizer fallback output |
| `wandb/` | Local W&B run cache |
| `dataset_dumps/` | Generated datasets and archived dataset variants |
| `.claude/worktrees/` | Agent worktree copies and their generated artifacts |

Active Git worktrees are not moved automatically. Inspect them with:

```bash
git worktree list
```

Archive only after confirming the branch is no longer needed.

## Migrating Existing Local Artifacts

Preview a migration:

```bash
python scripts/migrate_artifacts.py
```

Apply it:

```bash
python scripts/migrate_artifacts.py --apply
```

The migration moves ignored/generated legacy roots into `artifacts/legacy/`,
leaves compatibility symlinks for moved directories, skips tracked files, and
writes a path map under `artifacts/migration/`. The artifact catalog reads these
maps, so records for moved files keep both:

- `path` - current location under `artifacts/legacy/...`
- `legacy_path` - the original path before migration

## Cataloging Existing Artifacts

Generate a path-preserving catalog before archiving or moving legacy material:

```bash
python scripts/catalog_artifacts.py
```

This writes:

- `artifacts/catalog.jsonl` - one JSON record per discovered artifact
- `artifacts/catalog.md` - counts and sizes by root/type

Each record keeps `legacy_path`. Regenerate the catalog after migration so the
old and new paths remain searchable.

## Guardrail

Before committing, check that generated artifacts are not being added outside
the approved artifact roots:

```bash
python scripts/check_artifact_paths.py
```

The check looks at staged files by default, which avoids failing on the current
legacy clutter while still blocking new generated files from entering random
locations.

## Archiving Rule

Do not recursively delete generated material. Move old artifacts into
`artifacts/legacy/` or an `_archive/` subtree after they have been cataloged.
