---
name: cleanup-checkpoints
description: List or delete Tinker checkpoints not used by any eval log. Queries the Tinker API to enumerate all checkpoints, cross-references with eval logs, and deletes unused ones.
argument-hint: [--dry-run] [--keep-run UUID_PREFIX ...]
---

# Cleanup Unused Tinker Checkpoints

Delete remote Tinker LoRA checkpoints that aren't referenced by any eval log.

## Arguments

- `$ARGUMENTS` (optional):
  - `--dry-run` — List what would be deleted without deleting
  - `--keep-run UUID_PREFIX` — Keep all checkpoints for runs matching this prefix (repeatable)

## How it works

1. **Scan eval logs** in `sycophancy_eval_inspect/logs/` (all subdirectories). For each `.eval` (zip) or `.json` log file, extract `eval.task_args.metadata.checkpoint_path` to build the set of used `tinker://` paths.

2. **Fetch all training runs** from `GET /api/v1/training_runs` (paginated, 100 per page). Auth via `TINKER_API_KEY` from `.env`. Base URL: `https://tinker.thinkingmachines.dev/services/tinker-prod`.

3. **List checkpoints** for each run via `GET /api/v1/training_runs/{run_id}/checkpoints`.

4. **Show the user a summary BEFORE deleting**, including:
   - How many checkpoints will be deleted
   - Which checkpoints will be kept, and which eval log directory each corresponds to
   - Ask for explicit confirmation before proceeding

5. **Delete unused** (after user confirms) via `DELETE /api/v1/training_runs/{run_id}/checkpoints/{checkpoint_id}` for every checkpoint whose `tinker://{run_id}/{checkpoint_id}` path is NOT in the used set and NOT in a `--keep-run` prefix.

## Key implementation details

- `.eval` files are ZIP archives; read `_journal/start.json` inside them for metadata
- The checkpoint path is at `eval.task_args.metadata.checkpoint_path`
- Training runs list is paginated: use `?offset=N&limit=100` params
- Checkpoint list response has a `checkpoints` key containing objects with `checkpoint_id`
- The `tinker://` path format is `tinker://{training_run_id}/{checkpoint_id}` where checkpoint_id includes the prefix like `sampler_weights/` or `weights/`
- Auth header is `X-Api-Key` (not Bearer)
- Needs `httpx` for REST calls and `dotenv` for loading `.env`

## Examples

```bash
# Dry run - see what would be deleted
/cleanup-checkpoints --dry-run

# Delete all unused, but keep a specific training run's checkpoints
/cleanup-checkpoints --keep-run bce61b0a

# Delete everything not used in evals
/cleanup-checkpoints
```
