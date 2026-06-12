"""Rerun parse-failed samples across all evals at higher max_tokens.

For each (checkpoint_dir, bias_type) with at least one failed sample:
  1. Collect failed sample hashes (union across biased+unbiased variants).
  2. Write a per-(ckpt, bias) hash file.
  3. Invoke run_tinker_evals with --bias-types <single_bias> and that hash file
     against the same checkpoint URI and base model.

The new .eval files land in the same checkpoint dir; visualize_results iterates
all .eval files in chronological order and last-read wins on duplicate sample ids.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from inspect_ai.log import read_eval_log


def scan_failures(log_dir: Path) -> dict:
    """Return: {
        ckpt_name: {
            'checkpoint_path': str | None,
            'base_model': str,
            'model_name': str,
            'prompt_style': str,
            'dataset': str,  # e.g. 'hle'
            'failed_by_bias': {bias_type: set[hash]},
        }
    }
    (Union across variants — re-run runs both biased and unbiased for these hashes.)
    """
    out: dict = {}
    for ckpt_dir in sorted(log_dir.iterdir()):
        if not ckpt_dir.is_dir():
            continue
        eval_files = list(ckpt_dir.glob("*.eval"))
        if not eval_files:
            continue
        info = {
            "checkpoint_path": None,
            "base_model": None,
            "model_name": ckpt_dir.name,
            "prompt_style": None,
            "dataset": None,
            "failed_by_bias": defaultdict(set),
        }
        for f in eval_files:
            try:
                log = read_eval_log(str(f))
            except Exception as e:
                print(f"  warn: failed to read {f}: {e}", file=sys.stderr)
                continue
            md = log.eval.metadata or {}
            ta = log.eval.task_args or {}
            ds_path = ta.get("dataset_path", "")
            bias = Path(ds_path).parent.name
            dataset = Path(ds_path).stem.replace(f"_{bias}", "") if bias else ""
            info["checkpoint_path"] = info["checkpoint_path"] or md.get("checkpoint_path")
            info["base_model"] = info["base_model"] or md.get("base_model")
            info["prompt_style"] = info["prompt_style"] or ta.get("prompt_style")
            info["dataset"] = info["dataset"] or dataset
            for s in log.samples or []:
                if not s.scores:
                    continue
                sc = s.scores.get("mcq_bias_scorer")
                lsc = s.scores.get("mcq_bias_scorer_fallback")
                strict = bool(sc.value.get("answer_parsed")) if sc and sc.value else False
                lenient = bool(lsc.value.get("lenient_answer_parsed", 0)) if lsc and lsc.value else False
                if not strict and not lenient:
                    info["failed_by_bias"][bias].add(str(s.id))
        info["failed_by_bias"] = {b: hs for b, hs in info["failed_by_bias"].items() if hs}
        if info["failed_by_bias"]:
            out[ckpt_dir.name] = info
    return out


def build_cmd(ckpt_info: dict, bias: str, hash_file: Path, log_dir: Path,
              max_tokens: int, max_tasks: int) -> list[str]:
    cmd = [sys.executable, "-m", "sycophancy_eval_inspect.run_tinker_evals"]
    if ckpt_info.get("checkpoint_path"):
        cmd += ["--checkpoint", ckpt_info["checkpoint_path"]]
    cmd += ["--base-model", ckpt_info["base_model"]]
    cmd += ["--name", ckpt_info["model_name"]]
    if ckpt_info.get("prompt_style"):
        cmd += ["--prompt-styles", ckpt_info["prompt_style"]]
    cmd += ["--bias-types", bias]
    cmd += ["--datasets", ckpt_info["dataset"]]
    cmd += ["--log-dir", str(log_dir)]
    cmd += ["--max-tokens", str(max_tokens)]
    cmd += ["--max-tasks", str(max_tasks)]
    cmd += ["--hash-file", str(hash_file)]
    return cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", action="append", required=True)
    ap.add_argument("--max-tokens", type=int, default=24576)
    ap.add_argument("--max-tasks", type=int, default=50)
    ap.add_argument("--rerun-dir-name", default="rerun_hashes")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-parallel", type=int, default=4, help="Concurrent rerun subprocesses across all configs")
    ap.add_argument("--only-checkpoint", action="append", default=None)
    ap.add_argument("--skip-checkpoint", action="append", default=[])
    args = ap.parse_args()

    plans: list[tuple[str, str, int, list[str]]] = []  # (ckpt, bias, n_hashes, cmd)
    for ld in args.log_dir:
        log_dir = Path(ld)
        print(f"=== scan {log_dir} ===")
        failures = scan_failures(log_dir)
        rerun_dir = log_dir / args.rerun_dir_name
        rerun_dir.mkdir(exist_ok=True)
        for ckpt_name, info in failures.items():
            if args.only_checkpoint and ckpt_name not in args.only_checkpoint:
                continue
            if ckpt_name in args.skip_checkpoint:
                continue
            for bias, hashes in sorted(info["failed_by_bias"].items()):
                hash_file = rerun_dir / f"{ckpt_name}__{bias}.json"
                payload = {info["dataset"]: sorted(hashes)}
                if not hash_file.exists() or not args.dry_run:
                    with open(hash_file, "w") as fh:
                        json.dump(payload, fh, indent=2)
                cmd = build_cmd(info, bias, hash_file, log_dir, args.max_tokens, args.max_tasks)
                plans.append((ckpt_name, bias, len(hashes), cmd))
        print(f"  {len(failures)} checkpoints with failures")

    total_resamples = sum(n * 2 for _, _, n, _ in plans)  # biased + unbiased
    print(f"\nTotal invocations: {len(plans)}")
    print(f"Total re-samples (both variants): {total_resamples:,}")

    if args.dry_run:
        print("\n--- first 3 commands ---")
        for ckpt, bias, n, cmd in plans[:3]:
            print(f"\n# {ckpt} × {bias}  (n={n})")
            print("  " + " \\\n    ".join(cmd))
        return

    # Execute with limited concurrency
    import concurrent.futures as cf
    failures_log: list[str] = []

    def _run(entry):
        ckpt, bias, n, cmd = entry
        try:
            rc = subprocess.call(cmd)
            return (ckpt, bias, n, rc)
        except Exception as e:
            return (ckpt, bias, n, -1)

    with cf.ThreadPoolExecutor(max_workers=args.max_parallel) as ex:
        futures = [ex.submit(_run, p) for p in plans]
        for i, fut in enumerate(cf.as_completed(futures), 1):
            ckpt, bias, n, rc = fut.result()
            status = "OK" if rc == 0 else f"FAIL rc={rc}"
            print(f"[{i}/{len(plans)}] {ckpt} × {bias} (n={n}) — {status}", flush=True)
            if rc != 0:
                failures_log.append(f"{ckpt}/{bias}")

    if failures_log:
        print(f"\n{len(failures_log)} invocations failed:")
        for x in failures_log:
            print(f"  {x}")


if __name__ == "__main__":
    main()
