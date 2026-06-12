"""Run evals on Llama RLCT step 8 (midway = 64 datapoints) checkpoints.

Evaluates the 12 step 8 checkpoints from rlct_da_aw0_llama and
rlct_da_wfs_aw0_llama (3 seeds × main+ctrl × 2 configs). Eval params match
the original Llama RLCT runs (cot, 7 biases, 16k tokens) so midway results
plot alongside the existing final-checkpoint results.
"""

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
LOG_BASE = PROJECT_ROOT / "artifacts" / "legacy" / "logs"
EVAL_LOG_DIR = "artifacts/eval_suites/consistency_hle_llama/eval_logs"

EXPERIMENTS = ["rlct_da_aw0_llama", "rlct_da_wfs_aw0_llama"]
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
BIAS_TYPES = (
    "suggested_answer,are_you_sure,distractor_fact,distractor_argument_g4,"
    "post_hoc,spurious_few_shot_squares,wrong_few_shot"
)


def collect_step8_checkpoints() -> list[dict]:
    """Read each run's checkpoints.jsonl and return [{name, sampler_path}, ...]."""
    out: list[dict] = []
    for exp in EXPERIMENTS:
        for run_dir in sorted((LOG_BASE / exp).iterdir()):
            ckpt_file = run_dir / "checkpoints.jsonl"
            if not ckpt_file.exists():
                continue
            step8_lines = [
                json.loads(l) for l in ckpt_file.read_text().splitlines() if l.strip()
            ]
            step8_lines = [c for c in step8_lines if c.get("step") == 8]
            if not step8_lines:
                continue
            # Use the most recent (last entry, matches state.json's final UUID)
            latest = step8_lines[-1]
            out.append({
                "experiment": exp,
                "run_name": run_dir.name,
                "model_name": f"{run_dir.name}-step8",
                "sampler_path": latest["sampler_path"],
            })
    return out


def build_eval_cmd(ckpt: dict) -> list[str]:
    return [
        sys.executable, "-m", "sycophancy_eval_inspect.run_tinker_evals",
        "--checkpoint", ckpt["sampler_path"],
        "--base-model", BASE_MODEL,
        "--name", ckpt["model_name"],
        "--bias-types", BIAS_TYPES,
        "--datasets", "hle",
        "--prompt-styles", "cot",
        "--limit", "100",
        "--max-tokens", "16384",
        "--max-tasks", "50",
        "--log-dir", EVAL_LOG_DIR,
        "--hash-file", f"{EVAL_LOG_DIR}/common_hashes.json",
    ]


def run_eval(ckpt: dict) -> tuple[str, int, str]:
    cmd = build_eval_cmd(ckpt)
    result = subprocess.run(cmd, capture_output=True, text=True)
    tail = result.stdout[-500:] if result.stdout else result.stderr[-500:]
    return ckpt["model_name"], result.returncode, tail


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ckpts = collect_step8_checkpoints()
    print(f"Found {len(ckpts)} step 8 checkpoints:")
    for c in ckpts:
        print(f"  {c['model_name']}")

    if args.dry_run:
        print("\nSample command:")
        print(" \\\n    ".join(build_eval_cmd(ckpts[0])))
        return

    print(f"\nRunning evals (max_parallel={args.max_parallel})...")
    with ThreadPoolExecutor(max_workers=args.max_parallel) as ex:
        futures = {ex.submit(run_eval, c): c["model_name"] for c in ckpts}
        for f in as_completed(futures):
            name, rc, tail = f.result()
            status = "OK" if rc == 0 else f"FAIL({rc})"
            print(f"[{status}] {name}")
            if rc != 0:
                print(f"  tail: {tail}")


if __name__ == "__main__":
    main()
