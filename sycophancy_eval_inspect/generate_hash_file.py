"""CLI to precompute a hash file before running evals.

Eliminates the need for the first eval run to generate the hash file as a side
effect, which forced sequential execution when running multiple models against
the same eval set. With a precomputed hash file, all model evals can run in
parallel from the start.

Usage:
    python -m sycophancy_eval_inspect.generate_hash_file \\
        --datasets hellaswag,logiqa \\
        --bias-types suggested_answer,distractor_fact,post_hoc \\
        --limit 200 \\
        --output sycophancy_eval_inspect/logs/base_200/common_hashes.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from sycophancy_eval_inspect.eval_common import compute_hash_filter

logger = logging.getLogger(__name__)


def resolve_dataset_paths(
    datasets: list[str],
    bias_types: list[str],
    dataset_dir: Path,
) -> list[Path]:
    """Resolve (dataset, bias) combinations to JSONL paths.

    Raises ValueError if any expected path doesn't exist.
    """
    paths: list[Path] = []
    missing: list[str] = []
    for bias in bias_types:
        for ds in datasets:
            p = dataset_dir / bias / f"{ds}_{bias}.jsonl"
            if p.exists():
                paths.append(p)
            else:
                missing.append(str(p))
    if missing:
        raise ValueError(
            f"Missing dataset files:\n  "
            + "\n  ".join(missing)
            + "\nCheck --datasets and --bias-types against what's in --dataset-dir."
        )
    return paths


def main():
    parser = argparse.ArgumentParser(
        description="Precompute a common_hashes.json file for consistent cross-model evaluation."
    )
    parser.add_argument(
        "--datasets", required=True,
        help="Comma-separated list of original datasets (e.g. hellaswag,logiqa)",
    )
    parser.add_argument(
        "--bias-types", required=True,
        help="Comma-separated list of bias types (e.g. suggested_answer,distractor_fact)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Number of samples per dataset (None = use full intersection)",
    )
    parser.add_argument(
        "--output", required=True,
        help="Path to write common_hashes.json",
    )
    parser.add_argument(
        "--dataset-dir", default="dataset_dumps/test",
        help="Base directory containing bias subdirs (default: dataset_dumps/test)",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite output file if it already exists and has different content. "
             "Without this flag, existing files with identical content are reused "
             "and existing files with different content cause an error.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Print hash coverage report",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    bias_types = [b.strip() for b in args.bias_types.split(",") if b.strip()]
    dataset_dir = Path(args.dataset_dir)
    output_path = Path(args.output)

    paths = resolve_dataset_paths(datasets, bias_types, dataset_dir)

    # Compute hashes (in memory, no file write yet)
    result = compute_hash_filter(
        datasets=paths,
        limit=args.limit,
        save_to=None,
        print_report=args.verbose,
    )
    new_content = {k: sorted(v) for k, v in result.common_hashes.items()}

    # Idempotent write: compare to existing file if present
    if output_path.exists():
        try:
            with open(output_path) as f:
                existing = json.load(f)
            existing_sorted = {k: sorted(v) for k, v in existing.items()}
            if existing_sorted == new_content:
                print(f"✓ {output_path} already up-to-date ({sum(len(v) for v in new_content.values())} hashes)")
                return
            if not args.overwrite:
                print(
                    f"✗ {output_path} exists but content differs.\n"
                    f"  Existing: {{{', '.join(f'{k}: {len(v)}' for k, v in existing_sorted.items())}}}\n"
                    f"  New:      {{{', '.join(f'{k}: {len(v)}' for k, v in new_content.items())}}}\n"
                    f"  Pass --overwrite to replace, or choose a different --output.",
                    file=sys.stderr,
                )
                sys.exit(1)
        except (json.JSONDecodeError, OSError):
            # Malformed existing file — treat as missing and overwrite
            pass

    # Atomic write: write to .tmp, then rename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(new_content, f, indent=2)
    tmp_path.replace(output_path)

    total = sum(len(v) for v in new_content.values())
    per_ds = ", ".join(f"{k}: {len(v)}" for k, v in new_content.items())
    print(f"✓ Wrote {output_path} ({total} hashes across {len(new_content)} datasets: {per_ds})")


if __name__ == "__main__":
    main()
