"""Common utilities for running sycophancy evaluations.

This module provides shared functionality for eval runner scripts:
- Dataset discovery and grouping
- Hash-based sample filtering for consistent BRR calculation
- Common CLI argument handling
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from sycophancy_eval_inspect.hash_utils import (
    compute_common_hashes_for_eval,
    group_datasets_by_original,
    print_hash_coverage_report,
    save_common_hashes,
)

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for a dataset evaluation."""
    path: Path
    variant: str  # "biased" or "unbiased"
    allowed_hashes: set[str] | None = None

    @property
    def original_dataset(self) -> str:
        """Extract original dataset name (e.g., 'mmlu' from 'mmlu_suggested_answer')."""
        return get_original_dataset_name(self.path)

    @property
    def bias_name(self) -> str:
        """Extract bias type name from path."""
        return self.path.parent.name


def discover_mcq_datasets(base_dir: str = "dataset_dumps/test") -> list[Path]:
    """Find all MCQ bias JSONL files (excludes positional_bias)."""
    base = Path(base_dir)
    return sorted([f for f in base.glob("**/*.jsonl") if "positional_bias" not in str(f)])


def discover_positional_datasets(base_dir: str = "dataset_dumps/test") -> list[Path]:
    """Find positional bias JSONL files."""
    base = Path(base_dir)
    return sorted(list(base.glob("**/positional_bias/*.jsonl")))


def get_original_dataset_name(path: Path) -> str:
    """Extract original dataset name from a bias dataset path.

    E.g., 'mmlu' from 'dataset_dumps/test/suggested_answer/mmlu_suggested_answer.jsonl'
    """
    stem = path.stem
    bias_name = path.parent.name
    return stem.replace(f"_{bias_name}", "")


def get_unique_original_datasets(datasets: list[Path]) -> list[Path]:
    """
    Get one dataset file per original dataset for unbiased evaluation.

    Prefers bias types with more samples. Sorts alphabetically descending
    so 'wrong_few_shot' (full coverage) is preferred over 'are_you_sure' (partial).
    """
    groups = group_datasets_by_original(datasets)
    unique = []
    for original, paths in groups.items():
        # Sort descending to prefer 'wrong_few_shot' over 'are_you_sure'
        paths_sorted = sorted(paths, key=lambda p: p.parent.name, reverse=True)
        unique.append(paths_sorted[0])
    return unique


@dataclass
class HashFilterResult:
    """Result of hash filtering computation."""
    common_hashes: dict[str, set[str]]  # original_dataset -> set of hashes
    hash_filter_file: Path | None  # Path to saved JSON file (for CLI usage)

    def get_hashes_for_dataset(self, path: Path) -> set[str] | None:
        """Get allowed hashes for a dataset path."""
        if not self.common_hashes:
            return None
        original = get_original_dataset_name(path)
        return self.common_hashes.get(original)


def compute_hash_filter(
    datasets: list[Path],
    limit: int | None = None,
    save_to: Path | None = None,
    print_report: bool = True,
) -> HashFilterResult:
    """
    Compute hash-based filtering for consistent samples across bias types.

    Args:
        datasets: All MCQ dataset paths to evaluate
        limit: If specified, require at least this many common hashes per original dataset
        save_to: If provided, save common hashes to this JSON file (for CLI-based runners)
        print_report: Whether to print the hash coverage report

    Returns:
        HashFilterResult with common hashes and optional file path

    Raises:
        ValueError: If limit is specified and not enough common hashes exist
    """
    if print_report:
        print_hash_coverage_report(datasets, limit)

    common_hashes = compute_common_hashes_for_eval(datasets, limit=limit)

    for original, hashes in common_hashes.items():
        logger.info(f"  {original}: {len(hashes)} common samples")

    hash_filter_file = None
    if save_to is not None:
        save_to.parent.mkdir(parents=True, exist_ok=True)
        save_common_hashes(common_hashes, save_to)
        hash_filter_file = save_to

    return HashFilterResult(common_hashes=common_hashes, hash_filter_file=hash_filter_file)


def load_hash_filter_from_file(hash_file: str | Path) -> HashFilterResult:
    """Load pre-computed hashes from a JSON file.

    Args:
        hash_file: Path to JSON file with format {dataset_name: [hash1, hash2, ...]}

    Returns:
        HashFilterResult with loaded hashes
    """
    hash_path = Path(hash_file)
    with open(hash_path) as f:
        data = json.load(f)

    # Convert lists to sets
    common_hashes = {k: set(v) for k, v in data.items()}
    logger.info(f"Loaded hashes from {hash_path}:")
    for original, hashes in common_hashes.items():
        logger.info(f"  {original}: {len(hashes)} samples")

    return HashFilterResult(common_hashes=common_hashes, hash_filter_file=hash_path)


def build_dataset_configs(
    datasets: list[Path],
    run_biased: bool = True,
    run_unbiased: bool = True,
    hash_filter: HashFilterResult | None = None,
) -> list[DatasetConfig]:
    """
    Build list of DatasetConfig objects for evaluation.

    Args:
        datasets: All MCQ dataset paths
        run_biased: Whether to include biased variant evaluations
        run_unbiased: Whether to include unbiased variant evaluations
        hash_filter: Optional hash filtering result for consistent sampling

    Returns:
        List of DatasetConfig objects ready for evaluation
    """
    configs = []

    # Biased: run all datasets
    if run_biased:
        for path in datasets:
            allowed = hash_filter.get_hashes_for_dataset(path) if hash_filter else None
            configs.append(DatasetConfig(path=path, variant="biased", allowed_hashes=allowed))

    # Unbiased: run only one file per original dataset
    if run_unbiased:
        unbiased_datasets = get_unique_original_datasets(datasets)
        for path in unbiased_datasets:
            allowed = hash_filter.get_hashes_for_dataset(path) if hash_filter else None
            configs.append(DatasetConfig(path=path, variant="unbiased", allowed_hashes=allowed))

    return configs


def add_hash_filter_args(parser: argparse.ArgumentParser) -> None:
    """Add common hash filtering arguments to an argument parser."""
    parser.add_argument(
        "--skip-hash-filter",
        action="store_true",
        help="Disable hash-based filtering (not recommended - breaks BRR calculation)",
    )
    parser.add_argument(
        "--hash-file",
        type=str,
        default=None,
        help="Load pre-computed hashes from JSON file instead of computing them",
    )
    parser.add_argument(
        "--save-hash-file",
        type=str,
        default=None,
        help="Save computed hashes to this path (overrides default common_hashes.json in log dir)",
    )


def add_common_eval_args(parser: argparse.ArgumentParser) -> None:
    """Add common evaluation arguments to an argument parser."""
    parser.add_argument(
        "--bias-types",
        type=str,
        default=None,
        help="Filter to specific bias types (comma-separated)",
    )
    parser.add_argument(
        "--biased-only",
        action="store_true",
        help="Only run biased variant evaluations",
    )
    parser.add_argument(
        "--unbiased-only",
        action="store_true",
        help="Only run unbiased variant evaluations",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit samples per evaluation",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="dataset_dumps/test",
        help="Directory containing test datasets",
    )
    parser.add_argument(
        "--exclude-datasets",
        type=str,
        default=None,
        help="Exclude specific original datasets (comma-separated, e.g., 'logiqa,mmlu')",
    )
    add_hash_filter_args(parser)


def prepare_evaluation(
    args: argparse.Namespace,
    log_base_dir: str | None = None,
) -> tuple[list[DatasetConfig], HashFilterResult | None]:
    """
    Prepare for evaluation based on parsed arguments.

    This is the main entry point for eval scripts. It:
    1. Discovers datasets based on args.dataset_dir and args.bias_types
    2. Computes hash filtering if not disabled
    3. Builds dataset configs for evaluation

    Args:
        args: Parsed arguments (should include common eval args from add_common_eval_args)
        log_base_dir: Optional base directory for saving hash filter file

    Returns:
        Tuple of (dataset_configs, hash_filter_result)

    Raises:
        ValueError: If hash filtering fails (e.g., not enough common samples for limit)
    """
    # Discover datasets
    datasets = discover_mcq_datasets(args.dataset_dir)
    if args.bias_types:
        allowed_types = set(args.bias_types.split(","))
        datasets = [d for d in datasets if d.parent.name in allowed_types]

    # Filter out excluded original datasets
    exclude_datasets = getattr(args, 'exclude_datasets', None)
    if exclude_datasets:
        excluded = set(exclude_datasets.split(","))
        datasets = [d for d in datasets if get_original_dataset_name(d) not in excluded]

    if not datasets:
        raise ValueError(f"No datasets found in {args.dataset_dir}")

    # Determine variants to run
    run_biased = not getattr(args, 'unbiased_only', False)
    run_unbiased = not getattr(args, 'biased_only', False)

    # Hash filtering: either load from file or compute
    hash_filter = None
    skip_hash_filter = getattr(args, 'skip_hash_filter', False)
    hash_file = getattr(args, 'hash_file', None)

    if hash_file:
        # Load pre-computed hashes from file
        logger.info(f"Loading hashes from file: {hash_file}")
        hash_filter = load_hash_filter_from_file(hash_file)
    elif not skip_hash_filter:
        logger.info("Computing common hashes across bias types...")
        save_hash_file = getattr(args, 'save_hash_file', None)
        if save_hash_file:
            save_path = Path(save_hash_file)
        elif log_base_dir:
            save_path = Path(log_base_dir) / "common_hashes.json"
            if save_path.exists():
                raise FileExistsError(
                    f"Hash file already exists at {save_path}. "
                    f"Use --hash-file to load it, or specify a different path with --save-hash-file."
                )
        else:
            save_path = None

        hash_filter = compute_hash_filter(
            datasets=datasets,
            limit=args.limit,
            save_to=save_path,
            print_report=True,
        )

    # Build dataset configs
    dataset_configs = build_dataset_configs(
        datasets=datasets,
        run_biased=run_biased,
        run_unbiased=run_unbiased,
        hash_filter=hash_filter,
    )

    # Log summary
    biased_count = sum(1 for c in dataset_configs if c.variant == "biased")
    unbiased_count = sum(1 for c in dataset_configs if c.variant == "unbiased")
    logger.info(f"Datasets found: {len(datasets)}")
    logger.info(f"Evaluations: {biased_count} biased + {unbiased_count} unbiased = {len(dataset_configs)} total")
    if hash_file:
        logger.info(f"Hash filtering: loaded from {hash_file}")
    else:
        logger.info(f"Hash filtering: {'disabled' if skip_hash_filter else 'computed'}")

    return dataset_configs, hash_filter
