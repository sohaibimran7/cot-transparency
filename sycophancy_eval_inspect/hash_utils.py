"""Utilities for hash-based sample matching across bias types.

This module ensures that when evaluating multiple bias types, we only include
samples that have matching questions across all requested biases. This is
critical for computing Biased Reasoning Rate (BRR), which requires comparing
model responses to the same question under biased vs unbiased conditions.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def get_hashes_from_file(path: Path, filter_bias_on_wrong: bool = True) -> set[str]:
    """Extract original_question_hash values from a JSONL file.

    Args:
        path: Path to JSONL file
        filter_bias_on_wrong: If True, only include samples where biased_option != ground_truth.
            This ensures consistency with the dataset loader filtering.
    """
    hashes = set()
    with open(path) as f:
        for line in f:
            record = json.loads(line)

            # Skip positional bias files (different format)
            if "first_judge_prompt" in record:
                continue

            # Skip records without biased_question (same as dataset.py)
            if not record.get("biased_question"):
                continue

            # Apply same filter as dataset.py to ensure consistency
            if filter_bias_on_wrong:
                biased_option = record.get("biased_option", "")
                ground_truth = record.get("ground_truth", "")
                # Handle "NOT X" format from are_you_sure bias
                if not biased_option.startswith("NOT ") and biased_option == ground_truth:
                    continue

            h = record.get("original_question_hash")
            if h:
                hashes.add(h)
    return hashes


def get_hash_order_from_file(path: Path) -> list[str]:
    """Get hashes in file order from a JSONL file."""
    hashes = []
    with open(path) as f:
        for line in f:
            record = json.loads(line)
            h = record.get("original_question_hash")
            if h:
                hashes.append(h)
    return hashes


def group_datasets_by_original(datasets: list[Path]) -> dict[str, list[Path]]:
    """
    Group dataset paths by their original dataset name.

    E.g., mmlu_wrong_few_shot.jsonl and mmlu_are_you_sure.jsonl both map to "mmlu"
    """
    groups: dict[str, list[Path]] = {}
    for path in datasets:
        stem = path.stem
        bias_name = path.parent.name
        original = stem.replace(f"_{bias_name}", "")
        if original not in groups:
            groups[original] = []
        groups[original].append(path)
    return groups


def compute_common_hashes(
    datasets: list[Path],
    limit: int | None = None,
) -> dict[str, tuple[list[str], int]]:
    """
    Compute common hashes across bias types for each original dataset.

    Args:
        datasets: List of JSONL file paths to analyze
        limit: If specified, require at least this many common hashes

    Returns:
        Dict mapping original dataset name to (sorted_common_hashes, total_before_intersection)

    Raises:
        ValueError: If limit is specified and there aren't enough common hashes
    """
    groups = group_datasets_by_original(datasets)
    result: dict[str, tuple[list[str], int]] = {}

    for original, paths in groups.items():
        # Get hashes from each file
        hash_sets = [get_hashes_from_file(p) for p in paths]

        # Track the largest set size (before intersection)
        max_size = max(len(s) for s in hash_sets)

        # Compute intersection
        common = hash_sets[0]
        for s in hash_sets[1:]:
            common = common & s

        # Sort for consistent ordering
        sorted_hashes = sorted(common)

        # Validate against limit
        if limit is not None and len(sorted_hashes) < limit:
            bias_names = [p.parent.name for p in paths]
            raise ValueError(
                f"Not enough common hashes for {original}: "
                f"found {len(sorted_hashes)} common across {bias_names}, "
                f"but limit={limit} requested. "
                f"Reduce --limit or exclude bias types with fewer samples."
            )

        result[original] = (sorted_hashes, max_size)

        # Log warning if significant reduction
        if len(sorted_hashes) < max_size:
            reduction_pct = (1 - len(sorted_hashes) / max_size) * 100
            logger.warning(
                f"{original}: Using {len(sorted_hashes)}/{max_size} questions "
                f"({reduction_pct:.1f}% reduction) - common subset across {len(paths)} bias types"
            )

    return result


def compute_common_hashes_for_eval(
    datasets: list[Path],
    limit: int | None = None,
) -> dict[str, set[str]]:
    """
    Compute the set of allowed hashes for each original dataset.

    This is the main entry point for the eval pipeline.

    Args:
        datasets: All dataset paths that will be evaluated (biased + unbiased sources)
        limit: Optional limit on samples per evaluation

    Returns:
        Dict mapping original dataset name to set of allowed hashes.
        If limit is specified, returns only the first `limit` hashes (sorted).
    """
    common_by_dataset = compute_common_hashes(datasets, limit=limit)

    result: dict[str, set[str]] = {}
    for original, (sorted_hashes, max_size) in common_by_dataset.items():
        if limit is not None:
            # Take first `limit` hashes from the sorted list
            result[original] = set(sorted_hashes[:limit])
        else:
            result[original] = set(sorted_hashes)

    return result


def validate_hash_coverage(
    datasets: list[Path],
    limit: int | None = None,
) -> dict[str, dict]:
    """
    Analyze and report on hash coverage across bias types.

    Returns detailed statistics for logging/debugging.
    """
    groups = group_datasets_by_original(datasets)
    stats: dict[str, dict] = {}

    for original, paths in groups.items():
        bias_stats = {}
        hash_sets = {}

        for path in paths:
            bias_name = path.parent.name
            hashes = get_hashes_from_file(path)
            hash_sets[bias_name] = hashes
            bias_stats[bias_name] = len(hashes)

        # Compute intersection
        all_hashes = list(hash_sets.values())
        common = all_hashes[0]
        for s in all_hashes[1:]:
            common = common & s

        stats[original] = {
            "bias_counts": bias_stats,
            "common_count": len(common),
            "max_count": max(bias_stats.values()),
            "min_count": min(bias_stats.values()),
            "coverage_pct": len(common) / max(bias_stats.values()) * 100 if bias_stats else 0,
            "limit_feasible": limit is None or len(common) >= limit,
        }

    return stats


def save_common_hashes(
    common_hashes: dict[str, set[str]],
    output_path: Path,
) -> None:
    """Save common hashes to a JSON file for use with CLI hash_filter_file parameter."""
    # Convert sets to sorted lists for JSON serialization
    serializable = {k: sorted(v) for k, v in common_hashes.items()}
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    logger.info(f"Saved common hashes to {output_path}")


def print_hash_coverage_report(datasets: list[Path], limit: int | None = None) -> None:
    """Print a human-readable hash coverage report."""
    stats = validate_hash_coverage(datasets, limit)

    print("\n" + "=" * 70)
    print("HASH COVERAGE REPORT")
    print("=" * 70)

    for original, s in stats.items():
        print(f"\n{original.upper()}:")
        print(f"  Common hashes: {s['common_count']} / {s['max_count']} ({s['coverage_pct']:.1f}%)")
        print(f"  By bias type:")
        for bias, count in sorted(s['bias_counts'].items()):
            marker = "  " if count == s['max_count'] else "* "
            print(f"    {marker}{bias}: {count}")

        if limit:
            status = "✓ OK" if s['limit_feasible'] else f"✗ INSUFFICIENT (need {limit})"
            print(f"  Limit check: {status}")

    print("\n" + "=" * 70)
