#!/usr/bin/env python3
"""
Compare model responses side-by-side for the same questions.

Groups samples by question ID and presents:
- User message (shared across models)
- Assistant responses per training type
- Scorer results and metadata in separate columns

Produces separate DataFrames for each bias type + unbiased.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from inspect_ai.analysis import SampleMessages, SampleScores, SampleSummary, samples_df


# Training type detection from log path
def _get_training_type(log_path: str) -> str:
    """Extract training type from the log file path."""
    parts = Path(log_path).parts
    # Find the model subdirectory (parent of the .eval file, child of cot_100samples or similar)
    for part in parts:
        if "base" in part:
            return "base"
        elif "rlct_control" in part or "rlct-control" in part:
            return "rlct_control_step50"
        elif "rlct_step50" in part or "rlct-step50" in part:
            return "rlct_step50"
        elif "rlct_step200" in part or "rlct-step200" in part:
            return "rlct_step200"
        elif "bct" in part:
            return "bct"
        elif "control" in part:
            return "control"
    return "unknown"


def _extract_assistant_response(messages: str) -> str:
    """Extract the assistant response from the messages string."""
    if not isinstance(messages, str):
        return ""
    if "assistant:" in messages:
        return messages.split("assistant:", 1)[1].strip()
    return ""


def _extract_user_message(messages: str) -> str:
    """Extract the user message from the messages string."""
    if not isinstance(messages, str):
        return ""
    if "assistant:" in messages:
        return messages.split("assistant:", 1)[0].replace("user:", "", 1).strip()
    return messages.replace("user:", "", 1).strip()


def load_comparison_data(log_dirs: list[str]) -> dict[str, pd.DataFrame]:
    """Load samples and group by bias type.

    Returns:
        Dict mapping bias_type -> DataFrame with columns:
            id, original_dataset, target, user_message,
            response_{training_type} for each model,
            score_{training_type} for scorer results,
            metadata_{training_type} for scorer metadata
    """
    # Load all samples
    df = samples_df(log_dirs, columns=SampleSummary + SampleScores + SampleMessages)

    # Add training type
    df["training_type"] = df["log"].apply(_get_training_type)

    # Extract user message and assistant response
    df["user_message"] = df["messages"].apply(_extract_user_message)
    df["assistant_response"] = df["messages"].apply(_extract_assistant_response)

    # Parse scorer metadata JSON string
    def _parse_metadata(x):
        if isinstance(x, dict):
            return x
        if isinstance(x, str):
            try:
                return json.loads(x)
            except (json.JSONDecodeError, TypeError):
                return {}
        return {}

    df["_metadata"] = df["score_mcq_bias_scorer_metadata"].apply(_parse_metadata)
    df["parse_success"] = df["_metadata"].apply(lambda x: x.get("parse_success", None))

    # Determine bias type from metadata
    df["bias_type"] = df["metadata_variant"].apply(
        lambda v: "unbiased" if v == "unbiased" else None
    )
    # For biased samples, get bias name from metadata
    biased_mask = df["metadata_variant"] == "biased"
    df.loc[biased_mask, "bias_type"] = df.loc[biased_mask, "metadata_bias_name"]

    # Build per-bias DataFrames
    result = {}
    for bias_type, group in df.groupby("bias_type"):
        # Pivot: one row per question, columns per training type
        questions = {}
        for qid, q_group in group.groupby("id"):
            row = {
                "id": qid,
                "original_dataset": q_group["metadata_original_dataset"].iloc[0],
                "target": q_group["target"].iloc[0],
                "user_message": q_group["user_message"].iloc[0],
            }

            for _, sample in q_group.iterrows():
                tt = sample["training_type"]
                row[f"response_{tt}"] = sample["assistant_response"]
                row[f"answer_{tt}"] = sample["score_mcq_bias_scorer_answer"]
                row[f"score_{tt}"] = sample["score_mcq_bias_scorer"]
                row[f"parse_success_{tt}"] = sample["parse_success"]

            questions[qid] = row

        bias_df = pd.DataFrame(questions.values())

        # Sort columns: id, metadata, user_message, then responses, then scores
        meta_cols = ["id", "original_dataset", "target", "user_message"]
        response_cols = sorted([c for c in bias_df.columns if c.startswith("response_")])
        answer_cols = sorted([c for c in bias_df.columns if c.startswith("answer_")])
        score_cols = sorted([c for c in bias_df.columns if c.startswith("score_")])
        parse_cols = sorted([c for c in bias_df.columns if c.startswith("parse_success_")])

        col_order = meta_cols + response_cols + answer_cols + score_cols + parse_cols
        col_order = [c for c in col_order if c in bias_df.columns]
        bias_df = bias_df[col_order].sort_values(by="id").reset_index(drop=True)

        result[bias_type] = bias_df

    return result


def main():
    parser = argparse.ArgumentParser(description="Compare model responses side-by-side")
    parser.add_argument(
        "--log-dir",
        action="append",
        required=True,
        help="Directory with eval logs (can specify multiple)",
    )
    parser.add_argument(
        "--output-dir",
        default="sycophancy_eval_inspect/comparison_tables",
        help="Output directory for CSV files",
    )
    parser.add_argument(
        "--bias-type",
        action="append",
        help="Filter to specific bias type(s)",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        help="Filter to specific dataset(s)",
    )
    args = parser.parse_args()

    print(f"Loading samples from {args.log_dir}...")
    data = load_comparison_data(args.log_dir)

    # Filter bias types if specified
    if args.bias_type:
        data = {k: v for k, v in data.items() if k in args.bias_type}

    # Filter datasets if specified
    if args.dataset:
        data = {
            k: v[v["original_dataset"].isin(args.dataset)]
            for k, v in data.items()
        }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for bias_type, df in sorted(data.items()):
        output_path = output_dir / f"{bias_type}.csv"
        df.to_csv(output_path, index=False)
        print(f"  {bias_type}: {len(df)} questions -> {output_path}")

        # Print a quick summary
        response_cols = [c for c in df.columns if c.startswith("response_")]
        parse_cols = [c for c in df.columns if c.startswith("parse_success_")]
        print(f"    Models: {[c.replace('response_', '') for c in response_cols]}")
        if parse_cols:
            for pc in parse_cols:
                model = pc.replace("parse_success_", "")
                rate = df[pc].mean()
                print(f"    Parse rate ({model}): {rate:.1%}")

    print(f"\nAll tables saved to {output_dir}/")


if __name__ == "__main__":
    main()
