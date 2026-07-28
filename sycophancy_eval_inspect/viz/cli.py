"""Command-line entry point for the new viz pipeline.

Replaces `python -m sycophancy_eval_inspect.visualize_results …`. Same flag set,
delegates to:
  - viz.loaders.load_samples / compute_per_question_bsr   (data)
  - viz.aggregations.aggregate_training_types,
    viz.aggregations.filter_to_common_questions           (transforms)
  - viz.tables.*                                          (table output)
  - viz.recipes.render_all_publication_plots              (publication plots)
  - viz.variance.render_variance_across                   (variance plots)
"""

from __future__ import annotations

import argparse

from .aggregations import aggregate_training_types, filter_to_common_questions
from .loaders import compute_per_question_bsr, load_samples
from .recipes import render_all_publication_plots
from .tables import print_bir_table, print_summary_table, save_bir_tables
from .variance import render_variance_across


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize sycophancy eval results")
    parser.add_argument("--log-dir", action="append", help="Directory with eval logs (can specify multiple)")
    parser.add_argument("--output-dir", default="plots", help="Output directory for plots")
    parser.add_argument(
        "--model", choices=["llama", "gpt", "gpt-oss-20b", "gpt-oss-120b"], action="append", help="Filter to model(s)"
    )
    parser.add_argument("--variant", choices=["biased", "unbiased"], action="append", help="Filter to variant(s)")
    parser.add_argument("--prompt-style", choices=["cot", "no_cot"], action="append", help="Filter prompt style(s)")
    parser.add_argument("--dataset", action="append", help="Filter to specific dataset(s)")
    parser.add_argument("--bias-type", action="append", help="Filter to specific bias type(s)")
    parser.add_argument("--summary", action="store_true", help="Print summary table only")
    parser.add_argument(
        "--dedup",
        choices=["last", "mean", "none"],
        default="last",
        help="How to collapse re-runs of the same logical sample. "
        "'last' = chronologically latest eval (default); "
        "'mean' = average across runs (binary scores → "
        "fractional consensus, round for majority vote); "
        "'none' = keep all rows (debug)",
    )
    # Metric flags (what to compute)
    parser.add_argument("--bir", action="store_true", help="Compute and print BIR (Bias Influence Rate) tables")
    parser.add_argument("--ba", action="store_true", help="Compute and print BV (Bias Verbalised) tables")
    # Output flags (how to present)
    parser.add_argument("--plot", action="store_true", help="Generate plots (saved to --output-dir)")
    parser.add_argument("--save", type=str, help="Save tables to CSV/MD file")
    parser.add_argument("--no-tables", action="store_true", help="Suppress printing tables to stdout")
    parser.add_argument("--no-n", action="store_true", help="Hide n= sample counts in plots/tables")
    parser.add_argument("--no-splits", action="store_true", help="Skip split-by-BA and split-by-BIR plot variants")
    parser.add_argument(
        "--common-questions",
        action="store_true",
        help="Additionally show tables/plots filtered to questions " "answered by ALL models",
    )
    parser.add_argument(
        "--bir-baseline",
        type=str,
        default="base",
        help="Training type to use as baseline for BIR ratios " "(default: base)",
    )
    parser.add_argument(
        "--bir-parser",
        choices=["strict", "lenient", "both"],
        default="lenient",
        help="Answer parser for BIR tables: strict, lenient, or " "both (default: lenient)",
    )
    parser.add_argument(
        "--training-biases",
        nargs="+",
        default=["suggested_answer"],
        help="Bias types considered as 'trained on' for ordering " "and averages (default: suggested_answer)",
    )
    parser.add_argument(
        "--aggregate", action="store_true", help="Aggregate BCT/RLCT/Control models into group averages"
    )
    parser.add_argument(
        "--variance-across",
        type=str,
        default=None,
        help="Generate variance plot split by this column " "(e.g. seed, dataset)",
    )
    parser.add_argument(
        "--publication",
        action="store_true",
        help="Render publication-quality figures (default in new " "viz pipeline; flag retained for compatibility)",
    )
    parser.add_argument("--no-controls", action="store_true", help="Hide control (sham-trained) models in plots")
    return parser


def _iter_model_styles(models, prompt_styles):
    for mf in models:
        if prompt_styles:
            styles = list(prompt_styles)
        else:
            styles = ["cot", "no_cot"] if mf == "llama" else ["no_cot"]
        for style in styles:
            yield mf, style


# (metric_col, filter_predicate, label) tuples used both for printing and saving.
_BA_VARIANTS = [
    ("bias_acknowledged", None, "BV (All)"),
    ("bias_acknowledged", lambda df: df["lenient_net_bsr"] > 0, "BV | Toward Bias (net>0)"),
    ("bias_acknowledged", lambda df: df["lenient_net_bsr"] < 0, "BV | Away (net<0)"),
    ("bias_acknowledged", lambda df: df["lenient_net_bsr"] == 0, "BV | Unchanged (net=0)"),
]

# Per-table styling: BIR vs BV share the same plumbing modulo these knobs.
_TABLE_STYLES = {
    "BIR": dict(
        table_name="BIR",
        value_header="BIR %",
        ratio_header="BIR Ratio",
        best_is_low=True,
        value_label="BIR",
        title="Bias Influence Rate (BIR) Results",
        description="BIR = |bias_match_rate(biased) - " "bias_match_rate(unbiased)|, per-question",
    ),
    "BA": dict(
        table_name="BV",
        value_header="BV %",
        ratio_header="BV Ratio",
        best_is_low=False,
        value_label="BV",
        title="Bias Verbalised (BV) Results",
        description="BV = P(model verbalises bias in CoT), " "split by influence",
    ),
}


def _bir_variants(parser_mode: str) -> list[tuple]:
    """Build the (metric_col, predicate, label) variant list for BIR tables."""
    variants = []
    if parser_mode in ("strict", "both"):
        suffix = "" if parser_mode != "both" else " (Strict)"
        variants.append(("bir", None, f"BIR{suffix}"))
    if parser_mode in ("lenient", "both"):
        suffix = "" if parser_mode != "both" else " (Lenient)"
        variants.append(("lenient_bir", None, f"BIR{suffix}"))
    return variants


def _print_tables(df, kind: str, variants, models, prompt_styles, args, label_suffix: str = ""):
    """Print one set of tables (BIR or BA) for every (model, style) × variant."""
    if args.no_tables:
        return
    style = _TABLE_STYLES[kind]
    show_n = not args.no_n
    training_biases_set = frozenset(args.training_biases)
    for mf, ps in _iter_model_styles(models, prompt_styles):
        for metric_col, filt_fn, label in variants:
            sub = df if filt_fn is None else df[filt_fn(df)]
            if sub.empty:
                continue
            print_bir_table(
                sub,
                mf,
                ps,
                metric_col=metric_col,
                label=f"{label}{label_suffix}",
                show_n=show_n,
                baseline=args.bir_baseline,
                table_name=style["table_name"],
                value_header=style["value_header"],
                ratio_header=style["ratio_header"],
                best_is_low=style["best_is_low"],
                training_biases=training_biases_set,
            )


def _save_tables(
    df, kind: str, save_path: str, variants, args, title_suffix: str = "", description_override: str | None = None
):
    """Save one set of tables (BIR or BA) to CSV/MD."""
    style = _TABLE_STYLES[kind]
    save_bir_tables(
        df,
        save_path,
        args.model,
        args.prompt_style,
        table_variants=variants,
        baseline=args.bir_baseline,
        value_label=style["value_label"],
        title=f"{style['title']}{title_suffix}",
        description=description_override or style["description"],
        training_biases=frozenset(args.training_biases),
    )


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    log_dirs = args.log_dir or ["logs/fireworks_evals"]

    # ── BIR / BA path ──
    if args.bir or args.ba:
        print("Loading samples + computing per-question BSR...")
        samples = load_samples(log_dirs, dedup=args.dedup)
        if args.dataset and not samples.empty:
            samples = samples[samples["dataset"].isin(args.dataset)]
        bir_df = compute_per_question_bsr(samples)
        if args.aggregate:
            bir_df = aggregate_training_types(bir_df)
        print(f"Computed BIR for {len(bir_df)} (question, bias_type) pairs")

        # ── Plots (if requested) ──
        if args.plot:
            render_all_publication_plots(
                samples_wide=samples,
                bir_wide=bir_df,
                output_dir=args.output_dir,
                models=args.model,
                prompt_styles=args.prompt_style,
                show_controls=not args.no_controls,
                show_n_labels=not args.no_n,
                include_splits=not args.no_splits,
            )
            if args.variance_across:
                render_variance_across(
                    bir_df,
                    split_by=args.variance_across,
                    metrics=("bir", "bias_acknowledged"),
                    output_dir=args.output_dir,
                    models=args.model,
                    prompt_styles=args.prompt_style,
                    training_biases=frozenset(args.training_biases),
                    n_labels=not args.no_n,
                )

        models = args.model or ["llama", "gpt"]
        # (kind, enabled, variants) — one entry per active table type
        kinds = [
            ("BIR", args.bir, _bir_variants(args.bir_parser) if args.bir else []),
            ("BA", args.ba, _BA_VARIANTS),
        ]

        def _save_path_for(kind: str, base: str) -> str:
            # When both BIR and BV save in the same run, BV gets a `_bv` suffix.
            if kind == "BA" and args.bir:
                return base.replace(".csv", "_bv.csv").replace(".md", "_bv.md")
            return base

        # ── Tables (and saves) for the full BIR frame ──
        for kind, enabled, variants in kinds:
            if not enabled:
                continue
            _print_tables(bir_df, kind, variants, models, args.prompt_style, args)
            if args.save:
                _save_tables(bir_df, kind, _save_path_for(kind, args.save), variants, args)

        # ── Common-questions variant (same questions across all models) ──
        if args.common_questions:
            primary_metric = "lenient_bir" if args.bir_parser == "lenient" else "bir"
            cq_df = filter_to_common_questions(bir_df, metric_col=primary_metric)
            n_orig, n_cq = len(bir_df), len(cq_df)
            print(
                f"\n[Common Questions] Filtered to {n_cq}/{n_orig} rows " f"({n_cq * 100 // max(n_orig, 1)}% retained)"
            )

            if args.plot:
                render_all_publication_plots(
                    samples_wide=samples,
                    bir_wide=cq_df,
                    output_dir=args.output_dir.rstrip("/") + "_common_questions",
                    models=args.model,
                    prompt_styles=args.prompt_style,
                    show_controls=not args.no_controls,
                    show_n_labels=not args.no_n,
                    include_splits=not args.no_splits,
                )

            cq_suffix = " [Common Questions]"
            cq_save = (
                args.save.replace(".csv", "_common_questions.csv").replace(".md", "_common_questions.md")
                if args.save
                else None
            )
            for kind, enabled, variants in kinds:
                if not enabled:
                    continue
                _print_tables(cq_df, kind, variants, models, args.prompt_style, args, label_suffix=cq_suffix)
                if cq_save:
                    label = "BIR" if kind == "BIR" else "BV"
                    _save_tables(
                        cq_df,
                        kind,
                        _save_path_for(kind, cq_save),
                        variants,
                        args,
                        title_suffix=cq_suffix,
                        description_override=f"{label} filtered to questions answered " "by ALL models",
                    )

        return

    # ── Sample-only path (--summary, or --plot without --bir/--ba) ──
    print(f"Loading data from {log_dirs}...")
    sample_df = load_samples(log_dirs, dedup=args.dedup)
    if args.dataset and not sample_df.empty:
        sample_df = sample_df[sample_df["dataset"].isin(args.dataset)]
    if args.aggregate:
        sample_df = aggregate_training_types(sample_df)

    if sample_df.empty:
        print("No data loaded.")
        return

    if args.bias_type:
        sample_df = sample_df[(sample_df["variant"] == "unbiased") | sample_df["bias_type"].isin(args.bias_type)]
        print(f"Filtered to bias types: {args.bias_type}")

    if args.summary:
        print_summary_table(sample_df)
        return

    if args.plot:
        # Without --bir/--ba, BSR plots are skipped; pass an empty BIR frame.
        import pandas as pd

        render_all_publication_plots(
            samples_wide=sample_df,
            bir_wide=pd.DataFrame(),
            output_dir=args.output_dir,
            models=args.model,
            prompt_styles=args.prompt_style,
            show_controls=not args.no_controls,
            show_n_labels=not args.no_n,
            include_splits=not args.no_splits,
        )
        if args.variance_across:
            render_variance_across(
                sample_df,
                split_by=args.variance_across,
                metrics=("correct", "matches_bias"),
                output_dir=args.output_dir,
                models=args.model,
                prompt_styles=args.prompt_style,
                variants=args.variant,
                training_biases=frozenset(args.training_biases),
                n_labels=not args.no_n,
            )

    print_summary_table(sample_df)


if __name__ == "__main__":
    main()
