#!/usr/bin/env python3
"""Summarize a local MLflow experiment directory into long-format CSV tables."""

import argparse
from pathlib import Path

from src.utils.reporting import write_experiment_summary_csvs


def main():
    parser = argparse.ArgumentParser(description="Summarize an MLflow experiment directory")
    parser.add_argument("experiment_dir", type=str, help="Path to <workspace_root>/mlruns/<experiment_id>")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for CSV files (default: <experiment_dir>/metrics)",
    )
    parser.add_argument(
        "--include-ids",
        action="store_true",
        help="Include run_id/run_name in params_summary_long and metrics_summary_long.",
    )
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (experiment_dir / "metrics")

    summary = write_experiment_summary_csvs(
        experiment_dir=experiment_dir,
        output_dir=output_dir,
        include_ids=bool(args.include_ids),
    )

    print(f"Wrote: {summary['params_path']} ({summary['params_rows']} rows)")
    print(f"Wrote: {summary['metrics_long_path']} ({summary['metrics_long_rows']} rows)")
    print(f"Wrote: {summary['metrics_summary_path']} ({summary['metrics_summary_rows']} rows)")


if __name__ == "__main__":
    main()
