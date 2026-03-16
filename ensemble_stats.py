import argparse
from pathlib import Path

from src.config import load_ensemble_defaults
from src.ensemble import evaluate_samples_dir, evaluate_samples_stats
from src.io import write_ensemble_metadata


ENSEMBLE_DEFAULTS = load_ensemble_defaults()


def default_stats_output_path(output_path):
    output_path = Path(output_path)
    if output_path.name == "ensemble.csv":
        return output_path.with_name("ensemble_stats.csv")
    return output_path.with_name(f"{output_path.stem}_stats.csv")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ensemble statistics from raw samples.")
    parser.add_argument("--samples-dir", required=True, help="Directory containing *.pt sample files")
    parser.add_argument("--output", default="", help="Optional CSV output path")
    parser.add_argument("--trials", type=int, default=ENSEMBLE_DEFAULTS["trials"], help="Number of consensus trials")
    parser.add_argument(
        "--consensus",
        default=",".join(str(value) for value in ENSEMBLE_DEFAULTS["consensus_sizes"]),
        help="Comma-separated consensus sizes",
    )
    parser.add_argument("--seed", type=int, default=ENSEMBLE_DEFAULTS["base_seed"], help="Random seed for consensus sampling")
    return parser.parse_args()


def main():
    args = parse_args()
    consensus_sizes = [int(value) for value in args.consensus.split(",") if value]
    samples_dir = Path(args.samples_dir)
    output_path = Path(args.output) if args.output else samples_dir.parent / "ensemble.csv"

    stats = evaluate_samples_dir(
        samples_dir=str(samples_dir),
        consensus_sizes=consensus_sizes,
        trials=args.trials,
        seed=args.seed,
    )
    stats.to_csv(output_path, index=False)

    summary_stats = evaluate_samples_stats(output_path, consensus_sizes=consensus_sizes)
    stats_output_path = default_stats_output_path(output_path)
    summary_stats.to_csv(stats_output_path, index=False)

    write_ensemble_metadata(
        output_path=output_path,
        samples_dir=samples_dir,
        trials=args.trials,
        consensus_sizes=consensus_sizes,
        seed=args.seed,
    )
    print(f"Saved ensemble statistics to {output_path}")
    print(f"Saved ensemble summary stats to {stats_output_path}")

if __name__ == "__main__":
    main()
