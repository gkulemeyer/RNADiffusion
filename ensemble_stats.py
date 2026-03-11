import argparse
from pathlib import Path

from src.ensemble import DEFAULT_CONSENSUS, DEFAULT_TRIALS, evaluate_samples_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ensemble statistics from raw samples.")
    parser.add_argument("--samples-dir", required=True, help="Directory containing *.pt sample files")
    parser.add_argument("--output", default="", help="Optional CSV output path")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help="Number of consensus trials")
    parser.add_argument(
        "--consensus",
        default=",".join(str(value) for value in DEFAULT_CONSENSUS),
        help="Comma-separated consensus sizes",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for consensus sampling")
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
    print(f"Saved ensemble statistics to {output_path}")

if __name__ == "__main__":
    main()
