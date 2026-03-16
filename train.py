import argparse

from src.config import load_config
from src.experiment import run_experiment


def parse_args():
    parser = argparse.ArgumentParser(description="Train RNADiffusion with PyTorch Lightning.")
    parser.add_argument(
        "--config",
        default="configs/train/default.yaml",
        help="Path to a YAML config file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    run_experiment(config)


if __name__ == "__main__":
    main()
