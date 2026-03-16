import argparse
from pathlib import Path

from src.config import load_config
from src.data import build_dataloader
from src.ensemble import generate_raw_samples
from src.io import load_model_checkpoint, write_samples_metadata


def default_samples_dir(checkpoint_path):
    checkpoint_file = Path(checkpoint_path)
    return str(checkpoint_file.parent.parent / "raw_samples")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate raw ensemble samples for one checkpoint.")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to .ckpt file")
    parser.add_argument("--samples-dir", default="", help="Output directory for raw samples")
    parser.add_argument("--num-samples", type=int, default=0, help="Number of samples per sequence")
    parser.add_argument("--base-seed", type=int, default=None, help="Base seed for sampling")
    parser.add_argument("--batch-size", type=int, default=0, help="Optional batch size override")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    loader = build_dataloader(
        config,
        partition="test",
        batch_size=args.batch_size or None,
        shuffle=False,
    )
    model = load_model_checkpoint(config, args.checkpoint, eval_mode=True)
    samples_dir = args.samples_dir or default_samples_dir(args.checkpoint)
    ensemble_config = config["ensemble"]
    num_samples = args.num_samples or ensemble_config["num_samples"]
    base_seed = ensemble_config["base_seed"] if args.base_seed is None else args.base_seed

    generate_raw_samples(
        model=model,
        loader=loader,
        output_dir=samples_dir,
        num_samples=num_samples,
        base_seed=base_seed,
        chunk_size=ensemble_config["chunk_size"],
    )
    write_samples_metadata(
        samples_dir=samples_dir,
        checkpoint_path=args.checkpoint,
        num_samples=num_samples,
        base_seed=base_seed,
        chunk_size=ensemble_config["chunk_size"],
    )

if __name__ == "__main__":
    main()
