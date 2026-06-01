#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config
from src.run_io import RunIO
from supervised.backbone_experiment import (
    evaluate_backbone_checkpoint,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate one supervised backbone checkpoint.")
    parser.add_argument("--run-dir", default="", help="Run root directory")
    parser.add_argument("--config", default="", help="Path to config.yaml")
    parser.add_argument("--checkpoint", default="", help="Path to .ckpt file")
    parser.add_argument("--output", default="", help="Optional CSV path for the summary row")
    parser.add_argument("--batch-size", type=int, default=0, help="Optional batch size override")
    args = parser.parse_args()

    if args.run_dir:
        return args
    if not args.config:
        parser.error("use --run-dir or provide --config")
    return args


def resolve_paths(args):
    if args.run_dir:
        run = RunIO(args.run_dir)
        config_path = run.config_path
        checkpoint_path = run.last_ckpt_path
        output_path = Path(args.output) if args.output else run.best_eval_dir / "test_summary.csv"
        return run.root, config_path, checkpoint_path, output_path

    config_path = Path(args.config)
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        run = RunIO.from_train_dir(config_path.parent)
    else:
        run = RunIO.from_train_dir(config_path.parent)
        checkpoint_path = run.last_ckpt_path
    output_path = Path(args.output) if args.output else run.best_eval_dir / "test_summary.csv"
    return run.root, config_path, checkpoint_path, output_path


def main():
    args = parse_args()
    run_dir, config_path, checkpoint_path, output_path = resolve_paths(args)

    config = load_config(config_path)
    summary = evaluate_backbone_checkpoint(
        config,
        checkpoint_path,
        batch_size=args.batch_size,
        output_path=output_path,
        run_dir=run_dir,
    )
    print(summary)


if __name__ == "__main__":
    main()
