import argparse
import csv
from pathlib import Path

import numpy as np
import torch as tr
from tqdm import tqdm

from src.config import load_config
from src.data import build_dataloader
from src.io import load_model_checkpoint
from src.metrics import contact_f1

DEVICE = tr.device("cuda" if tr.cuda.is_available() else "cpu")

@tr.no_grad()
def evaluate_model(model, loader, device):
    model.eval()
    results = {"loss": [], "f1": []}

    for batch in tqdm(loader, desc="Testing", leave=False):
        condition = batch["conditioning"].to(device)
        target = batch["contact_one_hot"].to(device)
        mask = batch["mask"].to(device)
        lengths = batch["length"]

        loss = model.forward_all_timesteps(target, condition, mask=mask)
        results["loss"].append(loss.item())

        predictions = model._sample(condition)
        f1_scores = contact_f1(predictions, target, lengths=lengths, reduce=False)
        results["f1"].extend(f1_scores.cpu().tolist())

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate one RNADiffusion checkpoint.")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to .ckpt file")
    parser.add_argument("--output", default="", help="Optional CSV path for the summary row")
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
    results = evaluate_model(model, loader, DEVICE)

    summary = {
        "checkpoint": args.checkpoint,
        "timesteps": config["model"]["timesteps"],
        "epochs": config["training"]["max_epochs"],
        "test_loss": float(np.mean(results["loss"])),
        "test_f1": float(np.mean(results["f1"])),
        "test_f1_std": float(np.std(results["f1"])),
    }

    print(summary)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
            writer.writeheader()
            writer.writerow(summary)

if __name__ == "__main__":
    main()
