#!/usr/bin/env python
from __future__ import annotations

import copy
import subprocess
import sys
from pathlib import Path

import yaml

# ------------------------------------------------------------
# setup (edit this block)
# ------------------------------------------------------------
EXPERIMENT_NAME = "ArchiveII_sim_sweep"
BASE_CONFIG_PATH = "configs/sim60.yaml"
PARTITIONS = ["sim60", "sim70", "sim80", "sim90"]
TIMESTEPS = [5, 10]
EPOCHS = 1
FOLD = 0
CONSENSUS = "1,3"
RESUME = False
DRY_RUN = False


def latest_run_dir(save_dir: Path, run_name: str):
    matches = list(save_dir.glob(f"{run_name}*"))
    if not matches:
        return None
    matches.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return matches[0]


def run(cmd, cwd, dry_run):
    print("[CMD]", " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, cwd=cwd, check=True)


def build_run_config(base_config, experiment_name, partition, timestep, epochs, fold):
    config = copy.deepcopy(base_config)
    run_name = f"{partition}_t{timestep}_e{epochs}"
    config["experiment"]["name"] = run_name
    config["experiment"]["note"] = f"{experiment_name} | {partition} fold={fold}, t={timestep}, e={epochs}"
    config["data"]["partition_path"] = (
        f"data/simfolds/simfolds_max128/ArchiveII_partitions_{partition}.csv"
    )
    config["data"]["fold"] = fold
    config["model"]["timesteps"] = timestep
    config["training"]["max_epochs"] = epochs
    config["logging"]["save_dir"] = f"logs/{experiment_name}/{partition}/t{timestep}"
    return config, run_name


def main():
    repo_root = Path(__file__).resolve().parent.parent
    config_dir = repo_root / f"configs/experiments/{EXPERIMENT_NAME}"
    config_dir.mkdir(parents=True, exist_ok=True)

    with (repo_root / BASE_CONFIG_PATH).open("r", encoding="utf-8") as handle:
        base_config = yaml.safe_load(handle) or {}
    jobs = [(partition, timestep) for partition in PARTITIONS for timestep in TIMESTEPS]

    for index, (partition, timestep) in enumerate(jobs, start=1):
        config, run_name = build_run_config(
            base_config=base_config,
            experiment_name=EXPERIMENT_NAME,
            partition=partition,
            timestep=timestep,
            epochs=EPOCHS,
            fold=FOLD,
        )
        print(f"\n[{index}/{len(jobs)}] {run_name}")

        partition_path = repo_root / config["data"]["partition_path"]
        if not partition_path.exists():
            raise FileNotFoundError(f"Partition file not found: {partition_path}")

        save_dir = repo_root / config["logging"]["save_dir"]
        latest = latest_run_dir(save_dir, run_name)
        if RESUME and latest and (latest / "ensemble.csv").exists():
            print(f"[SKIP] {latest} already completed")
            continue

        config_path = config_dir / f"{partition}_t{timestep}.yaml"
        with config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)

        run([sys.executable, "train.py", "--config", str(config_path)], cwd=repo_root, dry_run=DRY_RUN)

        latest = latest_run_dir(save_dir, run_name) or (save_dir / run_name)
        run(
            [
                sys.executable,
                "ensemble_stats.py",
                "--samples-dir",
                str(latest / "raw_samples"),
                "--consensus",
                CONSENSUS,
                "--output",
                str(latest / "ensemble.csv"),
            ],
            cwd=repo_root,
            dry_run=DRY_RUN,
        )


if __name__ == "__main__":
    main()
