import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments.pipeline import run_pipeline


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
