from pathlib import Path

from src.config import load_config
from src.run_loop import run_training_and_evaluation


CONFIG_PATH = Path("configs/train/default.yaml")
JOB_DIR = Path("logs/manual_train")
RESUME = True


def main():
    config = load_config(CONFIG_PATH)
    run_training_and_evaluation(config, JOB_DIR, resume=RESUME)


if __name__ == "__main__":
    main()
