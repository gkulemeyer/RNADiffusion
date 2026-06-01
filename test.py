from pathlib import Path

from src.config import load_config
from src.experiment import test_checkpoint
from src.run_io import RunIO


RUN_DIR = Path("")
CONFIG_PATH = Path("configs/train/default.yaml")
CHECKPOINT_PATH = Path("")
OUTPUT_PATH = Path("")
BATCH_SIZE = None


def resolve_inputs():
    if str(RUN_DIR):
        run = RunIO(RUN_DIR)
        return run.config_path, run.best_ckpt_path
    if not str(CHECKPOINT_PATH):
        raise ValueError("Set RUN_DIR or CHECKPOINT_PATH before running test.py")
    return CONFIG_PATH, CHECKPOINT_PATH


def main():
    config_path, checkpoint_path = resolve_inputs()
    summary = test_checkpoint(
        config=load_config(config_path),
        checkpoint=checkpoint_path,
        output_path=OUTPUT_PATH if str(OUTPUT_PATH) else None,
        batch_size=BATCH_SIZE,
    )
    print(summary)


if __name__ == "__main__":
    main()
