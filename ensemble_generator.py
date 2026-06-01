from pathlib import Path

from src.config import load_config
from src.experiment import generate_ensemble_samples
from src.run_io import RunIO


RUN_DIR = Path("")
CONFIG_PATH = Path("configs/train/default.yaml")
CHECKPOINT_PATH = Path("")
SAMPLES_DIR = Path("")

NUM_SAMPLES = None
BASE_SEED = None
BATCH_SIZE = None
CLEAR_SAMPLES = False


def resolve_inputs():
    if str(RUN_DIR):
        run = RunIO(RUN_DIR)
        samples_dir = SAMPLES_DIR if str(SAMPLES_DIR) else run.best_eval_dir / "raw_samples"
        return run.config_path, run.best_ckpt_path, samples_dir
    if not str(CHECKPOINT_PATH):
        raise ValueError("Set RUN_DIR or CHECKPOINT_PATH before running ensemble_generator.py")
    samples_dir = SAMPLES_DIR if str(SAMPLES_DIR) else CHECKPOINT_PATH.parent.parent / "raw_samples"
    return CONFIG_PATH, CHECKPOINT_PATH, samples_dir


def main():
    config_path, checkpoint_path, samples_dir = resolve_inputs()
    samples_dir = generate_ensemble_samples(
        config=load_config(config_path),
        checkpoint=checkpoint_path,
        samples_dir=samples_dir,
        num_samples=NUM_SAMPLES,
        base_seed=BASE_SEED,
        batch_size=BATCH_SIZE,
        clear_samples=CLEAR_SAMPLES,
    )
    print(f"Saved raw samples to {samples_dir}")


if __name__ == "__main__":
    main()
