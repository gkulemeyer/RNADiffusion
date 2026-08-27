from pathlib import Path

from src.config import load_config
from src.experiment import generate_ensemble_samples
from src.run_io import RunIO


REPO_ROOT = Path(__file__).resolve().parent

RUN_DIR = Path("logs/ArchiveII_famfold/srp/t2/seed42")
run = RunIO(REPO_ROOT / RUN_DIR)


CHECKPOINT_PATH = None
NUM_SAMPLES = None
BASE_SEED = None
BATCH_SIZE = None
THRESHOLD = 0.1
CLEAR_SAMPLES = False


def main():
    run = RunIO(REPO_ROOT / RUN_DIR)
    tresholds = [0.08, 0.12, 0.15]

    checkpoint = (
        REPO_ROOT / CHECKPOINT_PATH
        if CHECKPOINT_PATH is not None
        else run.best_ckpt_path
    )
    for threshold in tresholds:
        samples_dir = run.best_eval_dir / f"samples_thr{threshold}"

    
    # samples_dir = (
    #     REPO_ROOT / SAMPLES_DIR
    #     if SAMPLES_DIR is not None
    #     else run.best_eval_dir / "samples"
    # )

        generate_ensemble_samples(
            config=load_config(run.config_path),
            checkpoint=checkpoint,
            samples_dir=samples_dir,
            num_samples=NUM_SAMPLES,
            base_seed=BASE_SEED,
            batch_size=BATCH_SIZE,
            threshold=threshold,
            clear_samples=CLEAR_SAMPLES,
        )
        print(f"Saved raw and processed samples to {samples_dir}")


if __name__ == "__main__":
    main()
