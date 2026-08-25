from pathlib import Path

from src.config import load_config
from src.experiment import (
    build_dataloader,
    load_model_checkpoint,
)
from src.ensemble import generate_ensemble_db
from src.run_io import RunIO


RUN_DIR = Path(
    "/home/gkulemeyer/Documents/Repos/RNADiffusion/"
    "logs/ArchiveII_ff128/srp/t10/bd32/bs1_acc4/seed42"
) 
CONFIG_PATH = None
CHECKPOINT_PATH = None
OUTPUT_CSV = None

NUM_SAMPLES = 10
SEED = 42
CHUNK_SIZE = 1
PARTITION = "test"
SAVE_AS = "db"
thresholds = [0, 0.1,  0.5,  0.8,]


def resolve_inputs(threshold=0):
    # if has a "dot, remove it"
    if threshold == 0:
        threshold = 0
    else:
        threshold = str(threshold).replace(".", "")
    if RUN_DIR is not None:
        run = RunIO(RUN_DIR)

        output_csv = (
            Path(OUTPUT_CSV)
            if OUTPUT_CSV is not None
            else run.best_eval_dir / f"generated_ensemble_db_thr{threshold}.csv"
        )

        return run.config_path, run.best_ckpt_path,output_csv  
    return Path(CONFIG_PATH), Path(CHECKPOINT_PATH), Path(OUTPUT_CSV)


def main():

    for threshold in thresholds:
        config_path, checkpoint_path, output_csv = resolve_inputs(threshold=threshold)

        config = load_config(config_path)
        model = load_model_checkpoint(config,checkpoint_path,eval_mode=True)
        loader = build_dataloader( config, partition=PARTITION, shuffle=False)

        print(f"Config:     {config_path}"); print(f"Checkpoint: {checkpoint_path}")
        print(f"Output:     {output_csv}")
        ensemble = generate_ensemble_db(
            model=model,
            loader=loader,
            output_csv=output_csv,
            num_samples=NUM_SAMPLES,
            seed=SEED,
            chunk_size=CHUNK_SIZE,
            save_as=SAVE_AS,
            threshold=threshold
        )

        print(f"Generated rows: {len(ensemble)}")
        print(f"Saved ensemble to: {output_csv}")


if __name__ == "__main__":
    main()