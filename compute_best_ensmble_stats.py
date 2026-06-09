from pathlib import Path

from src.config import load_config, load_ensemble_defaults
from src.experiment import generate_ensemble_samples, evaluate_ensemble_samples
from src.run_io import RunIO


ENSEMBLE_DEFAULTS = load_ensemble_defaults()
EXP_PATH = Path("/home/gkulemeyer/Documents/Repos/RNADiffusion/logs/bprna_simfold_512/")

NUM_SAMPLES = None
BASE_SEED = None
BATCH_SIZE = None
CLEAR_SAMPLES = False
KEEP_SAMPLES = False

TRIALS = ENSEMBLE_DEFAULTS["trials"]
CONSENSUS_SIZES = ENSEMBLE_DEFAULTS["consensus_sizes"]
SEED = ENSEMBLE_DEFAULTS["base_seed"]
GET_BEST_AND_WORST = ENSEMBLE_DEFAULTS["get_best_and_worst"]

def main():
    for sim in EXP_PATH.glob("sim*"):

        for ts in sim.glob("t*"):
            print(sim, ts)
            run_dir = ts / "vb_stochastic"
            output_dir = run_dir / "best_ensemble"

            run = RunIO(run_dir)
            # generate and evaluate best ckpt ensemble
            samples_dir = generate_ensemble_samples(
                config=load_config(run.config_path),
                checkpoint=run.best_ckpt_path,
                samples_dir=run.best_eval_dir / "raw_samples",
                num_samples=NUM_SAMPLES,
                base_seed=BASE_SEED,
                batch_size=BATCH_SIZE,
                clear_samples=CLEAR_SAMPLES,
            )
            print(f"Saved raw samples to {samples_dir}")
        
        
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path, stats_path = evaluate_ensemble_samples(
                samples_dir=samples_dir,
                output_path=str(output_dir / "ensemble.csv"),
                trials=TRIALS,
                consensus_sizes=CONSENSUS_SIZES,
                seed=SEED,
                get_best_and_worst=GET_BEST_AND_WORST,
            )
            print(f"Saved ensemble statistics to {output_path}")
            print(f"Saved ensemble summary stats to {stats_path}")

            if not KEEP_SAMPLES:
                for file in samples_dir.glob("*"):
                    file.unlink()
                samples_dir.rmdir()
                print(f"Deleted raw samples from {samples_dir}")

if __name__ == "__main__":
    main()
