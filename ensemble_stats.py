from pathlib import Path

from src.config import load_ensemble_defaults
from src.experiment import evaluate_ensemble_samples


ENSEMBLE_DEFAULTS = load_ensemble_defaults()

SAMPLES_DIR = Path("")
OUTPUT_PATH = Path("")
TRIALS = ENSEMBLE_DEFAULTS["trials"]
CONSENSUS_SIZES = ENSEMBLE_DEFAULTS["consensus_sizes"]
SEED = ENSEMBLE_DEFAULTS["base_seed"]
GET_BEST_AND_WORST = ENSEMBLE_DEFAULTS["get_best_and_worst"]

def main():
    if not str(SAMPLES_DIR):
        raise ValueError("Set SAMPLES_DIR before running ensemble_stats.py")

    output_path, stats_path = evaluate_ensemble_samples(
        samples_dir=SAMPLES_DIR,
        output_path=OUTPUT_PATH if str(OUTPUT_PATH) else None,
        trials=TRIALS,
        consensus_sizes=CONSENSUS_SIZES,
        seed=SEED,
        get_best_and_worst=GET_BEST_AND_WORST,
    )
    print(f"Saved ensemble statistics to {output_path}")
    print(f"Saved ensemble summary stats to {stats_path}")


if __name__ == "__main__":
    main()
