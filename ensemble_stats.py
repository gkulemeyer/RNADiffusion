from pathlib import Path

from src.config import load_ensemble_defaults
from src.experiment import evaluate_ensemble_samples


REPO_ROOT = Path(__file__).resolve().parent
ENSEMBLE_DEFAULTS = load_ensemble_defaults()

SAMPLES_DIR = Path("logs/ArchiveII_famfold/srp/t2/seed42/eval/best/samples_thr0.15")
SAMPLE_TYPE = "processed"
OUTPUT_FILENAME = "processed_ensemble_metrics_thr015.csv"

TRIALS = ENSEMBLE_DEFAULTS["trials"]
CONSENSUS_SIZES = ENSEMBLE_DEFAULTS["consensus_sizes"]
SEED = ENSEMBLE_DEFAULTS["base_seed"]
GET_BEST_AND_WORST = ENSEMBLE_DEFAULTS["get_best_and_worst"]


def main():
    samples_dir = REPO_ROOT / SAMPLES_DIR
    output_dir = samples_dir.parent / "samples_stats015"
    output_path = output_dir / OUTPUT_FILENAME

    metrics_path, stats_path = evaluate_ensemble_samples(
        samples_dir=samples_dir,
        output_path=output_path,
        trials=TRIALS,
        consensus_sizes=CONSENSUS_SIZES,
        seed=SEED,
        metadata_path=output_dir / "ensemble_metadata.yaml",
        get_best_and_worst=GET_BEST_AND_WORST,
        sample_type=SAMPLE_TYPE,
    )
    print(f"Saved ensemble metrics to {metrics_path}")
    print(f"Saved ensemble summary to {stats_path}")


if __name__ == "__main__":
    main()
