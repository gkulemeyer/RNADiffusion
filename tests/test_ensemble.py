from pathlib import Path

import yaml

from src.experiment import archive_epoch_artifacts
from src.io import load_samples_metadata, write_ensemble_metadata, write_samples_metadata


def test_write_samples_metadata(tmp_path: Path):
    samples_dir = tmp_path / "raw_samples"
    samples_dir.mkdir()

    metadata = write_samples_metadata(
        samples_dir=samples_dir,
        checkpoint_path="checkpoints/best.ckpt",
        num_samples=5,
        base_seed=42,
        chunk_size=25,
    )

    metadata_path = samples_dir / "samples_metadata.yaml"
    assert metadata_path.exists()
    assert metadata["num_samples"] == 5
    assert metadata["base_seed"] == 42

    loaded = load_samples_metadata(samples_dir)
    assert loaded["checkpoint_path"] == "checkpoints/best.ckpt"


def test_write_ensemble_metadata_uses_samples_metadata(tmp_path: Path):
    samples_dir = tmp_path / "raw_samples"
    samples_dir.mkdir()
    write_samples_metadata(
        samples_dir=samples_dir,
        checkpoint_path="checkpoints/best.ckpt",
        num_samples=5,
        base_seed=42,
        chunk_size=25,
    )

    output_path = tmp_path / "ensemble.csv"
    metadata = write_ensemble_metadata(
        output_path=output_path,
        samples_dir=samples_dir,
        trials=20,
        consensus_sizes=[3],
        seed=42,
    )

    metadata_path = output_path.with_name("ensemble_metadata.yaml")
    assert metadata_path.exists()
    assert metadata["num_samples"] == 5
    assert metadata["consensus_sizes"] == [3]

    with metadata_path.open("r", encoding="utf-8") as handle:
        stored = yaml.safe_load(handle)

    assert stored["checkpoint_path"] == "checkpoints/best.ckpt"
    assert stored["trials"] == 20


def test_evaluate_samples_stats_writes_expected_summary(tmp_path: Path):
    samples_csv = tmp_path / "ensemble.csv"
    samples_csv.write_text(
        "\n".join(
            [
                "seq_id,cons_k1_mean,cons_k1_std,cons_k3_mean,cons_k3_std",
                "a,0.1,0.01,0.4,0.04",
                "b,0.3,0.03,0.6,0.06",
            ]
        ),
        encoding="utf-8",
    )

    from src.ensemble import evaluate_samples_stats

    stats = evaluate_samples_stats(samples_csv, consensus_sizes=[1, 3])

    assert stats.columns.tolist() == ["consensus", "mean", "std", "std_mean", "std_std"]
    assert stats["consensus"].tolist() == [1, 3]


def test_archive_epoch_artifacts_moves_summary_and_checkpoint(tmp_path: Path):
    experiment_dir = tmp_path / "run"
    checkpoint_dir = experiment_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    source_path = experiment_dir / "ensemble_stats.csv"
    source_path.write_text("consensus,mean\n1,0.5\n", encoding="utf-8")
    checkpoint_path = checkpoint_dir / "last.ckpt"
    checkpoint_path.write_text("checkpoint", encoding="utf-8")

    class DummyLogger:
        def info(self, *args, **kwargs):
            return None

    archived_checkpoint = archive_epoch_artifacts(
        experiment_dir,
        2,
        checkpoint_path,
        DummyLogger(),
    )

    archived_path = experiment_dir / "epoch_2" / "ensemble_stats.csv"
    archived_checkpoint_path = experiment_dir / "epoch_2" / "last.ckpt"
    assert archived_path.exists()
    assert archived_path.read_text(encoding="utf-8") == "consensus,mean\n1,0.5\n"
    assert archived_checkpoint == archived_checkpoint_path
    assert archived_checkpoint_path.exists()
    assert archived_checkpoint_path.read_text(encoding="utf-8") == "checkpoint"
    assert not source_path.exists()
    assert not checkpoint_path.exists()
