from pathlib import Path

import ensemble_stats


def test_ensemble_stats_writes_next_to_samples(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(ensemble_stats, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(ensemble_stats, "SAMPLES_DIR", Path("run/eval/best/samples"))
    monkeypatch.setattr(ensemble_stats, "SAMPLE_TYPE", "processed")
    monkeypatch.setattr(
        ensemble_stats,
        "OUTPUT_FILENAME",
        "processed_ensemble_metrics.csv",
    )
    monkeypatch.setattr(
        ensemble_stats,
        "evaluate_ensemble_samples",
        lambda **kwargs: calls.append(kwargs)
        or (kwargs["output_path"], kwargs["output_path"].with_name("processed_ensemble_stats.csv")),
    )

    ensemble_stats.main()

    output_dir = tmp_path / "run/eval/best/samples_stats"
    assert calls[0]["samples_dir"] == tmp_path / "run/eval/best/samples"
    assert calls[0]["output_path"] == output_dir / "processed_ensemble_metrics.csv"
    assert calls[0]["metadata_path"] == output_dir / "ensemble_metadata.yaml"
    assert calls[0]["sample_type"] == "processed"
