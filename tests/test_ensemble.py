from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml

from src.ensemble import (
    SequenceEnsemble,
    _generate_ensemble_batch,
    evaluate_samples_dir,
    evaluate_samples_stats,
    export_db_ensemble,
    process_sample,
    save_ensemble_samples,
)
from src.io import load_samples_metadata, write_ensemble_metadata, write_samples_metadata


class FakeSampleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))
        self.seeds = []
        self.eval_called = False

    def eval(self):
        self.eval_called = True
        return super().eval()

    def sample(self, conditioning, lengths=None, return_logits=False):
        assert return_logits is True
        self.seeds.append(torch.initial_seed())
        batch_size, _, height, width = conditioning.shape
        raw = torch.zeros((batch_size, height, width), device=conditioning.device)
        raw[:, 0, 1] = 1
        logits = torch.full(
            (batch_size, 2, height, width),
            -10.0,
            device=conditioning.device,
        )
        logits[:, 1, 0, 1] = 10.0
        return raw, logits


class RandomSampleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))

    def sample(self, conditioning, lengths=None, return_logits=False):
        batch_size, _, height, width = conditioning.shape
        raw = torch.randint(0, 2, (batch_size, height, width))
        logits = torch.zeros((batch_size, 2, height, width))
        return raw, logits


def make_ensemble_payload():
    raw = torch.zeros((2, 4, 4), dtype=torch.int8)
    raw[:, 0, 1] = 1
    processed = torch.zeros_like(raw)
    processed[:, 0, 1] = 1
    processed[:, 1, 0] = 1
    target = processed[0].clone()
    return {
        "id": "seq-a",
        "raw_samples": raw,
        "processed_samples": processed,
        "target": target,
        "length": 4,
        "sample_seeds": [42, 43],
    }


def test_process_sample_is_symmetric_binary_greedy_matching():
    raw = torch.zeros((4, 4), dtype=torch.float32)
    raw[0, 1] = raw[0, 2] = raw[1, 3] = 1
    raw[3, 2] = 1  # Lower triangle must be ignored.
    logits = torch.full((2, 4, 4), -10.0)
    logits[1, 0, 1] = 2.0
    logits[1, 0, 2] = 3.0
    logits[1, 1, 3] = 1.0

    processed = process_sample(raw, logits, threshold=0.5)

    expected = torch.zeros_like(raw)
    expected[0, 2] = expected[2, 0] = 1
    expected[1, 3] = expected[3, 1] = 1
    assert torch.equal(processed, expected)
    assert processed.dtype == raw.dtype
    assert torch.equal(processed, processed.T)
    assert torch.count_nonzero(processed.diag()) == 0
    assert torch.all(processed.sum(dim=0) <= 1)
    assert set(processed.unique().tolist()) <= {0.0, 1.0}


def test_generate_ensemble_batch_uses_one_call_per_seed_and_zero_padding():
    model = FakeSampleModel()
    conditioning = torch.zeros((2, 1, 4, 4))

    raw, processed = _generate_ensemble_batch(
        model,
        conditioning,
        lengths=torch.tensor([3, 1]),
        sample_seeds=[10, 11, 12],
        threshold=0.5,
    )

    assert model.seeds == [10, 11, 12]
    assert raw.shape == processed.shape == (2, 3, 4, 4)
    assert raw.dtype == processed.dtype == torch.int8
    assert raw.device.type == processed.device.type == "cpu"
    assert torch.count_nonzero(processed[1]) == 0
    assert torch.count_nonzero(processed[:, :, 3, :]) == 0
    assert torch.count_nonzero(processed[:, :, :, 3]) == 0


def test_raw_generation_matches_previous_chunk_size_one_seed_behavior():
    model = RandomSampleModel()
    conditioning = torch.zeros((2, 1, 3, 3))
    lengths = torch.tensor([3, 3])
    sample_seeds = [20, 21, 22]

    expected_members = []
    for sample_seed in sample_seeds:
        torch.manual_seed(sample_seed)
        expected, _ = model.sample(conditioning, lengths=lengths, return_logits=True)
        expected_members.append(expected)
    expected_raw = torch.stack(expected_members, dim=1).to(torch.int8)

    raw, _ = _generate_ensemble_batch(
        model,
        conditioning,
        lengths,
        sample_seeds,
        threshold=0.5,
    )

    assert torch.equal(raw, expected_raw)


def test_save_ensemble_samples_skips_existing_and_writes_exact_schema(tmp_path: Path):
    model = FakeSampleModel()
    existing_path = tmp_path / "seq-a.pt"
    torch.save({"sentinel": True}, existing_path)
    contact_oh = torch.zeros((2, 2, 4, 4))
    contact_oh[:, 0] = 1
    contact_oh[1, 0, 0, 1] = 0
    contact_oh[1, 1, 0, 1] = 1
    loader = [
        {
            "id": ["seq-a", "seq-b"],
            "conditioning": torch.zeros((2, 1, 4, 4)),
            "length": [4, 3],
            "contact_oh": contact_oh,
        }
    ]

    save_ensemble_samples(
        model,
        loader,
        tmp_path,
        sample_seeds=[42, 43],
        threshold=0.5,
    )

    assert torch.load(existing_path)["sentinel"] is True
    saved = torch.load(tmp_path / "seq-b.pt", map_location="cpu")
    assert set(saved) == {
        "id",
        "raw_samples",
        "processed_samples",
        "target",
        "length",
        "sample_seeds",
    }
    assert saved["id"] == "seq-b"
    assert saved["raw_samples"].shape == saved["processed_samples"].shape == (2, 3, 3)
    assert saved["raw_samples"].dtype == torch.int8
    assert saved["processed_samples"].dtype == torch.int8
    assert saved["target"].shape == (3, 3)
    assert saved["target"].dtype == torch.int8
    assert saved["target"][0, 1] == 1
    assert saved["length"] == 3
    assert saved["sample_seeds"] == [42, 43]
    assert model.eval_called is True


def test_sequence_ensemble_loads_new_and_legacy_formats(tmp_path: Path):
    new_path = tmp_path / "new.pt"
    torch.save(make_ensemble_payload(), new_path)
    new = SequenceEnsemble(new_path)

    assert new.raw_samples.device.type == "cpu"
    assert new.processed_samples.device.type == "cpu"
    assert new.sample_seeds == [42, 43]
    assert new.num_samples == 2
    assert torch.equal(new.get_samples("raw"), new.raw_samples)
    assert torch.equal(new.get_samples("processed"), new.processed_samples)
    assert torch.equal(new.consensus(), new.consensus(sample_type="raw"))

    legacy_path = tmp_path / "legacy.pt"
    payload = make_ensemble_payload()
    one_hot_target = torch.nn.functional.one_hot(
        payload["target"].long(), num_classes=2
    ).permute(2, 0, 1)
    torch.save(
        {
            "samples": payload["raw_samples"],
            "seeds": [42, 43],
            "target": one_hot_target,
            "length": 4,
        },
        legacy_path,
    )
    legacy = SequenceEnsemble(legacy_path)

    assert legacy.processed_samples is None
    assert legacy.sample_seeds == [42, 43]
    assert legacy.target.ndim == 2
    assert torch.equal(legacy.get_samples("raw"), payload["raw_samples"])
    assert legacy.get_samples("processed") is None


def test_export_db_ensemble_uses_saved_processed_samples(tmp_path: Path):
    samples_dir = tmp_path / "samples"
    samples_dir.mkdir()
    payload_a = make_ensemble_payload()
    payload_b = make_ensemble_payload()
    payload_b["id"] = "seq-b"
    torch.save(payload_b, samples_dir / "b.pt")
    torch.save(payload_a, samples_dir / "a.pt")
    output_csv = tmp_path / "generated_ensemble.csv"

    dataframe = export_db_ensemble(samples_dir, output_csv)

    assert dataframe.columns.tolist() == [
        "id",
        "sample_id",
        "seed",
        "sampled_structure",
    ]
    assert len(dataframe) == 4
    assert not dataframe.duplicated(["id", "sample_id"]).any()
    assert dataframe["seed"].tolist() == [42, 43, 42, 43]
    assert dataframe["sampled_structure"].tolist() == ["().."] * 4
    assert pd.read_csv(output_csv).columns.tolist() == dataframe.columns.tolist()


def test_evaluate_samples_dir_selects_raw_or_processed_samples(tmp_path: Path):
    payload = make_ensemble_payload()
    payload["raw_samples"] = torch.zeros_like(payload["raw_samples"])
    torch.save(payload, tmp_path / "seq-a.pt")

    raw_metrics = evaluate_samples_dir(
        tmp_path,
        consensus_sizes=[2],
        trials=1,
        sample_type="raw",
    )
    processed_metrics = evaluate_samples_dir(
        tmp_path,
        consensus_sizes=[2],
        trials=1,
        sample_type="processed",
    )

    assert raw_metrics.loc[0, "cons_k2_mean"] == 0.0
    assert processed_metrics.loc[0, "cons_k2_mean"] == pytest.approx(1.0)


def test_write_samples_metadata(tmp_path: Path):
    samples_dir = tmp_path / "samples"
    samples_dir.mkdir()

    metadata = write_samples_metadata(
        samples_dir=samples_dir,
        checkpoint_path="checkpoints/best.ckpt",
        num_samples=3,
        base_seed=42,
        sample_seeds=[42, 43, 44],
        threshold=0.5,
        checkpoint_epoch=7,
    )

    assert "chunk_size" not in metadata
    assert metadata["sample_seeds"] == [42, 43, 44]
    assert metadata["stored_samples"] == ["raw", "processed"]
    assert metadata["processing"] == {
        "score": "sigmoid_channel_1",
        "threshold": 0.5,
        "triangle": "upper",
        "multiplet_resolution": "greedy_highest_score",
        "symmetric_output": True,
    }
    loaded = load_samples_metadata(samples_dir)
    assert loaded["checkpoint_path"] == "checkpoints/best.ckpt"
    assert loaded["checkpoint_epoch"] == 7


def test_write_ensemble_metadata_uses_samples_metadata(tmp_path: Path):
    samples_dir = tmp_path / "samples"
    samples_dir.mkdir()
    write_samples_metadata(
        samples_dir=samples_dir,
        checkpoint_path="checkpoints/best.ckpt",
        num_samples=3,
        base_seed=42,
        sample_seeds=[42, 43, 44],
        threshold=0.5,
        checkpoint_epoch=7,
    )

    output_path = tmp_path / "raw_ensemble_metrics.csv"
    metadata = write_ensemble_metadata(
        output_path=output_path,
        samples_dir=samples_dir,
        trials=20,
        consensus_sizes=[3],
        seed=42,
    )

    metadata_path = tmp_path / "raw_ensemble_metrics_metadata.yaml"
    assert metadata_path.exists()
    assert metadata["num_samples"] == 3
    assert metadata["consensus_sizes"] == [3]
    with metadata_path.open("r", encoding="utf-8") as handle:
        stored = yaml.safe_load(handle)
    assert stored["checkpoint_path"] == "checkpoints/best.ckpt"
    assert stored["checkpoint_epoch"] == 7
    assert stored["trials"] == 20


def test_evaluate_samples_stats_writes_expected_summary(tmp_path: Path):
    samples_csv = tmp_path / "raw_ensemble_metrics.csv"
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

    stats = evaluate_samples_stats(samples_csv, consensus_sizes=[1, 3])

    assert stats.columns.tolist() == ["consensus", "mean", "std", "std_mean", "std_std"]
    assert stats["consensus"].tolist() == [1, 3]
