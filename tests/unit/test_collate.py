import torch as tr

from src.data.collate import pad_batch


def test_pad_batch_shapes():
    batch = [
        {
            "id": "a",
            "sequence": "ACGU",
            "length": 4,
            "embedding": tr.zeros(4, 4),
            "outer": tr.zeros(16, 4, 4),
            "contact": tr.zeros(4, 4, dtype=tr.long),
            "contact_oh": tr.zeros(2, 4, 4),
        },
        {
            "id": "b",
            "sequence": "ACG",
            "length": 3,
            "embedding": tr.zeros(4, 3),
            "outer": tr.zeros(16, 3, 3),
            "contact": tr.zeros(3, 3, dtype=tr.long),
            "contact_oh": tr.zeros(2, 3, 3),
        },
    ]

    out = pad_batch(batch)
    assert out["embedding"].shape[0] == 2
    assert out["outer"].shape[0] == 2
    assert out["mask"].shape[0] == 2
