from src.utils.io import save_config, load_config


def test_config_roundtrip(tmp_path):
    cfg = {"a": 1, "b": {"c": [1, 2, 3]}}
    save_config(cfg, tmp_path)
    loaded = load_config(tmp_path)
    assert loaded["a"] == 1
    assert loaded["b"]["c"] == [1, 2, 3]
