"""Tests for src/config.py and src/utils.py."""

import numpy as np
import pytest

from src.config import Config, load_config, save_config
from src.utils import Timer, load_json, save_csv, save_json, set_seed


def test_default_config_matches_model_contract():
    cfg = Config()
    assert cfg.model.input_dim == 29
    assert cfg.model.hidden_dims == (256, 256, 128)
    assert cfg.model.n_bins == 10
    assert cfg.data.split_seed == 42


def test_load_config_yaml_overrides_and_tuples(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text(
        "train:\n  batch_size: 512\nmodel:\n  hidden_dims: [128, 128]\n", encoding="utf-8"
    )
    cfg = load_config(str(path))
    assert cfg.train.batch_size == 512
    assert cfg.model.hidden_dims == (128, 128)
    assert cfg.train.learning_rate == Config().train.learning_rate  # untouched default


def test_unknown_keys_are_rejected(tmp_path):
    bad_key = tmp_path / "bad_key.yaml"
    bad_key.write_text("train:\n  batchsize: 512\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(str(bad_key))
    bad_section = tmp_path / "bad_section.yaml"
    bad_section.write_text("trainn:\n  epochs: 1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(str(bad_section))


def test_save_and_load_config_roundtrip(tmp_path):
    cfg = Config()
    cfg.train.epochs = 3
    cfg.model.hidden_dims = (64, 32)
    path = tmp_path / "run" / "config.yaml"
    save_config(cfg, str(path))
    assert load_config(str(path)) == cfg


def test_set_seed_is_reproducible():
    set_seed(123)
    a = np.random.rand(5)
    set_seed(123)
    b = np.random.rand(5)
    assert np.array_equal(a, b)


def test_set_seed_torch_if_available():
    torch = pytest.importorskip("torch")
    set_seed(7)
    a = torch.rand(4)
    set_seed(7)
    assert torch.equal(a, torch.rand(4))


def test_json_roundtrip_with_numpy_types(tmp_path):
    path = tmp_path / "sub" / "x.json"
    save_json(path, {"i": np.int64(3), "f": np.float32(0.5), "a": np.arange(3), "t": (1, 2)})
    assert load_json(path) == {"i": 3, "f": 0.5, "a": [0, 1, 2], "t": [1, 2]}


def test_save_csv(tmp_path):
    path = tmp_path / "history.csv"
    save_csv(path, [{"epoch": 1, "val_mae": np.float32(0.25)}, {"epoch": 2, "val_mae": 0.2}])
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0] == "epoch,val_mae"
    assert len(lines) == 3


def test_timer_measures_elapsed():
    with Timer("noop") as t:
        sum(range(1000))
    assert t.elapsed >= 0.0
