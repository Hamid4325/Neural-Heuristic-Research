"""Smoke and unit tests for src/train.py.

The training tests use small synthetic data, so they run anywhere in a few seconds.
They use Hamid's real model (src/model.py) when it exists, otherwise a tiny stand-in
network with the same interface: forward(x) -> (y_pred (B, 1), bin_logits (B, n_bins)).
"""

import csv
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from src.config import Config, load_config
from src.train import (
    HISTORY_COLUMNS,
    assign_bins,
    default_model_factory,
    macro_f1,
    make_bin_edges,
    resolve_outputs,
    train,
)

CHECKPOINT_KEYS = {
    "model_state", "model_config", "target_mean", "target_std",
    "bin_edges", "config", "git_commit", "seed",
}


class StandInNet(nn.Module):
    """Minimal network with the contract interface (used until src/model.py exists)."""

    def __init__(self, cfg):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(cfg.input_dim, 32), nn.ReLU())
        self.reg = nn.Linear(32, 1)
        self.bins = nn.Linear(32, cfg.n_bins)

    def forward(self, x):
        h = self.trunk(x)
        return self.reg(h), self.bins(h)


def _real_model_available() -> bool:
    try:
        import src.model  # noqa: F401
        return True
    except ModuleNotFoundError as exc:
        if exc.name == "src.model":
            return False
        raise


MODEL_FACTORY = default_model_factory if _real_model_available() else StandInNet


def synthetic(n: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.random((n, 29), dtype=np.float32)
    y = (2.0 * X[:, 0] + X[:, 1] + 0.3 * X[:, 2] + 0.05 * rng.standard_normal(n))
    return X, np.abs(y).astype(np.float32)


@pytest.fixture
def small_cfg():
    cfg = Config()
    cfg.train.epochs = 1
    cfg.train.batch_size = 256
    cfg.train.max_train_cells_per_epoch = 10_000
    return cfg


@pytest.fixture
def small_data():
    X_tr, y_tr = synthetic(10_000, seed=0)
    X_va, y_va = synthetic(2_000, seed=1)
    return X_tr, y_tr, X_va, y_va


# ---------------------------------------------------------------------------
# Smoke test: 10k cells, 1 epoch
# ---------------------------------------------------------------------------
def test_training_smoke(tmp_path, small_cfg, small_data):
    out = train(small_cfg, str(tmp_path / "run"), model_factory=MODEL_FACTORY, data=small_data)

    ckpt_path = Path(out["checkpoint"])
    assert ckpt_path.exists()
    ckpt = torch.load(ckpt_path, map_location="cpu")  # default weights_only loader
    assert CHECKPOINT_KEYS <= set(ckpt)
    assert ckpt["seed"] == 42
    assert ckpt["target_std"] > 0.0
    assert len(ckpt["bin_edges"]) == 9
    assert all(a < b for a, b in zip(ckpt["bin_edges"], ckpt["bin_edges"][1:]))
    assert ckpt["model_config"]["input_dim"] == 29
    assert ckpt["model_config"]["hidden_dims"] == [256, 256, 128]
    assert isinstance(ckpt["git_commit"], str)

    # The saved weights load back into a freshly built network.
    net = MODEL_FACTORY(small_cfg.model)
    net.load_state_dict(ckpt["model_state"])

    assert out["epochs_run"] == 1
    assert np.isfinite(out["history"][0]["val_loss"])
    assert np.isfinite(out["best_val_mae"])


def test_history_csv_and_run_files(tmp_path, small_cfg, small_data):
    out = train(small_cfg, str(tmp_path / "run"), model_factory=MODEL_FACTORY, data=small_data)
    run = Path(out["run_dir"])
    for name in ("history.csv", "config.yaml", "metrics.json"):
        assert (run / name).exists()
    with open(run / "history.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0].keys()) == HISTORY_COLUMNS
    assert len(rows) == 1 and rows[0]["epoch"] == "1"
    assert load_config(str(run / "config.yaml")) == small_cfg


def test_early_stopping(tmp_path, small_cfg, small_data):
    # lr = 0 -> the weights never change, so val MAE never improves after epoch 1.
    small_cfg.train.learning_rate = 0.0
    small_cfg.train.epochs = 10
    small_cfg.train.patience = 2
    out = train(small_cfg, str(tmp_path / "run"), model_factory=MODEL_FACTORY, data=small_data)
    assert out["stopped_early"] is True
    assert out["epochs_run"] == 3  # epoch 1 = best, epochs 2-3 without improvement
    assert out["best_epoch"] == 1


def test_training_reduces_validation_error(tmp_path, small_cfg, small_data):
    small_cfg.train.epochs = 6
    small_cfg.train.learning_rate = 3e-3
    out = train(small_cfg, str(tmp_path / "run"), model_factory=MODEL_FACTORY, data=small_data)
    first, best = out["history"][0]["val_mae"], out["best_val_mae"]
    assert best < first


def test_training_is_reproducible(tmp_path, small_cfg, small_data):
    a = train(small_cfg, str(tmp_path / "a"), model_factory=MODEL_FACTORY, data=small_data)
    b = train(small_cfg, str(tmp_path / "b"), model_factory=MODEL_FACTORY, data=small_data)
    assert a["history"][0]["val_mae"] == pytest.approx(b["history"][0]["val_mae"], abs=1e-6)
    sa = torch.load(a["checkpoint"], map_location="cpu")["model_state"]
    sb = torch.load(b["checkpoint"], map_location="cpu")["model_state"]
    for key in sa:
        assert torch.allclose(sa[key], sb[key], atol=1e-6)


def test_rejects_wrong_feature_count(tmp_path, small_cfg, small_data):
    X_tr, y_tr, X_va, y_va = small_data
    with pytest.raises(ValueError):
        train(small_cfg, str(tmp_path / "run"), model_factory=MODEL_FACTORY,
              data=(X_tr[:, :28], y_tr, X_va[:, :28], y_va))


def test_rejects_nan_in_data(tmp_path, small_cfg, small_data):
    X_tr, y_tr, X_va, y_va = small_data
    X_bad = X_tr.copy()
    X_bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        train(small_cfg, str(tmp_path / "run"), model_factory=MODEL_FACTORY,
              data=(X_bad, y_tr, X_va, y_va))


# ---------------------------------------------------------------------------
# Unit tests for helpers
# ---------------------------------------------------------------------------
def test_quantile_bins_are_balanced():
    y = np.random.default_rng(0).gamma(2.0, 0.4, size=50_000).astype(np.float32)
    edges = make_bin_edges(y, 10, "quantile")
    assert len(edges) == 9 and edges == sorted(edges)
    counts = np.bincount(assign_bins(y, edges), minlength=10)
    assert counts.min() > 0.9 * len(y) / 10 and counts.max() < 1.1 * len(y) / 10


def test_uniform_bins_and_assignment():
    y = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    edges = make_bin_edges(y, 4, "uniform")
    assert edges == [1.0, 2.0, 3.0]
    assert assign_bins(y, edges).tolist() == [0, 1, 2, 3, 3]
    with pytest.raises(ValueError):
        make_bin_edges(y, 4, "other")


def test_macro_f1_matches_hand_computation():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])
    # class 0: F1 = 0.5, class 1: F1 = 0.8, class 2: F1 = 2/3
    assert macro_f1(y_true, y_pred, 3) == pytest.approx((0.5 + 0.8 + 2 / 3) / 3)
    assert macro_f1(y_true, y_true, 3) == pytest.approx(1.0)


def test_resolve_outputs_for_directory_and_pt_file():
    ckpt, run = resolve_outputs("models/run_default")
    assert Path(ckpt) == Path("models/run_default/best.pt")
    assert Path(run) == Path("models/run_default")
    ckpt, run = resolve_outputs("models/neural_heuristic_seed42.pt")
    assert Path(ckpt) == Path("models/neural_heuristic_seed42.pt")
    assert Path(run) == Path("results/training/neural_heuristic_seed42")


def test_default_yaml_in_sync_with_code():
    path = Path("configs/default.yaml")
    if not path.exists():
        pytest.skip("configs/default.yaml not created yet")
    assert load_config(str(path)) == Config()
