"""Tests for src/tune.py. They need neither Hamid's model nor the real dataset:
the pipeline tests run on small synthetic data with the built-in StandInNet."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import src.tune as tune
from src.config import Config, load_config
from src.tune import (
    GRID_KEYS,
    StandInNet,
    apply_hyperparams,
    full_grid,
    hyperparams_of,
    load_tuning_spec,
    make_run_id,
    rank_records,
    run_tuning,
    sample_configs,
    validate_spec,
)

GRID_FILE = Path("configs/tuning_grid.yaml")


@pytest.fixture(scope="module")
def spec():
    return load_tuning_spec(str(GRID_FILE))


# ---------------------------------------------------------------------------
# configs/tuning_grid.yaml follows the plan
# ---------------------------------------------------------------------------
def test_grid_matches_the_plan(spec):
    g = spec["grid"]
    assert g["hidden_dims"] == [[128, 128], [256, 256], [256, 256, 128], [512, 256]]
    assert g["activation"] == ["relu", "gelu"]
    assert g["dropout"] == [0.0, 0.1, 0.2]
    assert g["batch_size"] == [256, 1024, 2048]
    assert g["learning_rate"] == [0.0001, 0.0003, 0.001]
    assert g["weight_decay"] == [0.0, 0.00001, 0.0001]
    assert g["reg_loss"] == ["mse", "smooth_l1"]
    assert g["lambda_aux"] == [0.0, 0.1, 0.5]
    assert g["target_transform"] == ["raw", "standardised"]
    assert len(full_grid(g)) == 4 * 2 * 3 * 3 * 3 * 3 * 2 * 3 * 2 == 7776


def test_stage_settings_match_the_plan(spec):
    a, b, c = spec["stage_a"], spec["stage_b"], spec["stage_c"]
    assert (a["n_configs"], a["max_train_cells_per_epoch"], a["max_val_cells"], a["epochs"]) \
        == (12, 400_000, 100_000, 5)
    assert (b["top_k"], b["max_train_cells_per_epoch"], b["epochs"]) == (3, 1_500_000, 10)
    assert c["epochs"] == 20 and c["max_train_cells_per_epoch"] is None  # full train split


def test_default_config_is_inside_the_grid(spec):
    default = hyperparams_of(load_config("configs/default.yaml"))
    assert default in full_grid(spec["grid"])


def test_validate_spec_rejects_mistakes(spec):
    bad = json.loads(json.dumps(spec))
    bad["grid"]["learning_rate"] = ["1e-4", 0.001]  # what PyYAML gives for "1e-4"
    with pytest.raises(ValueError):
        validate_spec(bad)
    bad = json.loads(json.dumps(spec))
    bad["grid"]["momentum"] = [0.9]
    with pytest.raises(ValueError):
        validate_spec(bad)
    bad = json.loads(json.dumps(spec))
    del bad["stage_b"]
    with pytest.raises(ValueError):
        validate_spec(bad)


# ---------------------------------------------------------------------------
# Sampling, config building, run ids, ranking
# ---------------------------------------------------------------------------
def test_sampling_is_deterministic_unique_and_starts_with_default(spec):
    default = hyperparams_of(load_config("configs/default.yaml"))
    a = sample_configs(spec["grid"], 12, 42, default)
    assert a == sample_configs(spec["grid"], 12, 42, default)
    assert a != sample_configs(spec["grid"], 12, 43, default)
    assert len(a) == 12 and a[0] == default
    assert len({json.dumps(c, sort_keys=True) for c in a}) == 12
    grid = full_grid(spec["grid"])
    assert all(c in grid for c in a)


def test_sampling_more_than_the_grid_returns_everything():
    grid = {k: [v] for k, v in hyperparams_of(Config()).items()}
    assert len(sample_configs(grid, 5, 0)) == 1


def test_apply_hyperparams_sets_fields_without_touching_base(spec):
    base = Config()
    hp = {"hidden_dims": [512, 256], "activation": "gelu", "dropout": 0.2, "batch_size": 2048,
          "learning_rate": 0.001, "weight_decay": 0.0001, "reg_loss": "mse",
          "lambda_aux": 0.5, "target_transform": "raw"}
    stage = {"epochs": 3, "max_train_cells_per_epoch": 1000, "max_val_cells": 500, "patience": 2}
    cfg = apply_hyperparams(base, hp, stage, seed=7)
    assert cfg.model.hidden_dims == (512, 256) and cfg.model.activation == "gelu"
    assert cfg.train.batch_size == 2048 and cfg.train.reg_loss == "mse"
    assert cfg.train.epochs == 3 and cfg.train.max_val_cells == 500 and cfg.train.seed == 7
    assert base == Config()  # untouched


def test_run_id_is_stable_and_depends_on_stage_settings():
    hp = hyperparams_of(Config())
    s1 = {"epochs": 5, "max_train_cells_per_epoch": 400, "max_val_cells": 100, "patience": 5}
    s2 = dict(s1, epochs=10)
    assert make_run_id("A01", hp, s1) == make_run_id("A01", hp, s1)
    assert make_run_id("A01", hp, s1) != make_run_id("A01", hp, s2)
    assert make_run_id("A01", hp, s1).startswith("A01_")


def test_rank_records_orders_by_mae_and_drops_failures():
    recs = [
        {"run_id": "x", "best_val_mae": 0.3, "_ok": True},
        {"run_id": "y", "best_val_mae": 0.1, "_ok": True},
        {"run_id": "z", "best_val_mae": "", "_ok": False},
    ]
    assert [r["run_id"] for r in rank_records(recs)] == ["y", "x"]


def test_standin_net_follows_the_model_config():
    cfg = Config().model
    net = StandInNet(cfg)
    y, logits = net(torch.zeros(4, 29))
    assert y.shape == (4, 1) and logits.shape == (4, 10)
    assert sum(p.numel() for p in net.parameters()) == 107_787
    with pytest.raises(ValueError):
        StandInNet(type(cfg)(activation="swish"))


# ---------------------------------------------------------------------------
# End-to-end pipeline on tiny synthetic data
# ---------------------------------------------------------------------------
def _synthetic(n, seed):
    rng = np.random.default_rng(seed)
    X = rng.random((n, 29), dtype=np.float32)
    y = np.abs(2.0 * X[:, 0] + X[:, 1] + 0.05 * rng.standard_normal(n)).astype(np.float32)
    return X, y


TINY_SPEC = {
    "seed": 42,
    "include_default": True,
    "grid": {
        "hidden_dims": [[16, 16], [32, 16]], "activation": ["relu", "gelu"],
        "dropout": [0.0, 0.1], "batch_size": [128, 256], "learning_rate": [0.001, 0.003],
        "weight_decay": [0.0, 0.0001], "reg_loss": ["mse", "smooth_l1"],
        "lambda_aux": [0.0, 0.1], "target_transform": ["raw", "standardised"],
    },
}
TINY_STAGES = {
    "stage_a": {"n_configs": 3, "epochs": 1, "max_train_cells_per_epoch": 3000,
                "max_val_cells": 500, "patience": 5},
    "stage_b": {"top_k": 2, "epochs": 1, "max_train_cells_per_epoch": 4000,
                "max_val_cells": None, "patience": 5},
    "stage_c": {"epochs": 2, "max_train_cells_per_epoch": None,
                "max_val_cells": None, "patience": 5},
}


@pytest.fixture(scope="module")
def tiny_data():
    X_tr, y_tr = _synthetic(5000, 0)
    X_va, y_va = _synthetic(1000, 1)
    return X_tr, y_tr, X_va, y_va


def _run(tmp_path, tiny_data, **kwargs):
    return run_tuning(
        dict(TINY_SPEC), Config(), model_factory=kwargs.pop("model_factory", StandInNet),
        model_source="standin", data=tiny_data, results_dir=str(tmp_path / "tuning"),
        final_checkpoint=str(tmp_path / "tuned_best.pt"), stages=TINY_STAGES, **kwargs,
    )


def test_pipeline_writes_all_outputs(tmp_path, tiny_data):
    out = _run(tmp_path, tiny_data)
    root = tmp_path / "tuning"
    runs = out["records"]
    assert [r["stage"] for r in runs] == ["A", "A", "A", "B", "B", "C"]

    # Plan: results/tuning/<run_id>/{config.json, history.csv, metrics.json}
    for r in runs:
        for name in ("config.json", "history.csv", "metrics.json"):
            assert (root / r["run_id"] / name).exists(), (r["run_id"], name)
        cfg_json = json.loads((root / r["run_id"] / "config.json").read_text())
        assert set(cfg_json["hyperparameters"]) == set(GRID_KEYS)

    # Plan: results/tuning/tuning_summary.csv, one row per run
    with open(root / "tuning_summary.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 6
    assert list(rows[0].keys()) == tune.SUMMARY_COLUMNS
    assert all(float(r["best_val_mae"]) > 0 for r in rows)
    assert {r["model_source"] for r in rows} == {"standin"}

    # Stage logic: B reuses the best configs of A; C reuses the best config of B.
    a = [r for r in runs if r["stage"] == "A"]
    b = [r for r in runs if r["stage"] == "B"]
    top_a = [r["_hp"] for r in rank_records(a)[:2]]
    assert [r["_hp"] for r in b] == top_a
    assert runs[-1]["_hp"] == rank_records(b)[0]["_hp"]
    # The default config (256-256-128) is not in the tiny grid, so it must not be forced in.
    assert hyperparams_of(Config()) not in [r["_hp"] for r in a]

    # Winner files
    best = json.loads((root / "best_config.json").read_text())
    assert best["hyperparameters"] == runs[-1]["_hp"]
    assert load_config(str(root / "best_config.yaml")).train.epochs == 2
    ckpt = torch.load(tmp_path / "tuned_best.pt", map_location="cpu")
    assert {"model_state", "bin_edges", "target_mean", "target_std"} <= set(ckpt)


def test_resume_skips_finished_runs(tmp_path, tiny_data):
    first = _run(tmp_path, tiny_data)
    built = []

    def counting_factory(cfg):
        built.append(1)
        return StandInNet(cfg)

    second = _run(tmp_path, tiny_data, model_factory=counting_factory, resume=True)
    assert len(built) == 1  # only the fail-fast probe; no training run built a model
    assert all("reused" in r["status"] for r in second["records"])
    assert [r["best_val_mae"] for r in second["records"]] == \
        [r["best_val_mae"] for r in first["records"]]


def test_diverged_run_is_recorded_and_pipeline_continues(tmp_path, tiny_data, monkeypatch):
    real_train, calls = tune.train, []

    def flaky_train(cfg, output, **kw):
        calls.append(1)
        if len(calls) == 1:
            raise FloatingPointError("Non-finite loss at epoch 1")
        return real_train(cfg, output, **kw)

    monkeypatch.setattr(tune, "train", flaky_train)
    out = _run(tmp_path, tiny_data)
    statuses = [r["status"] for r in out["records"]]
    assert statuses[0] == "failed" and statuses.count("ok") == 5
    metrics = json.loads((tmp_path / "tuning" / out["records"][0]["run_id"]
                          / "metrics.json").read_text())
    assert metrics["failed"] is True
    assert all(r["_hp"] != out["records"][0]["_hp"] for r in out["records"] if r["stage"] == "B")


def test_all_stage_a_failures_stop_the_search(tmp_path, tiny_data, monkeypatch):
    def always_diverges(cfg, output, **kw):
        raise FloatingPointError("Non-finite loss at epoch 1")

    monkeypatch.setattr(tune, "train", always_diverges)
    with pytest.raises(RuntimeError):
        _run(tmp_path, tiny_data)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def test_unknown_model_choice_is_rejected():
    with pytest.raises(ValueError):
        tune.get_model_factory("other")


def test_cli_fails_fast_without_hamids_model(tmp_path):
    try:
        import src.model  # noqa: F401
        pytest.skip("src/model.py exists, so the 'hamid' model is available")
    except ModuleNotFoundError:
        pass
    with pytest.raises(ModuleNotFoundError, match="src/model.py"):
        tune.main(["--config", str(GRID_FILE), "--results-dir", str(tmp_path)])
