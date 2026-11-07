"""Tests for src/data.py (split manifest, cell loader, maze loader).

Run from the repository root:  pytest tests/test_data.py -q
Tests that need generated data are skipped (with a reason) if data/ has not been built.
"""

from pathlib import Path

import numpy as np
import pytest

from src.data import (
    N_FEATURES,
    SPLIT_NAMES,
    build_split_manifest,
    load_cell_split,
    load_maze_split,
    manifest_cell_counts,
)
from src.utils import load_json, save_json

CLEANED = Path("data/cleaned")
PROCESSED = Path("data/processed")
COMMITTED_MANIFEST = Path("results/manifests/split_manifest.json")

needs_cleaned = pytest.mark.skipif(
    not CLEANED.exists() or not any(CLEANED.glob("*.npz")),
    reason="data/cleaned is empty: run data_scripts/clean_data.py",
)
needs_processed = pytest.mark.skipif(
    not (PROCESSED / "train.npz").exists(),
    reason="data/processed is empty: run data_scripts/features.py",
)


@pytest.fixture(scope="module")
def manifest():
    return build_split_manifest()


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
@needs_cleaned
def test_split_counts(manifest):
    assert manifest["total_mazes"] == 1876
    assert manifest["splits"]["train"]["maze_count"] == 1313
    assert manifest["splits"]["val"]["maze_count"] == 281
    assert manifest["splits"]["test"]["maze_count"] == 282


@needs_cleaned
def test_no_overlap(manifest):
    sets = {
        k: {(e["file"], e["local_index"]) for e in v["entries"]}
        for k, v in manifest["splits"].items()
    }
    assert not (sets["train"] & sets["val"])
    assert not (sets["train"] & sets["test"])
    assert not (sets["val"] & sets["test"])
    assert sum(len(s) for s in sets.values()) == manifest["total_mazes"]


@needs_cleaned
def test_manifest_schema(manifest):
    assert set(manifest) == {"seed", "total_mazes", "splits"}
    assert manifest["seed"] == 42
    assert tuple(manifest["splits"]) == SPLIT_NAMES
    for block in manifest["splits"].values():
        assert block["maze_count"] == len(block["entries"])
        assert set(block["entries"][0]) == {"file", "local_index"}


@needs_cleaned
def test_manifest_is_deterministic(manifest):
    assert build_split_manifest() == manifest
    assert build_split_manifest(seed=7) != manifest


@needs_cleaned
def test_committed_manifest_is_up_to_date(manifest):
    if not COMMITTED_MANIFEST.exists():
        pytest.skip("results/manifests/split_manifest.json not created yet")
    assert load_json(COMMITTED_MANIFEST) == manifest


# ---------------------------------------------------------------------------
# Cell loader
# ---------------------------------------------------------------------------
@needs_processed
def test_cell_sample_counts():
    for split, expected in [("train", 4407160), ("val", 896328), ("test", 911949)]:
        X, y = load_cell_split(split)
        assert len(X) == expected and len(y) == expected
        assert X.shape[1] == N_FEATURES == 29
        assert X.dtype == np.float32 and y.dtype == np.float32


@needs_processed
def test_target_range_is_about_3_18():
    _, y = load_cell_split("train")
    assert y.min() >= 0.0
    assert 3.0 < float(y.max()) < 3.3  # corrects the "[0, 1]" claim in the D1 report


def test_cell_loader_rejects_bad_split():
    with pytest.raises(ValueError):
        load_cell_split("holdout")


# ---------------------------------------------------------------------------
# Manifest <-> processed data consistency, and maze loader
# ---------------------------------------------------------------------------
@needs_cleaned
@needs_processed
def test_manifest_matches_processed_cell_counts(manifest):
    expected = manifest_cell_counts(manifest)
    for split in SPLIT_NAMES:
        _, y = load_cell_split(split)
        assert expected[split] == len(y)


@needs_cleaned
def test_load_maze_split_contract(manifest):
    data = load_maze_split("test")
    assert set(data) == {"grids", "starts", "goals", "h_star", "opt_cost", "file_names"}
    m = manifest["splits"]["test"]["maze_count"]
    assert m == 282
    assert len(data["grids"]) == len(data["h_star"]) == len(data["opt_cost"]) == m
    assert len(data["file_names"]) == m
    assert data["starts"].shape == (m, 2) and data["goals"].shape == (m, 2)
    for grid, h, start, goal in zip(data["grids"], data["h_star"], data["starts"], data["goals"]):
        assert grid.ndim == 2 and grid.shape == h.shape
        assert grid[tuple(start)] == 0 and grid[tuple(goal)] == 0  # start/goal are free
        assert h[tuple(goal)] == 0.0                              # distance goal -> goal
        assert h[tuple(start)] >= 0.0                             # start can reach the goal


@needs_cleaned
def test_maze_loader_uses_manifest_file(manifest, tmp_path):
    path = tmp_path / "manifest.json"
    save_json(path, manifest)
    from_file = load_maze_split("val", manifest_path=str(path))
    assert len(from_file["grids"]) == 281
    # Same entries as the rebuilt default (missing-manifest fallback).
    fallback = load_maze_split("val", manifest_path=str(tmp_path / "missing.json"))
    assert list(from_file["file_names"]) == list(fallback["file_names"])
    assert np.array_equal(from_file["starts"], fallback["starts"])
