"""Tests for the ablation study configuration helpers."""

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from src.ablations import (
    ablation_configs,
    asymmetric_smooth_l1_loss,
    feature_indices,
    output_file_for_kind,
    rows_to_csv,
    select_features,
)


def test_feature_indices_drop_named_groups() -> None:
    assert len(feature_indices()) == 29
    assert len(feature_indices(["wall_window"])) == 4
    assert len(feature_indices(["goal_offsets"])) == 27
    assert len(feature_indices(["coordinates"])) == 27
    assert len(feature_indices(["wall_window", "goal_offsets"])) == 2


def test_select_features_matches_column_order() -> None:
    features = np.arange(5 * 29, dtype=np.float32).reshape(5, 29)

    selected = select_features(features, drop=["coordinates"])

    assert selected.shape == (5, 27)
    np.testing.assert_array_equal(
        selected,
        features[:, [2, 3, *range(4, 29)]],
    )


def test_select_features_validates_shape_and_group_names() -> None:
    with pytest.raises(ValueError):
        select_features(np.zeros((3, 28)))
    with pytest.raises(ValueError):
        select_features(np.zeros((3, 29)), drop=["unknown"])


def test_asymmetric_loss_penalises_overestimation_more() -> None:
    overpredict = torch.tensor([0.0, 2.0])
    underpredict = torch.tensor([0.0, 0.0])
    targets = torch.tensor([0.0, 0.0])

    over_loss = asymmetric_smooth_l1_loss(overpredict, targets)
    under_loss = asymmetric_smooth_l1_loss(underpredict, targets)

    assert over_loss > under_loss
    assert torch.isclose(
        over_loss,
        F.smooth_l1_loss(overpredict, targets, beta=0.1)
        + torch.tensor(2.0),
    )


def test_asymmetric_loss_matches_smooth_l1_without_penalty() -> None:
    predictions = torch.tensor([0.5, -0.5, 2.0])
    targets = torch.tensor([0.0, 0.0, 0.0])

    loss = asymmetric_smooth_l1_loss(
        predictions, targets, margin=0.0, lambda_over=0.0
    )

    assert torch.isclose(
        loss,
        F.smooth_l1_loss(predictions, targets, beta=0.1),
    )


def test_margin_suppresses_small_overestimates() -> None:
    predictions = torch.tensor([0.1])
    targets = torch.tensor([0.0])

    penalised = asymmetric_smooth_l1_loss(
        predictions, targets, margin=0.0, lambda_over=1.0
    )
    ignored = asymmetric_smooth_l1_loss(
        predictions, targets, margin=1.0, lambda_over=1.0
    )

    assert penalised > ignored


def test_ablation_plan_has_expected_groups() -> None:
    configs = ablation_configs()

    assert len(configs) == 7
    assert [c["kind"] for c in configs].count("feature") == 3
    assert [c["kind"] for c in configs].count("loss") == 2
    assert [c["kind"] for c in configs].count("architecture") == 2


def test_output_file_for_kind() -> None:
    assert output_file_for_kind("feature") == "feature_ablation.csv"
    assert output_file_for_kind("loss") == "loss_ablation.csv"
    assert output_file_for_kind("architecture") == "architecture_ablation.csv"
    with pytest.raises(ValueError):
        output_file_for_kind("unknown")


def test_rows_to_csv_writes_header_and_rows(tmp_path) -> None:
    rows = [
        {"name": "full", "val_mae": 0.1},
        {"name": "no_walls", "val_mae": 0.2},
    ]
    path = tmp_path / "ablations.csv"

    rows_to_csv(rows, str(path))

    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "name,val_mae"
    assert lines[1] == "full,0.1"
    assert lines[2] == "no_walls,0.2"
