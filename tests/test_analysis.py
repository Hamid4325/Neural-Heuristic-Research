"""Unit tests for error analysis and bias auditing module.

Part of Deliverable 2 for Neural Heuristic Research.
Author: Abdullah Zia
"""

import os
import pytest
from src.analysis import perform_error_analysis, perform_bias_analysis


def test_perform_error_analysis(tmp_path):
    """Test error analysis computes deciles and outputs JSON."""
    test_data = "data/processed/test.npz"
    reg_path = "models/fallback_mlp_regressor.joblib"
    if not (os.path.exists(test_data) and os.path.exists(reg_path)):
        pytest.skip("Test data or model missing")

    out_dir = str(tmp_path / "analysis_test")
    res = perform_error_analysis(
        test_data_path=test_data,
        regressor_path=reg_path,
        output_dir=out_dir,
        sample_size=1000,
    )

    assert os.path.exists(os.path.join(out_dir, "error_analysis.json"))
    assert "decile_breakdown" in res
    assert len(res["decile_breakdown"]) > 0
    assert "wall_density_breakdown" in res


def test_perform_bias_analysis(tmp_path):
    """Test bias auditing computes environmental subgroup statistics."""
    test_data = "data/processed/test.npz"
    reg_path = "models/fallback_mlp_regressor.joblib"
    per_maze_csv = "results/evaluation/test_per_maze.csv"

    if not (os.path.exists(test_data) and os.path.exists(reg_path) and os.path.exists(per_maze_csv)):
        pytest.skip("Required evaluation files missing")

    out_dir = str(tmp_path / "bias_test")
    df = perform_bias_analysis(
        per_maze_csv=per_maze_csv,
        test_data_path=test_data,
        regressor_path=reg_path,
        output_dir=out_dir,
        sample_size=1000,
    )

    assert os.path.exists(os.path.join(out_dir, "bias_table.csv"))
    assert not df.empty
    assert "subgroup" in df.columns
    assert "Generator" in df["subgroup"].values
    assert "Grid Size" in df["subgroup"].values
