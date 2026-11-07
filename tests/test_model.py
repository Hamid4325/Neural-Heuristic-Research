"""Tests for the neural heuristic model."""

import pytest
import torch

from src.model import ModelConfig, NeuralHeuristicNet


def test_forward_output_shapes() -> None:
    net = NeuralHeuristicNet(ModelConfig())
    net.eval()

    y_pred, bin_logits = net(torch.zeros(8, 29))

    assert y_pred.shape == (8, 1)
    assert bin_logits.shape == (8, 10)


def test_forward_outputs_are_finite() -> None:
    net = NeuralHeuristicNet(ModelConfig())
    net.eval()

    y_pred, bin_logits = net(torch.randn(16, 29))

    assert torch.isfinite(y_pred).all()
    assert torch.isfinite(bin_logits).all()


def test_parameter_count_is_reasonable() -> None:
    net = NeuralHeuristicNet(ModelConfig())

    n_params = sum(p.numel() for p in net.parameters())

    assert 50_000 < n_params < 500_000


def test_custom_hidden_dims_and_activation() -> None:
    cfg = ModelConfig(hidden_dims=(64, 32), activation="gelu", n_bins=5)
    net = NeuralHeuristicNet(cfg)
    net.eval()

    y_pred, bin_logits = net(torch.randn(4, 29))

    assert y_pred.shape == (4, 1)
    assert bin_logits.shape == (4, 5)


def test_invalid_activation_raises() -> None:
    with pytest.raises(ValueError):
        NeuralHeuristicNet(ModelConfig(activation="not-an-activation"))


def test_wrong_input_features_raises() -> None:
    net = NeuralHeuristicNet(ModelConfig())
    net.eval()

    with pytest.raises(RuntimeError):
        net(torch.zeros(2, 10))
