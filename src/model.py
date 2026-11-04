"""Feed-forward neural heuristic with regression and distance-bin heads."""

from dataclasses import dataclass

from torch import Tensor, nn

_ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
}


@dataclass(frozen=True)
class ModelConfig:
    """Architecture hyperparameters for :class:`NeuralHeuristicNet`."""

    input_dim: int = 29
    hidden_dims: tuple[int, ...] = (256, 256, 128)
    dropout: float = 0.1
    activation: str = "relu"
    n_bins: int = 10


class NeuralHeuristicNet(nn.Module):
    """Shared MLP trunk with a regression head and a distance-bin classifier.

    The regression head predicts the standardised normalised ground-truth
    heuristic ``h*(n) / N``. The auxiliary bin head predicts which distance
    regime the cell belongs to, which supplies the classification metrics
    required by the empirical-evaluation rubric.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        if config.activation not in _ACTIVATIONS:
            choices = ", ".join(sorted(_ACTIVATIONS))
            raise ValueError(
                f"Unknown activation {config.activation!r}; choose one of: {choices}"
            )
        if not config.hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer")

        self.config = config

        layers: list[nn.Module] = []
        in_dim = config.input_dim
        activation_cls = _ACTIVATIONS[config.activation]
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(activation_cls())
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            in_dim = hidden_dim

        self.trunk = nn.Sequential(*layers)
        self.regression_head = nn.Linear(in_dim, 1)
        self.bin_head = nn.Linear(in_dim, config.n_bins)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(y_pred, bin_logits)`` for a batch of feature vectors."""
        features = self.trunk(x)
        return self.regression_head(features), self.bin_head(features)
