"""Training entry point for pharmacokinetics GNN models."""

import torch
import torch.nn as nn
from torch.optim import Adam

from src.models.gnn_model import PlaceholderGNN
from src.utils.config import TrainingConfig, get_device


def train_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Single dummy epoch on synthetic data (replace with real loaders)."""
    model.train()
    x = torch.randn(32, model.layers[0].in_features, device=device)
    target = torch.randn(32, model.layers[-1].out_features, device=device)

    optimizer.zero_grad()
    pred = model(x)
    loss = criterion(pred, target)
    loss.backward()
    optimizer.step()
    return float(loss.item())


def main(config: TrainingConfig | None = None) -> None:
    config = config or TrainingConfig()
    device = get_device(config.device)
    print(f"Training on device: {device}")

    model = PlaceholderGNN(
        in_channels=config.in_channels,
        hidden_channels=config.hidden_channels,
        out_channels=config.out_channels,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(1, config.epochs + 1):
        loss = train_epoch(model, optimizer, criterion, device)
        print(f"Epoch {epoch}/{config.epochs} — loss: {loss:.4f}")

    print("Training complete.")


if __name__ == "__main__":
    main()
