"""Configuration dataclasses and helpers."""

from dataclasses import dataclass

import torch


@dataclass
class TrainingConfig:
    """Default hyperparameters for training scripts."""

    in_channels: int = 16
    hidden_channels: int = 32
    out_channels: int = 4
    learning_rate: float = 1e-3
    epochs: int = 3
    device: str = "auto"


def get_device(device: str = "auto") -> torch.device:
    """
    Resolve torch device from a string.

    Args:
        device: 'auto', 'cpu', 'cuda', or 'mps' (Apple Silicon).
    """
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)
