"""Inspect ``results/best_model.pt`` parameter names and shapes."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

DEFAULT_CHECKPOINT = Path("results/best_model.pt")


def load_state_dict(checkpoint: Path) -> dict[str, torch.Tensor]:
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}\n"
            "Run `python -m src.train` first to create it."
        )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError(f"Expected state_dict dict, got {type(state).__name__}")
    return state


def print_summary(state: dict[str, torch.Tensor]) -> None:
    total_params = sum(t.numel() for t in state.values())
    print(f"Parameters: {len(state)} tensors | {total_params:,} values total\n")
    print(f"{'Name':<60} {'Shape':<20} dtype")
    print("-" * 90)
    for name, tensor in sorted(state.items()):
        print(f"{name:<60} {str(tuple(tensor.shape)):<20} {tensor.dtype}")


def print_tensor(name: str, tensor: torch.Tensor) -> None:
    print(f"\n{name}")
    print(f"  shape: {tuple(tensor.shape)}")
    print(f"  dtype: {tensor.dtype}")
    print(f"  min / max / mean: {tensor.min().item():.6g} / {tensor.max().item():.6g} / {tensor.mean().item():.6g}")
    print("  values:")
    print(tensor)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect trained checkpoint weights.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"Path to .pt file (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--tensor",
        type=str,
        default=None,
        help="Show one parameter by exact name (e.g. param_head.weight)",
    )
    parser.add_argument(
        "--values",
        action="store_true",
        help="Print full tensor values for every parameter",
    )
    args = parser.parse_args()

    state = load_state_dict(args.checkpoint)
    print(f"Checkpoint: {args.checkpoint.resolve()}\n")
    print_summary(state)

    if args.tensor is not None:
        if args.tensor not in state:
            matches = [k for k in state if args.tensor in k]
            hint = f"\nPartial matches: {matches}" if matches else ""
            raise KeyError(f"Unknown parameter: {args.tensor!r}{hint}")
        print_tensor(args.tensor, state[args.tensor])
        return

    if args.values:
        for name, tensor in sorted(state.items()):
            print_tensor(name, tensor)


if __name__ == "__main__":
    main()
