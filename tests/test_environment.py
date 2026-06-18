#!/usr/bin/env python3
"""Verify that core scientific / ML dependencies import and report versions."""

import sys


def main() -> int:
    packages = [
        ("torch", "torch"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("scipy", "scipy"),
        ("sklearn", "scikit-learn"),
        ("matplotlib", "matplotlib"),
        ("networkx", "networkx"),
    ]

    print(f"Python: {sys.version}")
    print("-" * 50)

    optional_gnn = [
        ("torch_geometric", "torch-geometric"),
        ("torch_scatter", "torch-scatter"),
        ("torch_sparse", "torch-sparse"),
    ]

    failed = False
    for module_name, display_name in packages:
        try:
            mod = __import__(module_name)
            version = getattr(mod, "__version__", "unknown")
            print(f"  {display_name:20s} {version}")
        except ImportError as e:
            print(f"  {display_name:20s} MISSING ({e})")
            failed = True

    print("-" * 50)
    print("Optional GNN packages:")
    for module_name, display_name in optional_gnn:
        try:
            mod = __import__(module_name)
            version = getattr(mod, "__version__", "unknown")
            print(f"  {display_name:20s} {version}")
        except ImportError:
            print(f"  {display_name:20s} not installed (optional)")

    if "torch" in sys.modules:
        import torch

        print("-" * 50)
        print(f"  PyTorch CUDA available: {torch.cuda.is_available()}")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None:
            print(f"  PyTorch MPS available:  {mps.is_available()}")

    print("-" * 50)
    if failed:
        print("Environment check FAILED — install missing packages.")
        return 1
    print("Environment check PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
