"""Plot parent-drug compartment amounts (mg); dashed line is APAP + caffeine sum."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from src.metrics import pk_metrics_for_drug
from src.simulate import plasma_volume_L, run_simulation


def plot_drug_amounts(
    t: torch.Tensor,
    apap: dict[str, torch.Tensor],
    caff: dict[str, torch.Tensor],
    output: Path,
    metric: str = "track",
    show: bool = False,
) -> None:
    """Per-drug amount curves; combined trace sums masses (not concentrations)."""
    hours = t.detach().numpy()
    apap_y = apap[metric].detach().numpy()
    caff_y = caff[metric].detach().numpy()
    combined = apap_y + caff_y

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(hours, apap_y, color="tab:blue", linewidth=2, label="APAP")
    ax.plot(hours, caff_y, color="tab:orange", linewidth=2, label="Caffeine")
    ax.plot(hours, combined, color="tab:green", linewidth=2, linestyle="--", label="APAP + Caffeine")

    ax.set_xlabel("Time (h)")
    ax.set_ylabel(f"Amount ({metric}, mg)")
    ax.set_title("APAP and caffeine amounts over time (GNN-ODE)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved {output}")
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot APAP and caffeine amounts vs time.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pk_drug_amounts.png"),
    )
    parser.add_argument("--hours", type=float, default=24.0)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--metric", choices=["gut", "plasma", "liver", "sys", "track"], default="track")
    parser.add_argument("--use-gnn-factors", action="store_true")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    t, traj, data = run_simulation(
        hours=args.hours,
        steps=args.steps,
        use_gnn_factors=args.use_gnn_factors,
    )
    v_plasma = plasma_volume_L(data)

    apap = pk_metrics_for_drug(traj, t, "apap", v_plasma)
    caff = pk_metrics_for_drug(traj, t, "caffeine", v_plasma)

    plot_drug_amounts(t, apap, caff, args.output, metric=args.metric, show=args.show)

    print(f"APAP  Cmax plasma={apap['plasma'].max():.1f} mg")
    print(f"Caff  Cmax plasma={caff['plasma'].max():.1f} mg")


if __name__ == "__main__":
    main()
