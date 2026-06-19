"""Plot predicted plasma C_p (ng/mL) for APAP and caffeine; report terminal NCA metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from src.metrics import pk_metrics_for_drug
from src.simulate import patient_weight_kg, plasma_volume_L, run_simulation


def plot_pk_metrics(
    t: torch.Tensor,
    apap: dict[str, torch.Tensor],
    caff: dict[str, torch.Tensor],
    output: Path,
    show: bool = False,
) -> None:
    """APAP and caffeine plasma C_p (ng/mL) from integrated ODE trajectory."""
    hours = t.detach().numpy()

    fig, ax = plt.subplots(1, 1, figsize=(9, 5))

    ax.plot(hours, apap["C_p_ng_mL"].detach().numpy(), label="APAP", color="tab:blue", linewidth=2)
    ax.plot(hours, caff["C_p_ng_mL"].detach().numpy(), label="Caffeine", color="tab:orange", linewidth=2)
    ax.set_ylabel("Plasma concentration (ng/mL)")
    ax.set_xlabel("Time (h)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    fig.suptitle("Plasma concentration (GNN-ODE)")
    fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved {output}")
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Vd and t½ vs time for APAP and caffeine.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pk_metrics.png"),
    )
    parser.add_argument("--hours", type=float, default=24.0)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--use-gnn-factors", action="store_true")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    t, traj, data = run_simulation(
        hours=args.hours,
        steps=args.steps,
        use_gnn_factors=args.use_gnn_factors,
    )
    v_plasma = plasma_volume_L(data)
    weight = patient_weight_kg(data)

    apap = pk_metrics_for_drug(traj, t, "apap", v_plasma, weight_kg=weight)
    caff = pk_metrics_for_drug(traj, t, "caffeine", v_plasma, weight_kg=weight)

    plot_pk_metrics(t, apap, caff, args.output, show=args.show)

    print(f"\nTerminal-phase PK parameters (last 30% of time window, patient {weight:.1f} kg):")
    for name, m in [("APAP", apap), ("Caffeine", caff)]:
        t_half = m["t_half_terminal_h"].item()
        vd_l = m["vd_terminal_L"].item()
        vd_l_kg = m["vd_terminal_L_kg"].item()
        print(f"  {name:<9} t½ = {t_half:6.2f} h | Vd_sys = {vd_l_kg:5.2f} L/kg ({vd_l:6.2f} L)")


if __name__ == "__main__":
    main()
