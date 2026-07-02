"""DDI visualization: monotherapy vs. co-administration plasma PK.

Demonstrates the competitive-enzyme-inhibition drug-drug interaction (DDI)
mechanistically: each drug's plasma concentration when dosed ALONE is compared
against the same drug when CO-ADMINISTERED. Output arrays are never summed
across drugs; the interaction emerges from the shared enzyme denominators in
the ODE (``src/models/gnn_ode.py``).

Run: ``python -m src.plot_ddi``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from src.metrics import pk_metrics_for_drug
from src.models.gnn_ode import STATE_IDX
from src.simulate import patient_weight_kg, plasma_volume_L, run_simulation


def napqi_amount_mg(traj: torch.Tensor) -> torch.Tensor:
    """Liver NAPQI pool (mg) from ODE trajectory ``[T, S]``."""
    return traj[:, STATE_IDX["A_napqi"]].clamp(min=0.0)


def _to_numpy(curve: torch.Tensor) -> "list[float]":
    """Detach a 1-D tensor to a NumPy array for plotting."""
    return curve.detach().numpy()


def plot_ddi_comparison(
    t: torch.Tensor,
    apap_alone: torch.Tensor,
    apap_co: torch.Tensor,
    caff_alone: torch.Tensor,
    caff_co: torch.Tensor,
    napqi_alone: torch.Tensor,
    napqi_co: torch.Tensor,
    output: Path,
    show: bool = False,
) -> None:
    """Stacked APAP / caffeine plasma C_p (ng/mL) and NAPQI (mg) over time."""
    hours = t.detach().numpy()

    fig, (ax_apap, ax_caff, ax_napqi) = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

    ax_apap.plot(
        hours, _to_numpy(apap_alone),
        label="Acetaminophen alone", color="tab:blue", linewidth=2,
    )
    ax_apap.plot(
        hours, _to_numpy(apap_co),
        label="APAP + caffeine", color="tab:red", linestyle="--", linewidth=2,
    )
    ax_apap.set_title("Acetaminophen PK: Impact of Caffeine Co-administration")
    ax_apap.set_ylabel("APAP Concentration (ng/mL)")
    ax_apap.grid(True, alpha=0.3)
    ax_apap.legend(loc="upper right")

    ax_caff.plot(
        hours, _to_numpy(caff_alone),
        label="Caffeine alone", color="tab:orange", linewidth=2,
    )
    ax_caff.plot(
        hours, _to_numpy(caff_co),
        label="Caffeine + APAP", color="tab:red", linestyle="--", linewidth=2,
    )
    ax_caff.set_title("Caffeine PK: Impact of Acetaminophen Co-administration")
    ax_caff.set_ylabel("Caffeine Concentration (ng/mL)")
    ax_caff.grid(True, alpha=0.3)
    ax_caff.legend(loc="upper right")

    ax_napqi.plot(
        hours, _to_numpy(napqi_alone),
        label="APAP alone", color="tab:blue", linewidth=2,
    )
    ax_napqi.plot(
        hours, _to_numpy(napqi_co),
        label="APAP + caffeine", color="tab:red", linestyle="--", linewidth=2,
    )
    ax_napqi.set_title("NAPQI: Impact of Caffeine Co-administration")
    ax_napqi.set_ylabel("NAPQI amount (mg)")
    ax_napqi.set_xlabel("Time (h)")
    ax_napqi.grid(True, alpha=0.3)
    ax_napqi.legend(loc="upper right")

    fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved {output}")
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DDI plot: APAP/caffeine plasma PK, monotherapy vs. co-administration."
    )
    parser.add_argument("--output", type=Path, default=Path("results/ddi_comparison.png"))
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--use-gnn-factors", action="store_true")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    hours = 24.0

    # Step 1: three independent 24 h ODE runs via dose_overrides (full drug names).
    t, traj_apap_alone, data = run_simulation(
        hours=hours,
        steps=args.steps,
        use_gnn_factors=args.use_gnn_factors,
        dose_overrides={"acetaminophen": 3000.0, "caffeine": 0.0},
    )
    _, traj_caff_alone, _ = run_simulation(
        hours=hours,
        steps=args.steps,
        use_gnn_factors=args.use_gnn_factors,
        dose_overrides={"acetaminophen": 0.0, "caffeine": 200.0},
    )
    _, traj_coadmin, _ = run_simulation(
        hours=hours,
        steps=args.steps,
        use_gnn_factors=args.use_gnn_factors,
        dose_overrides={"acetaminophen": 3000.0, "caffeine": 200.0},
    )

    v_plasma = plasma_volume_L(data)
    weight = patient_weight_kg(data)

    # Step 2: per-drug plasma C_p extraction (no cross-drug array summing).
    m_apap_alone = pk_metrics_for_drug(traj_apap_alone, t, "apap", v_plasma, weight_kg=weight)
    m_caff_alone = pk_metrics_for_drug(traj_caff_alone, t, "caffeine", v_plasma, weight_kg=weight)
    m_apap_co = pk_metrics_for_drug(traj_coadmin, t, "apap", v_plasma, weight_kg=weight)
    m_caff_co = pk_metrics_for_drug(traj_coadmin, t, "caffeine", v_plasma, weight_kg=weight)

    napqi_alone = napqi_amount_mg(traj_apap_alone)
    napqi_co = napqi_amount_mg(traj_coadmin)

    # Step 3: build and save the stacked comparison figure.
    plot_ddi_comparison(
        t,
        apap_alone=m_apap_alone["C_p_ng_mL"],
        apap_co=m_apap_co["C_p_ng_mL"],
        caff_alone=m_caff_alone["C_p_ng_mL"],
        caff_co=m_caff_co["C_p_ng_mL"],
        napqi_alone=napqi_alone,
        napqi_co=napqi_co,
        output=args.output,
        show=args.show,
    )

    print(
        f"\nPeak NAPQI: APAP alone = {float(napqi_alone.max()):.4f} mg | "
        f"APAP + caffeine = {float(napqi_co.max()):.4f} mg"
    )

    # Step 4: quantify the DDI half-life shift via terminal NCA t1/2.
    print("\nTerminal t1/2 (post-Tmax log-linear NCA fit):")
    rows = [
        ("Acetaminophen alone", m_apap_alone),
        ("APAP + caffeine", m_apap_co),
        ("Caffeine alone", m_caff_alone),
        ("Caffeine + APAP", m_caff_co),
    ]
    for label, m in rows:
        t_half = m["t_half_terminal_h"].item()
        print(f"  {label:<18} t1/2 = {t_half:6.2f} h")


if __name__ == "__main__":
    main()
