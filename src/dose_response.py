"""APAP dose-response sweep using the trained GNN weights and learned f_GNN.

Only the APAP dose varies across runs; caffeine and all kinetics are held fixed,
so differences isolate the dose effect. Reports parent PK plus the NAPQI/GSH
toxicity sinks and a mass-balance residual for each dose.

Run: ``python -m src.dose_response``
"""

from __future__ import annotations

import argparse

import torch

from src.metrics import pk_metrics_for_drug
from src.models.gnn_ode import STATE_IDX, STATE_NAMES
from src.simulate import patient_weight_kg, plasma_volume_L, run_simulation

GSH_IDX = STATE_IDX["A_gsh"]
# Drug-derived mass conserves the dose; GSH regenerates and is excluded.
DRUG_DERIVED_IDX = [i for i in range(len(STATE_NAMES)) if i != GSH_IDX]


def mass_balance_residual(traj: torch.Tensor, dose_total_mg: float) -> float:
    """|sum(drug-derived final states) - administered dose| (mg)."""
    final = traj[-1]
    drug_mass = float(final[DRUG_DERIVED_IDX].sum())
    return abs(drug_mass - dose_total_mg)


def evaluate_dose(
    apap_mg: float,
    caffeine_mg: float,
    hours: float,
    steps: int,
    use_gnn_factors: bool,
) -> dict[str, float]:
    """Integrate one dose scenario and extract scalar PK / toxicity metrics."""
    t, traj, data = run_simulation(
        hours=hours,
        steps=steps,
        use_gnn_factors=use_gnn_factors,
        dose_overrides={"acetaminophen": apap_mg, "caffeine": caffeine_mg},
    )
    v_plasma = plasma_volume_L(data)
    weight = patient_weight_kg(data)

    apap = pk_metrics_for_drug(traj, t, "apap", v_plasma, weight_kg=weight)
    caff = pk_metrics_for_drug(traj, t, "caffeine", v_plasma, weight_kg=weight)
    final = traj[-1]

    return {
        "apap_dose_mg": apap_mg,
        "apap_cmax_ng_mL": float(apap["cmax_ng_mL"]),
        "apap_tmax_h": float(apap["tmax_h"]),
        "apap_t_half_h": float(apap["t_half_terminal_h"]),
        "apap_vd_L_kg": float(apap["vd_terminal_L_kg"]),
        "caff_cmax_ng_mL": float(caff["cmax_ng_mL"]),
        "caff_t_half_h": float(caff["t_half_terminal_h"]),
        "final_gsh_mg": float(final[STATE_IDX["A_gsh"]]),
        "final_napqi_mg": float(final[STATE_IDX["A_napqi"]]),
        "final_adduct_sink_mg": float(final[STATE_IDX["A_napqi_adduct_sink"]]),
        "final_urine_sink_mg": float(final[STATE_IDX["A_urine_sink"]]),
        "mass_balance_residual_mg": mass_balance_residual(traj, apap_mg + caffeine_mg),
    }


def _fmt(value: float) -> str:
    return "   nan" if value != value else f"{value:7.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="APAP dose-response sweep.")
    parser.add_argument("--doses", type=float, nargs="+", default=[1000.0, 3000.0, 5000.0])
    parser.add_argument("--caffeine", type=float, default=200.0)
    parser.add_argument("--hours", type=float, default=24.0)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--use-gnn-factors", action="store_true")
    args = parser.parse_args()

    factor_mode = "trained f_GNN" if args.use_gnn_factors else "neutral f_GNN=1"
    print(f"APAP dose-response sweep ({factor_mode}, caffeine {args.caffeine:.0f} mg fixed)\n")

    rows = [
        evaluate_dose(d, args.caffeine, args.hours, args.steps, args.use_gnn_factors)
        for d in args.doses
    ]

    header = (
        f"{'APAP mg':>8} | {'Cmax':>7} {'tmax':>7} {'t1/2':>7} {'Vd':>7} | "
        f"{'cafCmax':>7} {'caf t1/2':>8} | {'GSH':>8} {'NAPQI':>7} {'adduct':>7} "
        f"{'urine':>8} | {'massbal':>8}"
    )
    units = (
        f"{'':>8} | {'ng/mL':>7} {'h':>7} {'h':>7} {'L/kg':>7} | "
        f"{'ng/mL':>7} {'h':>8} | {'mg':>8} {'mg':>7} {'mg':>7} {'mg':>8} | {'mg':>8}"
    )
    print(header)
    print(units)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['apap_dose_mg']:8.0f} | {r['apap_cmax_ng_mL']:7.0f} {_fmt(r['apap_tmax_h'])} "
            f"{_fmt(r['apap_t_half_h'])} {_fmt(r['apap_vd_L_kg'])} | "
            f"{r['caff_cmax_ng_mL']:7.0f} {_fmt(r['caff_t_half_h']):>8} | "
            f"{r['final_gsh_mg']:8.1f} {r['final_napqi_mg']:7.2f} {r['final_adduct_sink_mg']:7.2f} "
            f"{r['final_urine_sink_mg']:8.1f} | {r['mass_balance_residual_mg']:8.1e}"
        )

    # Sanity checks on the expected dose-dependent toxicity trend.
    adduct = [r["final_adduct_sink_mg"] for r in rows]
    gsh = [r["final_gsh_mg"] for r in rows]
    adduct_rises = all(b >= a for a, b in zip(adduct, adduct[1:]))
    gsh_falls = all(b <= a for a, b in zip(gsh, gsh[1:]))
    print(
        f"\nadduct sink rises with dose: {adduct_rises} | "
        f"final GSH falls with dose: {gsh_falls}"
    )


if __name__ == "__main__":
    main()
