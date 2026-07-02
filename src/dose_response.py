"""APAP dose-response sweep using the trained GNN weights and learned f_GNN.

Only the APAP dose varies across runs; caffeine and all kinetics are held fixed,
so differences isolate the dose effect. Reports parent PK, clearance route
fractions, and NAPQI/GSH/adduct toxicity metrics.

Run: ``python -m src.dose_response``
"""

from __future__ import annotations

import argparse

import torch
from torch_geometric.data import HeteroData
from torchdiffeq import odeint

from src.data.build_graph import NODE_NAMES, build_dummy_graph
from src.metrics import pk_metrics_for_drug
from src.models.gnn_ode import (
    GNNODEModel,
    SPECIES_MW,
    STATE_IDX,
    STATE_NAMES,
    UM_PER_MG_PER_L,
    napqi_formation_scale,
)
from src.simulate import patient_weight_kg, plasma_volume_L

GSH_IDX = STATE_IDX["A_gsh"]
# Drug-derived mass conserves the dose; GSH regenerates and is excluded.
DRUG_DERIVED_IDX = [i for i in range(len(STATE_NAMES)) if i != GSH_IDX]

APAP_CLEARANCE_RXNS = frozenset(
    {"rxn_cyp_oxidation", "rxn_glucuronidation", "rxn_sulfation"}
)


def mass_balance_residual(traj: torch.Tensor, dose_total_mg: float) -> float:
    """|sum(drug-derived final states) - administered dose| (mg)."""
    final = traj[-1]
    drug_mass = float(final[DRUG_DERIVED_IDX].sum())
    return abs(drug_mass - dose_total_mg)


def integrate_apap_routes(
    traj: torch.Tensor,
    t: torch.Tensor,
    data: HeteroData,
    factors: torch.Tensor,
) -> dict[str, float]:
    """Time-integrate APAP liver clearance fluxes (mg) over ``traj``."""
    model = GNNODEModel(data)
    idx = model.ode_idx
    rxn_names = NODE_NAMES["reaction"]
    dt = float(t[1] - t[0])
    v_liver = float(idx["v_liver"])
    routes = {"glucuronidation": 0.0, "sulfation": 0.0, "oxidation": 0.0, "parent renal": 0.0}

    for ti in range(len(t) - 1):
        y = traj[ti].clamp(min=0.0)
        vmax = idx["edge_vmax_base"] * factors[idx["edge_rxn"], 0]
        c_sub_uM = (
            y[idx["edge_sub_state"]].clamp(min=0.0)
            / idx["edge_mw_sub"]
            / v_liver
            * UM_PER_MG_PER_L
        )
        rho = c_sub_uM / idx["edge_km"]
        denom_enz = torch.ones(idx["n_groups"]).index_add(0, idx["edge_enz_local"], rho)
        if idx["inhib_state"].numel() > 0:
            c_inh_uM = (
                y[idx["inhib_state"]].clamp(min=0.0)
                / idx["inhib_mw"]
                / v_liver
                * UM_PER_MG_PER_L
            )
            denom_enz = denom_enz.index_add(
                0, idx["inhib_enz_local"], c_inh_uM / idx["inhib_ki"]
            )
        v_uM_hr = vmax * rho / denom_enz[idx["edge_enz_local"]]

        a_apap_liver = y[STATE_IDX["A_liver_apap"]].clamp(min=0.0)
        c_apap_liver_uM = (
            a_apap_liver / SPECIES_MW["acetaminophen"] / v_liver * UM_PER_MG_PER_L
        )
        napqi_scale = napqi_formation_scale(c_apap_liver_uM)
        ox_mask = idx["edge_is_cyp_oxidation"].to(dtype=v_uM_hr.dtype)
        v_uM_hr = v_uM_hr * (1.0 - ox_mask + ox_mask * napqi_scale)

        sub_mg_hr = v_uM_hr * idx["edge_mw_sub"] * v_liver / UM_PER_MG_PER_L
        for k in range(idx["edge_rxn"].numel()):
            if int(idx["edge_sub_state"][k]) != STATE_IDX["A_liver_apap"]:
                continue
            rn = rxn_names[int(idx["edge_rxn"][k])]
            flux_mg = float(sub_mg_hr[k]) * dt
            if rn == "rxn_glucuronidation":
                routes["glucuronidation"] += flux_mg
            elif rn == "rxn_sulfation":
                routes["sulfation"] += flux_mg
            elif rn == "rxn_cyp_oxidation":
                routes["oxidation"] += flux_mg

        if idx["clear_state"].numel() > 0:
            clear_flux = idx["clear_k"] * y[idx["clear_state"]].clamp(min=0.0)
            for cs, cf in zip(idx["clear_state"].tolist(), clear_flux.tolist()):
                if cs == STATE_IDX["A_plasma_apap"]:
                    routes["parent renal"] += cf * dt

    return routes


def evaluate_dose(
    apap_mg: float,
    caffeine_mg: float,
    hours: float,
    steps: int,
    step_size: float,
    use_gnn_factors: bool,
    checkpoint: str,
) -> dict[str, float]:
    """Integrate one dose scenario and extract scalar PK / toxicity metrics."""
    data = build_dummy_graph(
        dose_overrides={"acetaminophen": apap_mg, "caffeine": caffeine_mg},
    )
    t = torch.linspace(0.0, hours, steps)
    model = GNNODEModel(data, hidden_channels=32, heads=2)

    with torch.no_grad():
        if use_gnn_factors:
            model.predict_params(data)
            state = torch.load(checkpoint)
            model.load_state_dict(state)
            model.eval()
            factors = model.predict_params(data)
        else:
            factors = torch.ones(data["reaction"].num_nodes, 1)
        y0 = model.initial_state(data)
        traj = odeint(
            model.build_ode(factors),
            y0,
            t,
            method="rk4",
            options={"step_size": step_size},
        )

    v_plasma = plasma_volume_L(data)
    weight = patient_weight_kg(data)
    apap = pk_metrics_for_drug(traj, t, "apap", v_plasma, weight_kg=weight)
    routes = integrate_apap_routes(traj, t, data, factors)
    dose = apap_mg
    final = traj[-1]
    napqi_traj = traj[:, STATE_IDX["A_napqi"]]
    gsh_traj = traj[:, STATE_IDX["A_gsh"]]
    gsh_baseline_mg = float(data["endogenous_molecule"].x_static[0, 0].abs())
    peak_napqi_idx = int(napqi_traj.argmax())
    min_gsh_idx = int(gsh_traj.argmin())
    urine_sink_mg = float(final[STATE_IDX["A_urine_sink"]])
    adduct_sink_mg = float(final[STATE_IDX["A_napqi_adduct_sink"]])
    min_gsh_mg = float(gsh_traj.min())

    return {
        "apap_dose_mg": apap_mg,
        "apap_cmax_ng_mL": float(apap["cmax_ng_mL"]),
        "apap_tmax_h": float(apap["tmax_h"]),
        "apap_t_half_h": float(apap["t_half_terminal_h"]),
        "pct_gluc": 100.0 * routes["glucuronidation"] / dose,
        "pct_sulf": 100.0 * routes["sulfation"] / dose,
        "pct_ox": 100.0 * routes["oxidation"] / dose,
        "pct_renal": 100.0 * routes["parent renal"] / dose,
        "peak_napqi_mg": float(napqi_traj.max()),
        "t_peak_napqi_h": float(t[peak_napqi_idx]),
        "min_gsh_mg": min_gsh_mg,
        "t_min_gsh_h": float(t[min_gsh_idx]),
        "pct_gsh_remaining": 100.0 * min_gsh_mg / gsh_baseline_mg,
        "final_gsh_mg": float(final[STATE_IDX["A_gsh"]]),
        "final_napqi_mg": float(final[STATE_IDX["A_napqi"]]),
        "final_adduct_sink_mg": adduct_sink_mg,
        "total_eliminated_sink_mg": urine_sink_mg + adduct_sink_mg,
        "mass_balance_residual_mg": mass_balance_residual(traj, apap_mg + caffeine_mg),
    }


def _fmt(value: float) -> str:
    return "   nan" if value != value else f"{value:7.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="APAP dose-response sweep.")
    parser.add_argument(
        "--doses",
        type=float,
        nargs="+",
        default=[3000.0, 7500.0, 10000.0, 15000.0, 20000.0, 30000.0, 50000.0, 120000.0],
    )
    parser.add_argument("--caffeine", type=float, default=0.0)
    parser.add_argument("--hours", type=float, default=48.0)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--step-size", type=float, default=0.01)
    parser.add_argument("--use-gnn-factors", action="store_true")
    parser.add_argument("--checkpoint", type=str, default="results/best_model.pt")
    args = parser.parse_args()

    factor_mode = "trained f_GNN" if args.use_gnn_factors else "neutral f_GNN=1"
    print(
        f"APAP dose-response ({factor_mode}, caffeine {args.caffeine:.0f} mg, "
        f"{args.hours:.0f} h)\n"
    )

    rows = [
        evaluate_dose(
            d,
            args.caffeine,
            args.hours,
            args.steps,
            args.step_size,
            args.use_gnn_factors,
            args.checkpoint,
        )
        for d in args.doses
    ]

    header = (
        f"{'APAP mg':>8} | {'tmax':>5} {'t1/2':>5} | "
        f"{'gluc%':>5} {'sulf%':>5} {'ox%':>4} | "
        f"{'pkNAPQI':>7} {'t_pkN':>5} | "
        f"{'minGSH':>6} {'t_min':>5} {'%GSH':>5} | "
        f"{'adduct':>7} {'sink':>7} {'MBres':>7}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['apap_dose_mg']:8.0f} | "
            f"{_fmt(r['apap_tmax_h']):>5} {_fmt(r['apap_t_half_h']):>5} | "
            f"{r['pct_gluc']:5.1f} {r['pct_sulf']:5.1f} {r['pct_ox']:4.1f} | "
            f"{r['peak_napqi_mg']:7.1f} {_fmt(r['t_peak_napqi_h']):>5} | "
            f"{r['min_gsh_mg']:6.0f} {_fmt(r['t_min_gsh_h']):>5} "
            f"{r['pct_gsh_remaining']:5.1f} | "
            f"{r['final_adduct_sink_mg']:7.1f} {r['total_eliminated_sink_mg']:7.0f} "
            f"{r['mass_balance_residual_mg']:7.2e}"
        )

    adduct = [r["final_adduct_sink_mg"] for r in rows]
    gsh_min = [r["min_gsh_mg"] for r in rows]
    ox_pct = [r["pct_ox"] for r in rows]
    adduct_rises = all(b >= a for a, b in zip(adduct, adduct[1:]))
    gsh_falls = all(b <= a for a, b in zip(gsh_min, gsh_min[1:]))
    ox_rises = all(b >= a for a, b in zip(ox_pct, ox_pct[1:]))
    print(
        f"\nTrend checks: ox% rises={ox_rises} | min GSH falls={gsh_falls} | "
        f"adduct rises={adduct_rises}"
    )

    r3k = next((r for r in rows if abs(r["apap_dose_mg"] - 3000.0) < 1.0), None)
    if r3k is not None:
        print(
            f"\n3000 mg anchor: gluc={r3k['pct_gluc']:.1f}% sulf={r3k['pct_sulf']:.1f}% "
            f"ox={r3k['pct_ox']:.1f}% renal={r3k['pct_renal']:.1f}% "
            f"t1/2={r3k['apap_t_half_h']:.2f} h"
        )


if __name__ == "__main__":
    main()
