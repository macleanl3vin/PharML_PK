"""GNN-ODE: edge-aware GNN IVIVE + competitive Michaelis-Menten PK integration.

The GNN predicts per-reaction Vmax modulation factors from static graph features
(Km/Kcat/Ki on edges, fingerprints, enzyme abundance). ``odeint`` integrates
coupled APAP/caffeine mass states; shared enzyme denominators encode DDIs.

Run: ``python -m src.models.gnn_ode``
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATv2Conv, HeteroConv, Linear
from torchdiffeq import odeint

from src.data.build_graph import NODE_NAMES, build_dummy_graph

# ODE state: amounts (mg). Indices 0–12 stable for downstream metrics code.
# Sinks A_necrosis and A_urine_sink close mass balance for bound NAPQI and renal loss.
STATE_NAMES = [
    "A_gut_apap", "A_plasma_apap", "A_liver_apap", "A_napqi", "A_gsh",
    "A_gut_caffeine", "A_plasma_caffeine", "A_liver_caffeine", "A_paraxanthine",
    "A_liver_apap_gluc", "A_liver_apap_sulf", "A_liver_theobromine", "A_liver_theophylline",
    "A_necrosis", "A_urine_sink",
]
STATE_IDX = {name: i for i, name in enumerate(STATE_NAMES)}

# Hand-maintained species → ODE index map (graph topology supplies the rest).
DRUG_GUT_STATE = {"acetaminophen": STATE_IDX["A_gut_apap"], "caffeine": STATE_IDX["A_gut_caffeine"]}
DRUG_PLASMA_STATE = {"acetaminophen": STATE_IDX["A_plasma_apap"], "caffeine": STATE_IDX["A_plasma_caffeine"]}
# Liver pool drives C/Km in enzymatic reactions.
SUBSTRATE_STATE = {
    "acetaminophen": STATE_IDX["A_liver_apap"],
    "caffeine": STATE_IDX["A_liver_caffeine"],
    "NAPQI": STATE_IDX["A_napqi"],
}
# Metabolite products with dedicated ODE states; NAPQI_glutathione → urine sink.
PRODUCT_STATE = {
    "NAPQI": STATE_IDX["A_napqi"],
    "paraxanthine": STATE_IDX["A_paraxanthine"],
    "acetaminophen_glucuronide": STATE_IDX["A_liver_apap_gluc"],
    "acetaminophen_sulfate": STATE_IDX["A_liver_apap_sulf"],
    "theobromine": STATE_IDX["A_liver_theobromine"],
    "theophylline": STATE_IDX["A_liver_theophylline"],
    "NAPQI_glutathione": STATE_IDX["A_urine_sink"],
}
# Co-substrates consumed 1:1 with flux (e.g. GSH in conjugation).
COSUBSTRATE_STATE = {"glutathione": STATE_IDX["A_gsh"]}

# MW (g/mol) for mg ↔ μM conversion at enzymatic sites.
SPECIES_MW = {
    "acetaminophen": 151.16,
    "caffeine": 194.19,
    "NAPQI": 149.15,
    "paraxanthine": 180.16,
    "glutathione": 307.32,
    "acetaminophen_glucuronide": 327.29,
    "acetaminophen_sulfate": 231.22,
    "theobromine": 180.16,
    "theophylline": 180.16,
}

# Graph stores Kcat in min⁻¹; ODE time is hours.
KCAT_PER_MIN_TO_PER_HR = 60.0

LIVER_COMPARTMENT_IDX = 1  # [plasma, liver, urine_sink]

# IVIVE: pmol/mg microsomal protein → hepatic enzyme concentration (μM).
# [E]_μM = abundance × MPPGL × liver_wt × 1e-6 / V_liver
MPPGL_MG_PER_G = 40.0
LIVER_WEIGHT_G = 1500.0
PMOL_TO_UMOL = 1e-6

# Co-substrate availability gate; stalls conjugation as GSH depletes (mg half-max).
COSUB_GATE_MG = 50.0

NG_PER_MG_PER_L = 1000.0   # mg/L → ng/mL
UM_PER_MG_PER_L = 1000.0   # C[μM] = A[mg] / MW / V[L] × 1000


def edge_dim_of(data: HeteroData, edge_type) -> int | None:
    store = data[edge_type]
    return store.edge_attr.size(-1) if "edge_attr" in store else None


def build_ode_index(data: HeteroData) -> dict:
    """Wire graph topology to ODE indices, mechanistic Vmax_base, and DDI groups."""
    rxn_names = NODE_NAMES["reaction"]
    drug_names = NODE_NAMES["drug"]
    met_names = NODE_NAMES["metabolite"]
    endo_names = NODE_NAMES["endogenous_molecule"]

    # Substrate / product / co-substrate species per reaction node.
    sub_species: dict[int, str] = {}
    for et in [("drug", "reactant_in", "reaction"), ("metabolite", "reactant_in", "reaction")]:
        names = NODE_NAMES[et[0]]
        for s, r in data[et].edge_index.t().tolist():
            sub_species[r] = names[s]
    cosub_species: dict[int, str] = {}
    et = ("endogenous_molecule", "reactant_in", "reaction")
    if et in data.edge_types:
        for s, r in data[et].edge_index.t().tolist():
            cosub_species[r] = endo_names[s]
    prod_species: dict[int, str] = {}
    for r, m in data["reaction", "produces", "metabolite"].edge_index.t().tolist():
        prod_species[r] = met_names[m]

    # effective_abundance = baseline × PGx × is_active × activity_multiplier (static).
    enz_static = data["enzyme"].x_static
    enz_abundance = enz_static[:, 3]                        # baseline_abundance_pmol_mg
    enz_pgx = enz_static[:, 0]                              # PGx_phenotype_multiplier
    enz_active = enz_static[:, 1]                           # is_active (0/1)
    n_enzymes = data["enzyme"].num_nodes
    act_mult = torch.ones(n_enzymes)
    et = ("patient", "expresses", "enzyme")
    if et in data.edge_types and "edge_attr" in data[et]:
        ea = data[et].edge_attr
        for k, (_p, e) in enumerate(data[et].edge_index.t().tolist()):
            act_mult[e] = ea[k, 0]
    effective_abundance = (enz_abundance * enz_pgx * enz_active * act_mult)

    v_liver = float(data["compartment"].x_static[LIVER_COMPARTMENT_IDX, 0].abs().clamp(min=1e-3))
    enzyme_conc_factor = MPPGL_MG_PER_G * LIVER_WEIGHT_G * PMOL_TO_UMOL / v_liver

    # Per catalytic edge: Vmax_base (μM/hr) = Kcat[hr⁻¹] × [E]_μM.
    cat_et = ("enzyme", "catalyzes", "reaction")
    cat_attr = data[cat_et].edge_attr                     # [E, 3] = Km, Ki, Kcat
    (edge_enz_global, edge_rxn, edge_km, edge_vmax_base, edge_sub_state, edge_mw_sub,
     edge_prod_state, edge_prod_mask, edge_mw_prod,
     edge_cosub_state, edge_cosub_mask, edge_mw_cosub) = ([] for _ in range(12))
    for k, (e, r) in enumerate(data[cat_et].edge_index.t().tolist()):
        km = max(float(cat_attr[k, 0]), 1e-6)
        kcat_per_hr = float(cat_attr[k, 2]) * KCAT_PER_MIN_TO_PER_HR
        edge_enz_global.append(e)
        edge_rxn.append(r)
        edge_km.append(km)
        edge_vmax_base.append(kcat_per_hr * float(effective_abundance[e]) * enzyme_conc_factor)
        edge_sub_state.append(SUBSTRATE_STATE[sub_species[r]])
        edge_mw_sub.append(SPECIES_MW[sub_species[r]])
        p = prod_species.get(r)
        if p in PRODUCT_STATE:
            edge_prod_state.append(PRODUCT_STATE[p]); edge_prod_mask.append(True)
            # Parent-equivalent deposition: product MW defaults to substrate MW.
            edge_mw_prod.append(SPECIES_MW.get(p, SPECIES_MW[sub_species[r]]))
        else:
            edge_prod_state.append(0); edge_prod_mask.append(False); edge_mw_prod.append(1.0)
        c = cosub_species.get(r)
        if c in COSUBSTRATE_STATE:
            edge_cosub_state.append(COSUBSTRATE_STATE[c]); edge_cosub_mask.append(True)
            edge_mw_cosub.append(SPECIES_MW[c])
        else:
            edge_cosub_state.append(0); edge_cosub_mask.append(False); edge_mw_cosub.append(1.0)

    # Reactions on the same enzyme share one MM denominator (DDI).
    uniq_enz = sorted(set(edge_enz_global))
    enz_group_of = {e: i for i, e in enumerate(uniq_enz)}
    edge_enz_local = [enz_group_of[e] for e in edge_enz_global]
    n_groups = len(uniq_enz)

    enz_rxn_rows = sorted(set(edge_rxn))

    # ka from graph edge; not GNN-predicted.
    abs_et = ("drug", "absorbed_via", "reaction")
    abs_ka_attr = data[abs_et].edge_attr if "edge_attr" in data[abs_et] else None
    abs_gut, abs_plasma, abs_ka = [], [], []
    for k, (d, r) in enumerate(data[abs_et].edge_index.t().tolist()):
        name = drug_names[d]
        if name in DRUG_GUT_STATE:
            abs_gut.append(DRUG_GUT_STATE[name])
            abs_plasma.append(DRUG_PLASMA_STATE[name])
            abs_ka.append(float(abs_ka_attr[k, 0]) if abs_ka_attr is not None else 1.0)

    # k_p2l, k_l2p from distributes_to edges (Vd target × weight in build_graph).
    dist_et = ("drug", "distributes_to", "compartment")
    dist_p_idx, dist_l_idx, dist_k_p2l, dist_k_l2p = [], [], [], []
    if dist_et in data.edge_types and "edge_attr" in data[dist_et]:
        dist_attr = data[dist_et].edge_attr
        for k, (d, _comp) in enumerate(data[dist_et].edge_index.t().tolist()):
            name = drug_names[d]
            if name in DRUG_PLASMA_STATE and name in SUBSTRATE_STATE:
                dist_p_idx.append(DRUG_PLASMA_STATE[name])
                dist_l_idx.append(SUBSTRATE_STATE[name])
                dist_k_p2l.append(float(dist_attr[k, 0]))
                dist_k_l2p.append(float(dist_attr[k, 1]))

    # Competitive inhibition: liver C/Ki added to shared enzyme denominator.
    inhib_et = ("drug", "competitively_inhibits", "enzyme")
    inhib_state, inhib_enz_local, inhib_ki, inhib_mw = [], [], [], []
    if inhib_et in data.edge_types and "edge_attr" in data[inhib_et]:
        inhib_attr = data[inhib_et].edge_attr
        for k, (d, e) in enumerate(data[inhib_et].edge_index.t().tolist()):
            name = drug_names[d]
            if name in SUBSTRATE_STATE and e in enz_group_of:
                inhib_state.append(SUBSTRATE_STATE[name])
                inhib_enz_local.append(enz_group_of[e])
                inhib_ki.append(max(float(inhib_attr[k, 0]), 1e-6))
                inhib_mw.append(SPECIES_MW[name])

    # Terminal metabolite renal clearance → urine sink.
    clr_et = ("metabolite", "cleared_via", "reaction")
    clear_state, clear_k = [], []
    if clr_et in data.edge_types and "edge_attr" in data[clr_et]:
        clr_attr = data[clr_et].edge_attr
        for k, (m, _r) in enumerate(data[clr_et].edge_index.t().tolist()):
            name = met_names[m]
            if name in PRODUCT_STATE:
                clear_state.append(PRODUCT_STATE[name])
                clear_k.append(float(clr_attr[k, 0]))

    # GSH regeneration parameters from endogenous_molecule.x_static.
    endo_static = data["endogenous_molecule"].x_static
    gsh_baseline_mg = float(endo_static[0, 0].abs())
    k_syn_gsh = float(endo_static[0, 1].abs())

    L = lambda x: torch.tensor(x, dtype=torch.long)
    return {
        "enz_rxn_rows": L(enz_rxn_rows),
        "enz_rxn_names": [rxn_names[r] for r in enz_rxn_rows],
        "edge_rxn": L(edge_rxn),
        "edge_enz_local": L(edge_enz_local),
        "edge_km": torch.tensor(edge_km, dtype=torch.float),
        "edge_vmax_base": torch.tensor(edge_vmax_base, dtype=torch.float),
        "edge_sub_state": L(edge_sub_state),
        "edge_mw_sub": torch.tensor(edge_mw_sub, dtype=torch.float),
        "edge_prod_state_valid": L([s for s, m in zip(edge_prod_state, edge_prod_mask) if m]),
        "edge_prod_local_valid": L([i for i, m in enumerate(edge_prod_mask) if m]),
        "edge_mw_prod_valid": torch.tensor([mw for mw, m in zip(edge_mw_prod, edge_prod_mask) if m], dtype=torch.float),
        "edge_cosub_state_valid": L([s for s, m in zip(edge_cosub_state, edge_cosub_mask) if m]),
        "edge_cosub_local_valid": L([i for i, m in enumerate(edge_cosub_mask) if m]),
        "edge_mw_cosub_valid": torch.tensor([mw for mw, m in zip(edge_mw_cosub, edge_cosub_mask) if m], dtype=torch.float),
        "edge_cosub_state_all": L(edge_cosub_state),
        "edge_has_cosub": torch.tensor([1.0 if m else 0.0 for m in edge_cosub_mask], dtype=torch.float),
        "n_groups": n_groups,
        "v_liver": v_liver,
        "abs_gut": L(abs_gut),
        "abs_plasma": L(abs_plasma),
        "abs_ka": torch.tensor(abs_ka, dtype=torch.float),
        "dist_p_idx": L(dist_p_idx),
        "dist_l_idx": L(dist_l_idx),
        "dist_k_p2l": torch.tensor(dist_k_p2l, dtype=torch.float),
        "dist_k_l2p": torch.tensor(dist_k_l2p, dtype=torch.float),
        "inhib_state": L(inhib_state),
        "inhib_enz_local": L(inhib_enz_local),
        "inhib_ki": torch.tensor(inhib_ki, dtype=torch.float),
        "inhib_mw": torch.tensor(inhib_mw, dtype=torch.float),
        "clear_state": L(clear_state),
        "clear_k": torch.tensor(clear_k, dtype=torch.float),
        "urine_state": STATE_IDX["A_urine_sink"],
        "necrosis_state": STATE_IDX["A_napqi"],
        "gsh_state": STATE_IDX["A_gsh"],
        "gsh_baseline_mg": gsh_baseline_mg,
        "k_syn_gsh": k_syn_gsh,
    }


def trajectory_to_curves(traj: torch.Tensor, v_plasma: torch.Tensor) -> torch.Tensor:
    """Map ODE trajectory to ``[7, T, 2]`` metabolite/parent concentrations (ng/mL).

    Column 0 = metabolite; column 1 = parent plasma. NAPQI_glutathione uses
    GSH consumed as a proxy (no dedicated state).
    """
    c = traj / v_plasma * NG_PER_MG_PER_L  # [T, S] ng/mL
    i_gsh = STATE_IDX["A_gsh"]
    gsh_consumed = (traj[0, i_gsh] - traj[:, i_gsh]) / v_plasma * NG_PER_MG_PER_L  # [T]
    metab = torch.stack([
        c[:, STATE_IDX["A_napqi"]],               # NAPQI
        c[:, STATE_IDX["A_liver_apap_gluc"]],     # acetaminophen_glucuronide
        c[:, STATE_IDX["A_liver_apap_sulf"]],
        gsh_consumed,
        c[:, STATE_IDX["A_paraxanthine"]],        # paraxanthine
        c[:, STATE_IDX["A_liver_theobromine"]],   # theobromine
        c[:, STATE_IDX["A_liver_theophylline"]],  # theophylline
    ], dim=0)
    i_apap_p = STATE_IDX["A_plasma_apap"]
    i_caff_p = STATE_IDX["A_plasma_caffeine"]
    parent = torch.stack([
        c[:, i_apap_p], c[:, i_apap_p], c[:, i_apap_p], c[:, i_apap_p],
        c[:, i_caff_p], c[:, i_caff_p], c[:, i_caff_p],
    ], dim=0)
    return torch.stack([metab, parent], dim=-1).clamp(min=0.0)  # [7, T, 2]


class MichaelisMentenODE(nn.Module):
    """Competitive MM kinetics for APAP + caffeine (amounts in mg, time in hr).

    Vmax_edge = Kcat × [E] × f_GNN; shared enzyme denominators couple DDIs.
    Absorption, distribution, renal clearance, and GSH regeneration are
    graph-derived; k_tox (NAPQI → necrosis) is the sole hardcoded rate.
    """

    def __init__(self, factors: torch.Tensor, idx: dict, k_tox: float = 0.5):
        super().__init__()
        self.factors = factors
        self.idx = idx
        self.k_tox = k_tox  # hr⁻¹; NAPQI covalent binding when GSH is depleted

    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        idx = self.idx
        y = y.clamp(min=0.0)  # non-negative amounts for kinetics
        dydt = torch.zeros_like(y)

        # Enzymatic flux: mg → μM → MM (μM/hr) → mg/hr.
        V = self.idx["v_liver"]
        f = self.factors[idx["edge_rxn"], 0]
        vmax = idx["edge_vmax_base"] * f

        a_sub = y[idx["edge_sub_state"]].clamp(min=0.0)
        c_sub_uM = a_sub / idx["edge_mw_sub"] / V * UM_PER_MG_PER_L
        rho = c_sub_uM / idx["edge_km"]

        denom_enz = torch.ones(idx["n_groups"], dtype=y.dtype).index_add(
            0, idx["edge_enz_local"], rho)
        # Competitive inhibition: add C_inhibitor/Ki to shared enzyme denominator.
        if idx["inhib_state"].numel() > 0:
            a_inh = y[idx["inhib_state"]].clamp(min=0.0)
            c_inh_uM = a_inh / idx["inhib_mw"] / V * UM_PER_MG_PER_L
            rho_inh = c_inh_uM / idx["inhib_ki"]
            denom_enz = denom_enz.index_add(0, idx["inhib_enz_local"], rho_inh)
        denom_edge = denom_enz[idx["edge_enz_local"]]
        v_uM_hr = vmax * rho / denom_edge

        # Scale flux by co-substrate availability (e.g. GSH gate).
        if idx["edge_cosub_local_valid"].numel() > 0:
            loc = idx["edge_cosub_local_valid"]
            c_cosub = y[idx["edge_cosub_state_valid"]]
            avail = c_cosub / (c_cosub + COSUB_GATE_MG)
            gate = torch.ones_like(v_uM_hr)
            gate = gate.index_put((loc,), avail)
            v_uM_hr = v_uM_hr * gate

        sub_mg_hr = v_uM_hr * idx["edge_mw_sub"] * V / UM_PER_MG_PER_L
        dydt = dydt.index_add(0, idx["edge_sub_state"], -sub_mg_hr)
        # Parent-equivalent product deposition (mg 1:1 from substrate).
        if idx["edge_prod_local_valid"].numel() > 0:
            loc = idx["edge_prod_local_valid"]
            prod_mg_hr = sub_mg_hr[loc]
            dydt = dydt.index_add(0, idx["edge_prod_state_valid"], prod_mg_hr)
        if idx["edge_cosub_local_valid"].numel() > 0:
            loc = idx["edge_cosub_local_valid"]
            cosub_mg_hr = v_uM_hr[loc] * idx["edge_mw_cosub_valid"] * V / UM_PER_MG_PER_L
            dydt = dydt.index_add(0, idx["edge_cosub_state_valid"], -cosub_mg_hr)

        if idx["abs_gut"].numel() > 0:
            abs_flux = idx["abs_ka"] * y[idx["abs_gut"]].clamp(min=0.0)
            dydt = dydt.index_add(0, idx["abs_gut"], -abs_flux)
            dydt = dydt.index_add(0, idx["abs_plasma"], abs_flux)

        p_idx = idx["dist_p_idx"]
        l_idx = idx["dist_l_idx"]
        k_p2l = idx["dist_k_p2l"].to(y.device)
        k_l2p = idx["dist_k_l2p"].to(y.device)
        dist_flux = k_p2l * y[p_idx] - k_l2p * y[l_idx]
        dydt = dydt.index_add(0, p_idx, -dist_flux)
        dydt = dydt.index_add(0, l_idx, dist_flux)

        gsh_i = idx["gsh_state"]
        dydt[gsh_i] = dydt[gsh_i] + idx["k_syn_gsh"] * (idx["gsh_baseline_mg"] - y[gsh_i])

        napqi_i = idx["necrosis_state"]
        necrosis_flux = self.k_tox * y[napqi_i]
        dydt[napqi_i] = dydt[napqi_i] - necrosis_flux
        dydt[STATE_IDX["A_necrosis"]] = dydt[STATE_IDX["A_necrosis"]] + necrosis_flux

        if idx["clear_state"].numel() > 0:
            clear_flux = idx["clear_k"] * y[idx["clear_state"]].clamp(min=0.0)
            dydt = dydt.index_add(0, idx["clear_state"], -clear_flux)
            dydt[idx["urine_state"]] = dydt[idx["urine_state"]] + clear_flux.sum()

        return dydt


class GNNODEModel(nn.Module):
    def __init__(self, data: HeteroData, hidden_channels: int = 32, heads: int = 2):
        super().__init__()
        node_types, edge_types = data.metadata()

        self.encoders = nn.ModuleDict(
            {nt: Linear(-1, hidden_channels) for nt in node_types}
        )

        def make_layer() -> HeteroConv:
            return HeteroConv(
                {
                    et: GATv2Conv(
                        (-1, -1), hidden_channels, heads=heads, concat=False,
                        edge_dim=edge_dim_of(data, et), add_self_loops=False,
                    )
                    for et in edge_types
                },
                aggr="sum",
            )

        self.conv1 = make_layer()
        self.conv2 = make_layer()

        # Per-reaction softplus head → dimensionless Vmax modulation factor.
        self.param_head = nn.Linear(hidden_channels, 1)

        self.ode_idx = build_ode_index(data)

    def encode(self, data: HeteroData) -> dict:
        """Two edge-aware HeteroConv layers; reaction embeddings retained (no pooling)."""
        x_dict = {
            nt: F.relu(self.encoders[nt](torch.cat([data[nt].x_state, data[nt].x_static], dim=-1)))
            for nt in self.encoders
        }
        out = self.conv1(x_dict, data.edge_index_dict, edge_attr_dict=data.edge_attr_dict)
        x_dict = {**x_dict, **{k: F.relu(v) for k, v in out.items()}}
        out = self.conv2(x_dict, data.edge_index_dict, edge_attr_dict=data.edge_attr_dict)
        x_dict = {**x_dict, **{k: F.relu(v) for k, v in out.items()}}
        return x_dict

    def predict_params(self, data: HeteroData) -> torch.Tensor:
        """Reaction-level readout: ``[R, 1]`` positive Vmax modulation factors."""
        x_dict = self.encode(data)
        h_rxn = x_dict["reaction"]
        return F.softplus(self.param_head(h_rxn)) + 1e-6

    def initial_state(self, data: HeteroData) -> torch.Tensor:
        """y₀: doses from admin→drug edges seed gut; GSH at baseline; all else zero."""
        drug_names = NODE_NAMES["drug"]
        dose_by_drug: dict[str, torch.Tensor] = {}
        dose_et = ("administration_event", "releases", "drug")
        if dose_et in data.edge_types and "edge_attr" in data[dose_et]:
            for k, (_admin, d) in enumerate(data[dose_et].edge_index.t().tolist()):
                dose_by_drug[drug_names[d]] = data[dose_et].edge_attr[k, 0].clamp(min=0.0)
        zero = torch.zeros(())
        dose_apap = dose_by_drug.get("acetaminophen", zero)
        dose_caff = dose_by_drug.get("caffeine", zero)
        gsh0 = data["endogenous_molecule"].x_state[0, 0].abs()
        return torch.stack([
            dose_apap, zero, zero, zero, gsh0,
            dose_caff, zero, zero, zero,
            zero, zero, zero, zero,
            zero, zero,
        ])

    def build_ode(self, factors: torch.Tensor) -> MichaelisMentenODE:
        return MichaelisMentenODE(factors, self.ode_idx)

    def forward(self, data: HeteroData, t: torch.Tensor):
        factors = self.predict_params(data)
        y0 = self.initial_state(data)

        # Fixed-step rk4: bounds memory vs adaptive solver during backprop through odeint.
        traj = odeint(self.build_ode(factors), y0, t, method="rk4",
                      options={"step_size": 0.025})

        v_plasma = data["compartment"].x_static[0, 0].abs().clamp(min=1.0)
        curves = trajectory_to_curves(traj, v_plasma)

        return traj, curves, factors


def main() -> None:
    data = build_dummy_graph()
    model = GNNODEModel(data, hidden_channels=32, heads=2)

    t = torch.linspace(0.0, 24.0, steps=10)
    traj, curves, factors = model(data, t)
    metabolite_pred = curves[:, -1, :]
    idx = model.ode_idx

    print(f"ODE trajectory shape : {tuple(traj.shape)}  (T x state)")
    print(f"concentration curves : {tuple(curves.shape)}  (7 x T x 2, ng/mL)")
    print(f"reaction factors     : {tuple(factors.shape)}  (R x 1, GNN Vmax modulation)")
    print(f"metabolite pred      : {tuple(metabolite_pred.shape)} | target {tuple(data['metabolite'].y.shape)}")
    print("per catalytic edge: mechanistic Vmax_base (Kcat*abundance), Km, GNN factor:")
    fac = factors.detach()
    for k in range(idx["edge_rxn"].numel()):
        rxn = idx["enz_rxn_names"][0] if False else NODE_NAMES["reaction"][int(idx["edge_rxn"][k])]
        print(f"  {rxn:28s} Vmax_base={float(idx['edge_vmax_base'][k]):9.3f}  "
              f"Km={float(idx['edge_km'][k]):6.3f}  f_GNN={float(fac[int(idx['edge_rxn'][k]), 0]):6.3f}")

    # Scale-normalized MSE smoke test: confirms gradients reach GNN via odeint.
    scale = metabolite_pred.detach().abs().max().clamp(min=1.0)
    loss = F.mse_loss(metabolite_pred / scale, data["metabolite"].y / scale)
    loss.backward()

    grad_total = sum(
        p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None
    )
    head_grad = sum(
        p.grad.abs().sum().item()
        for n, p in model.named_parameters()
        if "param_head" in n and p.grad is not None
    )
    print(f"loss = {loss.item():.4f} | total grad = {grad_total:.2f} | param_head grad = {head_grad:.4f}")

    assert grad_total > 0 and head_grad > 0, "Gradients did not flow through the ODE solver."
    print("Reaction-level Michaelis-Menten GNN-ODE forward and backward pass succeeded.")

    # 24 h mass-balance check with neutral f_GNN = 1.
    with torch.no_grad():
        t24 = torch.linspace(0.0, 24.0, steps=200)
        neutral = torch.ones(data["reaction"].num_nodes, 1)
        y0 = model.initial_state(data)
        traj24 = odeint(model.build_ode(neutral), y0, t24, method="rk4",
                        options={"step_size": 0.01})
        final = traj24[-1]

        dose_et = ("administration_event", "releases", "drug")
        dose_total = float(data[dose_et].edge_attr[:, 0].clamp(min=0.0).sum())
        gsh_i = STATE_IDX["A_gsh"]
        drug_idx = [i for i in range(len(STATE_NAMES)) if i != gsh_i]  # exclude regenerating GSH
        drug_mass = float(final[drug_idx].sum())
        residual = abs(drug_mass - dose_total)

        print("\n24 h final states (mg):")
        for name, val in zip(STATE_NAMES, final.tolist()):
            print(f"  {name:22s} {val:12.4f}")
        print(f"\nadministered dose          = {dose_total:.4f} mg")
        print(f"sum(drug-derived states)   = {drug_mass:.4f} mg")
        print(f"mass-balance residual      = {residual:.3e} mg")
        print(f"A_urine_sink               = {float(final[STATE_IDX['A_urine_sink']]):.4f} mg")
        print(f"A_necrosis                 = {float(final[STATE_IDX['A_necrosis']]):.4f} mg")
        assert residual < 1e-2, f"Mass balance violated: residual {residual:.3e} mg"
        print("Mass balance holds: drug-derived states conserve the administered dose.")


if __name__ == "__main__":
    main()
