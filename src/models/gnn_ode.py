"""Phase 3 (refactored): reaction-level Michaelis-Menten IVIVE engine.

The GNN no longer mean-pools the graph into a single vector and emits global
first-order rate constants. Instead it acts as an In-Vitro to In-Vivo
Extrapolation (IVIVE) engine:

  Stage 1-2  Edge-aware HeteroConv message passing keeps the literature
             constants (Km/Kcat/Ki) flowing as edge_attr so the GNN can learn
             context (genetics x abundance x chemistry).
  Stage 3    Reaction-level readout: h_rxn = x_dict['reaction'].  NO pooling --
             the bipartite topology (who competes for which enzyme) is kept.
  Stage 4    A Linear+softplus head emits, per reaction, two positive apparent
             in-vivo parameters: [V_max_app, K_m_app].
  Stage 5    A competitive Michaelis-Menten ODE integrates the system.  Each
             enzyme's reactions share a denominator
             (1 + sum_substrate C/Km + sum_inhibitor C/Ki), so a spike in one
             substrate (or a pure inhibitor) instantly throttles every other
             reaction on that enzyme -- a mechanistic Drug-Drug Interaction.

The reaction -> substrate / product / enzyme groupings are derived from the
graph topology (see ``build_ode_index``); the only hand-maintained bridge is the
species -> ODE-state-index map below.

Run from project root:
    python -m src.models.gnn_ode
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATv2Conv, HeteroConv, Linear
from torchdiffeq import odeint

from src.data.build_graph import NODE_NAMES, build_dummy_graph

# 15-dim ODE state vector (amounts, e.g. mg). Indices 0-12 are kept stable
# (downstream code in metrics.py / simulate.py keys off STATE_IDX). Two terminal
# sink states are appended: A_necrosis (covalently bound NAPQI / liver damage)
# and A_urine_sink (renally excreted mass). Together with the metabolite states
# they close the mass balance: nothing disappears, it moves into a tracked sink.
STATE_NAMES = [
    "A_gut_apap", "A_plasma_apap", "A_liver_apap", "A_napqi", "A_gsh",
    "A_gut_caffeine", "A_plasma_caffeine", "A_liver_caffeine", "A_paraxanthine",
    "A_liver_apap_gluc", "A_liver_apap_sulf", "A_liver_theobromine", "A_liver_theophylline",
    "A_necrosis", "A_urine_sink",
]
STATE_IDX = {name: i for i, name in enumerate(STATE_NAMES)}

# --- the species -> ODE-state bridge (the one hand-maintained mapping) --------
# Where each drug's mass lives along gut -> plasma -> liver.
DRUG_GUT_STATE = {"acetaminophen": STATE_IDX["A_gut_apap"], "caffeine": STATE_IDX["A_gut_caffeine"]}
DRUG_PLASMA_STATE = {"acetaminophen": STATE_IDX["A_plasma_apap"], "caffeine": STATE_IDX["A_plasma_caffeine"]}
# Concentration that drives the C/Km ratio of an enzymatic reaction (site of metabolism).
SUBSTRATE_STATE = {
    "acetaminophen": STATE_IDX["A_liver_apap"],
    "caffeine": STATE_IDX["A_liver_caffeine"],
    "NAPQI": STATE_IDX["A_napqi"],
}
# Reaction products with a dedicated, mass-conserving ODE state. Every metabolite
# node in the graph now maps to its own state, so enzymatic flux deposits real
# mass instead of disappearing. The GSH conjugate (NAPQI_glutathione) is routed
# straight to the urine sink (mercapturate excretion), closing the NAPQI balance
# while keeping the endogenous GSH pool as a separate regenerating species.
PRODUCT_STATE = {
    "NAPQI": STATE_IDX["A_napqi"],
    "paraxanthine": STATE_IDX["A_paraxanthine"],
    "acetaminophen_glucuronide": STATE_IDX["A_liver_apap_gluc"],
    "acetaminophen_sulfate": STATE_IDX["A_liver_apap_sulf"],
    "theobromine": STATE_IDX["A_liver_theobromine"],
    "theophylline": STATE_IDX["A_liver_theophylline"],
    "NAPQI_glutathione": STATE_IDX["A_urine_sink"],
}
# Co-substrates consumed 1:1 with the reaction flux (e.g. GSH in conjugation).
COSUBSTRATE_STATE = {"glutathione": STATE_IDX["A_gsh"]}

# Molecular weights (g/mol == mg/mmol) used for the mg <-> uM unit conversion in
# the metabolism math. Needed for every species that is consumed or produced by
# an enzymatic reaction.
SPECIES_MW = {
    "acetaminophen": 151.16,
    "caffeine": 194.19,
    "NAPQI": 149.15,
    "paraxanthine": 180.16,
    "glutathione": 307.32,
    # Explicit conjugate / demethylation products (now tracked as ODE states).
    "acetaminophen_glucuronide": 327.29,
    "acetaminophen_sulfate": 231.22,
    "theobromine": 180.16,
    "theophylline": 180.16,
}

# Kcat is stored per-minute in the graph; the ODE runs in hours.
KCAT_PER_MIN_TO_PER_HR = 60.0

# Liver is the metabolic compartment (its volume converts liver amounts <-> conc).
LIVER_COMPARTMENT_IDX = 1  # COMPARTMENT_NODES = [plasma, liver, urine_sink]

# --- IVIVE scale-up: in-vitro enzyme abundance -> in-vivo enzyme concentration ---
# baseline_abundance is pmol enzyme / mg microsomal protein (in-vitro). Scaling to
# an organ-level enzyme concentration (uM) requires the microsomal recovery factor
# (MPPGL, mg microsomal protein / g liver) and the liver mass (g):
#     [E]_uM = abundance[pmol/mg] * MPPGL[mg/g] * liver_wt[g] * 1e-6[umol/pmol] / V_liver[L]
# Without this step Vmax is ~1000x too large and clearance is non-physical.
MPPGL_MG_PER_G = 40.0      # mg microsomal protein per gram liver
LIVER_WEIGHT_G = 1500.0    # liver mass (g)
PMOL_TO_UMOL = 1e-6

# Co-substrate gating: a conjugation reaction stalls as its co-substrate (e.g.
# GSH) is exhausted. This smooth availability factor -> 0 as the pool empties,
# which both keeps amounts non-negative and reproduces the NAPQI-accumulation
# (toxicity) regime once GSH is depleted. Half-max at COSUB_GATE_MG (mg).
COSUB_GATE_MG = 50.0

# Unit conversion: state amounts are mg, volumes are L, so A/V is mg/L.
# 1 mg/L == 1000 ng/mL, so concentrations reported in ng/mL need this factor.
NG_PER_MG_PER_L = 1000.0
# mg -> uM:  C[uM] = A[mg] / MW[g/mol] / V[L] * 1000.   (and the inverse for uM -> mg)
UM_PER_MG_PER_L = 1000.0


def edge_dim_of(data: HeteroData, edge_type) -> int | None:
    store = data[edge_type]
    return store.edge_attr.size(-1) if "edge_attr" in store else None


def build_ode_index(data: HeteroData) -> dict:
    """Derive the reaction <-> state/enzyme wiring from the graph topology.

    Returns a dict of LongTensors/FloatTensors describing, for the enzymatic
    subset of reactions, which ODE states they consume/produce, which reactions
    share an enzyme (the competition mask), and which inhibitors throttle them.
    Also returns the first-order absorption wiring. Everything is keyed by the
    reaction-node row order so it lines up with the GNN's [R, 2] output.
    """
    rxn_names = NODE_NAMES["reaction"]
    drug_names = NODE_NAMES["drug"]
    met_names = NODE_NAMES["metabolite"]
    endo_names = NODE_NAMES["endogenous_molecule"]

    # --- species attached to each reaction (from topology) ---
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

    # --- effective enzyme abundance (mechanistic Vmax scaling) ---
    #   effective_abundance = baseline_abundance * PGx * is_active * activity_multiplier
    # baseline_abundance_pmol_mg is static (does not change with time); a separate
    # current_abundance state column would be needed for mechanism-based inhibition.
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
    effective_abundance = (enz_abundance * enz_pgx * enz_active * act_mult)  # [n_enzymes]

    # metabolic compartment volume (L) for the mg <-> uM conversion + IVIVE scale-up.
    v_liver = float(data["compartment"].x_static[LIVER_COMPARTMENT_IDX, 0].abs().clamp(min=1e-3))
    # in-vitro abundance (pmol/mg) -> in-vivo enzyme concentration (uM)
    enzyme_conc_factor = MPPGL_MG_PER_G * LIVER_WEIGHT_G * PMOL_TO_UMOL / v_liver

    # --- per (enzyme, reaction) catalytic edge: mechanistic Vmax_base = Kcat * [E] ---
    cat_et = ("enzyme", "catalyzes", "reaction")
    cat_attr = data[cat_et].edge_attr                     # [E, 3] = Km, Ki, Kcat
    (edge_enz_global, edge_rxn, edge_km, edge_vmax_base, edge_sub_state, edge_mw_sub,
     edge_prod_state, edge_prod_mask, edge_mw_prod,
     edge_cosub_state, edge_cosub_mask, edge_mw_cosub) = ([] for _ in range(12))
    for k, (e, r) in enumerate(data[cat_et].edge_index.t().tolist()):
        km = max(float(cat_attr[k, 0]), 1e-6)             # Km (μM), floored for safety
        kcat_per_hr = float(cat_attr[k, 2]) * KCAT_PER_MIN_TO_PER_HR
        edge_enz_global.append(e)
        edge_rxn.append(r)
        edge_km.append(km)
        # mechanistic Vmax base (uM/hr): Kcat[/hr] * [E]_uM, pre-GNN-factor.
        #   [E]_uM = effective_abundance[pmol/mg] * enzyme_conc_factor (IVIVE scale-up)
        edge_vmax_base.append(kcat_per_hr * float(effective_abundance[e]) * enzyme_conc_factor)
        edge_sub_state.append(SUBSTRATE_STATE[sub_species[r]])
        edge_mw_sub.append(SPECIES_MW[sub_species[r]])
        p = prod_species.get(r)
        if p in PRODUCT_STATE:
            edge_prod_state.append(PRODUCT_STATE[p]); edge_prod_mask.append(True)
            # Product MW falls back to the substrate's MW: deposition is
            # parent-equivalent (mg 1:1), so the product's own MW is not needed
            # (and some products like NAPQI_glutathione have no tabulated MW).
            edge_mw_prod.append(SPECIES_MW.get(p, SPECIES_MW[sub_species[r]]))
        else:
            edge_prod_state.append(0); edge_prod_mask.append(False); edge_mw_prod.append(1.0)
        c = cosub_species.get(r)
        if c in COSUBSTRATE_STATE:
            edge_cosub_state.append(COSUBSTRATE_STATE[c]); edge_cosub_mask.append(True)
            edge_mw_cosub.append(SPECIES_MW[c])
        else:
            edge_cosub_state.append(0); edge_cosub_mask.append(False); edge_mw_cosub.append(1.0)

    # enzyme group index: all reactions on one enzyme share a denominator (DDI).
    uniq_enz = sorted(set(edge_enz_global))
    enz_group_of = {e: i for i, e in enumerate(uniq_enz)}
    edge_enz_local = [enz_group_of[e] for e in edge_enz_global]
    n_groups = len(uniq_enz)

    enz_rxn_rows = sorted(set(edge_rxn))                  # reactions exposed to the GNN factor

    # first-order absorption: drug absorbed_via reaction, entering plasma. The
    # rate constant ka is read straight from the edge attribute (absorption_rate_ka),
    # NOT predicted by the GNN -- absorption is input kinetics, not metabolism.
    abs_et = ("drug", "absorbed_via", "reaction")
    abs_ka_attr = data[abs_et].edge_attr if "edge_attr" in data[abs_et] else None
    abs_gut, abs_plasma, abs_ka = [], [], []
    for k, (d, r) in enumerate(data[abs_et].edge_index.t().tolist()):
        name = drug_names[d]
        if name in DRUG_GUT_STATE:
            abs_gut.append(DRUG_GUT_STATE[name])
            abs_plasma.append(DRUG_PLASMA_STATE[name])
            abs_ka.append(float(abs_ka_attr[k, 0]) if abs_ka_attr is not None else 1.0)

    # --- patient-specific plasma <-> liver distribution (read from the graph) ---
    # Drug-agnostic: k_p2l / k_l2p come from the distributes_to edge_attr (derived
    # from each drug's Vd target x patient weight in build_graph), NOT hardcoded.
    dist_et = ("drug", "distributes_to", "compartment")
    dist_p_idx, dist_l_idx, dist_k_p2l, dist_k_l2p = [], [], [], []
    if dist_et in data.edge_types and "edge_attr" in data[dist_et]:
        dist_attr = data[dist_et].edge_attr
        for k, (d, _comp) in enumerate(data[dist_et].edge_index.t().tolist()):
            name = drug_names[d]
            if name in DRUG_PLASMA_STATE and name in SUBSTRATE_STATE:
                dist_p_idx.append(DRUG_PLASMA_STATE[name])   # plasma state (e.g. 1, 6)
                dist_l_idx.append(SUBSTRATE_STATE[name])     # liver state  (e.g. 2, 7)
                dist_k_p2l.append(float(dist_attr[k, 0]))
                dist_k_l2p.append(float(dist_attr[k, 1]))

    # --- competitive inhibition (DDI): drug competitively_inhibits enzyme ---
    # The inhibitor's liver concentration adds C_inh/Ki to the SHARED denominator
    # of its target enzyme, throttling every reaction that enzyme catalyses.
    inhib_et = ("drug", "competitively_inhibits", "enzyme")
    inhib_state, inhib_enz_local, inhib_ki, inhib_mw = [], [], [], []
    if inhib_et in data.edge_types and "edge_attr" in data[inhib_et]:
        inhib_attr = data[inhib_et].edge_attr
        for k, (d, e) in enumerate(data[inhib_et].edge_index.t().tolist()):
            name = drug_names[d]
            if name in SUBSTRATE_STATE and e in enz_group_of:
                inhib_state.append(SUBSTRATE_STATE[name])     # liver pool drives inhibition
                inhib_enz_local.append(enz_group_of[e])       # shared per-enzyme denom group
                inhib_ki.append(max(float(inhib_attr[k, 0]), 1e-6))
                inhib_mw.append(SPECIES_MW[name])

    # --- dynamic renal clearance: metabolite cleared_via reaction (k_clear) ---
    # Terminal metabolites drain by k_clear * amount into the urine sink state.
    clr_et = ("metabolite", "cleared_via", "reaction")
    clear_state, clear_k = [], []
    if clr_et in data.edge_types and "edge_attr" in data[clr_et]:
        clr_attr = data[clr_et].edge_attr
        for k, (m, _r) in enumerate(data[clr_et].edge_index.t().tolist()):
            name = met_names[m]
            if name in PRODUCT_STATE:
                clear_state.append(PRODUCT_STATE[name])
                clear_k.append(float(clr_attr[k, 0]))

    # --- GSH homeostatic regeneration params (from the endogenous-pool node) ---
    # x_static cols = [baseline_homeostatic_pool_glut, synthesis_rate, depletion_rate].
    endo_static = data["endogenous_molecule"].x_static
    gsh_baseline_mg = float(endo_static[0, 0].abs())
    k_syn_gsh = float(endo_static[0, 1].abs())

    L = lambda x: torch.tensor(x, dtype=torch.long)
    return {
        "enz_rxn_rows": L(enz_rxn_rows),
        "enz_rxn_names": [rxn_names[r] for r in enz_rxn_rows],
        # per catalytic (enzyme, reaction) edge
        "edge_rxn": L(edge_rxn),                       # -> gather the GNN Vmax factor
        "edge_enz_local": L(edge_enz_local),           # -> shared per-enzyme denominator
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
        # per-edge co-substrate gating helpers (vectorized, no scatter)
        "edge_cosub_state_all": L(edge_cosub_state),
        "edge_has_cosub": torch.tensor([1.0 if m else 0.0 for m in edge_cosub_mask], dtype=torch.float),
        "n_groups": n_groups,
        "v_liver": v_liver,
        "abs_gut": L(abs_gut),
        "abs_plasma": L(abs_plasma),
        "abs_ka": torch.tensor(abs_ka, dtype=torch.float),
        # patient-specific plasma <-> liver distribution (dynamic, from the graph).
        "dist_p_idx": L(dist_p_idx),
        "dist_l_idx": L(dist_l_idx),
        "dist_k_p2l": torch.tensor(dist_k_p2l, dtype=torch.float),
        "dist_k_l2p": torch.tensor(dist_k_l2p, dtype=torch.float),
        # competitive inhibition (DDI) wiring.
        "inhib_state": L(inhib_state),
        "inhib_enz_local": L(inhib_enz_local),
        "inhib_ki": torch.tensor(inhib_ki, dtype=torch.float),
        "inhib_mw": torch.tensor(inhib_mw, dtype=torch.float),
        # dynamic renal clearance wiring.
        "clear_state": L(clear_state),
        "clear_k": torch.tensor(clear_k, dtype=torch.float),
        "urine_state": STATE_IDX["A_urine_sink"],
        "necrosis_state": STATE_IDX["A_napqi"],  # source of the necrosis shunt
        # GSH homeostatic regeneration.
        "gsh_state": STATE_IDX["A_gsh"],
        "gsh_baseline_mg": gsh_baseline_mg,
        "k_syn_gsh": k_syn_gsh,
    }


def trajectory_to_curves(traj: torch.Tensor, v_plasma: torch.Tensor) -> torch.Tensor:
    """Map an ODE trajectory to per-metabolite concentration curves.

    Args:
        traj: [T, len(STATE_NAMES)] state amounts (mg) over time.
        v_plasma: plasma volume (L) for amount -> concentration conversion.

    Returns:
        [7, T, 2] concentrations in ng/mL, aligned with the 7 metabolite nodes;
        column 0 = metabolite concentration, column 1 = parent concentration.

    Each metabolite row reads its TRUE, dedicated ODE state index (no placeholder
    species reuse). NAPQI_glutathione has no dedicated state, so it is reported via
    the GSH-consumed proxy (GSH consumed == GSH-conjugate formed). Fully
    differentiable.
    """
    c = traj / v_plasma * NG_PER_MG_PER_L  # [T, S] ng/mL
    i_gsh = STATE_IDX["A_gsh"]
    gsh_consumed = (traj[0, i_gsh] - traj[:, i_gsh]) / v_plasma * NG_PER_MG_PER_L  # [T]
    metab = torch.stack([
        c[:, STATE_IDX["A_napqi"]],               # NAPQI
        c[:, STATE_IDX["A_liver_apap_gluc"]],     # acetaminophen_glucuronide
        c[:, STATE_IDX["A_liver_apap_sulf"]],     # acetaminophen_sulfate
        gsh_consumed,                             # NAPQI_glutathione (GSH-consumed proxy)
        c[:, STATE_IDX["A_paraxanthine"]],        # paraxanthine
        c[:, STATE_IDX["A_liver_theobromine"]],   # theobromine
        c[:, STATE_IDX["A_liver_theophylline"]],  # theophylline
    ], dim=0)           # [7, T]
    i_apap_p = STATE_IDX["A_plasma_apap"]
    i_caff_p = STATE_IDX["A_plasma_caffeine"]
    parent = torch.stack([
        c[:, i_apap_p], c[:, i_apap_p], c[:, i_apap_p], c[:, i_apap_p],  # APAP plasma
        c[:, i_caff_p], c[:, i_caff_p], c[:, i_caff_p],                  # caffeine plasma
    ], dim=0)           # [7, T]
    # Reported concentrations are non-negative (guards small fixed-step overshoots).
    return torch.stack([metab, parent], dim=-1).clamp(min=0.0)  # [7, T, 2]


class MichaelisMentenODE(nn.Module):
    """dy/dt for the APAP + caffeine system, competitive MM kinetics.

    Mechanistic, per (enzyme, reaction) catalytic edge:

        Vmax_edge = Kcat_edge * effective_abundance_enzyme * f_GNN(reaction)
        v_edge    = Vmax_edge * (C_sub / Km_edge) / D_enzyme
        D_enzyme  = 1 + sum_{edges on that enzyme} C_sub / Km_edge

    where f_GNN is the GNN's positive, dimensionless modulation factor (the GNN
    never predicts raw Vmax). Reactions catalysed by the same enzyme share a
    denominator -> mechanistic DDIs; multi-enzyme reactions sum their edge fluxes.
    Competitive inhibitors (competitively_inhibits edge) add C/Ki to that shared
    denominator. Absorption is first-order with ka from the absorbed_via edge.
    Plasma<->liver distribution rates are graph-derived per drug (distributes_to
    edge: drug Vd target x patient weight). Terminal-metabolite renal clearance is
    graph-derived (cleared_via edge, k_clear) and drains into A_urine_sink; GSH
    regenerates toward its baseline; NAPQI shunts to A_necrosis when GSH is low.
    All transfers are mg 1:1 (parent-equivalent), so total drug mass is conserved.
    """

    def __init__(self, factors: torch.Tensor, idx: dict, k_tox: float = 0.5):
        super().__init__()
        self.factors = factors                                # [R, 1] GNN Vmax modulation
        self.idx = idx
        # Renal clearance and plasma<->liver distribution rates are graph-derived
        # (live in idx); the only remaining hardcoded constant is the first-order
        # NAPQI -> necrosis (covalent binding) toxicity rate (hr⁻¹).
        self.k_tox = k_tox

    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        idx = self.idx
        # Positivity guard: amounts are physically non-negative. Reading every flux
        # off a clamped view stops the solver from feeding negative amounts back
        # into the kinetics (which otherwise drives runaway negative overshoot).
        y = y.clamp(min=0.0)
        dydt = torch.zeros_like(y)

        # --- Stage 5: competitive Michaelis-Menten, per catalytic edge ---
        # Units: amounts are mg. Convert substrate mg -> uM, run MM in uM/hr,
        # convert the molar flux back to mg/hr per species before applying it.
        V = self.idx["v_liver"]
        f = self.factors[idx["edge_rxn"], 0]                   # [E] GNN Vmax modulation
        vmax = idx["edge_vmax_base"] * f                       # [E] uM/hr (Kcat[/hr]*abundance*f)

        a_sub = y[idx["edge_sub_state"]].clamp(min=0.0)        # [E] substrate amount (mg)
        c_sub_uM = a_sub / idx["edge_mw_sub"] / V * UM_PER_MG_PER_L   # mg -> uM
        rho = c_sub_uM / idx["edge_km"]                        # [E] C/Km (uM/uM, Km in uM)

        # shared per-enzyme denominator: 1 + sum(C/Km over the enzyme's edges).
        denom_enz = torch.ones(idx["n_groups"], dtype=y.dtype).index_add(
            0, idx["edge_enz_local"], rho)
        # --- Component 1: competitive inhibition (DDI) ---
        # Add sum(C_inhibitor / Ki) to the inhibited enzyme's denominator. The
        # inhibitor amount (mg) is converted to uM with the liver volume + its MW,
        # exactly like substrates, so caffeine dynamically chokes CYP1A2 (and thus
        # APAP oxidation to NAPQI) as its liver level rises.
        if idx["inhib_state"].numel() > 0:
            a_inh = y[idx["inhib_state"]].clamp(min=0.0)       # [I] inhibitor amount (mg)
            c_inh_uM = a_inh / idx["inhib_mw"] / V * UM_PER_MG_PER_L
            rho_inh = c_inh_uM / idx["inhib_ki"]               # [I] C/Ki
            denom_enz = denom_enz.index_add(0, idx["inhib_enz_local"], rho_inh)
        denom_edge = denom_enz[idx["edge_enz_local"]]          # [E]
        v_uM_hr = vmax * rho / denom_edge                      # [E] molar flux (uM/hr)

        # Co-substrate gating: scale the whole reaction (substrate, product AND
        # co-substrate fluxes) by co-substrate availability so e.g. GSH cannot go
        # negative and NAPQI accumulates once GSH is exhausted. Only the co-substrate
        # edges enter the division, so non-cosub edges never create a 0*inf grad.
        if idx["edge_cosub_local_valid"].numel() > 0:
            loc = idx["edge_cosub_local_valid"]
            c_cosub = y[idx["edge_cosub_state_valid"]]
            avail = c_cosub / (c_cosub + COSUB_GATE_MG)
            gate = torch.ones_like(v_uM_hr)
            gate = gate.index_put((loc,), avail)
            v_uM_hr = v_uM_hr * gate

        # substrate consumption -> mg/hr (uM -> mg via the substrate's own MW).
        sub_mg_hr = v_uM_hr * idx["edge_mw_sub"] * V / UM_PER_MG_PER_L
        dydt = dydt.index_add(0, idx["edge_sub_state"], -sub_mg_hr)
        # Parent-equivalent product deposition: the product state gains EXACTLY the
        # mg removed from the substrate (substrate MW for both sides). This keeps a
        # strict mg mass balance even across MW-changing conjugations (the added
        # glucuronyl/sulfate/GSH group mass comes from untracked co-substrates).
        if idx["edge_prod_local_valid"].numel() > 0:
            loc = idx["edge_prod_local_valid"]
            prod_mg_hr = sub_mg_hr[loc]
            dydt = dydt.index_add(0, idx["edge_prod_state_valid"], prod_mg_hr)
        if idx["edge_cosub_local_valid"].numel() > 0:
            loc = idx["edge_cosub_local_valid"]
            cosub_mg_hr = v_uM_hr[loc] * idx["edge_mw_cosub_valid"] * V / UM_PER_MG_PER_L
            dydt = dydt.index_add(0, idx["edge_cosub_state_valid"], -cosub_mg_hr)

        # --- first-order absorption: ka read from the graph edge (input kinetics) ---
        if idx["abs_gut"].numel() > 0:
            abs_flux = idx["abs_ka"] * y[idx["abs_gut"]].clamp(min=0.0)
            dydt = dydt.index_add(0, idx["abs_gut"], -abs_flux)
            dydt = dydt.index_add(0, idx["abs_plasma"], abs_flux)

        # --- patient-specific plasma <-> liver distribution (vectorized) ---
        # Rates are graph-derived (drug Vd target x patient weight); one tensor op
        # routes mass for any number of drugs without per-drug branching.
        p_idx = idx["dist_p_idx"]
        l_idx = idx["dist_l_idx"]
        k_p2l = idx["dist_k_p2l"].to(y.device)
        k_l2p = idx["dist_k_l2p"].to(y.device)
        dist_flux = k_p2l * y[p_idx] - k_l2p * y[l_idx]    # leave plasma, enter liver
        dydt = dydt.index_add(0, p_idx, -dist_flux)
        dydt = dydt.index_add(0, l_idx, dist_flux)

        # --- Component 2: GSH homeostatic regeneration ---
        # The hepatic GSH pool refills toward its baseline as it is consumed by
        # NAPQI conjugation: dA_gsh/dt += k_syn * (baseline - A_gsh). GSH is an
        # endogenous regenerating species, kept outside the drug-mass invariant.
        gsh_i = idx["gsh_state"]
        dydt[gsh_i] = dydt[gsh_i] + idx["k_syn_gsh"] * (idx["gsh_baseline_mg"] - y[gsh_i])

        # --- Component 2: NAPQI -> necrosis toxicity shunt ---
        # A parallel first-order flux binds NAPQI to liver proteins. When GSH is
        # plentiful the GST conjugation dominates; once GSH is exhausted NAPQI
        # accumulates and this shunt (covalent binding / damage) takes over.
        napqi_i = idx["necrosis_state"]
        necrosis_flux = self.k_tox * y[napqi_i]
        dydt[napqi_i] = dydt[napqi_i] - necrosis_flux
        dydt[STATE_IDX["A_necrosis"]] = dydt[STATE_IDX["A_necrosis"]] + necrosis_flux

        # --- Component 3: dynamic renal clearance into the urine sink ---
        # Rates are graph-derived (metabolite cleared_via reaction, k_clear). Mass
        # leaves each terminal metabolite and is accumulated in A_urine_sink, so
        # eliminated mass is conserved rather than discarded.
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

        # Stage 4: per-reaction head -> ONE positive Vmax modulation factor per
        # reaction (dimensionless). The GNN never predicts raw Vmax; it only
        # scales the mechanistic Kcat * abundance product.
        self.param_head = nn.Linear(hidden_channels, 1)

        # Graph-derived reaction <-> state/enzyme wiring + mechanistic Vmax base.
        self.ode_idx = build_ode_index(data)

    def encode(self, data: HeteroData) -> dict:
        """Stages 1-3: edge-aware message passing, NO pooling. Returns x_dict."""
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
        """Stage 3-4: reaction-level readout + positive Vmax modulation factor.

        Returns [R, 1]: a positive, dimensionless factor per reaction node
        (bipartite topology preserved -- no global pooling). The factor scales
        the mechanistic Vmax (Kcat * effective_abundance) inside the ODE.
        """
        x_dict = self.encode(data)
        h_rxn = x_dict["reaction"]                          # [R, hidden]
        return F.softplus(self.param_head(h_rxn)) + 1e-6    # [R, 1]

    def initial_state(self, data: HeteroData) -> torch.Tensor:
        """y(t0): doses seed gut compartments; GSH seeds its pool; rest start at 0.

        Dose is read from the administration_event -> drug edge (dose_amount_mg),
        not from drug node state — it serves only as the gut initial-condition seed.
        """
        drug_names = NODE_NAMES["drug"]
        dose_by_drug: dict[str, torch.Tensor] = {}
        dose_et = ("administration_event", "releases", "drug")
        if dose_et in data.edge_types and "edge_attr" in data[dose_et]:
            for k, (_admin, d) in enumerate(data[dose_et].edge_index.t().tolist()):
                dose_by_drug[drug_names[d]] = data[dose_et].edge_attr[k, 0].clamp(min=0.0)
        zero = torch.zeros(())
        dose_apap = dose_by_drug.get("acetaminophen", zero)
        dose_caff = dose_by_drug.get("caffeine", zero)
        # Seed GSH at its homeostatic baseline (the regeneration target).
        gsh0 = data["endogenous_molecule"].x_state[0, 0].abs()
        return torch.stack([
            dose_apap, zero, zero, zero, gsh0,
            dose_caff, zero, zero, zero,
            # explicit metabolite states start empty; mass arrives via metabolism.
            zero, zero, zero, zero,
            # terminal sinks (necrosis, urine) start empty.
            zero, zero,
        ])

    def build_ode(self, factors: torch.Tensor) -> MichaelisMentenODE:
        return MichaelisMentenODE(factors, self.ode_idx)

    def forward(self, data: HeteroData, t: torch.Tensor):
        factors = self.predict_params(data)                  # [R, 1]
        y0 = self.initial_state(data)

        # Integrate the GNN-parameterized competitive MM ODE (differentiable).
        # Fixed-step rk4: the nonlinear MM saturation makes the system stiffer
        # than a linear model, so an adaptive solver (dopri5) blows up its
        # internal step count -> the stored backprop graph OOMs during training.
        # A fixed step bounds compute/memory deterministically.
        traj = odeint(self.build_ode(factors), y0, t, method="rk4",
                      options={"step_size": 0.025})  # [T, S]

        # Amounts -> per-metabolite concentration curves (ng/mL), [7, T, 2].
        v_plasma = data["compartment"].x_static[0, 0].abs().clamp(min=1.0)
        curves = trajectory_to_curves(traj, v_plasma)

        return traj, curves, factors


def main() -> None:
    data = build_dummy_graph()
    model = GNNODEModel(data, hidden_channels=32, heads=2)

    t = torch.linspace(0.0, 24.0, steps=10)
    traj, curves, factors = model(data, t)
    metabolite_pred = curves[:, -1, :]  # final-timestep slice -> [7, 2]
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

    # ---- dummy loss + backward through the ODE solver ----
    # Scale-normalize: concentrations are ~1e3-1e4 ng/mL, so an unnormalized MSE
    # produces ~1e16 gradients that overflow to NaN. Real training (train.py)
    # normalizes the same way; here we just want to confirm gradients flow.
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

    # ---- Phase 3 validation: 24 h mass balance (neutral factors) ----
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
        # Drug-derived invariant excludes the regenerating endogenous GSH pool.
        drug_idx = [i for i in range(len(STATE_NAMES)) if i != gsh_i]
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
