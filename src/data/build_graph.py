import torch
from collections import defaultdict
from torch_geometric.data import HeteroData

node_types = [
    'patient',
    'administration_event',
    'drug',
    'compartment',
    'enzyme',
    'reaction',
    'metabolite',
    'endogenous_molecule',
    'protein_target',
    'clinical_outcome'
]

edge_types = [
    ('patient', 'receives', 'administration_event'),
    ('patient', 'expresses', 'enzyme'),

    ('administration_event', 'releases', 'drug'),

    ('drug', 'absorbed_via', 'reaction'),
    ('reaction', 'enters', 'compartment'),

    ('compartment', 'supplies_mass_to', 'reaction'),
    ('reaction', 'supplies_mass_to', 'compartment'),

    ('reaction', 'partitions_into', 'compartment'),
    ('compartment', 'partitions_into', 'reaction'),

    ('reaction', 'restores_mass_to', 'endogenous_molecule'),
    ('endogenous_molecule', 'depleted_by', 'reaction'),
    ('metabolite', 'reactant_in', 'reaction'),

    ('protein_target', 'reactant_in', 'reaction'),

    ('compartment', 'contains', 'enzyme'),
    ('enzyme', 'catalyzes', 'reaction'),

    ('drug', 'distributes_to', 'compartment'),

    ('drug', 'reactant_in', 'reaction'),
    ('endogenous_molecule', 'reactant_in', 'reaction'),

    ('reaction', 'produces', 'metabolite'),
    ('metabolite', 'cleared_via', 'reaction'),
    ('reaction', 'excretes_to', 'compartment'),

    ('compartment', 'measured_liver_panel', 'clinical_outcome'),
    ('compartment', 'measured_plasma_panel', 'clinical_outcome')
]

DRUG_NODES = [
    "acetaminophen",
    "caffeine",
]

COMPARTMENT_NODES = [
    "plasma",
    "liver",
    "urine_sink",
]

ENZYME_NODES = [
    "CYP2E1",
    "CYP1A2",
    "CYP3A4",
    "UGT1A1",
    "SULT1A1",
    "GST",
]

REACTION_NODES = [
    "apap_absorption",
    "caffeine_absorption",
    "apap_distribution",
    "caffeine_distribution",
    "rxn_cyp_oxidation",
    "rxn_glucuronidation",
    "rxn_sulfation",
    "rxn_gsh_conjugation",
    "rxn_gsh_regeneration",
    "rxn_covalent_binding",
    "rxn_caff_n3_demethylation",
    "rxn_clearance",
]

METABOLITE_NODES = [
    "NAPQI",
    "acetaminophen_glucuronide",
    "acetaminophen_sulfate",
    "NAPQI_glutathione",
    "paraxanthine",
    "theobromine",
    "theophylline",
]

ENDOGENOUS_MOLECULE_NODES = [
    "glutathione",
]

PROTEIN_TARGET_NODES = [
    "hepatic_proteins",
]

PATIENT_NODES = [
    "patient_0",
]

ADMINISTRATION_EVENT_NODES = [
    "admin_event_0",
]

CLINICAL_OUTCOME_NODES = [
    "clinical_outcome_0",
]

NODE_NAMES = {
    "patient": PATIENT_NODES,
    "administration_event": ADMINISTRATION_EVENT_NODES,
    "drug": DRUG_NODES,
    "compartment": COMPARTMENT_NODES,
    "enzyme": ENZYME_NODES,
    "reaction": REACTION_NODES,
    "metabolite": METABOLITE_NODES,
    "endogenous_molecule": ENDOGENOUS_MOLECULE_NODES,
    "protein_target": PROTEIN_TARGET_NODES,
    "clinical_outcome": CLINICAL_OUTCOME_NODES,
}

node_to_idx = {
    node_type: {node_name: i for i, node_name in enumerate(names)}
    for node_type, names in NODE_NAMES.items()
}

EDGE_FEATURES = {
    ("patient", "receives", "administration_event"): [
        "creatinine",
        "ALT",
        "AST",
        "bilirubin",
        "albumin",
    ],

    ("administration_event", "releases", "drug"): [
        "dose_amount_mg",  # mg
    ],

    # Units must match gnn_ode.build_ode_index / MichaelisMentenODE:
    #   Km, Ki  -> micromolar (μM)
    #   Kcat    -> min⁻¹  (×60 in ODE → hr⁻¹)
    ("enzyme", "catalyzes", "reaction"): [
        "Km",
        "Ki",
        "Kcat",
    ],

    # First-order oral absorption rate constant ka (per hour). Lives on the
    # drug -> absorption-reaction edge; the ODE reads it directly (NOT metabolism).
    ("drug", "absorbed_via", "reaction"): [
        "absorption_rate_ka",
    ],

    # Patient-specific plasma<->liver distribution rates (per hour). Computed
    # dynamically from drug Vd target x patient weight; the ODE reads them
    # directly to route mass between the plasma and liver state compartments.
    ("drug", "distributes_to", "compartment"): [
        "k_p2l",
        "k_l2p",
    ],

    ("patient", "expresses", "enzyme"): [
        "activity_multiplier",
    ],

    ("drug", "reactant_in", "reaction"): [
        "consumption_rate",
    ],

    ("metabolite", "reactant_in", "reaction"): [
        "consumption_rate",
    ],

    ("reaction", "produces", "metabolite"): [
        "stoichiometric_yield",
    ],

    ("protein_target", "reactant_in", "reaction"): [
        "binding_rate",
        "necrosis_fraction",
    ],

    ("reaction", "restores_mass_to", "endogenous_molecule"): [
        "synthesis_rate",
    ],

    ("endogenous_molecule", "depleted_by", "reaction"): [
        "depletion_rate",
        "current_pool_mass",
    ],

    ("metabolite", "cleared_via", "reaction"): [
        "clearance_rate",
    ],

    ("reaction", "excretes_to", "compartment"): [
        "excretion_rate",
    ],

    ("reaction", "partitions_into", "compartment"): [
        "partition_rate",
        "blood_flow_rate",
    ],

    ("compartment", "partitions_into", "reaction"): [
        "partition_rate",
        "blood_flow_rate",
    ],
}

EDGES = {
    ("patient", "receives", "administration_event"): [("patient_0", "admin_event_0")],
    ("patient", "expresses", "enzyme"): [("patient_0", e) for e in ENZYME_NODES],

    ("administration_event", "releases", "drug"): [
        ("admin_event_0", "acetaminophen"), ("admin_event_0", "caffeine"),
    ],
    ("drug", "absorbed_via", "reaction"): [
        ("acetaminophen", "apap_absorption"), ("caffeine", "caffeine_absorption"),
    ],
    ("reaction", "enters", "compartment"): [
        ("apap_absorption", "plasma"), ("caffeine_absorption", "plasma"),
    ],

    ("compartment", "supplies_mass_to", "reaction"): [
        ("plasma", "apap_distribution"), ("plasma", "caffeine_distribution"),
    ],
    ("reaction", "supplies_mass_to", "compartment"): [
        ("apap_distribution", "plasma"), ("caffeine_distribution", "plasma"),
    ],
    ("reaction", "partitions_into", "compartment"): [
        ("apap_distribution", "liver"), ("caffeine_distribution", "liver"),
    ],
    ("compartment", "partitions_into", "reaction"): [
        ("liver", "apap_distribution"), ("liver", "caffeine_distribution"),
    ],
    ("reaction", "paritions_into", "compartment"): [
        ("apap_distribution", "plasma"), ("caffeine_distribution", "plasma"),
    ],

    ("drug", "distributes_to", "compartment"): [
        ("acetaminophen", "liver"), ("caffeine", "liver"),
    ],

    ("compartment", "contains", "enzyme"): [("liver", e) for e in ENZYME_NODES],
    ("enzyme", "catalyzes", "reaction"): [
        ("CYP2E1", "rxn_cyp_oxidation"), ("CYP3A4", "rxn_cyp_oxidation"),
        ("CYP1A2", "rxn_cyp_oxidation"),
        ("UGT1A1", "rxn_glucuronidation"), ("SULT1A1", "rxn_sulfation"),
        ("GST", "rxn_gsh_conjugation"), ("CYP1A2", "rxn_caff_n3_demethylation"),
    ],

    ("drug", "reactant_in", "reaction"): [
        ("acetaminophen", "rxn_cyp_oxidation"),
        ("acetaminophen", "rxn_glucuronidation"),
        ("acetaminophen", "rxn_sulfation"),
        ("caffeine", "rxn_caff_n3_demethylation"),
    ],
    ("metabolite", "reactant_in", "reaction"): [
        ("NAPQI", "rxn_gsh_conjugation"), ("NAPQI", "rxn_covalent_binding"),
    ],
    ("protein_target", "reactant_in", "reaction"): [
        ("hepatic_proteins", "rxn_covalent_binding"),
    ],
    ("endogenous_molecule", "reactant_in", "reaction"): [
        ("glutathione", "rxn_gsh_conjugation"),
    ],
    ("endogenous_molecule", "depleted_by", "reaction"): [
        ("glutathione", "rxn_gsh_conjugation"),
    ],
    ("reaction", "restores_mass_to", "endogenous_molecule"): [
        ("rxn_gsh_regeneration", "glutathione"),
    ],

    ("reaction", "produces", "metabolite"): [
        ("rxn_cyp_oxidation", "NAPQI"),
        ("rxn_glucuronidation", "acetaminophen_glucuronide"),
        ("rxn_sulfation", "acetaminophen_sulfate"),
        ("rxn_caff_n3_demethylation", "paraxanthine"),
        ("rxn_gsh_conjugation", "NAPQI_glutathione"),
    ],
    ("metabolite", "cleared_via", "reaction"): [
        (m, "rxn_clearance") for m in METABOLITE_NODES
    ],
    ("reaction", "excretes_to", "compartment"): [("rxn_clearance", "urine_sink")],

    ("compartment", "measured_liver_panel", "clinical_outcome"): [
        ("liver", "clinical_outcome_0"),
    ],
    ("compartment", "measured_plasma_panel", "clinical_outcome"): [
        ("plasma", "clinical_outcome_0"),
    ],
}

# ============================================================
# PHASE 3: feature schema split
# ============================================================
FILL = 0.0  # default for missing values (use float("nan") if you prefer)
NODE_FEATURES_STATE = {
    "endogenous_molecule": ["current_amount_glut"],
    "protein_target": ["current_amount_proteins"],
    "metabolite": ["initial_concentration_ng_mL"],  # seeded 0 at t0; unused by GNN-ODE (ODE state vector owns PK)
}

NODE_FEATURES_STATIC = {
    "patient": [
        "age_yrs", "weight_kg", "height_cm", "sex_encoded", "pH",
    ],
    "administration_event": ["route_of_admin", "time_of_admin"],
    "drug": ["molecular_weight", "is_parent_drug"],
    "enzyme": ["PGx_phenotype_multiplier", "is_active", "protein_half_life_hrs", "baseline_abundance_pmol_mg"],
    "reaction": ["E_k"],
    "metabolite": ["is_hepatoxic", "Molecular_weight_g_mol"],
    "endogenous_molecule": ["baseline_homeostatic_pool_glut", "synthesis_rate", "depletion_rate"],
    "compartment": ["volume_L"],
}
# Targets are NEVER fed into x (prevents leakage). -> data[nt].y
TARGET_FEATURES = {
    "clinical_outcome": ["ALT", "AST", "bilirubin", "toxicity_label"],
    "metabolite": ["target_metabolite_ng_mL", "target_parent_ng_mL"],
}
# ============================================================
# STEP 3: real-value scaffolds (populate incrementally)
# ============================================================
NODE_VALUES = {
    "patient": {
        # Body weight drives the patient-specific tissue-distribution volumes
        # (see the distribution-rate calculation block below).
        "patient_0": {"weight_kg": 70.0},
    },
    "drug": {
        # target_vd_L_kg: literature steady-state Vd (L/kg). Combined with patient
        # weight it sets the apparent Vd the plasma<->liver transfer must reproduce.
        "acetaminophen": {"molecular_weight": 151.16, "is_parent_drug": 1.0, "target_vd_L_kg": 0.9},
        "caffeine":      {"molecular_weight": 194.19, "is_parent_drug": 1.0, "target_vd_L_kg": 0.7},
    },
    "enzyme": {
        # baseline_abundance_pmol_mg, PGx multiplier, is_active, half-life (hrs)
        "CYP2E1":  {"baseline_abundance_pmol_mg": 49.0,  "PGx_phenotype_multiplier": 1.0, "is_active": 1.0, "protein_half_life_hrs": 27.0},
        "CYP1A2":  {"baseline_abundance_pmol_mg": 52.0,  "PGx_phenotype_multiplier": 1.0, "is_active": 1.0, "protein_half_life_hrs": 39.0},
        "CYP3A4":  {"baseline_abundance_pmol_mg": 137.0, "PGx_phenotype_multiplier": 1.0, "is_active": 1.0, "protein_half_life_hrs": 70.0},
        "UGT1A1":  {"baseline_abundance_pmol_mg": 70.0,  "PGx_phenotype_multiplier": 1.0, "is_active": 1.0, "protein_half_life_hrs": 30.0},
        "SULT1A1": {"baseline_abundance_pmol_mg": 40.0,  "PGx_phenotype_multiplier": 1.0, "is_active": 1.0, "protein_half_life_hrs": 24.0},
        "GST":     {"baseline_abundance_pmol_mg": 100.0, "PGx_phenotype_multiplier": 1.0, "is_active": 1.0, "protein_half_life_hrs": 48.0},
    },
    "compartment": {
        # Distribution volumes (L). Liver volume drives the metabolism mg<->uM conversion.
        "plasma":     {"volume_L": 3.0},
        "liver":      {"volume_L": 1.5},
        "urine_sink": {"volume_L": 1.0},
    },
    "endogenous_molecule": {
        # Hepatic glutathione pool (mg). ~10 umol/g liver * 1500 g * 307 mg/mmol.
        "glutathione": {
            "current_amount_glut": 3000.0,
            "baseline_homeostatic_pool_glut": 3000.0,
            "synthesis_rate": 0.0,
            "depletion_rate": 0.0,
        },
    },
}
# ============================================================
# Patient-specific plasma<->liver distribution rates (drug-agnostic)
# ------------------------------------------------------------
# The apparent Vd is moved OUT of the hardcoded ODE and INTO the graph: each
# drug's plasma->liver rate k_p2l is derived from its literature Vd target and
# the patient's body weight, so a heavier patient automatically gets wider
# tissue distribution. With k_l2p held fixed, a 2-compartment system at
# pseudo-equilibrium gives  Vd_apparent = V_plasma_L * (1 + k_p2l / k_l2p),
# hence  k_p2l = (Vd_target_L / V_plasma_L - 1) * k_l2p.
# ============================================================
PATIENT_WEIGHT_KG = NODE_VALUES["patient"]["patient_0"]["weight_kg"]
V_PLASMA_L = NODE_VALUES["compartment"]["plasma"]["volume_L"]
K_L2P_FIXED = 1.0


def _k_p2l_for_drug(drug_name: str) -> float:
    """Plasma->liver rate (hr⁻¹) reproducing the drug's weight-scaled Vd target."""
    vd_target_L = NODE_VALUES["drug"][drug_name]["target_vd_L_kg"] * PATIENT_WEIGHT_KG
    return (vd_target_L / V_PLASMA_L - 1.0) * K_L2P_FIXED


DISTRIBUTION_RATES = {
    name: {"k_p2l": _k_p2l_for_drug(name), "k_l2p": K_L2P_FIXED}
    for name in ("acetaminophen", "caffeine")
}

# Catalytic edge units (must match gnn_ode.py):
#   Km, Ki  -> μM
#   Kcat    -> min⁻¹
#   ka      -> hr⁻¹
EDGE_VALUES = {
    ("drug", "distributes_to", "compartment"): {
        ("acetaminophen", "liver"): DISTRIBUTION_RATES["acetaminophen"],
        ("caffeine", "liver"):      DISTRIBUTION_RATES["caffeine"],
    },
    ("patient", "receives", "administration_event"): {
        ("patient_0", "admin_event_0"): {
            "creatinine": 1.0, "ALT": 25.0, "AST": 22.0,
            "bilirubin": 0.8, "albumin": 4.2,
        },
    },
    ("administration_event", "releases", "drug"): {
        ("admin_event_0", "acetaminophen"): {"dose_amount_mg": 1000.0},
        ("admin_event_0", "caffeine"):      {"dose_amount_mg": 200.0},
    },
    ("endogenous_molecule", "depleted_by", "reaction"): {
        ("glutathione", "rxn_gsh_conjugation"): {
            "depletion_rate": 0.0,
            "current_pool_mass": 3000.0,
        },
    },
    ("enzyme", "catalyzes", "reaction"): {
        ("CYP2E1", "rxn_cyp_oxidation"):          {"Km": 1290.0, "Ki": 1e6, "Kcat": 4.2},
        ("CYP3A4", "rxn_cyp_oxidation"):          {"Km": 6890.0, "Ki": 1e6, "Kcat": 2.1},
        ("CYP1A2", "rxn_cyp_oxidation"):          {"Km": 2700.0, "Ki": 1e6, "Kcat": 1.8},
        ("UGT1A1", "rxn_glucuronidation"):        {"Km": 3500.0, "Ki": 1e6, "Kcat": 5.0},
        ("SULT1A1", "rxn_sulfation"):             {"Km": 250.0,  "Ki": 1e6, "Kcat": 3.3},
        ("GST", "rxn_gsh_conjugation"):           {"Km": 900.0,  "Ki": 1e6, "Kcat": 6.1},
        ("CYP1A2", "rxn_caff_n3_demethylation"): {"Km": 500.0,  "Ki": 1e6, "Kcat": 2.9},
    },
    ("drug", "absorbed_via", "reaction"): {
        ("acetaminophen", "apap_absorption"):    {"absorption_rate_ka": 1.0},
        ("caffeine", "caffeine_absorption"):     {"absorption_rate_ka": 1.2},
    },
    ("patient", "expresses", "enzyme"): {
        ("patient_0", "CYP2E1"):  {"activity_multiplier": 1.0},
        ("patient_0", "CYP1A2"):  {"activity_multiplier": 1.0},
        ("patient_0", "CYP3A4"):  {"activity_multiplier": 1.0},
        ("patient_0", "UGT1A1"):  {"activity_multiplier": 1.0},
        ("patient_0", "SULT1A1"): {"activity_multiplier": 1.0},
        ("patient_0", "GST"):     {"activity_multiplier": 1.0},
    },
}
# ============================================================
# STEP 4: tensor builders
# ============================================================
def _to_float(v, default=FILL):
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        return float(v)
    return default  # strings / datetime / None -> encode later
def build_node_tensors(node_type, value_table):
    """Return (x_state, x_static) for a node type, ordered by NODE_NAMES."""
    names = NODE_NAMES[node_type]
    state_cols = NODE_FEATURES_STATE.get(node_type, [])
    static_cols = NODE_FEATURES_STATIC.get(node_type, [])
    # fallback so data.validate() still passes for unpopulated types
    if node_type not in value_table:
        x_state = torch.randn(len(names), max(len(state_cols), 1))
        x_static = torch.randn(len(names), max(len(static_cols), 1))
        return x_state, x_static
    table = value_table[node_type]
    state_rows, static_rows = [], []
    for name in names:
        rec = table.get(name, {})
        state_rows.append([_to_float(rec.get(c, FILL)) for c in state_cols])
        static_rows.append([_to_float(rec.get(c, FILL)) for c in static_cols])
    if len(state_cols) == 0:
        x_state = torch.zeros(len(names), 0)
    else:
        x_state = torch.tensor(state_rows, dtype=torch.float).reshape(len(names), len(state_cols))
    x_static = torch.tensor(static_rows, dtype=torch.float).reshape(len(names), len(static_cols))
    return x_state, x_static
def build_target_tensor(node_type):
    """Placeholder y. Will become a [N, T] / [N, T, k] time-series later."""
    cols = TARGET_FEATURES.get(node_type)
    if not cols:
        return None
    return torch.zeros(len(NODE_NAMES[node_type]), len(cols))  # -> [N, T] in Phase 5
def build_edge_attr(edge_type, pairs, value_table):
    """Map kinetic constants onto edges; randn fallback if not yet populated."""
    cols = EDGE_FEATURES.get(edge_type)
    if not cols:
        return None
    if edge_type not in value_table:
        return torch.randn(len(pairs), len(cols))
    table = value_table[edge_type]
    rows = [[_to_float(table.get((s, d), {}).get(c, FILL)) for c in cols] for s, d in pairs]
    return torch.tensor(rows, dtype=torch.float).reshape(len(pairs), len(cols))

CATALYTIC_EDGE_TYPE = ("enzyme", "catalyzes", "reaction")

def validate_catalytic_edge_values(edge_values=EDGE_VALUES) -> None:
    """Assert catalytic kinetics are positive and finite (Km μM, Kcat min⁻¹)."""
    table = edge_values.get(CATALYTIC_EDGE_TYPE, {})
    for pair, rec in table.items():
        km = rec.get("Km")
        kcat = rec.get("Kcat")
        assert km is not None and km > 0, f"{pair}: Km must be positive (μM), got {km!r}"
        assert kcat is not None and kcat > 0, f"{pair}: Kcat must be positive (min⁻¹), got {kcat!r}"
        for key in ("Km", "Ki", "Kcat"):
            if key in rec:
                assert torch.isfinite(torch.tensor(float(rec[key]))), f"{pair}: {key} must be finite"

# ============================================================
# STEP 4/5: assemble the graph
# ============================================================
def build_dummy_graph(node_values=NODE_VALUES, edge_values=EDGE_VALUES) -> HeteroData:
    validate_catalytic_edge_values(edge_values)
    data = HeteroData()
    # --- nodes: separate dynamic state vs static params, isolate targets ---
    for nt in node_types:
        data[nt].num_nodes = len(NODE_NAMES[nt])
        data[nt].node_names = NODE_NAMES[nt]
        x_state, x_static = build_node_tensors(nt, node_values)
        data[nt].x_state = x_state
        data[nt].x_static = x_static
        y = build_target_tensor(nt)
        if y is not None:
            data[nt].y = y
    # --- edges: real connectivity + kinetic edge_attr ---
    for (src, rel, dst), pairs in EDGES.items():
        s = [node_to_idx[src][a] for a, _ in pairs]
        d = [node_to_idx[dst][b] for _, b in pairs]
        data[src, rel, dst].edge_index = torch.tensor([s, d], dtype=torch.long)
        ea = build_edge_attr((src, rel, dst), pairs, edge_values)
        if ea is not None:
            data[src, rel, dst].edge_attr = ea
    data.validate(raise_on_error=True)
    return data

def validate_graph(data):
    print("Running graph validation...")

    # Built-in PyG validation
    data.validate(raise_on_error=True)

    # Validate node tensors
    for node_type in data.node_types:
        store = data[node_type]

        assert store.num_nodes > 0, f"{node_type} has no nodes"

        if hasattr(store, "x_state"):
            assert torch.isfinite(store.x_state).all(), f"{node_type}.x_state has NaN or inf"
            assert store.x_state.size(0) == store.num_nodes, f"{node_type}.x_state row mismatch"

        if hasattr(store, "x_static"):
            assert torch.isfinite(store.x_static).all(), f"{node_type}.x_static has NaN or inf"
            assert store.x_static.size(0) == store.num_nodes, f"{node_type}.x_static row mismatch"

        if hasattr(store, "y"):
            assert torch.isfinite(store.y).all(), f"{node_type}.y has NaN or inf"
            assert store.y.size(0) == store.num_nodes, f"{node_type}.y row mismatch"

    # Validate edge tensors
    for edge_type in data.edge_types:
        store = data[edge_type]
        src_type, rel_type, dst_type = edge_type

        edge_index = store.edge_index
        num_edges = edge_index.size(1)

        assert edge_index.size(0) == 2, f"{edge_type} edge_index must have shape [2, num_edges]"

        if num_edges > 0:
            assert edge_index[0].max().item() < data[src_type].num_nodes, f"{edge_type} source index out of range"
            assert edge_index[1].max().item() < data[dst_type].num_nodes, f"{edge_type} destination index out of range"

        if hasattr(store, "edge_attr"):
            assert store.edge_attr.size(0) == num_edges, f"{edge_type} edge_attr row mismatch"
            assert torch.isfinite(store.edge_attr).all(), f"{edge_type}.edge_attr has NaN or inf"
            if edge_type == CATALYTIC_EDGE_TYPE:
                km, kcat = store.edge_attr[:, 0], store.edge_attr[:, 2]
                assert (km > 0).all(), f"{edge_type}: Km must be positive (μM)"
                assert (kcat > 0).all(), f"{edge_type}: Kcat must be positive (min⁻¹)"

    print("Graph validation passed.")


if __name__ == "__main__":
    data = build_dummy_graph()
    print(data)
    validate_graph(data)
    print("drug x_static cols :", NODE_FEATURES_STATIC["drug"])
    print("admin releases dose:", data["administration_event", "releases", "drug"].edge_attr)
    print("enzyme catalyzes attr [0]:", data["enzyme", "catalyzes", "reaction"].edge_attr[0])

