import torch
from collections import defaultdict
from torch_geometric.data import HeteroData

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    _RDKIT_AVAILABLE = True
except Exception:  # rdkit is optional; the builder degrades to deterministic zeros
    _RDKIT_AVAILABLE = False

# 2048-bit Morgan fingerprints encode chemical structure for every chemical
# entity (parent drugs AND metabolites) so the GNN sees real chemistry instead
# of random noise.
MORGAN_N_BITS = 2048
MORGAN_RADIUS = 2

# Canonical SMILES for parent drugs and metabolites. Drives the deterministic
# Morgan fingerprints below. If RDKit is unavailable, or a SMILES is missing /
# unparseable, the builder falls back to a zero vector -- never random noise.
SMILES = {
    # parent drugs
    "acetaminophen": "CC(=O)Nc1ccc(O)cc1",
    "caffeine": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    # acetaminophen metabolites
    "NAPQI": "CC(=O)N=C1C=CC(=O)C=C1",
    "acetaminophen_glucuronide": "CC(=O)Nc1ccc(OC2OC(C(=O)O)C(O)C(O)C2O)cc1",
    "acetaminophen_sulfate": "CC(=O)Nc1ccc(OS(=O)(=O)O)cc1",
    "NAPQI_glutathione": "CC(=O)Nc1ccc(O)c(SCC(NC(=O)CCC(N)C(=O)O)C(=O)NCC(=O)O)c1",
    # caffeine demethylation metabolites
    "paraxanthine": "Cn1cnc2c1c(=O)[nH]c(=O)n2C",
    "theobromine": "Cn1cnc2c1c(=O)n(C)c(=O)[nH]2",
    "theophylline": "Cn1c(=O)c2[nH]cnc2n(C)c1=O",
}


def morgan_fingerprint(
    name: str, n_bits: int = MORGAN_N_BITS, radius: int = MORGAN_RADIUS
) -> torch.Tensor:
    """Deterministic Morgan fingerprint (``[n_bits]`` float) for a chemical entity.

    Uses RDKit + the canonical ``SMILES`` table when both are available. Falls
    back to a deterministic zero vector (NEVER random) when RDKit is missing or
    the SMILES is absent/unparseable, keeping the graph fully reproducible.
    """
    smiles = SMILES.get(name)
    if _RDKIT_AVAILABLE and smiles is not None:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            try:
                # Modern, non-deprecated RDKit fingerprint generator.
                from rdkit.Chem import rdFingerprintGenerator
                gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
                bitvect = gen.GetFingerprint(mol)
            except Exception:
                bitvect = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            return torch.tensor(list(bitvect), dtype=torch.float)
    return torch.zeros(n_bits, dtype=torch.float)

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
    ('drug', 'competitively_inhibits', 'enzyme'),

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
    "rxn_caff_n1_demethylation",
    "rxn_caff_n7_demethylation",
    "rxn_clearance",
]

# Mechanistic class of each reaction node. The GNN gets this as a deterministic
# one-hot feature so it can learn the differences between enzyme/reaction classes
# (e.g. an oxidation behaves differently from a glucuronidation).
REACTION_TYPE = {
    "apap_absorption": "absorption",
    "caffeine_absorption": "absorption",
    "apap_distribution": "distribution",
    "caffeine_distribution": "distribution",
    "rxn_cyp_oxidation": "oxidation",
    "rxn_glucuronidation": "glucuronidation",
    "rxn_sulfation": "sulfation",
    "rxn_gsh_conjugation": "gsh_conjugation",
    "rxn_gsh_regeneration": "gsh_regeneration",
    "rxn_covalent_binding": "covalent_binding",
    "rxn_caff_n3_demethylation": "demethylation",
    "rxn_caff_n1_demethylation": "demethylation",
    "rxn_caff_n7_demethylation": "demethylation",
    "rxn_clearance": "clearance",
}
# Stable, sorted category vocabulary -> deterministic one-hot column order.
REACTION_TYPE_VOCAB = sorted(set(REACTION_TYPE.values()))


def reaction_one_hot() -> torch.Tensor:
    """Deterministic one-hot reaction-type matrix ``[R, len(REACTION_TYPE_VOCAB)]``."""
    vocab_idx = {category: i for i, category in enumerate(REACTION_TYPE_VOCAB)}
    onehot = torch.zeros(len(REACTION_NODES), len(REACTION_TYPE_VOCAB))
    for row, name in enumerate(REACTION_NODES):
        onehot[row, vocab_idx[REACTION_TYPE[name]]] = 1.0
    return onehot

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

    # Competitive inhibition constant Ki (μM). A pure/competitive inhibitor adds
    # C_inhibitor / Ki to the shared Michaelis-Menten denominator of its target
    # enzyme. Non-inhibitor pairs default to Ki = 1e6 (effectively infinite).
    ("drug", "competitively_inhibits", "enzyme"): [
        "Ki",
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

    # First-order renal clearance rate k_clear (hr⁻¹) per terminal metabolite.
    # The ODE drains the metabolite by k_clear * amount into the urine sink state.
    ("metabolite", "cleared_via", "reaction"): [
        "k_clear",
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
        # CYP1A2 also drives the minor caffeine demethylation routes.
        ("CYP1A2", "rxn_caff_n1_demethylation"),
        ("CYP1A2", "rxn_caff_n7_demethylation"),
    ],

    # Caffeine competitively inhibits CYP1A2 (the classic APAP<->caffeine DDI):
    # it throttles every CYP1A2 reaction, including APAP oxidation to NAPQI.
    ("drug", "competitively_inhibits", "enzyme"): [
        ("caffeine", "CYP1A2"),
    ],

    ("drug", "reactant_in", "reaction"): [
        ("acetaminophen", "rxn_cyp_oxidation"),
        ("acetaminophen", "rxn_glucuronidation"),
        ("acetaminophen", "rxn_sulfation"),
        ("caffeine", "rxn_caff_n3_demethylation"),
        ("caffeine", "rxn_caff_n1_demethylation"),
        ("caffeine", "rxn_caff_n7_demethylation"),
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
        ("rxn_caff_n1_demethylation", "theobromine"),
        ("rxn_caff_n7_demethylation", "theophylline"),
        ("rxn_gsh_conjugation", "NAPQI_glutathione"),
    ],
    # Only TERMINAL metabolites are renally cleared. NAPQI is reactive (consumed
    # by GSH conjugation + necrosis, not excreted); NAPQI_glutathione mass is
    # routed to the urine sink inside the ODE via the conjugation product.
    ("metabolite", "cleared_via", "reaction"): [
        ("acetaminophen_glucuronide", "rxn_clearance"),
        ("acetaminophen_sulfate", "rxn_clearance"),
        ("paraxanthine", "rxn_clearance"),
        ("theobromine", "rxn_clearance"),
        ("theophylline", "rxn_clearance"),
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
            # First-order homeostatic regeneration k_syn (hr⁻¹): the ODE adds
            # synthesis_rate * (baseline - current) so the pool refills as it is
            # consumed by NAPQI conjugation.
            "synthesis_rate": 0.1,
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
        # Caffeine demethylation split (CYP1A2). Kcat scaled to the clinical
        # formation ratio paraxanthine : theobromine : theophylline ~ 84 : 12 : 4.
        ("CYP1A2", "rxn_caff_n3_demethylation"): {"Km": 500.0,  "Ki": 1e6, "Kcat": 2.9},
        ("CYP1A2", "rxn_caff_n1_demethylation"): {"Km": 500.0,  "Ki": 1e6, "Kcat": 0.41},
        ("CYP1A2", "rxn_caff_n7_demethylation"): {"Km": 500.0,  "Ki": 1e6, "Kcat": 0.14},
    },
    ("drug", "competitively_inhibits", "enzyme"): {
        ("caffeine", "CYP1A2"): {"Ki": 150.0},  # μM
    },
    # Terminal-metabolite renal clearance rates (hr⁻¹) -> urine sink.
    ("metabolite", "cleared_via", "reaction"): {
        ("acetaminophen_glucuronide", "rxn_clearance"): {"k_clear": 0.5},
        ("acetaminophen_sulfate", "rxn_clearance"):     {"k_clear": 0.5},
        ("paraxanthine", "rxn_clearance"):              {"k_clear": 0.3},
        ("theobromine", "rxn_clearance"):               {"k_clear": 0.2},
        ("theophylline", "rxn_clearance"):              {"k_clear": 0.2},
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
def _build_x_state(node_type, value_table):
    """Deterministic dynamic-state features (seeded at t0); zeros when unpopulated."""
    names = NODE_NAMES[node_type]
    state_cols = NODE_FEATURES_STATE.get(node_type, [])
    if len(state_cols) == 0:
        return torch.zeros(len(names), 0)
    table = value_table.get(node_type, {})
    rows = [[_to_float(table.get(name, {}).get(c, FILL)) for c in state_cols] for name in names]
    return torch.tensor(rows, dtype=torch.float).reshape(len(names), len(state_cols))


def build_node_tensors(node_type, value_table):
    """Return (x_state, x_static) for a node type, ordered by NODE_NAMES.

    No randomness anywhere in the graph: metabolites carry deterministic 2048-bit
    Morgan fingerprints, reactions carry a one-hot reaction-type vector, and every
    other static feature is read from the value table or zero-filled. Random noise
    would destroy the gradient signal, so it is never used.
    """
    names = NODE_NAMES[node_type]
    static_cols = NODE_FEATURES_STATIC.get(node_type, [])

    x_state = _build_x_state(node_type, value_table)

    if node_type == "metabolite":
        # Treat metabolites exactly like parent drugs: real chemistry via Morgan FP.
        x_static = torch.stack([morgan_fingerprint(name) for name in names])
    elif node_type == "reaction":
        x_static = reaction_one_hot()
    elif node_type in value_table and len(static_cols) > 0:
        table = value_table[node_type]
        rows = [[_to_float(table.get(name, {}).get(c, FILL)) for c in static_cols] for name in names]
        x_static = torch.tensor(rows, dtype=torch.float).reshape(len(names), len(static_cols))
    else:
        # Unpopulated structural nodes (administration_event, protein_target,
        # clinical_outcome): deterministic zeros, never randn. Keep >=1 column so
        # the lazy encoder always receives a feature dimension.
        x_static = torch.zeros(len(names), max(len(static_cols), 1))

    # Guarantee the encoder input (concat of state + static) is never empty.
    if x_state.size(1) == 0 and x_static.size(1) == 0:
        x_static = torch.zeros(len(names), 1)

    return x_state, x_static
def build_target_tensor(node_type):
    """Placeholder y. Will become a [N, T] / [N, T, k] time-series later."""
    cols = TARGET_FEATURES.get(node_type)
    if not cols:
        return None
    return torch.zeros(len(NODE_NAMES[node_type]), len(cols))  # -> [N, T] in Phase 5
def build_edge_attr(edge_type, pairs, value_table):
    """Map kinetic constants onto edges; deterministic zero fallback if unpopulated."""
    cols = EDGE_FEATURES.get(edge_type)
    if not cols:
        return None
    if edge_type not in value_table:
        return torch.zeros(len(pairs), len(cols))
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

