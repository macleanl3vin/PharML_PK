"""Export graph schema and kinetic values to Excel with unit labels."""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.build_graph import (  # noqa: E402
    DISTRIBUTION_RATES,
    EDGE_FEATURES,
    EDGE_VALUES,
    EDGES,
    NODE_FEATURES_STATE,
    NODE_FEATURES_STATIC,
    NODE_NAMES,
    NODE_VALUES,
    REACTION_TYPE,
    REACTION_TYPE_VOCAB,
    SMILES,
    TARGET_FEATURES,
    morgan_fingerprint,
)

DEFAULT_OUTPUT = os.path.expanduser("~/Downloads/PharmMLPK_Graph_Data_With_Units.xlsx")

UNIT_MAPPING = {
    "Km": "Km (μM)",
    "Ki": "Ki (μM)",
    "Kcat": "Kcat (min⁻¹)",
    "k_clear": "k_clear (hr⁻¹)",
    "absorption_rate_ka": "absorption_rate_ka (hr⁻¹)",
    "k_p2l": "k_p2l (hr⁻¹)",
    "k_l2p": "k_l2p (hr⁻¹)",
    "consumption_rate": "consumption_rate (hr⁻¹)",
    "binding_rate": "binding_rate (hr⁻¹)",
    "clearance_rate": "clearance_rate (hr⁻¹)",
    "excretion_rate": "excretion_rate (hr⁻¹)",
    "partition_rate": "partition_rate (hr⁻¹)",
    "creatinine": "creatinine (mg/dL)",
    "ALT": "ALT (U/L)",
    "AST": "AST (U/L)",
    "bilirubin": "bilirubin (mg/dL)",
    "albumin": "albumin (g/dL)",
    "current_pool_mass": "current_pool_mass (mg)",
    "synthesis_rate": "synthesis_rate (mg/hr or hr⁻¹)",
    "depletion_rate": "depletion_rate (mg/hr)",
    "activity_multiplier": "activity_multiplier (fold)",
    "PGx_phenotype_multiplier": "PGx_phenotype_multiplier (fold)",
    "is_active": "is_active (bool)",
    "is_parent_drug": "is_parent_drug (bool)",
    "is_hepatoxic": "is_hepatoxic (bool)",
    "sex_encoded": "sex_encoded (categorical)",
    "target_metabolite_ng_mL": "target_metabolite_ng_mL (ng/mL)",
    "target_parent_ng_mL": "target_parent_ng_mL (ng/mL)",
    "toxicity_label": "toxicity_label (class)",
    "initial_concentration_ng_mL": "initial_concentration_ng_mL (ng/mL)",
}

# Fields wired into the current GNN-ODE pipeline vs schema-only placeholders.
NODE_SOURCE = {
    "patient.weight_kg": "INPUT",
    "drug.molecular_weight": "INPUT",
    "drug.is_parent_drug": "INPUT",
    "drug.target_vd_L_kg": "INPUT",
    "enzyme.baseline_abundance_pmol_mg": "INPUT",
    "enzyme.PGx_phenotype_multiplier": "INPUT",
    "enzyme.is_active": "INPUT",
    "compartment.volume_L": "INPUT",
    "endogenous_molecule.current_amount_glut": "INPUT",
    "endogenous_molecule.baseline_homeostatic_pool_glut": "INPUT",
    "endogenous_molecule.synthesis_rate": "INPUT",
    "metabolite.morgan_fingerprint": "INPUT (computed from SMILES)",
    "reaction.reaction_type": "INPUT (one-hot in model)",
    "metabolite.target_metabolite_ng_mL": "TARGET",
    "metabolite.target_parent_ng_mL": "TARGET",
    "clinical_outcome.ALT": "TARGET",
    "clinical_outcome.AST": "TARGET",
    "clinical_outcome.bilirubin": "TARGET",
    "clinical_outcome.toxicity_label": "TARGET",
}

EDGE_SOURCE = {
    "dose_amount_mg": "INPUT",
    "Km": "INPUT",
    "Ki": "INPUT",
    "Kcat": "INPUT",
    "absorption_rate_ka": "INPUT",
    "k_p2l": "COMPUTED_INPUT",
    "k_l2p": "COMPUTED_INPUT",
    "activity_multiplier": "INPUT",
    "k_clear": "INPUT",
    "creatinine": "INPUT",
    "ALT": "INPUT",
    "AST": "INPUT",
    "bilirubin": "INPUT",
    "albumin": "INPUT",
    "current_pool_mass": "INPUT",
    "depletion_rate": "SCHEMA_ONLY",
    "consumption_rate": "SCHEMA_ONLY",
    "stoichiometric_yield": "SCHEMA_ONLY",
    "binding_rate": "SCHEMA_ONLY",
    "necrosis_fraction": "SCHEMA_ONLY",
    "synthesis_rate": "SCHEMA_ONLY",
    "excretion_rate": "SCHEMA_ONLY",
    "partition_rate": "SCHEMA_ONLY",
    "blood_flow_rate": "SCHEMA_ONLY",
}


def apply_units(columns: list[str]) -> list[str]:
    return [UNIT_MAPPING.get(c, c) for c in columns]


def _get(node_type: str, name: str, key: str, default: str = "") -> str | float:
    rec = NODE_VALUES.get(node_type, {}).get(name, {})
    if key not in rec:
        return default
    return rec[key]


def build_nodes_df() -> pd.DataFrame:
    rows: list[dict] = []
    for node_type, names in NODE_NAMES.items():
        static_cols = NODE_FEATURES_STATIC.get(node_type, [])
        state_cols = NODE_FEATURES_STATE.get(node_type, [])
        target_cols = TARGET_FEATURES.get(node_type, [])

        for name in names:
            row: dict = {"Node_Type": node_type, "Node_Name": name}

            for col in static_cols:
                row[col] = _get(node_type, name, col, "")

            for col in state_cols:
                row[col] = _get(node_type, name, col, "")

            for col in target_cols:
                row[col] = ""

            if node_type == "drug":
                row["target_vd_L_kg"] = _get(node_type, name, "target_vd_L_kg", "")
            if node_type == "reaction":
                row["reaction_type"] = REACTION_TYPE.get(name, "")
            if node_type in ("drug", "metabolite"):
                row["SMILES"] = SMILES.get(name, "")
                fp = morgan_fingerprint(name)
                row["morgan_fp_n_bits"] = int(fp.numel())
                row["morgan_fp_n_on"] = int(fp.sum().item())

            rows.append(row)

    df = pd.DataFrame(rows)
    preferred = [
        "Node_Type", "Node_Name", "SMILES", "reaction_type",
        "morgan_fp_n_bits", "morgan_fp_n_on",
    ]
    other = [c for c in df.columns if c not in preferred]
    df = df[[c for c in preferred if c in df.columns] + other]
    return df


def build_edges_df() -> pd.DataFrame:
    rows: list[dict] = []
    for edge_type_tuple, pairs in EDGES.items():
        src_type, rel, dst_type = edge_type_tuple
        edge_label = f"{src_type} -> {rel} -> {dst_type}"
        attr_cols = EDGE_FEATURES.get(edge_type_tuple, [])

        for src_node, dst_node in pairs:
            row: dict = {
                "Edge_Type": edge_label,
                "Source_Node": src_node,
                "Relation": rel,
                "Target_Node": dst_node,
            }
            values = EDGE_VALUES.get(edge_type_tuple, {}).get((src_node, dst_node), {})
            for col in attr_cols:
                row[col] = values.get(col, "")
            if not attr_cols:
                row["Attributes"] = "(topology only)"
            rows.append(row)

    df = pd.DataFrame(rows)
    preferred = ["Edge_Type", "Source_Node", "Relation", "Target_Node"]
    other = [c for c in df.columns if c not in preferred]
    return df[preferred + other]


def build_smiles_df() -> pd.DataFrame:
    rows = [{"Entity": name, "SMILES": smiles} for name, smiles in sorted(SMILES.items())]
    return pd.DataFrame(rows)


def build_reaction_types_df() -> pd.DataFrame:
    rows = [
        {
            "Reaction_Name": name,
            "Reaction_Type": REACTION_TYPE[name],
            **{cat: 1.0 if REACTION_TYPE[name] == cat else 0.0 for cat in REACTION_TYPE_VOCAB},
        }
        for name in NODE_NAMES["reaction"]
    ]
    return pd.DataFrame(rows)


def build_distribution_df() -> pd.DataFrame:
    rows = [
        {
            "Drug": drug,
            "target_vd_L_kg": NODE_VALUES["drug"][drug]["target_vd_L_kg"],
            "patient_weight_kg": NODE_VALUES["patient"]["patient_0"]["weight_kg"],
            "V_plasma_L": NODE_VALUES["compartment"]["plasma"]["volume_L"],
            **rates,
        }
        for drug, rates in DISTRIBUTION_RATES.items()
    ]
    return pd.DataFrame(rows)


def build_legend_df() -> pd.DataFrame:
    rows = [
        {"Category": "INPUT", "Meaning": "User/literature value fed into graph or ODE"},
        {"Category": "COMPUTED_INPUT", "Meaning": "Derived from other inputs before simulation"},
        {"Category": "GNN", "Meaning": "Learned per-reaction f_GNN (not stored in this workbook)"},
        {"Category": "ODE", "Meaning": "Time-varying mass/concentration from integration"},
        {"Category": "TARGET", "Meaning": "Ground-truth label for training/evaluation"},
        {"Category": "SCHEMA_ONLY", "Meaning": "Defined in graph schema, not wired into current ODE"},
    ]
    node_rows = [
        {"Sheet": "Nodes", "Field": k, "Source": v} for k, v in sorted(NODE_SOURCE.items())
    ]
    edge_rows = [
        {"Sheet": "Edges", "Field": k, "Source": v} for k, v in sorted(EDGE_SOURCE.items())
    ]
    return pd.DataFrame(rows), pd.DataFrame(node_rows + edge_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export graph data to Excel.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output .xlsx path")
    args = parser.parse_args()

    nodes_df = build_nodes_df()
    edges_df = build_edges_df()
    nodes_df.columns = apply_units(list(nodes_df.columns))
    edges_df.columns = apply_units(list(edges_df.columns))

    legend_df, field_legend_df = build_legend_df()

    with pd.ExcelWriter(args.output, engine="xlsxwriter") as writer:
        nodes_df.to_excel(writer, sheet_name="Nodes_and_Attributes", index=False)
        edges_df.to_excel(writer, sheet_name="Edges_and_Attributes", index=False)
        build_smiles_df().to_excel(writer, sheet_name="SMILES", index=False)
        build_reaction_types_df().to_excel(writer, sheet_name="Reaction_Types", index=False)
        build_distribution_df().to_excel(writer, sheet_name="Distribution_Rates", index=False)
        legend_df.to_excel(writer, sheet_name="Source_Legend", index=False)
        field_legend_df.to_excel(writer, sheet_name="Field_Sources", index=False)

        for sheet in writer.sheets.values():
            sheet.set_column(0, 0, 22)
            sheet.set_column(1, 3, 28)

    print(f"Exported graph workbook -> {args.output}")


if __name__ == "__main__":
    main()
