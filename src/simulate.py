"""Shared GNN-ODE integration for plotting and analysis."""

from __future__ import annotations

from pathlib import Path

import torch
from torch_geometric.data import HeteroData
from torchdiffeq import odeint

from src.data.build_graph import build_dummy_graph
from src.models.gnn_ode import GNNODEModel

DEFAULT_CHECKPOINT = Path("results/best_model.pt")


def run_simulation(
    hours: float = 24.0,
    steps: int = 200,
    step_size: float = 0.025,
    use_gnn_factors: bool = False,
    checkpoint: Path | str = DEFAULT_CHECKPOINT,
) -> tuple[torch.Tensor, torch.Tensor, HeteroData]:
    """Integrate PK ODE; return ``(t, traj, data)``. traj ``[T, 15]`` mg, t in hr.

    With ``use_gnn_factors=True``, loads ``checkpoint`` before predicting f_GNN.
    """
    data = build_dummy_graph()
    t = torch.linspace(0.0, hours, steps)
    model = GNNODEModel(data, hidden_channels=32, heads=2)

    with torch.no_grad():
        if use_gnn_factors:
            model.predict_params(data)  # materialize lazy layers before load_state_dict
            checkpoint_path = Path(checkpoint)
            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"Trained checkpoint not found at {checkpoint_path}. "
                    "Run `python -m src.train` first, or omit --use-gnn-factors "
                    "to simulate with neutral (1.0) factors."
                )
            model.load_state_dict(torch.load(checkpoint_path))
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
    return t, traj, data


def plasma_volume_L(data: HeteroData) -> float:
    return float(data["compartment"].x_static[0, 0].abs().clamp(min=1.0))


def patient_weight_kg(data: HeteroData) -> float:
    """Body weight (kg) from ``patient.x_static`` column ``weight_kg``."""
    return float(data["patient"].x_static[0, 1].abs().clamp(min=1e-3))
