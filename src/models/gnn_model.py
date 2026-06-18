"""Phase 1 smoke test: prove HeteroGNN forward/backward plumbing works.

Goal is dimensional correctness only (not biological accuracy):
  - concatenate x_state + x_static per node type via lazy encoders
  - 2 heterogeneous message-passing layers (HeteroConv + SAGEConv), no edge_attr
  - two linear heads -> metabolite [7, 2] and clinical_outcome [1, 4]
  - dummy MSE loss + loss.backward() to confirm gradients flow

Run from project root:
    python -m src.models.smoke_test_gnn
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import HeteroConv, SAGEConv, Linear

from src.data.build_graph import build_dummy_graph


class HeteroGNN(nn.Module):
    def __init__(self, metadata, hidden_channels: int = 32):
        super().__init__()
        node_types, edge_types = metadata

        # Node encoders: lazy Linear(-1) handles each type's concat(state, static) width.
        self.encoders = nn.ModuleDict(
            {nt: Linear(-1, hidden_channels) for nt in node_types}
        )

        # Two basic heterogeneous message-passing layers. edge_attr is ignored.
        self.conv1 = HeteroConv(
            {et: SAGEConv((-1, -1), hidden_channels) for et in edge_types}, aggr="sum"
        )
        self.conv2 = HeteroConv(
            {et: SAGEConv((-1, -1), hidden_channels) for et in edge_types}, aggr="sum"
        )

        # Prediction heads sized to the target tensors.
        self.metabolite_head = nn.Linear(hidden_channels, 2)        # -> [num_metabolites, 2]
        self.clinical_head = nn.Linear(hidden_channels, 4)          # -> [num_outcomes, 4]

    def forward(self, x_state_dict, x_static_dict, edge_index_dict):
        # 1. encode: concat dynamic state + static params, project to hidden dim
        x_dict = {
            nt: F.relu(self.encoders[nt](torch.cat([x_state_dict[nt], x_static_dict[nt]], dim=-1)))
            for nt in self.encoders
        }

        # 2. message passing. Merge updates back so non-destination types persist
        #    (e.g. patient / protein_target are never destinations).
        out = self.conv1(x_dict, edge_index_dict)
        x_dict = {**x_dict, **{k: F.relu(v) for k, v in out.items()}}

        out = self.conv2(x_dict, edge_index_dict)
        x_dict = {**x_dict, **{k: F.relu(v) for k, v in out.items()}}

        # 3. heads
        metabolite_pred = self.metabolite_head(x_dict["metabolite"])
        clinical_pred = self.clinical_head(x_dict["clinical_outcome"])
        return metabolite_pred, clinical_pred


def main() -> None:
    data = build_dummy_graph()

    x_state_dict = {nt: data[nt].x_state for nt in data.node_types}
    x_static_dict = {nt: data[nt].x_static for nt in data.node_types}
    edge_index_dict = data.edge_index_dict

    model = HeteroGNN(data.metadata(), hidden_channels=32)

    # ---- forward pass ----
    metabolite_pred, clinical_pred = model(x_state_dict, x_static_dict, edge_index_dict)

    metabolite_y = data["metabolite"].y
    clinical_y = data["clinical_outcome"].y

    print(f"metabolite       pred {tuple(metabolite_pred.shape)} | target {tuple(metabolite_y.shape)}")
    print(f"clinical_outcome pred {tuple(clinical_pred.shape)} | target {tuple(clinical_y.shape)}")

    # ---- dummy loss + backward pass ----
    loss = F.mse_loss(metabolite_pred, metabolite_y) + F.mse_loss(clinical_pred, clinical_y)
    loss.backward()

    grad_total = sum(
        p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None
    )
    print(f"loss = {loss.item():.4f} | total gradient magnitude = {grad_total:.4f}")

    assert grad_total > 0, "No gradients flowed through the model."
    print("GNN forward and backward pass succeeded.")


if __name__ == "__main__":
    main()