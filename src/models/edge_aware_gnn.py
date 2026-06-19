"""Smoke test: GATv2Conv message passing with per-edge kinetic attributes."""

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATv2Conv, HeteroConv, Linear

from src.data.build_graph import build_dummy_graph


def edge_dim_of(data: HeteroData, edge_type) -> int | None:
    """Per-edge-type attribute width, or None if the edge has no edge_attr."""
    store = data[edge_type]
    return store.edge_attr.size(-1) if "edge_attr" in store else None


class EdgeAwareHeteroGNN(nn.Module):
    """HeteroConv + GATv2Conv; ``edge_dim`` varies by edge type (Km/Kcat width, etc.)."""

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
                        (-1, -1),
                        hidden_channels,
                        heads=heads,
                        concat=False,
                        edge_dim=edge_dim_of(data, et),
                        add_self_loops=False,  # invalid on bipartite hetero edges
                    )
                    for et in edge_types
                },
                aggr="sum",
            )

        self.conv1 = make_layer()
        self.conv2 = make_layer()

        self.metabolite_head = nn.Linear(hidden_channels, 2)
        self.clinical_head = nn.Linear(hidden_channels, 4)

    def forward(self, x_state_dict, x_static_dict, edge_index_dict, edge_attr_dict):
        x_dict = {
            nt: F.relu(self.encoders[nt](torch.cat([x_state_dict[nt], x_static_dict[nt]], dim=-1)))
            for nt in self.encoders
        }

        out = self.conv1(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
        x_dict = {**x_dict, **{k: F.relu(v) for k, v in out.items()}}

        out = self.conv2(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
        x_dict = {**x_dict, **{k: F.relu(v) for k, v in out.items()}}

        metabolite_pred = self.metabolite_head(x_dict["metabolite"])
        clinical_pred = self.clinical_head(x_dict["clinical_outcome"])
        return metabolite_pred, clinical_pred


def main() -> None:
    data = build_dummy_graph()

    x_state_dict = {nt: data[nt].x_state for nt in data.node_types}
    x_static_dict = {nt: data[nt].x_static for nt in data.node_types}
    edge_index_dict = data.edge_index_dict
    edge_attr_dict = data.edge_attr_dict

    model = EdgeAwareHeteroGNN(data, hidden_channels=32, heads=2)

    metabolite_pred, clinical_pred = model(
        x_state_dict, x_static_dict, edge_index_dict, edge_attr_dict
    )

    metabolite_y = data["metabolite"].y
    clinical_y = data["clinical_outcome"].y

    print(f"edge types with edge_attr: {len(edge_attr_dict)} / {len(data.edge_types)}")
    print(f"metabolite       pred {tuple(metabolite_pred.shape)} | target {tuple(metabolite_y.shape)}")
    print(f"clinical_outcome pred {tuple(clinical_pred.shape)} | target {tuple(clinical_y.shape)}")

    loss = F.mse_loss(metabolite_pred, metabolite_y) + F.mse_loss(clinical_pred, clinical_y)
    loss.backward()

    # lin_edge grads confirm edge_attr reached the conv layers.
    edge_param_grads = sum(
        p.grad.abs().sum().item()
        for n, p in model.named_parameters()
        if "lin_edge" in n and p.grad is not None
    )
    grad_total = sum(
        p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None
    )
    print(f"loss = {loss.item():.4f} | total grad = {grad_total:.4f} | edge-proj grad = {edge_param_grads:.4f}")

    assert grad_total > 0, "No gradients flowed through the model."
    assert edge_param_grads > 0, "edge_attr was not used (no gradient on edge projections)."
    print("Edge-attribute-aware GNN forward and backward pass succeeded; edge_attr_dict integrated.")


if __name__ == "__main__":
    main()