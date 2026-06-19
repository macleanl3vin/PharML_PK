"""End-to-end GNN-ODE training on synthetic concentration targets.

Teacher ODE supplies labels at sparse observation times; loss backpropagates
through ``odeint`` into GNN weights. Run: ``python -m src.train``
"""

import copy
from pathlib import Path

import torch
import torch.nn.functional as F
from torchdiffeq import odeint

from src.data.build_graph import NODE_NAMES, build_dummy_graph
from src.metrics import pk_metrics_for_drug
from src.models.gnn_ode import GNNODEModel, trajectory_to_curves
from src.simulate import patient_weight_kg, plasma_volume_L

# Hidden teacher modulation factors (not shown to the model); omitted reactions → 1.0.
TRUE_FACTORS = {
    "rxn_cyp_oxidation":          1.3,
    "rxn_glucuronidation":        0.8,
    "rxn_sulfation":              1.1,
    "rxn_gsh_conjugation":        0.7,
    "rxn_caff_n3_demethylation":  1.2,
}


def make_true_factors() -> torch.Tensor:
    """Assemble the teacher's [R, 1] Vmax-modulation tensor in reaction-row order."""
    rxn_names = NODE_NAMES["reaction"]
    f = torch.ones(len(rxn_names), 1)
    for name, val in TRUE_FACTORS.items():
        f[rxn_names.index(name), 0] = val
    return f


def make_synthetic_targets(model, data, true_factors, time_points, measure_idx, noise=0.02):
    """Integrate teacher ODE with ``true_factors``; sample curves at ``measure_idx`` → ``[7, M, 2]`` ng/mL."""
    with torch.no_grad():
        y0 = model.initial_state(data)
        teacher_traj = odeint(model.build_ode(true_factors), y0, time_points, method="rk4",
                              options={"step_size": 0.0025})
        v_plasma = data["compartment"].x_static[0, 0].abs().clamp(min=1.0)
        curves = trajectory_to_curves(teacher_traj, v_plasma)  # [7, T, 2]
        target = curves[:, measure_idx, :]
        target = (target * (1.0 + noise * torch.randn_like(target))).clamp(min=0.0)
    return target.detach()


def main() -> None:
    num_epochs = 120
    learning_rate = 2e-3
    time_points = torch.linspace(0.0, 24.0, steps=100)

    measure_hours = [1, 2, 4, 8, 12, 24]
    measure_idx = [round(h / 24.0 * (len(time_points) - 1)) for h in measure_hours]

    # Single-graph overfit; replace with patient DataLoader for multi-sample training.
    data = build_dummy_graph()
    model = GNNODEModel(data, hidden_channels=32, heads=2)

    # Materialize lazy layers before Adam sees parameters.
    with torch.no_grad():
        model(data, time_points)

    true_factors = make_true_factors()
    target = make_synthetic_targets(model, data, true_factors, time_points, measure_idx)
    print(f"target shape {tuple(target.shape)} | mean {target.mean().item():.2f} ng/mL "
          f"| measurement hours {measure_hours}")

    # Normalize MSE by target scale (~1e4 ng/mL) to stabilize ODE backprop.
    scale = target.abs().amax(dim=(0, 1)).clamp(min=1.0).detach()
    v_plasma = plasma_volume_L(data)
    weight = patient_weight_kg(data)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Roll back to best weights and halve LR on NaN loss or grad norm.
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        optimizer.zero_grad()

        traj, curves, factors = model(data, time_points)
        pred = curves[:, measure_idx, :]

        metabolite_concentration_loss = F.mse_loss(pred[..., 0] / scale[0], target[..., 0] / scale[0])
        parent_concentration_loss = F.mse_loss(pred[..., 1] / scale[1], target[..., 1] / scale[1])

        loss = parent_concentration_loss + metabolite_concentration_loss

        if not torch.isfinite(loss):
            model.load_state_dict(best_state)
            for g in optimizer.param_groups:
                g["lr"] *= 0.5
            continue

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = copy.deepcopy(model.state_dict())

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        if not torch.isfinite(grad_norm):
            model.load_state_dict(best_state)
            for g in optimizer.param_groups:
                g["lr"] *= 0.5
            continue
        optimizer.step()

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"epoch {epoch:03d} | loss {loss.item():.6e} "
                f"| parent {parent_concentration_loss.item():.6e} "
                f"| metab {metabolite_concentration_loss.item():.6e} "
                f"| lr {optimizer.param_groups[0]['lr']:.1e}"
            )
            with torch.no_grad():
                for drug in ("apap", "caffeine"):
                    m = pk_metrics_for_drug(traj, time_points, drug, v_plasma, weight_kg=weight)
                    vd_at = m["vd_sys_L_kg"][measure_idx]
                    th_at = m["t_half_h"][measure_idx]
                    vd_str = ", ".join(f"{h}h:{v:.2f}" for h, v in zip(measure_hours, vd_at.tolist()))
                    th_str = ", ".join(
                        f"{h}h:{v:.2f}" if v == v else f"{h}h:—"
                        for h, v in zip(measure_hours, th_at.tolist())
                    )
                    print(f"  {drug} Vd_sys(L/kg)  {vd_str}")
                    print(f"  {drug} t½(h)         {th_str}")

    model.load_state_dict(best_state)

    checkpoint_path = Path("results/best_model.pt")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, checkpoint_path)
    print(f"saved trained weights -> {checkpoint_path}")

    print(f"\nbest loss {best_loss:.6e}")
    factors = model.predict_params(data).detach()
    rxn_names = NODE_NAMES["reaction"]
    print("\nrecovered GNN Vmax modulation factors (learned vs true):")
    for name, tv in TRUE_FACTORS.items():
        i = rxn_names.index(name)
        print(f"  {name:28s} f learned={float(factors[i, 0]):7.3f}  true={tv:7.3f}")


if __name__ == "__main__":
    main()
