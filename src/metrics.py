"""NCA-style PK metrics from integrated ODE trajectories (APAP, caffeine).

Inputs: traj ``[T, ≥9]`` amounts (mg), t (hr), v_plasma (L).
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

from src.models.gnn_ode import STATE_IDX

DrugKey = Literal["apap", "caffeine"]

_DRUG_INDICES: dict[DrugKey, tuple[int, int, int]] = {
    "apap": (
        STATE_IDX["A_gut_apap"],
        STATE_IDX["A_plasma_apap"],
        STATE_IDX["A_liver_apap"],
    ),
    "caffeine": (
        STATE_IDX["A_gut_caffeine"],
        STATE_IDX["A_plasma_caffeine"],
        STATE_IDX["A_liver_caffeine"],
    ),
}

LN2 = 0.6931471805599453


def _resolve_drug(drug: DrugKey) -> tuple[int, int, int]:
    return _DRUG_INDICES[drug]


def _central_derivative(t: Tensor, y: Tensor) -> Tensor:
    """Central finite difference dy/dt; endpoints are NaN. t, y: [T]."""
    if t.ndim != 1 or y.ndim != 1 or t.shape != y.shape:
        raise ValueError(f"t and y must be 1-D with same length; got {t.shape}, {y.shape}")
    if t.numel() < 3:
        return torch.full_like(y, float("nan"))

    dy = torch.full_like(y, float("nan"))
    dy[1:-1] = (y[2:] - y[:-2]) / (t[2:] - t[:-2])
    return dy


def parent_amounts(traj: Tensor, drug: DrugKey) -> dict[str, Tensor]:
    """Return gut, plasma, liver, sys, track — each [T], in mg."""
    i_gut, i_plasma, i_liver = _resolve_drug(drug)

    if traj.ndim != 2 or traj.size(-1) < 9:
        raise ValueError(f"traj must be [T, >=9]; got {tuple(traj.shape)}")

    gut = traj[:, i_gut].clamp(min=0.0)
    plasma = traj[:, i_plasma].clamp(min=0.0)
    liver = traj[:, i_liver].clamp(min=0.0)

    return {
        "gut": gut,
        "plasma": plasma,
        "liver": liver,
        "sys": plasma + liver,
        "track": gut + plasma + liver,
    }


def plasma_concentration_mg_L(A_plasma: Tensor, v_plasma_L: Tensor | float) -> Tensor:
    """Plasma concentration C_p = A_plasma / V_p (mg/L)."""
    v = torch.as_tensor(v_plasma_L, dtype=A_plasma.dtype, device=A_plasma.device)
    v = v.abs().clamp(min=1e-9)
    return A_plasma.clamp(min=0.0) / v


def plasma_concentration_ng_mL(A_plasma: Tensor, v_plasma_L: Tensor | float) -> Tensor:
    """Plasma concentration in ng/mL (matches trajectory_to_curves)."""
    return plasma_concentration_mg_L(A_plasma, v_plasma_L) * 1000.0


def _plasma_conc_with_floor(
    A_plasma: Tensor,
    v_plasma_L: Tensor | float,
    *,
    c_min: float,
) -> tuple[Tensor, Tensor]:
    """Return (C_p mg/L, valid_mask) with numerical floor on concentration."""
    C_p = plasma_concentration_mg_L(A_plasma, v_plasma_L)
    valid = C_p >= c_min
    C_p_safe = torch.where(valid, C_p, torch.full_like(C_p, c_min))
    return C_p_safe, valid


def vd_sys(
    t: Tensor,
    A_plasma: Tensor,
    A_liver: Tensor,
    v_plasma_L: Tensor | float,
    *,
    c_min: float = 1e-6,
) -> Tensor:
    """Apparent systemic Vd(t) = (A_plasma + A_liver) / C_p [L]. Systemic = plasma + liver."""
    del t
    C_p, valid = _plasma_conc_with_floor(A_plasma, v_plasma_L, c_min=c_min)
    A_sys = A_plasma.clamp(min=0.0) + A_liver.clamp(min=0.0)
    vd = A_sys / C_p
    return torch.where(valid, vd, torch.full_like(vd, float("nan")))


def vd_track(
    t: Tensor,
    A_gut: Tensor,
    A_plasma: Tensor,
    A_liver: Tensor,
    v_plasma_L: Tensor | float,
    *,
    c_min: float = 1e-6,
) -> Tensor:
    """Apparent total tracked Vd(t) = (A_gut + A_plasma + A_liver) / C_p [L]."""
    del t
    C_p, valid = _plasma_conc_with_floor(A_plasma, v_plasma_L, c_min=c_min)
    A_track = (
        A_gut.clamp(min=0.0)
        + A_plasma.clamp(min=0.0)
        + A_liver.clamp(min=0.0)
    )
    vd = A_track / C_p
    return torch.where(valid, vd, torch.full_like(vd, float("nan")))


def t_half_instantaneous(
    t: Tensor,
    C_p_mg_L: Tensor,
    *,
    c_min: float = 1e-6,
) -> Tensor:
    """Instantaneous elimination t½(t) = ln(2) / k_elim [hours]; NaN during absorption."""
    valid = C_p_mg_L >= c_min
    lnC = torch.log(C_p_mg_L.clamp(min=c_min))
    k_elim = -_central_derivative(t, lnC)

    t_half = torch.full_like(C_p_mg_L, float("nan"))
    elim = valid & (k_elim > 0)
    return torch.where(elim, LN2 / k_elim, t_half)


def t_half_from_clearance(
    t: Tensor,
    A_sys: Tensor,
    C_p_mg_L: Tensor,
    vd_sys_vals: Tensor,
    *,
    c_min: float = 1e-6,
) -> Tensor:
    """Instantaneous t½ from CL_inst and Vd_sys: t½ = ln(2) * Vd / CL [hours]."""
    valid = C_p_mg_L >= c_min
    dA_dt = _central_derivative(t, A_sys)
    C_safe = C_p_mg_L.clamp(min=c_min)
    CL_inst = -dA_dt / C_safe

    t_half = torch.full_like(C_p_mg_L, float("nan"))
    elim = valid & (CL_inst > 0) & torch.isfinite(vd_sys_vals)
    return torch.where(elim, LN2 * vd_sys_vals / CL_inst, t_half)


def _terminal_window_mask(t: Tensor, terminal_fraction: float) -> Tensor:
    """Boolean mask selecting the last `terminal_fraction` of the time span."""
    if not 0.0 < terminal_fraction <= 1.0:
        raise ValueError(f"terminal_fraction must be in (0, 1]; got {terminal_fraction}")
    t_start = t.min()
    t_end = t.max()
    cutoff = t_end - (t_end - t_start) * terminal_fraction
    return t >= cutoff


def terminal_half_life(
    t: Tensor,
    C_p_mg_L: Tensor,
    *,
    terminal_fraction: float = 0.30,
    c_min: float = 1e-6,
) -> Tensor:
    """Terminal t½ via log-linear regression of ln(C_p) over the last 30% of t."""
    window = _terminal_window_mask(t, terminal_fraction)
    valid = window & (C_p_mg_L >= c_min)

    t_w = t[valid]
    lnC_w = torch.log(C_p_mg_L[valid].clamp(min=c_min))

    nan = torch.tensor(float("nan"), dtype=C_p_mg_L.dtype, device=C_p_mg_L.device)
    if t_w.numel() < 2:
        return nan

    t_mean = t_w.mean()
    lnC_mean = lnC_w.mean()
    dt = t_w - t_mean
    denom = (dt * dt).sum()
    if denom <= 0:
        return nan

    slope = (dt * (lnC_w - lnC_mean)).sum() / denom
    k_elim = -slope
    if k_elim <= 0:
        return nan
    return LN2 / k_elim


def terminal_vd_sys(
    vd_sys_vals: Tensor,
    t: Tensor,
    *,
    terminal_fraction: float = 0.30,
) -> Tensor:
    """Steady-state Vd_sys = median Vd over the terminal window."""
    window = _terminal_window_mask(t, terminal_fraction)
    vd_w = vd_sys_vals[window]
    vd_w = vd_w[torch.isfinite(vd_w)]

    nan = torch.tensor(float("nan"), dtype=vd_sys_vals.dtype, device=vd_sys_vals.device)
    if vd_w.numel() == 0:
        return nan
    return torch.median(vd_w)


def pk_metrics_for_drug(
    traj: Tensor,
    t: Tensor,
    drug: DrugKey,
    v_plasma_L: Tensor | float,
    *,
    c_min: float = 1e-6,
    weight_kg: Tensor | float | None = None,
) -> dict[str, Tensor]:
    """Parent PK metrics; optional ``weight_kg`` adds L/kg-normalized Vd."""
    amts = parent_amounts(traj, drug)
    C_p = plasma_concentration_mg_L(amts["plasma"], v_plasma_L)
    vd_s = vd_sys(t, amts["plasma"], amts["liver"], v_plasma_L, c_min=c_min)
    vd_t = vd_track(t, amts["gut"], amts["plasma"], amts["liver"], v_plasma_L, c_min=c_min)
    t_half = t_half_instantaneous(t, C_p, c_min=c_min)
    t_half_cl = t_half_from_clearance(t, amts["sys"], C_p, vd_s, c_min=c_min)
    t_half_terminal = terminal_half_life(t, C_p, c_min=c_min)
    vd_terminal = terminal_vd_sys(vd_s, t)
    metrics = {
        **amts,
        "C_p_mg_L": C_p,
        "C_p_ng_mL": C_p * 1000.0,
        "vd_sys_L": vd_s,
        "vd_track_L": vd_t,
        "t_half_h": t_half,
        "t_half_from_clearance_h": t_half_cl,
        "t_half_terminal_h": t_half_terminal,
        "vd_terminal_L": vd_terminal,
    }
    if weight_kg is not None:
        w = torch.as_tensor(weight_kg, dtype=vd_s.dtype, device=vd_s.device).abs().clamp(min=1e-3)
        metrics["vd_sys_L_kg"] = vd_s / w
        metrics["vd_track_L_kg"] = vd_t / w
        metrics["vd_terminal_L_kg"] = vd_terminal / w
    return metrics
