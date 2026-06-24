"""NCA-style PK metrics from integrated ODE trajectories (APAP, caffeine).

Inputs: traj ``[T, ≥9]`` amounts (mg), t (hr), v_plasma (L).
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

from src.models.gnn_ode import STATE_IDX

DrugKey = Literal["apap", "caffeine"]

_DRUG_INDICES: dict[DrugKey, tuple[int, int, int, int]] = {
    "apap": (
        STATE_IDX["A_gut_apap"],
        STATE_IDX["A_plasma_apap"],
        STATE_IDX["A_liver_apap"],
        STATE_IDX["A_periph_apap"],
    ),
    "caffeine": (
        STATE_IDX["A_gut_caffeine"],
        STATE_IDX["A_plasma_caffeine"],
        STATE_IDX["A_liver_caffeine"],
        STATE_IDX["A_periph_caffeine"],
    ),
}

LN2 = 0.6931471805599453


def _resolve_drug(drug: DrugKey) -> tuple[int, int, int, int]:
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
    """Return gut, plasma, liver, periph, sys, track — each [T], in mg.

    Systemic = plasma + liver + peripheral; tracked = systemic + gut.
    """
    i_gut, i_plasma, i_liver, i_periph = _resolve_drug(drug)

    if traj.ndim != 2 or traj.size(-1) <= i_periph:
        raise ValueError(f"traj must be [T, >{i_periph}]; got {tuple(traj.shape)}")

    gut = traj[:, i_gut].clamp(min=0.0)
    plasma = traj[:, i_plasma].clamp(min=0.0)
    liver = traj[:, i_liver].clamp(min=0.0)
    periph = traj[:, i_periph].clamp(min=0.0)

    return {
        "gut": gut,
        "plasma": plasma,
        "liver": liver,
        "periph": periph,
        "sys": plasma + liver + periph,
        "track": gut + plasma + liver + periph,
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
    A_periph: Tensor,
    v_plasma_L: Tensor | float,
    *,
    c_min: float = 1e-6,
) -> Tensor:
    """Apparent systemic Vd(t) = (A_plasma + A_liver + A_periph) / C_p [L]."""
    del t
    C_p, valid = _plasma_conc_with_floor(A_plasma, v_plasma_L, c_min=c_min)
    A_sys = A_plasma.clamp(min=0.0) + A_liver.clamp(min=0.0) + A_periph.clamp(min=0.0)
    vd = A_sys / C_p
    return torch.where(valid, vd, torch.full_like(vd, float("nan")))


def vd_track(
    t: Tensor,
    A_gut: Tensor,
    A_plasma: Tensor,
    A_liver: Tensor,
    A_periph: Tensor,
    v_plasma_L: Tensor | float,
    *,
    c_min: float = 1e-6,
) -> Tensor:
    """Apparent total tracked Vd(t) = (A_gut + A_plasma + A_liver + A_periph) / C_p [L]."""
    del t
    C_p, valid = _plasma_conc_with_floor(A_plasma, v_plasma_L, c_min=c_min)
    A_track = (
        A_gut.clamp(min=0.0)
        + A_plasma.clamp(min=0.0)
        + A_liver.clamp(min=0.0)
        + A_periph.clamp(min=0.0)
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


def terminal_phase_mask(
    t: Tensor,
    C_p_mg_L: Tensor,
    *,
    frac_low: float = 0.05,
    frac_high: float = 0.80,
    min_points: int = 3,
) -> Tensor:
    """Boolean mask of the post-Tmax log-linear elimination phase.

    Keeps points strictly after Tmax with ``frac_low*Cmax <= C_p <= frac_high*Cmax``
    (the upper bound drops the distribution shoulder, the lower bound drops the
    quantitation-limit tail). The upper bound is relaxed if too few points survive.
    Returns an all-False mask when no usable terminal phase exists (e.g. the drug
    is fully cleared), so callers degrade to NaN instead of fitting noise.
    """
    n = C_p_mg_L.numel()
    false_mask = torch.zeros(n, dtype=torch.bool, device=C_p_mg_L.device)
    if n < min_points:
        return false_mask

    cmax, tmax_idx = torch.max(C_p_mg_L, dim=0)
    if not torch.isfinite(cmax) or cmax <= 0:
        return false_mask

    after_peak = torch.arange(n, device=C_p_mg_L.device) > tmax_idx
    positive = C_p_mg_L > 0
    lower = frac_low * cmax
    upper = frac_high * cmax

    mask = after_peak & positive & (C_p_mg_L >= lower) & (C_p_mg_L <= upper)
    if int(mask.sum()) < min_points:  # relax the distribution-phase upper bound
        mask = after_peak & positive & (C_p_mg_L >= lower)
    if int(mask.sum()) < min_points:
        return false_mask
    return mask


def terminal_half_life(
    t: Tensor,
    C_p_mg_L: Tensor,
    *,
    frac_low: float = 0.05,
    frac_high: float = 0.80,
    min_points: int = 3,
) -> Tensor:
    """Terminal t½ via log-linear regression over the post-Tmax elimination phase."""
    nan = torch.tensor(float("nan"), dtype=C_p_mg_L.dtype, device=C_p_mg_L.device)
    mask = terminal_phase_mask(
        t, C_p_mg_L, frac_low=frac_low, frac_high=frac_high, min_points=min_points
    )
    if int(mask.sum()) < min_points:
        return nan

    t_w = t[mask]
    lnC_w = torch.log(C_p_mg_L[mask])

    dt = t_w - t_w.mean()
    denom = (dt * dt).sum()
    if denom <= 0:
        return nan

    slope = (dt * (lnC_w - lnC_w.mean())).sum() / denom
    if slope >= 0:  # flat or rising terminal phase is not an elimination slope
        return nan
    return LN2 / (-slope)


def terminal_vd_sys(
    vd_sys_vals: Tensor,
    t: Tensor,
    C_p_mg_L: Tensor,
    *,
    frac_low: float = 0.05,
    frac_high: float = 0.80,
    min_points: int = 3,
) -> Tensor:
    """Steady-state Vd_sys = median Vd over the post-Tmax elimination phase."""
    nan = torch.tensor(float("nan"), dtype=vd_sys_vals.dtype, device=vd_sys_vals.device)
    mask = terminal_phase_mask(
        t, C_p_mg_L, frac_low=frac_low, frac_high=frac_high, min_points=min_points
    )
    vd_w = vd_sys_vals[mask]
    vd_w = vd_w[torch.isfinite(vd_w)]
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
    vd_s = vd_sys(t, amts["plasma"], amts["liver"], amts["periph"], v_plasma_L, c_min=c_min)
    vd_t = vd_track(t, amts["gut"], amts["plasma"], amts["liver"], amts["periph"], v_plasma_L, c_min=c_min)
    t_half = t_half_instantaneous(t, C_p, c_min=c_min)
    t_half_cl = t_half_from_clearance(t, amts["sys"], C_p, vd_s, c_min=c_min)
    t_half_terminal = terminal_half_life(t, C_p)
    vd_terminal = terminal_vd_sys(vd_s, t, C_p)
    cmax, tmax_idx = torch.max(C_p, dim=0)
    metrics = {
        **amts,
        "C_p_mg_L": C_p,
        "C_p_ng_mL": C_p * 1000.0,
        "cmax_ng_mL": cmax * 1000.0,
        "tmax_h": t[tmax_idx],
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
