"""MGAC: MFD-Gated Adaptive Cost with learned weights."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class MGACWeights:
    omega: float
    kappa: float
    rho: float


@dataclass
class MGACParams:
    u_omega: np.ndarray
    u_kappa: np.ndarray
    u_rho: np.ndarray
    omega0: float = 0.5
    kappa0: float = 0.3
    rho0: float = 0.2


def mfd_embedding(avg_speed: float, accumulation: float, slope: float = 0.0) -> np.ndarray:
    return np.array([avg_speed / 100.0, accumulation, slope], dtype=np.float32)


def compute_mgac_weights(params: MGACParams, mfd_emb: np.ndarray) -> MGACWeights:
    omega = params.omega0 * float(np.exp(np.dot(params.u_omega, mfd_emb)))
    kappa = params.kappa0 * float(np.exp(np.dot(params.u_kappa, mfd_emb)))
    rho = params.rho0 * float(np.exp(np.dot(params.u_rho, mfd_emb)))
    return MGACWeights(omega=omega, kappa=kappa, rho=rho)


def mgac_edge_cost(
    tau: float,
    mu_ots: float,
    sigma_ots: float,
    cvar_ots: float,
    weights: MGACWeights,
) -> float:
    return tau + weights.omega * mu_ots + weights.kappa * cvar_ots + weights.rho * sigma_ots


def fit_mgac_params(
    mfd_states: list,
    rii_vals: list,
    time_vals: list,
    eta: float = 0.5,
    n_steps: int = 50,
    lr: float = 0.05,
) -> MGACParams:
    """Simple coordinate descent on validation routing objective."""
    params = MGACParams(
        u_omega=np.zeros(3, dtype=np.float32),
        u_kappa=np.zeros(3, dtype=np.float32),
        u_rho=np.zeros(3, dtype=np.float32),
    )
    if not mfd_states:
        return params

    best_loss = np.inf
    best = MGACParams(
        u_omega=params.u_omega.copy(),
        u_kappa=params.u_kappa.copy(),
        u_rho=params.u_rho.copy(),
    )
    rng = np.random.default_rng(42)
    for _ in range(n_steps):
        for attr in ("u_omega", "u_kappa", "u_rho"):
            u = getattr(params, attr).copy()
            u += rng.normal(0, lr, size=3).astype(np.float32)
            setattr(params, attr, u)
        loss = 0.0
        for emb, rii, tt in zip(mfd_states, rii_vals, time_vals):
            w = compute_mgac_weights(params, emb)
            loss += rii + eta * tt
        if loss < best_loss:
            best_loss = loss
            best = MGACParams(
                u_omega=params.u_omega.copy(),
                u_kappa=params.u_kappa.copy(),
                u_rho=params.u_rho.copy(),
            )
    return best
