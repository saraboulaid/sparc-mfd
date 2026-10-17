"""Baseline routing methods for SPARC-MFD experiments."""

from __future__ import annotations

from typing import List

from .routing import RoutingMethod

BASELINE_METHODS: List[RoutingMethod] = [
    RoutingMethod.TIME_ONLY,
    RoutingMethod.RISK_DET,
    RoutingMethod.RISK_CVAR,
    RoutingMethod.CQR_ROBUST,
    RoutingMethod.SPARC_MFD,
    RoutingMethod.ORACLE,
]

METHOD_LABELS = {
    RoutingMethod.TIME_ONLY: "B1 Time-only",
    RoutingMethod.RISK_DET: "B2 Risk-det",
    RoutingMethod.RISK_CVAR: "B3 Risk-CVaR fixed",
    RoutingMethod.CQR_ROBUST: "B4 CQR-style",
    RoutingMethod.SPARC_MFD: "SPARC-MFD",
    RoutingMethod.ORACLE: "B5 Oracle",
}
