"""Chaos Theory Lab package."""

from .analysis import analyze_series, estimate_lyapunov_from_separation
from .simulation import simulate_continuous, simulate_logistic

__all__ = [
    "analyze_series",
    "estimate_lyapunov_from_separation",
    "simulate_continuous",
    "simulate_logistic",
]

__version__ = "0.1.0"
