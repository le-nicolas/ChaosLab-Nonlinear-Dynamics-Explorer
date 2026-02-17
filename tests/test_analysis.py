import numpy as np

from chaos_theory.analysis import (
    estimate_lyapunov_from_separation,
    permutation_entropy,
)
from chaos_theory.simulation import simulate_logistic


def test_estimate_lyapunov_from_exponential_separation() -> None:
    dt = 0.1
    time = np.arange(600, dtype=float) * dt
    expected = 0.4
    separation = np.exp(expected * time)

    fit = estimate_lyapunov_from_separation(
        separation=separation,
        dt=dt,
        fit_start=30,
        fit_end=300,
    )
    assert abs(fit.exponent - expected) < 0.02
    assert fit.r2 > 0.99


def test_permutation_entropy_is_in_unit_interval() -> None:
    values = np.sin(np.linspace(0.0, 40.0, 500))
    entropy = permutation_entropy(values, order=5, delay=1)
    assert 0.0 <= entropy <= 1.0


def test_logistic_divergence_lyapunov_is_positive() -> None:
    result = simulate_logistic(steps=1600, x0=0.2, perturbation=1e-12, r=4.0)
    fit = estimate_lyapunov_from_separation(result.separation, dt=1.0, fit_start=1, fit_end=120)
    assert fit.exponent > 0.0
