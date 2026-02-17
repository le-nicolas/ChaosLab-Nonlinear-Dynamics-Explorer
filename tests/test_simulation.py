import numpy as np

from chaos_theory.simulation import simulate_continuous, simulate_logistic


def test_logistic_stays_bounded_for_r_leq_4() -> None:
    result = simulate_logistic(steps=1500, x0=0.21, perturbation=1e-9, r=3.95)
    values = result.primary[:, 0]
    assert np.all(values >= 0.0)
    assert np.all(values <= 1.0)
    assert values.shape[0] == 1500


def test_continuous_simulation_returns_expected_shape() -> None:
    result = simulate_continuous(system_name="lorenz", duration=2.0, dt=0.01, perturbation=1e-7)
    assert result.primary.shape == result.perturbed.shape
    assert result.primary.shape[1] == 3
    assert result.separation.shape[0] == result.time.shape[0]
