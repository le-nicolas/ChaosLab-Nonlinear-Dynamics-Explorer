from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .systems import LOGISTIC_DEFAULTS, get_continuous_system, logistic_next


@dataclass
class SimulationResult:
    system: str
    time: np.ndarray
    primary: np.ndarray
    perturbed: np.ndarray
    separation: np.ndarray
    axis_labels: tuple[str, ...]
    parameters: dict[str, float]


def rk4_step(
    derivative,
    t: float,
    state: np.ndarray,
    dt: float,
    params: dict[str, float],
) -> np.ndarray:
    k1 = derivative(t, state, params)
    k2 = derivative(t + dt / 2.0, state + dt * k1 / 2.0, params)
    k3 = derivative(t + dt / 2.0, state + dt * k2 / 2.0, params)
    k4 = derivative(t + dt, state + dt * k3, params)
    return state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def simulate_continuous(
    system_name: str = "lorenz",
    duration: float = 30.0,
    dt: float = 0.01,
    initial_state: tuple[float, ...] | None = None,
    perturbation: float = 1e-8,
    params: dict[str, float] | None = None,
) -> SimulationResult:
    if duration <= 0:
        raise ValueError("duration must be positive.")
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if perturbation <= 0:
        raise ValueError("perturbation must be positive.")

    system = get_continuous_system(system_name)
    merged_params = dict(system.defaults)
    if params:
        merged_params.update(params)

    base_state = np.array(initial_state or system.default_initial_state, dtype=float)
    if base_state.shape != (system.dimension,):
        raise ValueError(
            f"initial_state for {system.name} must have length {system.dimension}, "
            f"got {base_state.shape[0]}."
        )

    steps = int(np.floor(duration / dt)) + 1
    time = np.linspace(0.0, dt * (steps - 1), steps)
    primary = np.zeros((steps, system.dimension), dtype=float)
    perturbed = np.zeros_like(primary)

    primary[0] = base_state
    perturbed[0] = base_state
    perturbed[0, 0] += perturbation

    for i in range(1, steps):
        t_prev = time[i - 1]
        primary[i] = rk4_step(system.derivative, t_prev, primary[i - 1], dt, merged_params)
        perturbed[i] = rk4_step(system.derivative, t_prev, perturbed[i - 1], dt, merged_params)

    separation = np.linalg.norm(primary - perturbed, axis=1)

    return SimulationResult(
        system=system.name,
        time=time,
        primary=primary,
        perturbed=perturbed,
        separation=separation,
        axis_labels=system.axis_labels,
        parameters=merged_params,
    )


def simulate_logistic(
    steps: int = 1200,
    x0: float = LOGISTIC_DEFAULTS["x0"],
    perturbation: float = 1e-9,
    r: float = LOGISTIC_DEFAULTS["r"],
) -> SimulationResult:
    if steps < 10:
        raise ValueError("steps must be at least 10.")
    if perturbation <= 0:
        raise ValueError("perturbation must be positive.")

    primary = np.zeros((steps, 1), dtype=float)
    perturbed = np.zeros((steps, 1), dtype=float)
    primary[0, 0] = float(x0)
    perturbed[0, 0] = float(x0 + perturbation)

    for i in range(1, steps):
        primary[i, 0] = logistic_next(primary[i - 1, 0], r=r)
        perturbed[i, 0] = logistic_next(perturbed[i - 1, 0], r=r)

    separation = np.abs(primary[:, 0] - perturbed[:, 0])
    time = np.arange(steps, dtype=float)

    return SimulationResult(
        system="logistic",
        time=time,
        primary=primary,
        perturbed=perturbed,
        separation=separation,
        axis_labels=("x",),
        parameters={"r": r},
    )


def simulation_to_frame(result: SimulationResult) -> pd.DataFrame:
    frame = pd.DataFrame({"time": result.time})

    for index, label in enumerate(result.axis_labels):
        frame[label] = result.primary[:, index]
        frame[f"{label}_perturbed"] = result.perturbed[:, index]

    frame["separation"] = result.separation
    return frame
