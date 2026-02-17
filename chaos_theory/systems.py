from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

Derivative = Callable[[float, np.ndarray, dict[str, float]], np.ndarray]


@dataclass(frozen=True)
class ContinuousSystem:
    name: str
    axis_labels: tuple[str, ...]
    defaults: dict[str, float]
    default_initial_state: tuple[float, ...]
    derivative: Derivative

    @property
    def dimension(self) -> int:
        return len(self.axis_labels)


def lorenz_derivative(_: float, state: np.ndarray, params: dict[str, float]) -> np.ndarray:
    sigma = params["sigma"]
    beta = params["beta"]
    rho = params["rho"]
    x, y, z = state
    return np.array(
        [
            sigma * (y - x),
            x * (rho - z) - y,
            x * y - beta * z,
        ],
        dtype=float,
    )


def rossler_derivative(_: float, state: np.ndarray, params: dict[str, float]) -> np.ndarray:
    a = params["a"]
    b = params["b"]
    c = params["c"]
    x, y, z = state
    return np.array(
        [
            -y - z,
            x + a * y,
            b + z * (x - c),
        ],
        dtype=float,
    )


CONTINUOUS_SYSTEMS: dict[str, ContinuousSystem] = {
    "lorenz": ContinuousSystem(
        name="lorenz",
        axis_labels=("x", "y", "z"),
        defaults={"sigma": 10.0, "beta": 8.0 / 3.0, "rho": 28.0},
        default_initial_state=(1.0, 1.0, 1.0),
        derivative=lorenz_derivative,
    ),
    "rossler": ContinuousSystem(
        name="rossler",
        axis_labels=("x", "y", "z"),
        defaults={"a": 0.2, "b": 0.2, "c": 5.7},
        default_initial_state=(0.0, 1.0, 0.0),
        derivative=rossler_derivative,
    ),
}

LOGISTIC_DEFAULTS: dict[str, float] = {"r": 3.9, "x0": 0.2}


def get_continuous_system(name: str) -> ContinuousSystem:
    key = name.lower()
    if key not in CONTINUOUS_SYSTEMS:
        available = ", ".join(sorted(CONTINUOUS_SYSTEMS))
        raise ValueError(f"Unknown continuous system '{name}'. Expected one of: {available}.")
    return CONTINUOUS_SYSTEMS[key]


def logistic_next(x: float, r: float) -> float:
    return r * x * (1.0 - x)


def available_systems() -> dict[str, dict[str, object]]:
    systems: dict[str, dict[str, object]] = {}
    for name, system in CONTINUOUS_SYSTEMS.items():
        systems[name] = {
            "type": "continuous",
            "dimension": system.dimension,
            "axis_labels": system.axis_labels,
            "parameters": system.defaults,
            "initial_state": system.default_initial_state,
        }
    systems["logistic"] = {
        "type": "discrete",
        "dimension": 1,
        "axis_labels": ("x",),
        "parameters": {"r": LOGISTIC_DEFAULTS["r"]},
        "initial_state": (LOGISTIC_DEFAULTS["x0"],),
    }
    return systems
