"""Common LQI synthesis and scheduling for oracle controller comparisons."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve_discrete_are

from .oracle import OracleArchitecture


def augmented_tracking_matrices(
    a: NDArray[np.float64], b: NDArray[np.float64], sample_time: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    augmented_a = np.block([[a, np.zeros((2, 1))], [sample_time * np.array([[0.0, 1.0]]), np.ones((1, 1))]])
    augmented_b = np.vstack((b, np.zeros((1, 1))))
    return augmented_a, augmented_b


def design_lqi_gain(a, b, sample_time: float, q, r) -> tuple[NDArray[np.float64], float]:
    """Return augmented state-feedback gain and yaw-rate steady-state prefilter."""
    augmented_a, augmented_b = augmented_tracking_matrices(a, b, sample_time)
    controllability = np.column_stack(
        (augmented_b, augmented_a @ augmented_b, augmented_a @ augmented_a @ augmented_b)
    )
    if np.linalg.matrix_rank(controllability) != 3:
        raise ValueError("augmented tracking model is not controllable")
    p = solve_discrete_are(augmented_a, augmented_b, np.asarray(q, dtype=float), np.asarray(r, dtype=float))
    gain = np.linalg.solve(np.asarray(r, dtype=float) + augmented_b.T @ p @ augmented_b,
                           augmented_b.T @ p @ augmented_a)
    yaw_output = np.array([[0.0, 1.0]])
    equilibrium_system = np.block([[np.eye(2) - a, -b], [yaw_output, np.zeros((1, 1))]])
    equilibrium = np.linalg.solve(equilibrium_system, np.array([0.0, 0.0, 1.0]))
    state_equilibrium, input_equilibrium = equilibrium[:2], float(equilibrium[2])
    prefilter = input_equilibrium + float(gain.ravel()[:2] @ state_equilibrium)
    return gain.ravel(), prefilter


@dataclass(frozen=True)
class ScheduledController:
    architecture_name: str
    specialization: str
    interpolation: str
    speeds: NDArray[np.float64]
    gains: dict[str, NDArray[np.float64]]
    prefilters: dict[str, NDArray[np.float64]]

    def evaluate(self, family: str, speed: float) -> tuple[NDArray[np.float64], float]:
        key = "global" if self.specialization == "global" else family
        if self.interpolation == "constant":
            return self.gains[key][0], float(self.prefilters[key][0])
        if self.interpolation == "nearest":
            index = int(np.argmin(np.abs(self.speeds - speed)))
            return self.gains[key][index], float(self.prefilters[key][index])
        gain = np.asarray([np.interp(speed, self.speeds, self.gains[key][:, column]) for column in range(3)])
        prefilter = float(np.interp(speed, self.speeds, self.prefilters[key]))
        return gain, prefilter


def design_oracle_controllers(
    architectures: dict[str, OracleArchitecture],
    sample_time: float,
    q: NDArray[np.float64],
    r: NDArray[np.float64],
    scheduled_speeds: NDArray[np.float64],
    gridded_speeds: NDArray[np.float64],
) -> dict[str, ScheduledController]:
    controllers = {}
    for method, architecture in architectures.items():
        if method in {"M1", "M2"}:
            speeds, interpolation = np.asarray([20.0]), "constant"
        elif method in {"M3", "M4"}:
            speeds, interpolation = np.asarray(scheduled_speeds, dtype=float), "linear"
        else:
            speeds, interpolation = np.asarray(gridded_speeds, dtype=float), "nearest"
        keys = ("global",) if architecture.specialization == "global" else tuple(sorted(architecture.models_a))
        gains, prefilters = {}, {}
        for key in keys:
            family = "nominal" if key == "global" else key
            matrices_a, matrices_b = architecture.predict(family, speeds)
            designs = [design_lqi_gain(a, b, sample_time, q, r) for a, b in zip(matrices_a, matrices_b)]
            gains[key] = np.asarray([design[0] for design in designs])
            prefilters[key] = np.asarray([design[1] for design in designs])
        controllers[method] = ScheduledController(
            method, architecture.specialization, interpolation, speeds, gains, prefilters
        )
    return controllers
