"""Linear single-track vehicle model used by the oracle benchmark."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm


@dataclass(frozen=True)
class VehicleParameters:
    """Structural parameters that remain fixed for one simulated client."""

    mass: float
    yaw_inertia: float
    front_length: float
    rear_length: float
    front_stiffness: float
    rear_stiffness: float


def family_centers() -> dict[str, VehicleParameters]:
    """Return interpretable initial family centers for the feasibility study."""
    nominal = VehicleParameters(1500.0, 2500.0, 1.2, 1.6, 80000.0, 80000.0)
    return {
        "nominal": nominal,
        "heavy": VehicleParameters(1725.0, 3000.0, 1.2, 1.6, 84000.0, 80000.0),
        "handling": VehicleParameters(1425.0, 2250.0, 1.2, 1.6, 72000.0, 88000.0),
    }


def continuous_bicycle_matrices(
    speed: float, parameters: VehicleParameters
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return continuous-time matrices for x=[beta, yaw_rate], u=steering.

    The sign convention assumes positive cornering-stiffness magnitudes and the
    standard small-angle linear tire model. Speed must be strictly positive.
    """
    if speed <= 0:
        raise ValueError("speed must be strictly positive")

    p = parameters
    m, iz, lf, lr, cf, cr, vx = (
        p.mass,
        p.yaw_inertia,
        p.front_length,
        p.rear_length,
        p.front_stiffness,
        p.rear_stiffness,
        float(speed),
    )
    a = np.array(
        [
            [-(cf + cr) / (m * vx), (cr * lr - cf * lf) / (m * vx**2) - 1.0],
            [(cr * lr - cf * lf) / iz, -(cf * lf**2 + cr * lr**2) / (iz * vx)],
        ],
        dtype=float,
    )
    b = np.array([[cf / (m * vx)], [cf * lf / iz]], dtype=float)
    return a, b


def discrete_bicycle_matrices(
    speed: float, parameters: VehicleParameters, sample_time: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Discretize the frozen-speed bicycle model exactly under zero-order hold."""
    if sample_time <= 0:
        raise ValueError("sample_time must be strictly positive")
    a, b = continuous_bicycle_matrices(speed, parameters)
    augmented = np.block([[a, b], [np.zeros((1, 3))]])
    discrete = expm(augmented * sample_time)
    return discrete[:2, :2], discrete[:2, 2:]
