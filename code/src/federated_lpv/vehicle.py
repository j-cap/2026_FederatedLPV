"""Linear and nonlinear single-track models used by the oracle benchmark."""

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


def static_axle_loads(
    parameters: VehicleParameters, gravity: float = 9.81
) -> tuple[float, float]:
    """Return static front and rear axle loads for the single-track model."""
    if gravity <= 0:
        raise ValueError("gravity must be strictly positive")
    length = parameters.front_length + parameters.rear_length
    front = parameters.mass * gravity * parameters.rear_length / length
    rear = parameters.mass * gravity * parameters.front_length / length
    return front, rear


def tire_saturation_angles(
    parameters: VehicleParameters, friction_coefficient: float = 0.9
) -> tuple[float, float]:
    """Return tanh scale angles such that force limits equal static friction limits."""
    if friction_coefficient <= 0:
        raise ValueError("friction_coefficient must be strictly positive")
    front_load, rear_load = static_axle_loads(parameters)
    return (
        friction_coefficient * front_load / parameters.front_stiffness,
        friction_coefficient * rear_load / parameters.rear_stiffness,
    )


def nonlinear_tire_forces(
    state: NDArray[np.float64],
    steering: float,
    speed: float,
    parameters: VehicleParameters,
    friction_coefficient: float = 0.9,
) -> tuple[float, float, float, float]:
    """Return front/rear lateral forces and slip angles for tanh tires."""
    if speed <= 0:
        raise ValueError("speed must be strictly positive")
    beta, yaw_rate = np.asarray(state, dtype=float)
    front_slip = float(steering - beta - parameters.front_length * yaw_rate / speed)
    rear_slip = float(-beta + parameters.rear_length * yaw_rate / speed)
    front_sat, rear_sat = tire_saturation_angles(parameters, friction_coefficient)
    front_force = parameters.front_stiffness * front_sat * np.tanh(front_slip / front_sat)
    rear_force = parameters.rear_stiffness * rear_sat * np.tanh(rear_slip / rear_sat)
    return float(front_force), float(rear_force), front_slip, rear_slip


def nonlinear_bicycle_rhs(
    state: NDArray[np.float64],
    steering: float,
    speed: float,
    parameters: VehicleParameters,
    friction_coefficient: float = 0.9,
) -> NDArray[np.float64]:
    """Evaluate the nonlinear bicycle dynamics for x=[beta, yaw_rate]."""
    front_force, rear_force, _, _ = nonlinear_tire_forces(
        state, steering, speed, parameters, friction_coefficient
    )
    beta, yaw_rate = np.asarray(state, dtype=float)
    beta_dot = (front_force + rear_force) / (parameters.mass * speed) - yaw_rate
    yaw_rate_dot = (
        parameters.front_length * front_force - parameters.rear_length * rear_force
    ) / parameters.yaw_inertia
    return np.asarray([beta_dot, yaw_rate_dot], dtype=float)
