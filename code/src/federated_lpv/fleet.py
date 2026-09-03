"""Reproducible client-population construction for vehicle-family experiments."""

from dataclasses import dataclass, replace

import numpy as np

from .vehicle import VehicleParameters, family_centers


@dataclass(frozen=True)
class VehicleClient:
    client_id: str
    family: str
    parameters: VehicleParameters


def sample_fleet(
    seed: int,
    clients_per_family: int = 10,
    mass_inertia_half_range: float = 0.025,
    stiffness_half_range: float = 0.05,
) -> list[VehicleClient]:
    """Sample bounded, independent multiplicative scatter around each center."""
    if clients_per_family < 1:
        raise ValueError("clients_per_family must be positive")
    if not 0.0 <= mass_inertia_half_range < 1.0:
        raise ValueError("mass/inertia half range must lie in [0, 1)")
    if not 0.0 <= stiffness_half_range < 1.0:
        raise ValueError("stiffness half range must lie in [0, 1)")

    generator = np.random.default_rng(seed)
    clients: list[VehicleClient] = []
    for family, center in family_centers().items():
        for index in range(clients_per_family):
            mi_scale = generator.uniform(
                1.0 - mass_inertia_half_range, 1.0 + mass_inertia_half_range, size=2
            )
            stiffness_scale = generator.uniform(
                1.0 - stiffness_half_range, 1.0 + stiffness_half_range, size=2
            )
            parameters = replace(
                center,
                mass=center.mass * mi_scale[0],
                yaw_inertia=center.yaw_inertia * mi_scale[1],
                front_stiffness=center.front_stiffness * stiffness_scale[0],
                rear_stiffness=center.rear_stiffness * stiffness_scale[1],
            )
            clients.append(VehicleClient(f"{family}_{index:02d}", family, parameters))
    return clients

