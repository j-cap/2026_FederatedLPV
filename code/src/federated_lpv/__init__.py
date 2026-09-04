"""Federated LPV research package."""

from .fleet import VehicleClient, sample_fleet
from .control import ScheduledController, augmented_tracking_matrices, design_lqi_gain, design_oracle_controllers
from .lpv import BASIS_FUNCTIONS, LPVMatrixModel, fit_lpv_matrix_model
from .oracle import ConstantMatrixModel, GriddedMatrixModel, OracleArchitecture, fit_oracle_architectures
from .vehicle import (
    VehicleParameters,
    continuous_bicycle_matrices,
    discrete_bicycle_matrices,
    family_centers,
)

__all__ = [
    "VehicleParameters",
    "VehicleClient",
    "LPVMatrixModel",
    "BASIS_FUNCTIONS",
    "continuous_bicycle_matrices",
    "discrete_bicycle_matrices",
    "family_centers",
    "sample_fleet",
    "fit_lpv_matrix_model",
    "ConstantMatrixModel",
    "GriddedMatrixModel",
    "OracleArchitecture",
    "fit_oracle_architectures",
    "ScheduledController",
    "augmented_tracking_matrices",
    "design_lqi_gain",
    "design_oracle_controllers",
]
