"""Oracle global, family-specific, and gridded model architectures."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .fleet import VehicleClient
from .lpv import LPVMatrixModel, fit_lpv_matrix_model
from .vehicle import discrete_bicycle_matrices


@dataclass(frozen=True)
class ConstantMatrixModel:
    matrix: NDArray[np.float64]

    def predict(self, speeds: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.repeat(self.matrix[None, :, :], len(np.asarray(speeds)), axis=0)


@dataclass(frozen=True)
class GriddedMatrixModel:
    anchors: NDArray[np.float64]
    matrices: NDArray[np.float64]

    def predict(self, speeds: NDArray[np.float64]) -> NDArray[np.float64]:
        values = np.asarray(speeds, dtype=float)
        indices = np.abs(values[:, None] - self.anchors[None, :]).argmin(axis=1)
        return self.matrices[indices]


@dataclass(frozen=True)
class OracleArchitecture:
    name: str
    scheduling: str
    specialization: str
    model_count: int
    models_a: dict[str, ConstantMatrixModel | GriddedMatrixModel | LPVMatrixModel]
    models_b: dict[str, ConstantMatrixModel | GriddedMatrixModel | LPVMatrixModel]

    def predict(self, family: str, speeds: NDArray[np.float64]):
        key = "global" if self.specialization == "global" else family
        return self.models_a[key].predict(speeds), self.models_b[key].predict(speeds)


def _training_arrays(clients: list[VehicleClient], speeds: NDArray[np.float64], sample_time: float):
    rows = []
    for client in clients:
        for speed in speeds:
            a, b = discrete_bicycle_matrices(float(speed), client.parameters, sample_time)
            rows.append((client.family, float(speed), a, b))
    return rows


def fit_oracle_architectures(
    clients: list[VehicleClient], speeds: NDArray[np.float64], sample_time: float
) -> dict[str, OracleArchitecture]:
    """Fit M1--M5 with uniform support and oracle family labels."""
    if not clients:
        raise ValueError("at least one client is required")
    speeds = np.asarray(speeds, dtype=float)
    rows = _training_arrays(clients, speeds, sample_time)
    families = tuple(sorted({client.family for client in clients}))

    all_a = np.asarray([row[2] for row in rows])
    all_b = np.asarray([row[3] for row in rows])
    all_speeds = np.asarray([row[1] for row in rows])
    m1 = OracleArchitecture(
        "M1 global LTI", "none", "global", 1,
        {"global": ConstantMatrixModel(all_a.mean(axis=0))},
        {"global": ConstantMatrixModel(all_b.mean(axis=0))},
    )

    family_lti_a, family_lti_b = {}, {}
    family_lpv_a, family_lpv_b = {}, {}
    family_grid_a, family_grid_b = {}, {}
    for family in families:
        selected = [row for row in rows if row[0] == family]
        matrices_a = np.asarray([row[2] for row in selected])
        matrices_b = np.asarray([row[3] for row in selected])
        selected_speeds = np.asarray([row[1] for row in selected])
        family_lti_a[family] = ConstantMatrixModel(matrices_a.mean(axis=0))
        family_lti_b[family] = ConstantMatrixModel(matrices_b.mean(axis=0))
        family_lpv_a[family], _ = fit_lpv_matrix_model(selected_speeds, matrices_a, "reciprocal")
        family_lpv_b[family], _ = fit_lpv_matrix_model(selected_speeds, matrices_b, "reciprocal")
        grid_a = np.asarray([matrices_a[selected_speeds == speed].mean(axis=0) for speed in speeds])
        grid_b = np.asarray([matrices_b[selected_speeds == speed].mean(axis=0) for speed in speeds])
        family_grid_a[family] = GriddedMatrixModel(speeds.copy(), grid_a)
        family_grid_b[family] = GriddedMatrixModel(speeds.copy(), grid_b)

    global_lpv_a, _ = fit_lpv_matrix_model(all_speeds, all_a, "reciprocal")
    global_lpv_b, _ = fit_lpv_matrix_model(all_speeds, all_b, "reciprocal")
    return {
        "M1": m1,
        "M2": OracleArchitecture("M2 clustered LTI", "none", "family", len(families), family_lti_a, family_lti_b),
        "M3": OracleArchitecture("M3 global LPV", "continuous", "global", 1,
                                 {"global": global_lpv_a}, {"global": global_lpv_b}),
        "M4": OracleArchitecture("M4 clustered LPV", "continuous", "family", len(families),
                                 family_lpv_a, family_lpv_b),
        "M5": OracleArchitecture("M5 gridded LTI", "discrete", "family", len(families) * len(speeds),
                                 family_grid_a, family_grid_b),
    }
