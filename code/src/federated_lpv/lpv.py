"""Finite-dimensional LPV matrix models and scheduling bases."""

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray


BasisFunction = Callable[[NDArray[np.float64]], NDArray[np.float64]]


def _linear(speed: NDArray[np.float64]) -> NDArray[np.float64]:
    return speed[:, None]


def _reciprocal(speed: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.column_stack((1.0 / speed, 1.0 / speed**2))


def _mixed(speed: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.column_stack((speed, 1.0 / speed, 1.0 / speed**2))


BASIS_FUNCTIONS: dict[str, BasisFunction] = {
    "linear": _linear,
    "reciprocal": _reciprocal,
    "mixed": _mixed,
}


@dataclass(frozen=True)
class LPVMatrixModel:
    """Affine regression in standardized scheduling features."""

    basis_name: str
    feature_mean: NDArray[np.float64]
    feature_scale: NDArray[np.float64]
    coefficients: NDArray[np.float64]
    matrix_shape: tuple[int, int]

    def design_matrix(self, speeds: NDArray[np.float64]) -> NDArray[np.float64]:
        raw = BASIS_FUNCTIONS[self.basis_name](np.asarray(speeds, dtype=float))
        standardized = (raw - self.feature_mean) / self.feature_scale
        return np.column_stack((np.ones(len(standardized)), standardized))

    def predict(self, speeds: NDArray[np.float64]) -> NDArray[np.float64]:
        flat = self.design_matrix(np.asarray(speeds, dtype=float)) @ self.coefficients
        return flat.reshape((-1, *self.matrix_shape))


def fit_lpv_matrix_model(
    speeds: NDArray[np.float64], matrices: NDArray[np.float64], basis_name: str
) -> tuple[LPVMatrixModel, float]:
    """Fit an LPV matrix model and return its standardized design condition."""
    if basis_name not in BASIS_FUNCTIONS:
        raise ValueError(f"unknown LPV basis: {basis_name}")
    speeds = np.asarray(speeds, dtype=float)
    matrices = np.asarray(matrices, dtype=float)
    if speeds.ndim != 1 or matrices.shape[0] != len(speeds):
        raise ValueError("speeds and matrices must share their first dimension")
    if np.any(speeds <= 0.0):
        raise ValueError("speeds must be strictly positive")

    raw = BASIS_FUNCTIONS[basis_name](speeds)
    mean = raw.mean(axis=0)
    scale = raw.std(axis=0)
    if np.any(scale <= np.finfo(float).eps):
        raise ValueError("basis contains a constant or numerically degenerate feature")
    design = np.column_stack((np.ones(len(speeds)), (raw - mean) / scale))
    coefficients, _, rank, _ = np.linalg.lstsq(design, matrices.reshape(len(speeds), -1), rcond=None)
    if rank != design.shape[1]:
        raise ValueError("LPV design matrix is rank deficient")
    model = LPVMatrixModel(basis_name, mean, scale, coefficients, matrices.shape[1:])
    return model, float(np.linalg.cond(design))
