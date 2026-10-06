"""Constraint geometry, complete gradients, profiles, and client leakage checks."""

import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from federated_lpv.innovation_likelihood import (
    C,
    InnovationLikelihood,
    MeasuredDataset,
    physical_coupling_diagnostic,
    structured_matrices,
)
from federated_lpv.physical_coupled_fit import (
    COUPLING_ROW,
    LOG_MAP,
    CoupledLikelihood,
    PhysicalCoordinates,
    constraint_audit,
    optimize_coupled,
)

NOMINAL = np.log([0.6 * 1.2, 0.6 * 1.5, 100.0, 1.2 * 100.0, 90.0, 1.5 * 90.0, 2.0, 1.7, 8.0])
Q = np.diag(np.array([1e-7, 1e-6, 2e-4, 2e-4, 1e-6]) * 0.01)
R = np.diag([np.deg2rad(0.03) ** 2, 0.03**2, np.deg2rad(0.02) ** 2])


def coordinates():
    return PhysicalCoordinates(NOMINAL, NOMINAL + np.log(0.35), NOMINAL + np.log(3.0))


def fixture():
    rng = np.random.default_rng(733)
    speeds = np.array([10.0, 20.0, 30.0])
    commands = rng.normal(scale=0.008, size=(3, 90))
    measured = rng.normal(size=(3, 90, 3)) * np.sqrt(np.diag(R))
    for i, speed in enumerate(speeds):
        a, b = structured_matrices(NOMINAL, speed, 0.01)
        x = np.zeros(5)
        for t in range(90):
            x = a @ x + b * commands[i, t]
            measured[i, t] += C @ x
    return MeasuredDataset(speeds, commands, measured)


def test_mapping_has_exactly_eight_independent_physical_directions():
    assert np.linalg.matrix_rank(LOG_MAP) == 8
    assert_allclose(COUPLING_ROW @ LOG_MAP, np.zeros(8), atol=0)
    co = coordinates()
    rng = np.random.default_rng(2)
    for _ in range(20):
        eta = rng.normal(size=8)
        z = co.effective(eta)
        assert abs(physical_coupling_diagnostic(z)["log_coupling_residual"]) < 1e-13
        assert_allclose(LOG_MAP @ co.nominal_physical_logs, NOMINAL, atol=1e-14)
    with pytest.raises(ValueError, match="physical identity"):
        PhysicalCoordinates(NOMINAL + np.eye(9)[0] * 0.1, NOMINAL - 1, NOMINAL + 1)


def test_initialization_preserves_the_original_coefficient_polytope():
    co = coordinates()
    proposal = np.array([3.0, 4.0, -4.0, 2.0, -3.0, 1.0, 0.0, -2.0])
    eta = co.feasible_initial(proposal)
    z = co.effective(eta)
    assert np.all(z >= co.lower - 1e-12)
    assert np.all(z <= co.upper + 1e-12)
    nonzero = proposal != 0
    assert_allclose(eta[nonzero] / proposal[nonzero], eta[0] / proposal[0])


def test_complete_eight_coordinate_likelihood_gradient():
    co = coordinates()
    original = InnovationLikelihood(fixture(), 0.01, Q, R)
    evaluator = CoupledLikelihood(original, co)
    eta = np.linspace(-0.12, 0.10, 8)
    result = evaluator.evaluate(eta, gradient=True, information=True)
    numeric = np.array(
        [
            (
                evaluator.value(eta + np.eye(8)[j] * 1e-5)
                - evaluator.value(eta - np.eye(8)[j] * 1e-5)
            )
            / 2e-5
            for j in range(8)
        ]
    )
    assert_allclose(result["gradient"], numeric, atol=2e-6, rtol=2e-5)
    independent = original.evaluate(co.effective(eta), information=True)
    assert_allclose(result["information"], LOG_MAP.T @ independent["information"] @ LOG_MAP)


def test_kkt_handles_coupled_bound_normals_and_wrong_signs():
    co = coordinates()
    eta = np.zeros(8)
    eta[0] = np.log(3.0)  # Both yaw gains touch their upper bounds.
    outward = LOG_MAP[0] + LOG_MAP[1]
    audit = constraint_audit(eta, -outward, co)
    assert audit["active_coefficient_bounds"] == 2
    assert audit["kkt_residual"] < 1e-12
    assert audit["constraint_violation"] < 1e-12
    assert constraint_audit(eta, outward, co)["kkt_residual"] > 0.1
    assert audit["face_basis"].shape == (8, 6)


def test_profile_start_adjusts_nuisance_coordinates_instead_of_clipping():
    co = coordinates()
    eta = np.zeros(8)
    eta[0] = np.log(3.0)
    start = co.feasible_profile_start(eta, 1, 0.1)
    assert_allclose(start[1], 0.1, atol=1e-12)
    assert start[0] < eta[0]
    assert constraint_audit(start, np.zeros(8), co)["constraint_violation"] < 1e-9
    assert co.feasible_profile_start(eta, 5, np.log(3.0) + 0.1) is None


def test_nuisance_optimizer_respects_fixed_coordinate_and_kkt():
    class Quadratic:
        def evaluate(self, eta, gradient=False, information=False):
            residual = eta - np.array([0.3, -0.2, 0.1, 0.2, -0.1, 0.1, -0.1, 0.2])
            result = {"objective": float(1 + residual @ residual), "gradient": 2 * residual}
            if information:
                result["information"] = np.eye(8)
            return result

        def value_gradient(self, eta):
            result = self.evaluate(eta)
            return result["objective"], result["gradient"]

    cfg = json.loads((Path(__file__).parents[1] / "config/experiment_10hi.json").read_text())
    fitted, result, _, _, invalid, audit = optimize_coupled(
        np.zeros(8), Quadratic(), coordinates(), cfg, "joint", {1: 0.15}
    )
    assert result.success and invalid == 0
    assert_allclose(fitted[1], 0.15, atol=1e-12)
    assert_allclose(fitted[[0, 2, 3, 4, 5, 6, 7]], [0.3, 0.1, 0.2, -0.1, 0.1, -0.1, 0.2], atol=1e-9)
    assert audit["kkt_residual"] < 1e-9


def test_inner_folds_never_split_a_clients_speed_records():
    import sys

    sys.path.insert(0, str(Path(__file__).parents[1] / "experiments"))
    from experiment_10hi_physically_coupled_fit import inner_client_folds

    data = fixture()
    positions = [0, 0, 1]
    for _, fitting, validation, ids in inner_client_folds(data, positions):
        assert ids[0] == ids[1] != ids[2]
        assert len(fitting.speeds) + len(validation.speeds) == 3
        assert set(fitting.speeds).isdisjoint(validation.speeds)


def test_confirmation_seeds_are_sealed_and_grid_is_predeclared():
    cfg = json.loads((Path(__file__).parents[1] / "config/experiment_10hi.json").read_text())
    assert set(cfg["development_seeds"]).isdisjoint(cfg["reserved_confirmation_seeds"])
    assert set(range(591, 601)) | set(range(611, 621)) == set(cfg["reserved_confirmation_seeds"])
    assert cfg["covariance_grid"] == [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]


def test_zero_offset_profile_reuses_fit_despite_bound_roundoff(monkeypatch):
    import sys

    sys.path.insert(0, str(Path(__file__).parents[1] / "experiments"))
    from experiment_10hi_physically_coupled_fit import nuisance_profiles

    co = coordinates()
    evaluator = CoupledLikelihood(InnovationLikelihood(fixture(), 0.01, Q, R), co)
    fitted = np.zeros(8)
    fitted[3] = np.log(3.0) + 1e-14
    cfg = json.loads((Path(__file__).parents[1] / "config/experiment_10hi.json").read_text())
    cfg["profile_log_offsets"] = [0.0]

    def forbidden_projection(*args):
        raise AssertionError("The profile center must not be reprojected")

    monkeypatch.setattr(PhysicalCoordinates, "feasible_profile_start", forbidden_projection)
    _, profiles = nuisance_profiles(evaluator, fitted, co, cfg)
    assert len(profiles) == 8 and profiles.feasible.all()
    assert_allclose(profiles.actual_log_distance, 0.0, atol=0)
    assert profiles.constraint_violation.max() < 1e-12


def test_innovation_audit_uses_the_repaired_likelihood_covariance():
    import sys

    sys.path.insert(0, str(Path(__file__).parents[1] / "experiments"))
    from experiment_10hi_innovation_audit import whitened_diagnostics

    from federated_lpv.innovation_likelihood import steady_filter

    data = fixture()
    z = coordinates().effective(np.linspace(-.1, .12, 8))
    audit = whitened_diagnostics(data, z, .01, Q, R)
    evaluator = InnovationLikelihood(data, .01, Q, R)
    determinant_term = np.mean([
        steady_filter(z, speed, .01, Q, R)["logdet"] - np.linalg.slogdet(R)[1]
        for speed in data.speeds
    ])
    assert_allclose(audit["normalized_innovation_squared_mean"],
                    evaluator.value(z) - determinant_term, rtol=1e-10)
