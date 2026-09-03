import unittest

import numpy as np

from federated_lpv import (
    continuous_bicycle_matrices,
    discrete_bicycle_matrices,
    family_centers,
)


class VehicleModelTests(unittest.TestCase):
    def test_continuous_model_dimensions_and_finiteness(self) -> None:
        a, b = continuous_bicycle_matrices(20.0, family_centers()["nominal"])
        self.assertEqual(a.shape, (2, 2))
        self.assertEqual(b.shape, (2, 1))
        self.assertTrue(np.isfinite(a).all() and np.isfinite(b).all())

    def test_speed_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            continuous_bicycle_matrices(0.0, family_centers()["nominal"])

    def test_families_produce_distinct_dynamics(self) -> None:
        matrices = [continuous_bicycle_matrices(20.0, p)[0] for p in family_centers().values()]
        self.assertFalse(np.allclose(matrices[0], matrices[1]))
        self.assertFalse(np.allclose(matrices[0], matrices[2]))

    def test_zoh_discretization_is_consistent_for_small_sample_time(self) -> None:
        parameters = family_centers()["nominal"]
        a, b = continuous_bicycle_matrices(20.0, parameters)
        sample_time = 1e-7
        ad, bd = discrete_bicycle_matrices(20.0, parameters, sample_time)
        np.testing.assert_allclose(ad, np.eye(2) + sample_time * a, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(bd, sample_time * b, rtol=1e-6, atol=1e-12)

    def test_discrete_model_rejects_nonpositive_sample_time(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            discrete_bicycle_matrices(20.0, family_centers()["nominal"], 0.0)

    def test_all_family_centers_are_stable_and_controllable_on_envelope(self) -> None:
        for parameters in family_centers().values():
            for speed in np.linspace(10.0, 30.0, 41):
                a, b = continuous_bicycle_matrices(float(speed), parameters)
                self.assertTrue(np.all(np.linalg.eigvals(a).real < 0.0))
                self.assertEqual(np.linalg.matrix_rank(np.column_stack((b, a @ b))), 2)

    def test_yaw_gain_matches_understeer_formula(self) -> None:
        for parameters in family_centers().values():
            length = parameters.front_length + parameters.rear_length
            ku = parameters.mass * (
                parameters.rear_stiffness * parameters.rear_length
                - parameters.front_stiffness * parameters.front_length
            ) / (parameters.front_stiffness * parameters.rear_stiffness * length)
            for speed in (12.0, 20.0, 28.0):
                a, b = continuous_bicycle_matrices(speed, parameters)
                numerical_gain = -np.linalg.solve(a, b)[1, 0]
                analytical_gain = speed / (length + ku * speed**2)
                self.assertAlmostEqual(numerical_gain, analytical_gain, places=12)


if __name__ == "__main__":
    unittest.main()
