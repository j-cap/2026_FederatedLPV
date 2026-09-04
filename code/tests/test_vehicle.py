import unittest

import numpy as np

from federated_lpv import (
    continuous_bicycle_matrices,
    discrete_bicycle_matrices,
    family_centers,
    nonlinear_bicycle_rhs,
    nonlinear_tire_forces,
    sample_fleet,
    static_axle_loads,
    tire_saturation_angles,
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

    def test_fleet_sampling_is_reproducible_and_bounded(self) -> None:
        fleet_a = sample_fleet(seed=1)
        fleet_b = sample_fleet(seed=1)
        self.assertEqual(fleet_a, fleet_b)
        self.assertEqual(len(fleet_a), 30)
        for client in fleet_a:
            center = family_centers()[client.family]
            self.assertLessEqual(abs(client.parameters.mass / center.mass - 1.0), 0.025)
            self.assertLessEqual(abs(client.parameters.yaw_inertia / center.yaw_inertia - 1.0), 0.025)
            self.assertLessEqual(abs(client.parameters.front_stiffness / center.front_stiffness - 1.0), 0.05)
            self.assertLessEqual(abs(client.parameters.rear_stiffness / center.rear_stiffness - 1.0), 0.05)

    def test_static_loads_sum_to_vehicle_weight(self) -> None:
        parameters = family_centers()["nominal"]
        front, rear = static_axle_loads(parameters)
        self.assertAlmostEqual(front + rear, parameters.mass * 9.81)
        self.assertAlmostEqual(front * parameters.front_length, rear * parameters.rear_length)

    def test_tanh_force_is_odd_and_respects_friction_limit(self) -> None:
        parameters = family_centers()["nominal"]
        front_sat, _ = tire_saturation_angles(parameters)
        positive = nonlinear_tire_forces(
            np.zeros(2), 20.0 * front_sat, 20.0, parameters
        )[0]
        negative = nonlinear_tire_forces(
            np.zeros(2), -20.0 * front_sat, 20.0, parameters
        )[0]
        front_load, _ = static_axle_loads(parameters)
        self.assertAlmostEqual(positive, -negative, places=8)
        self.assertLessEqual(abs(positive), 0.9 * front_load * (1.0 + 1e-12))
        self.assertAlmostEqual(positive / (0.9 * front_load), 1.0, places=12)

    def test_nonlinear_small_signal_jacobian_matches_linear_model(self) -> None:
        for parameters in family_centers().values():
            for speed in (12.0, 20.0, 28.0):
                a, b = continuous_bicycle_matrices(speed, parameters)
                epsilon = 1e-7
                numerical_a = np.column_stack(
                    [
                        (
                            nonlinear_bicycle_rhs(np.eye(2)[j] * epsilon, 0.0, speed, parameters)
                            - nonlinear_bicycle_rhs(-np.eye(2)[j] * epsilon, 0.0, speed, parameters)
                        )
                        / (2.0 * epsilon)
                        for j in range(2)
                    ]
                )
                numerical_b = (
                    nonlinear_bicycle_rhs(np.zeros(2), epsilon, speed, parameters)
                    - nonlinear_bicycle_rhs(np.zeros(2), -epsilon, speed, parameters)
                )[:, None] / (2.0 * epsilon)
                np.testing.assert_allclose(numerical_a, a, rtol=1e-9, atol=1e-9)
                np.testing.assert_allclose(numerical_b, b, rtol=1e-9, atol=1e-9)

    def test_nonlinear_model_rejects_invalid_operating_conditions(self) -> None:
        parameters = family_centers()["nominal"]
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            nonlinear_bicycle_rhs(np.zeros(2), 0.0, 0.0, parameters)
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            tire_saturation_angles(parameters, 0.0)


if __name__ == "__main__":
    unittest.main()
