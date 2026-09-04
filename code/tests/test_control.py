import unittest

import numpy as np

from federated_lpv import (
    augmented_tracking_matrices,
    design_lqi_gain,
    design_oracle_controllers,
    fit_oracle_architectures,
    sample_fleet,
)


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.fit_speeds = np.asarray([12.0, 16.0, 20.0, 24.0, 28.0])
        self.architectures = fit_oracle_architectures(sample_fleet(1, 2), self.fit_speeds, 0.01)
        self.controllers = design_oracle_controllers(
            self.architectures, 0.01, np.diag([100.0, 10.0, 500.0]), np.asarray([[10.0]]),
            np.asarray([10.0, 15.0, 20.0, 25.0, 30.0]), self.fit_speeds,
        )

    def test_augmented_shapes_and_lqi_stability(self):
        a, b = self.architectures["M4"].predict("nominal", np.asarray([20.0]))
        aa, ba = augmented_tracking_matrices(a[0], b[0], 0.01)
        gain, prefilter = design_lqi_gain(a[0], b[0], 0.01, np.diag([100, 10, 500]), [[10]])
        self.assertEqual(aa.shape, (3, 3))
        self.assertEqual(ba.shape, (3, 1))
        self.assertEqual(gain.shape, (3,))
        self.assertTrue(np.isfinite(prefilter))
        self.assertLess(max(abs(np.linalg.eigvals(aa - ba @ gain[None, :]))), 1.0)

    def test_linear_schedule_is_continuous_at_grid_point(self):
        controller = self.controllers["M4"]
        left, _ = controller.evaluate("nominal", 19.999999)
        right, _ = controller.evaluate("nominal", 20.000001)
        np.testing.assert_allclose(left, right, atol=1e-6)

    def test_prefilter_supports_zero_integrator_equilibrium(self):
        a, b = self.architectures["M4"].predict("nominal", np.asarray([20.0]))
        gain, prefilter = design_lqi_gain(a[0], b[0], 0.01, np.diag([100, 10, 500]), [[10]])
        system = np.block([[np.eye(2) - a[0], -b[0]], [np.array([[0.0, 1.0]]), np.zeros((1, 1))]])
        equilibrium = np.linalg.solve(system, np.array([0.0, 0.0, 1.0]))
        commanded = -gain[:2] @ equilibrium[:2] + prefilter
        self.assertAlmostEqual(commanded, equilibrium[2])

    def test_controller_modes_match_architectures(self):
        self.assertEqual(self.controllers["M1"].interpolation, "constant")
        self.assertEqual(self.controllers["M4"].interpolation, "linear")
        self.assertEqual(self.controllers["M5"].interpolation, "nearest")


if __name__ == "__main__":
    unittest.main()
