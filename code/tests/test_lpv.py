import unittest

import numpy as np

from federated_lpv import fit_lpv_matrix_model


class LPVModelTests(unittest.TestCase):
    def test_linear_basis_recovers_affine_matrix_exactly(self) -> None:
        speeds = np.asarray([12.0, 16.0, 20.0, 24.0, 28.0])
        offset = np.asarray([[1.0, 2.0], [3.0, 4.0]])
        slope = np.asarray([[0.2, -0.1], [0.05, 0.3]])
        matrices = np.asarray([offset + speed * slope for speed in speeds])
        model, condition = fit_lpv_matrix_model(speeds, matrices, "linear")
        test_speeds = np.asarray([14.0, 22.0, 30.0])
        expected = np.asarray([offset + speed * slope for speed in test_speeds])
        np.testing.assert_allclose(model.predict(test_speeds), expected, atol=1e-12)
        self.assertLess(condition, 2.0)

    def test_reciprocal_basis_recovers_reciprocal_matrix_exactly(self) -> None:
        speeds = np.asarray([12.0, 16.0, 20.0, 24.0, 28.0])
        matrices = np.asarray([[[1 + 2 / speed + 3 / speed**2]] for speed in speeds])
        model, _ = fit_lpv_matrix_model(speeds, matrices, "reciprocal")
        test_speeds = np.asarray([14.0, 18.0, 26.0])
        expected = np.asarray([[[1 + 2 / speed + 3 / speed**2]] for speed in test_speeds])
        np.testing.assert_allclose(model.predict(test_speeds), expected, atol=1e-12)

    def test_unknown_basis_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown LPV basis"):
            fit_lpv_matrix_model(np.asarray([1.0, 2.0]), np.zeros((2, 1, 1)), "missing")


if __name__ == "__main__":
    unittest.main()
