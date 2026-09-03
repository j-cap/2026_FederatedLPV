import unittest

import numpy as np

from federated_lpv import continuous_bicycle_matrices, family_centers


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


if __name__ == "__main__":
    unittest.main()
