import unittest

import numpy as np

from federated_lpv import fit_oracle_architectures, sample_fleet


class OracleArchitectureTests(unittest.TestCase):
    def setUp(self):
        self.speeds = np.asarray([12.0, 16.0, 20.0, 24.0, 28.0])
        self.models = fit_oracle_architectures(sample_fleet(1, clients_per_family=2), self.speeds, 0.01)

    def test_all_five_architectures_are_constructed(self):
        self.assertEqual(set(self.models), {"M1", "M2", "M3", "M4", "M5"})
        self.assertEqual([self.models[name].model_count for name in self.models], [1, 3, 1, 3, 15])

    def test_predictions_have_state_space_shapes(self):
        query = np.asarray([14.0, 22.0])
        for model in self.models.values():
            a, b = model.predict("nominal", query)
            self.assertEqual(a.shape, (2, 2, 2))
            self.assertEqual(b.shape, (2, 2, 1))

    def test_gridded_model_uses_nearest_anchor(self):
        model = self.models["M5"]
        a_left, _ = model.predict("nominal", np.asarray([12.0]))
        a_near, _ = model.predict("nominal", np.asarray([13.0]))
        np.testing.assert_allclose(a_left, a_near)


if __name__ == "__main__":
    unittest.main()
