"""Unit tests for the first-pass screening pipeline."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generate_demo import demo
from pilot import cyclic_month, features, load_data, run, spatial_split


class PilotTests(unittest.TestCase):
    def setUp(self):
        self.df = demo(n=500, seed=21)
        self.sat = [f"s_{i}" for i in range(8)]
        self.ground = [f"g_{i}" for i in range(8)]

    def test_month_embedding_wraps(self):
        cyc = cyclic_month([1, 4, 7, 10])
        self.assertEqual(cyc.shape, (4, 2))
        self.assertTrue(np.allclose(np.sum(cyc ** 2, axis=1), 1.0))

    def test_spatial_split_disjoint(self):
        train, test = spatial_split(self.df, block_degrees=2)
        self.assertGreater(len(train), 0)
        self.assertGreater(len(test), 0)
        self.assertEqual(len(set(train).intersection(set(test))), 0)

    def test_feature_dimensions(self):
        x = features(self.df, self.sat, self.ground, "month_interaction")
        self.assertEqual(x.shape, (500, 8 + 8 + 2 + 16))
        quality = features(self.df, self.sat, self.ground, "quality_interaction")
        self.assertEqual(quality.shape, (500, 8 + 8 + 2 + 16 + 1 + 8))

    def test_full_pipeline_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "toy.csv"
            self.df.to_csv(path, index=False)
            frame, sat, ground = load_data(path)
            result = run(frame, sat, ground, block_degrees=2, seed=4)
            self.assertIn("month_shuffled", result["models"])
            self.assertIn("quality_interaction", result["models"])
            self.assertIn("ground_mismatch_0.1", result["corruptions"])
            self.assertIn("satellite_missing_0.6", result["corruptions"])
            self.assertTrue(0 <= result["models"]["concat"]["accuracy"] <= 1)


if __name__ == "__main__":
    unittest.main()
