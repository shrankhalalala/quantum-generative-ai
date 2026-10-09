"""Scientific configuration and reproducibility checks, without training."""
from dataclasses import replace
import json
import random
import tempfile
import unittest
from pathlib import Path
from config import DEFAULT_CONFIG, PAPER_TITLE
from utils.reproducibility import save_json, seed_everything

class Phase1Tests(unittest.TestCase):
    def test_approved_specification(self):
        c = DEFAULT_CONFIG
        c.validate()
        self.assertEqual(PAPER_TITLE, "Quantum Generative AI: A Survey of Hybrid Architectures and Emerging Applications")
        self.assertEqual(c.dataset.split_sizes, (1000, 200, 200))
        self.assertEqual(c.dataset.subset_sizes, (1000, 500, 250, 100))
        self.assertEqual(c.model.quantum_weight_count, 16)
        self.assertIsNone(c.training.epochs)
        self.assertIsNone(c.training.max_steps)
        self.assertEqual(json.loads(json.dumps(c.to_dict()))["diffusion"]["steps"], 20)

    def test_invalid_configuration_rejected(self):
        modifications = [
            replace(DEFAULT_CONFIG, dataset=replace(DEFAULT_CONFIG.dataset, split_sizes=(999, 200, 200))),
            replace(DEFAULT_CONFIG, dataset=replace(DEFAULT_CONFIG.dataset, subset_sizes=(100, 500))),
            replace(DEFAULT_CONFIG, model=replace(DEFAULT_CONFIG.model, n_qubits=5)),
            replace(DEFAULT_CONFIG, diffusion=replace(DEFAULT_CONFIG.diffusion, beta_end=1.0)),
            replace(DEFAULT_CONFIG, training=replace(DEFAULT_CONFIG.training, max_steps=0)),
            replace(DEFAULT_CONFIG, evaluation=replace(DEFAULT_CONFIG.evaluation, max_acf_lag=32)),
        ]
        for config in modifications:
            with self.subTest(config=config), self.assertRaises(ValueError):
                config.validate()

    def test_reproducible_random_streams(self):
        import numpy as np
        import torch
        def draw(seed):
            rng = seed_everything(seed)
            return random.random(), np.random.normal(size=5), rng.normal(size=5), torch.randn(5)
        a, b, c = draw(11), draw(11), draw(22)
        self.assertEqual(a[0], b[0])
        for i in (1, 2):
            np.testing.assert_array_equal(a[i], b[i])
            self.assertFalse(np.array_equal(a[i], c[i]))
        self.assertTrue(torch.equal(a[3], b[3]))
        self.assertFalse(torch.equal(a[3], c[3]))

    def test_seed_validation(self):
        for seed in (-1, 2**32, 1.5):
            with self.assertRaises(ValueError):
                seed_everything(seed)

    def test_strict_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.json"
            save_json(path, DEFAULT_CONFIG.to_dict())
            self.assertEqual(json.loads(path.read_text())["dataset"]["n_sequences"], 1400)
            with self.assertRaises(ValueError):
                save_json(path, {"metric": float("nan")})

if __name__ == "__main__":
    unittest.main()
