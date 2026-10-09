"""Dataset correctness, persistence, leakage prevention and ACF definitions."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from config import DEFAULT_CONFIG
from data.dataset import generate_dataset, make_splits, nested_subsets, validate_splits, clean_signal
from data.normalization import fit_normalization, prepare_splits
from data.artifacts import save_bundle, load_bundle, subset_key
from utils.time_series import sequence_acf, mean_acf


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = DEFAULT_CONFIG
        cls.dataset = generate_dataset(cls.config.dataset)
        cls.splits = make_splits(cls.config.dataset)

    def test_shape_grid_ranges_and_equation(self):
        dataset = self.dataset
        dataset.validate(self.config.dataset)
        self.assertEqual(dataset.sequences.shape, (1400, 32))
        self.assertEqual(dataset.sequences.dtype, np.float64)
        self.assertEqual(dataset.time[-1], 31/32)
        np.testing.assert_array_equal(dataset.sequences,
                                      clean_signal(dataset.time, dataset.parameters) + dataset.observation_noise)
        # Sanity-check realized noise, not a requirement of exact moments.
        self.assertLess(abs(dataset.observation_noise.mean()), .0015)
        self.assertLess(abs(dataset.observation_noise.std() - .05), .0015)

    def test_exact_repeatability_and_seed_sensitivity(self):
        repeat = generate_dataset(self.config.dataset)
        np.testing.assert_array_equal(repeat.sequences, self.dataset.sequences)
        for key in repeat.parameters:
            np.testing.assert_array_equal(repeat.parameters[key], self.dataset.parameters[key])
        changed = generate_dataset(replace(self.config.dataset, generation_seed=2025))
        self.assertFalse(np.array_equal(changed.sequences, repeat.sequences))
        self.assertFalse(np.array_equal(changed.parameters['amplitude'], repeat.parameters['amplitude']))

    def test_generation_does_not_consume_global_rng(self):
        np.random.seed(42)
        expected = np.random.random(5)
        np.random.seed(42)
        generate_dataset(self.config.dataset)
        make_splits(self.config.dataset)
        nested_subsets(self.splits['train'], self.config.dataset.subset_sizes, 11)
        np.testing.assert_array_equal(np.random.random(5), expected)

    def test_split_counts_disjoint_coverage_and_determinism(self):
        validate_splits(self.splits, self.config.dataset)
        self.assertEqual([len(self.splits[name]) for name in ('train', 'validation', 'test')], [1000, 200, 200])
        np.testing.assert_array_equal(np.sort(np.concatenate(list(self.splits.values()))), np.arange(1400))
        for name, repeat in make_splits(self.config.dataset).items():
            np.testing.assert_array_equal(repeat, self.splits[name])
        changed = make_splits(replace(self.config.dataset, split_seed=2028))
        self.assertFalse(np.array_equal(changed['train'], self.splits['train']))
        broken = {name: indices.copy() for name, indices in self.splits.items()}
        broken['validation'][0] = broken['train'][0]
        with self.assertRaises(ValueError):
            validate_splits(broken, self.config.dataset)

    def test_nested_subsets_all_seeds(self):
        for seed in self.config.training.seeds:
            subsets = nested_subsets(self.splits['train'], self.config.dataset.subset_sizes, seed)
            repeat = nested_subsets(self.splits['train'], self.config.dataset.subset_sizes, seed)
            for size, indices in subsets.items():
                self.assertEqual(indices.shape, (size,))
                self.assertEqual(np.unique(indices).size, size)
                self.assertTrue(set(indices) <= set(self.splits['train']))
                self.assertFalse(set(indices) & set(self.splits['validation']))
                self.assertFalse(set(indices) & set(self.splits['test']))
                np.testing.assert_array_equal(indices, subsets[1000][:size])
                np.testing.assert_array_equal(indices, repeat[size])
        seed11 = nested_subsets(self.splits['train'], (1000, 500, 250, 100), 11)
        seed22 = nested_subsets(self.splits['train'], (1000, 500, 250, 100), 22)
        self.assertFalse(np.array_equal(seed11[100], seed22[100]))

    def test_invalid_subsets_and_nonfinite_data_rejected(self):
        for sizes in ((1001,), (0,), (100, 500), (100, 100)):
            with self.assertRaises(ValueError):
                nested_subsets(self.splits['train'], sizes, 11)
        with self.assertRaises(ValueError):
            nested_subsets(np.array([0, 0]), (1,), 11)
        corrupt = replace(self.dataset, sequences=self.dataset.sequences.copy())
        corrupt.sequences[0, 0] = np.nan
        with self.assertRaises(ValueError):
            corrupt.validate(self.config.dataset)


class NormalizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = generate_dataset(DEFAULT_CONFIG.dataset)
        cls.splits = make_splits(DEFAULT_CONFIG.dataset)

    def test_training_statistics_and_round_trip_all_conditions(self):
        for seed in DEFAULT_CONFIG.training.seeds:
            for size, indices in nested_subsets(self.splits['train'], DEFAULT_CONFIG.dataset.subset_sizes, seed).items():
                with self.subTest(seed=seed, size=size):
                    train = self.dataset.sequences[indices]
                    stats = fit_normalization(train)
                    self.assertEqual(stats.mean, float(train.mean()))
                    self.assertEqual(stats.std, float(train.std(ddof=0)))
                    normalized = prepare_splits(self.dataset.sequences, indices, self.splits['validation'], self.splits['test'], stats)
                    self.assertEqual(normalized['train'].shape, (size, 32))
                    self.assertAlmostEqual(float(normalized['train'].mean()), 0., places=12)
                    self.assertAlmostEqual(float(normalized['train'].std()), 1., places=12)
                    for name, source_indices in (('train', indices), ('validation', self.splits['validation']), ('test', self.splits['test'])):
                        np.testing.assert_allclose(stats.inverse_transform(normalized[name]),
                                                   self.dataset.sequences[source_indices], rtol=1e-14, atol=1e-14)

    def test_evaluation_and_unused_training_values_cannot_affect_fit(self):
        indices = nested_subsets(self.splits['train'], (100,), 11)[100]
        before = fit_normalization(self.dataset.sequences[indices])
        changed = self.dataset.sequences.copy()
        excluded = np.setdiff1d(np.arange(1400), indices)
        changed[excluded] = 1e6
        after = fit_normalization(changed[indices])
        self.assertEqual(before, after)
        transformed = prepare_splits(changed, indices, self.splits['validation'], self.splits['test'], after)
        expected = (changed[self.splits['test']] - before.mean) / before.std
        np.testing.assert_array_equal(transformed['test'], expected)
        self.assertGreater(float(transformed['test'].mean()), 1e5)
        self.assertNotEqual(before.std, fit_normalization(changed).std)

    def test_bad_normalization_inputs(self):
        for values in (np.ones((3,32)), np.empty((0,32)), np.array([1.,2.]), np.full((2,32), np.nan), np.full((2,32), np.inf)):
            with self.assertRaises(ValueError):
                fit_normalization(values)
        stats = fit_normalization(self.dataset.sequences[:100])
        with self.assertRaises(ValueError):
            stats.transform(np.ones((2,31)))


class ACFTests(unittest.TestCase):
    def test_hand_calculated_biased_acf(self):
        values = np.array([[1.,2.,3.,4.]])
        np.testing.assert_allclose(sequence_acf(values, 3), [[1., .25, -.3, -.45]], atol=1e-15)

    def test_per_sequence_centering_and_aggregate(self):
        values = np.array([[1.,2.,3.,4.], [2.,-1.,-1.,2.]])
        acfs = sequence_acf(values, 3)
        np.testing.assert_allclose(mean_acf(values, 3), acfs.mean(axis=0))
        np.testing.assert_allclose(sequence_acf(values*np.array([[2.],[3.]])+np.array([[50.],[-20.]]), 3), acfs)
        # Row boundaries never affect results: computing each row separately is identical.
        np.testing.assert_allclose(acfs, np.vstack([sequence_acf(row[None,:],3) for row in values]))

    def test_undefined_acf_and_bad_lags_rejected(self):
        with self.assertRaisesRegex(ValueError, '1 near-constant'):
            sequence_acf(np.array([[1.,1.,1.,1.], [1.,2.,3.,4.]]), 2)
        for lag in (-1, 4, 1.5):
            with self.assertRaises(ValueError):
                sequence_acf(np.array([[1.,2.,3.,4.]]), lag)


class ArtifactTests(unittest.TestCase):
    def test_save_reload_normalization_and_overwrite_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            dataset = generate_dataset(DEFAULT_CONFIG.dataset)
            saved = save_bundle(path, dataset, DEFAULT_CONFIG)
            loaded = load_bundle(path, DEFAULT_CONFIG)
            np.testing.assert_array_equal(loaded.dataset.sequences, dataset.sequences)
            np.testing.assert_array_equal(loaded.dataset.observation_noise, dataset.observation_noise)
            self.assertEqual(len(loaded.subsets), 12)
            self.assertEqual(loaded.metadata['selection']['representative_indices'], loaded.splits['test'][:6].tolist())
            self.assertEqual(loaded.metadata['summary']['all']['shape'], [1400,32])
            for seed in DEFAULT_CONFIG.training.seeds:
                for size in DEFAULT_CONFIG.dataset.subset_sizes:
                    key = subset_key(seed, size)
                    self.assertEqual(saved.normalization[key], loaded.normalization[key])
                    np.testing.assert_array_equal(saved.normalized(seed,size)['train'], loaded.normalized(seed,size)['train'])
            with self.assertRaises(FileExistsError):
                save_bundle(path, dataset, DEFAULT_CONFIG)
            mismatched = replace(DEFAULT_CONFIG, dataset=replace(DEFAULT_CONFIG.dataset, generation_seed=42))
            with self.assertRaisesRegex(ValueError, 'configuration differs'):
                load_bundle(path, mismatched)

    def test_later_training_budget_does_not_invalidate_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            save_bundle(path, generate_dataset(DEFAULT_CONFIG.dataset), DEFAULT_CONFIG)
            with_budget = replace(DEFAULT_CONFIG, training=replace(DEFAULT_CONFIG.training, max_steps=5))
            bundle = load_bundle(path, with_budget)
            self.assertEqual(bundle.normalized(11,100)['train'].shape, (100,32))
            changed_seeds = replace(DEFAULT_CONFIG, training=replace(DEFAULT_CONFIG.training, seeds=(11,)))
            with self.assertRaisesRegex(ValueError, 'repetition seeds'):
                load_bundle(path, changed_seeds)

    def test_corrupted_artifact_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            save_bundle(path, generate_dataset(DEFAULT_CONFIG.dataset), DEFAULT_CONFIG)
            artifact = path/'normalization.json'
            records = json.loads(artifact.read_text())
            records['seed_11_n_100']['mean'] = 999.
            artifact.write_text(json.dumps(records))
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                load_bundle(path, DEFAULT_CONFIG)


if __name__ == '__main__':
    unittest.main()
