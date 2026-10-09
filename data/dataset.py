"""Deterministic synthetic sequences and index-only split/subset selection."""
from dataclasses import dataclass
import numpy as np
from config import DatasetConfig

PARAMETER_NAMES = ("amplitude", "frequency", "phase", "harmonic_amplitude", "harmonic_phase")

@dataclass
class SyntheticDataset:
    sequences: np.ndarray
    time: np.ndarray
    parameters: dict[str, np.ndarray]
    observation_noise: np.ndarray

    def validate(self, config: DatasetConfig) -> None:
        shape = (config.n_sequences, config.sequence_length)
        if self.sequences.shape != shape or self.observation_noise.shape != shape:
            raise ValueError("Incorrect sequence or noise shape")
        if self.sequences.dtype != np.float64 or self.observation_noise.dtype != np.float64:
            raise ValueError("Raw sequences/noise must retain float64 precision")
        if not np.array_equal(self.time, np.arange(config.sequence_length) / config.sequence_length):
            raise ValueError("Time grid must use endpoint=False over [0,1)")
        ranges = (config.amplitude_range, config.frequency_range, (0., 2*np.pi),
                  config.harmonic_range, (0., 2*np.pi))
        if set(self.parameters) != set(PARAMETER_NAMES):
            raise ValueError("Missing or unexpected signal parameters")
        for name, (low, high) in zip(PARAMETER_NAMES, ranges):
            values = self.parameters[name]
            if values.shape != (config.n_sequences,) or not np.isfinite(values).all():
                raise ValueError(f"Invalid parameter {name}")
            if np.any(values < low) or np.any(values >= high):
                raise ValueError(f"Parameter out of uniform sampling range: {name}")
        if not np.isfinite(self.sequences).all() or not np.isfinite(self.observation_noise).all():
            raise ValueError("Dataset contains non-finite values")
        reconstructed = clean_signal(self.time, self.parameters) + self.observation_noise
        if not np.array_equal(self.sequences, reconstructed):
            raise ValueError("Saved sequences do not match equation and stored noise")


def clean_signal(time: np.ndarray, parameters: dict[str, np.ndarray]) -> np.ndarray:
    """Broadcast per-sequence parameters against the time grid (N,L)."""
    p = {key: value[:, None] for key, value in parameters.items()}
    return (p["amplitude"] * np.sin(2*np.pi*p["frequency"]*time + p["phase"])
            + p["harmonic_amplitude"] * np.sin(4*np.pi*p["frequency"]*time + p["harmonic_phase"]))


def generate_dataset(config: DatasetConfig) -> SyntheticDataset:
    """Generate in documented A,f,phi,B,psi,noise RNG draw order."""
    if config.n_sequences <= 0 or config.sequence_length != 32 or config.noise_std <= 0:
        raise ValueError("Invalid dataset dimensions/noise")
    rng = np.random.default_rng(config.generation_seed)
    ranges = (config.amplitude_range, config.frequency_range, (0., 2*np.pi),
              config.harmonic_range, (0., 2*np.pi))
    parameters = {name: rng.uniform(low, high, config.n_sequences)
                  for name, (low, high) in zip(PARAMETER_NAMES, ranges)}
    time = np.arange(config.sequence_length, dtype=np.float64) / config.sequence_length
    noise = rng.normal(0., config.noise_std, (config.n_sequences, config.sequence_length))
    dataset = SyntheticDataset(clean_signal(time, parameters) + noise, time, parameters, noise)
    dataset.validate(config)
    return dataset


def make_splits(config: DatasetConfig) -> dict[str, np.ndarray]:
    if any(n <= 0 for n in config.split_sizes) or sum(config.split_sizes) != config.n_sequences:
        raise ValueError("Invalid split counts")
    order = np.random.default_rng(config.split_seed).permutation(config.n_sequences)
    first, second = np.cumsum(config.split_sizes)[:2]
    splits = dict(zip(("train", "validation", "test"), np.split(order, [first, second])))
    validate_splits(splits, config)
    return splits


def validate_splits(splits: dict[str, np.ndarray], config: DatasetConfig) -> None:
    if set(splits) != {"train", "validation", "test"}:
        raise ValueError("Expected train/validation/test indices")
    for name, size in zip(("train", "validation", "test"), config.split_sizes):
        indices = splits[name]
        if indices.shape != (size,) or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError(f"Invalid {name} index shape/dtype")
    combined = np.concatenate([splits[name] for name in ("train", "validation", "test")])
    if not np.array_equal(np.sort(combined), np.arange(config.n_sequences)):
        raise ValueError("Splits must be disjoint and cover the dataset exactly")


def nested_subsets(train_indices: np.ndarray, sizes: tuple[int, ...], seed: int) -> dict[int, np.ndarray]:
    """One permutation, then prefixes; indices refer to original raw rows."""
    if train_indices.ndim != 1 or not np.issubdtype(train_indices.dtype, np.integer):
        raise ValueError("Training indices must be a one-dimensional integer array")
    if np.unique(train_indices).size != train_indices.size:
        raise ValueError("Training indices must be unique")
    if not sizes or any(size <= 0 or size > len(train_indices) for size in sizes):
        raise ValueError("Subset size outside training pool")
    if tuple(sorted(set(sizes), reverse=True)) != sizes:
        raise ValueError("Subset sizes must be distinct and descending")
    order = np.random.default_rng(seed).permutation(train_indices)
    return {size: order[:size].copy() for size in sizes}
