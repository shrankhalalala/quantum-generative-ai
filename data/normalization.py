"""Scalar z-score fitted solely on a caller-supplied training subset."""
from dataclasses import asdict, dataclass
import numpy as np


def finite_sequences(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or 0 in values.shape or not np.isfinite(values).all():
        raise ValueError("Expected a nonempty finite (N,L) array")
    return values

@dataclass(frozen=True)
class NormalizationStats:
    mean: float
    std: float
    n_sequences: int
    sequence_length: int
    ddof: int = 0

    def validate(self) -> None:
        if not np.isfinite(self.mean) or not np.isfinite(self.std) or self.std <= 1e-12:
            raise ValueError("Invalid or near-zero normalization scale")
        if self.n_sequences <= 0 or self.sequence_length <= 0 or self.ddof != 0:
            raise ValueError("Invalid normalization metadata")

    def transform(self, values: np.ndarray) -> np.ndarray:
        self.validate()
        values = finite_sequences(values)
        if values.shape[1] != self.sequence_length:
            raise ValueError("Sequence length differs from fitted normalization")
        return (values - self.mean) / self.std

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        self.validate()
        values = finite_sequences(values)
        if values.shape[1] != self.sequence_length:
            raise ValueError("Sequence length differs from fitted normalization")
        return values * self.std + self.mean

    def to_dict(self) -> dict:
        self.validate()
        return asdict(self)


def fit_normalization(training_subset: np.ndarray) -> NormalizationStats:
    """Global pooled mean/std, ddof=0; no validation/test argument exists."""
    values = finite_sequences(training_subset)
    stats = NormalizationStats(float(values.mean()), float(values.std(ddof=0)), *values.shape)
    stats.validate()
    return stats


def prepare_splits(sequences: np.ndarray, subset_indices: np.ndarray,
                   validation_indices: np.ndarray, test_indices: np.ndarray,
                   stats: NormalizationStats) -> dict[str, np.ndarray]:
    """Reuse recorded subset-only statistics, never refit on evaluation data."""
    return {name: stats.transform(sequences[indices]) for name, indices in
            (("train", subset_indices), ("validation", validation_indices), ("test", test_indices))}
