"""Shared biased per-sequence ACF for dataset plots and later evaluation."""
import numpy as np
from data.normalization import finite_sequences


def sequence_acf(sequences: np.ndarray, max_lag: int = 10) -> np.ndarray:
    """r(l)=sum(y[t]*y[t+l])/sum(y[t]**2), after demeaning each row.

    Uses the same full-window denominator at every lag, no N-lag correction.
    Undefined near-constant rows cause an explicit error instead of exclusion.
    """
    values = finite_sequences(sequences)
    if not isinstance(max_lag, int) or not 0 <= max_lag < values.shape[1]:
        raise ValueError("ACF lag must be an integer in [0,L)")
    centered = values - values.mean(axis=1, keepdims=True)
    denominator = np.sum(centered**2, axis=1)
    bad = np.flatnonzero(denominator <= 1e-12)
    if bad.size:
        raise ValueError(f"ACF undefined for {bad.size} near-constant sequences: rows {bad.tolist()}")
    acf = np.empty((len(values), max_lag + 1))
    acf[:, 0] = 1.
    for lag in range(1, max_lag + 1):
        acf[:, lag] = np.sum(centered[:, :-lag] * centered[:, lag:], axis=1) / denominator
    return acf


def mean_acf(sequences: np.ndarray, max_lag: int = 10) -> np.ndarray:
    return sequence_acf(sequences, max_lag).mean(axis=0)
