"""Held-out generation diagnostics in original signal units, never training loss."""
import numpy as np
from scipy.stats import wasserstein_distance
from data.normalization import finite_sequences
from utils.time_series import mean_acf


def evaluate_samples(generated: np.ndarray, reference: np.ndarray, max_lag: int = 10) -> dict:
    generated, reference = finite_sequences(generated), finite_sequences(reference)
    if generated.shape[1] != reference.shape[1]:
        raise ValueError("Generated/reference sequence lengths must match")
    if not isinstance(max_lag, int) or not 1 <= max_lag < generated.shape[1]:
        raise ValueError("Evaluation lag must be an integer in [1,L)")
    result = {"n_generated": len(generated), "n_reference": len(reference),
              "marginal_wasserstein_1": float(wasserstein_distance(generated.ravel(), reference.ravel())),
              "max_acf_lag": max_lag}
    for name, operation in (("mean", np.mean), ("std", np.std), ("variance", np.var)):
        actual, expected = float(operation(generated)), float(operation(reference))
        result[name] = {"generated": actual, "reference": expected,
                        "difference": actual - expected, "absolute_error": abs(actual - expected)}
    bad_rows = {}
    for name, values in (("generated", generated), ("reference", reference)):
        energy = ((values - values.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
        bad_rows[name] = np.flatnonzero(energy <= 1e-12).tolist()
    result["near_constant_rows"] = bad_rows
    if any(bad_rows.values()):
        result.update(acf_status="undefined: near-constant rows; no rows excluded",
                      acf_mae=None, generated_mean_acf=None, reference_mean_acf=None)
    else:
        actual, expected = mean_acf(generated, max_lag), mean_acf(reference, max_lag)
        result.update(acf_status="ok", acf_mae=float(np.abs(actual[1:] - expected[1:]).mean()),
                      generated_mean_acf=actual.tolist(), reference_mean_acf=expected.tolist())
    return result
