"""Explicit seeding and JSON provenance; no global seed changes on import."""
import json
import os
import platform
import random
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

PACKAGES = ("numpy", "scipy", "matplotlib", "torch", "qiskit", "qiskit-machine-learning", "pytest")

def seed_everything(seed: int, deterministic: bool = True):
    """Seed Python, NumPy and Torch; return an independent NumPy Generator.

    Set PYTHONHASHSEED before launching Python if hash-order reproducibility matters.
    This function cannot change the running interpreter's hash seed.
    """
    if not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("Seed must be an integer in [0, 2**32)")
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    return np.random.default_rng(seed)

def environment_metadata() -> dict:
    """Record versions even when a dependency is absent."""
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return {
        "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "machine": platform.machine(),
        "processor": platform.processor(), "cpu_count": os.cpu_count(),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"), "packages": versions,
    }

def save_json(path: str | Path, payload: dict) -> None:
    """Write strict JSON; NaN/Inf are rejected rather than stored as results."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
