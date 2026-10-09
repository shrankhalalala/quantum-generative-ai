"""Portable NPZ/JSON artifacts with checksums and verified reload."""
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
from config import ExperimentConfig, PAPER_TITLE
from data.dataset import PARAMETER_NAMES, SyntheticDataset, make_splits, nested_subsets, validate_splits
from data.normalization import NormalizationStats, fit_normalization, prepare_splits
from utils.reproducibility import environment_metadata, save_json

SCHEMA_VERSION = 1


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(values: np.ndarray) -> dict:
    return {"shape": list(values.shape), "mean": float(values.mean()),
            "std": float(values.std(ddof=0)), "min": float(values.min()), "max": float(values.max())}


def subset_key(seed: int, size: int) -> str:
    return f"seed_{seed}_n_{size}"

@dataclass
class DatasetBundle:
    dataset: SyntheticDataset
    splits: dict[str, np.ndarray]
    subsets: dict[str, np.ndarray]
    normalization: dict[str, NormalizationStats]
    metadata: dict

    def normalized(self, seed: int, size: int) -> dict[str, np.ndarray]:
        key = subset_key(seed, size)
        return prepare_splits(self.dataset.sequences, self.subsets[key], self.splits["validation"],
                              self.splits["test"], self.normalization[key])


def save_bundle(directory: Path, dataset: SyntheticDataset, config: ExperimentConfig,
                overwrite: bool = False) -> DatasetBundle:
    config.validate()
    dataset.validate(config.dataset)
    directory = Path(directory)
    names = ("dataset.npz", "subsets.npz", "normalization.json", "metadata.json")
    if not overwrite and any((directory / name).exists() for name in names):
        raise FileExistsError("Dataset artifacts already exist; verify them or explicitly use --overwrite")
    directory.mkdir(parents=True, exist_ok=True)
    splits = make_splits(config.dataset)
    subsets, normalizers = {}, {}
    for seed in config.training.seeds:
        for size, indices in nested_subsets(splits["train"], config.dataset.subset_sizes, seed).items():
            key = subset_key(seed, size)
            subsets[key] = indices
            normalizers[key] = fit_normalization(dataset.sequences[indices])
    np.savez_compressed(directory / "dataset.npz", sequences=dataset.sequences, time=dataset.time,
                        observation_noise=dataset.observation_noise, **dataset.parameters,
                        **{f"{name}_indices": indices for name, indices in splits.items()})
    np.savez_compressed(directory / "subsets.npz", **subsets)
    save_json(directory / "normalization.json", {key: stats.to_dict() for key, stats in normalizers.items()})
    metadata = {
        "schema_version": SCHEMA_VERSION, "paper_title": PAPER_TITLE,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "config": config.to_dict(),
        "generation": {"equation": "A*sin(2*pi*f*t+phi)+B*sin(4*pi*f*t+psi)+epsilon(t)",
                       "parameter_columns": list(PARAMETER_NAMES),
                       "rng": "NumPy default_rng (PCG64)", "draw_order": "A,f,phi,B,psi,noise",
                       "phases": "independent Uniform[0,2*pi)", "endpoint": False,
                       "noise": "independent Normal(0, noise_std**2)", "dtype": str(dataset.sequences.dtype)},
        "selection": {"split": "seeded permutation: first train, then validation, then test",
                      "subsets": "one train-index permutation per repetition seed; nested prefixes",
                      "representative_indices": splits["test"][:6].tolist()},
        "normalization": "per-subset pooled scalar z-score; training entries only; ddof=0",
        "summary": {"all": summarize(dataset.sequences), **{name: summarize(dataset.sequences[indices])
                    for name, indices in splits.items()}}, "environment": environment_metadata(),
        "sha256": {name: sha256(directory/name) for name in names if name != "metadata.json"},
    }
    save_json(directory / "metadata.json", metadata)
    return load_bundle(directory, config)


def load_bundle(directory: Path, config: ExperimentConfig) -> DatasetBundle:
    """Check hashes, equation, exact split/subset selection and fitted stats."""
    config.validate()
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    if metadata["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported dataset schema")
    requested = json.loads(json.dumps(config.to_dict()))
    # Training budgets/model settings can change later without changing the data.
    if (metadata["config"]["dataset"] != requested["dataset"]
            or metadata["config"]["training"]["seeds"] != requested["training"]["seeds"]):
        raise ValueError("Artifact configuration differs in dataset settings or repetition seeds")
    for name in ("dataset.npz", "subsets.npz", "normalization.json"):
        if sha256(directory/name) != metadata["sha256"][name]:
            raise ValueError(f"Artifact checksum mismatch: {name}")
    with np.load(directory / "dataset.npz", allow_pickle=False) as archive:
        dataset = SyntheticDataset(archive["sequences"], archive["time"],
                  {name: archive[name] for name in PARAMETER_NAMES}, archive["observation_noise"])
        splits = {name: archive[f"{name}_indices"] for name in ("train", "validation", "test")}
    dataset.validate(config.dataset)
    validate_splits(splits, config.dataset)
    for name, expected in make_splits(config.dataset).items():
        if not np.array_equal(splits[name], expected):
            raise ValueError(f"Split seed/index mismatch: {name}")
    with np.load(directory / "subsets.npz", allow_pickle=False) as archive:
        subsets = {name: archive[name] for name in archive.files}
    records = json.loads((directory / "normalization.json").read_text())
    normalizers = {key: NormalizationStats(**value) for key, value in records.items()}
    expected_keys = {subset_key(seed, size) for seed in config.training.seeds for size in config.dataset.subset_sizes}
    if set(subsets) != expected_keys or set(normalizers) != expected_keys:
        raise ValueError("Missing or unexpected subset/normalization records")
    for seed in config.training.seeds:
        for size, expected in nested_subsets(splits["train"], config.dataset.subset_sizes, seed).items():
            key = subset_key(seed, size)
            if not np.array_equal(subsets[key], expected):
                raise ValueError(f"Subset seed/index mismatch: {key}")
            if normalizers[key] != fit_normalization(dataset.sequences[expected]):
                raise ValueError(f"Normalization does not match training-only statistics: {key}")
    return DatasetBundle(dataset, splits, subsets, normalizers, metadata)
