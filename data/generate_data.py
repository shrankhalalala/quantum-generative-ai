"""Phase 2 CLI: generate/verify data artifacts and dataset-only figures."""
import argparse
import csv
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import DEFAULT_CONFIG
from data.artifacts import load_bundle, save_bundle, sha256
from data.dataset import generate_dataset
from data.plotting import plot_dataset
from utils.reproducibility import save_json
from utils.time_series import mean_acf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/datasets")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--verify-only", action="store_true", help="Reload/check existing data; no plots or writes")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace existing dataset artifacts")
    args = parser.parse_args()
    if args.verify_only:
        load_bundle(args.output_dir, DEFAULT_CONFIG)
        print("Saved dataset, splits, nested subsets, normalization and checksums verified.")
        return
    bundle = save_bundle(args.output_dir, generate_dataset(DEFAULT_CONFIG.dataset), DEFAULT_CONFIG, args.overwrite)
    figures = plot_dataset(bundle, args.results_dir / "figures", DEFAULT_CONFIG.evaluation.max_acf_lag)
    summary = bundle.metadata["summary"]
    save_json(args.results_dir / "tables/dataset_summary.json", summary)
    table = args.results_dir / "tables/dataset_summary.csv"
    with table.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, lineterminator="\n", fieldnames=("split", "n_sequences", "sequence_length", "mean", "std", "min", "max"))
        writer.writeheader()
        for split, record in summary.items():
            writer.writerow({"split": split, "n_sequences": record["shape"][0], "sequence_length": record["shape"][1],
                             **{name: record[name] for name in ("mean", "std", "min", "max")}})
    # Track small provenance outside ignored bulk artifacts.
    save_json(args.results_dir / "dataset_metadata.json", bundle.metadata)
    save_json(args.results_dir / "tables/normalization.json", {key: stats.to_dict() for key, stats in bundle.normalization.items()})
    save_json(args.results_dir / "dataset_validation.json", {
        "status": "passed", "data_directory": str(args.output_dir.resolve()),
        "artifact_checks": ["hashes", "equation including saved noise", "fixed split indices", "nested prefixes",
                            "subset-only normalization"], "raw_dataset_summary": summary["all"],
        "observation_noise_mean": float(bundle.dataset.observation_noise.mean()),
        "observation_noise_std": float(bundle.dataset.observation_noise.std(ddof=0)),
        "representative_indices": bundle.splits["test"][:6].tolist(),
        "held_out_mean_acf": mean_acf(bundle.dataset.sequences[bundle.splits["test"]], DEFAULT_CONFIG.evaluation.max_acf_lag).tolist(),
        "figure_sha256": {path.name: sha256(path) for path in figures},
        "dataset_metadata_sha256": sha256(args.output_dir / "metadata.json"),
        "models_trained": False,
    })
    print(f"Dataset saved and verified: {args.output_dir.resolve()}")
    print(f"Summary: {summary['all']}")
    print(f"Created {len(figures)} PNG/PDF plot artifacts; no model or training executed.")

if __name__ == "__main__":
    main()
