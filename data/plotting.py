"""Dataset-only academic PNG/PDF figures from fixed held-out examples."""
from pathlib import Path
import numpy as np
from data.artifacts import DatasetBundle
from utils.time_series import sequence_acf, mean_acf


def plot_dataset(bundle: DatasetBundle, directory: Path, max_lag: int = 10) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    indices = bundle.splits["test"][:6]
    values = bundle.dataset.sequences[indices]
    time = bundle.dataset.time
    paths = []
    def export(fig, stem):
        for extension in ("png", "pdf"):
            path = directory / f"{stem}.{extension}"
            fig.savefig(path, dpi=300, bbox_inches="tight")
            paths.append(path)
        plt.close(fig)
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig, axes = plt.subplots(3, 2, figsize=(10, 7), sharex=True, sharey=True, layout="constrained")
        for ax, index, signal in zip(axes.flat, indices, values):
            ax.plot(time, signal, color="#2463a0", linewidth=1.6)
            ax.set_title(f"Held-out row {index}")
            ax.grid(alpha=.2)
        fig.supxlabel("Time (observation windows)")
        fig.supylabel("Signal amplitude (arbitrary units)")
        fig.suptitle("Real synthetic time series: six fixed held-out examples")
        export(fig, "dataset_representative_series")

        fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
        bins = np.linspace(bundle.dataset.sequences.min(), bundle.dataset.sequences.max(), 65)
        ax.hist(bundle.dataset.sequences.ravel(), bins=bins, density=True, color="#7d9cbe", alpha=.55,
                label="All 1,400 sequences (descriptive)")
        ax.hist(bundle.dataset.sequences[bundle.splits["test"]].ravel(), bins=bins, density=True,
                histtype="step", color="#ad4a20", linewidth=1.6, label="Held-out reference (200)")
        ax.set(xlabel="Signal amplitude (arbitrary units)", ylabel="Probability density (1/amplitude unit)",
               title="Pooled real-data amplitude distribution")
        ax.legend(frameon=False)
        ax.grid(alpha=.2)
        export(fig, "dataset_amplitude_distribution")

        fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
        lags = np.arange(max_lag+1)
        for index, acf in zip(indices, sequence_acf(values, max_lag)):
            ax.plot(lags, acf, linewidth=1., alpha=.65, label=f"Row {index}")
        ax.plot(lags, mean_acf(bundle.dataset.sequences[bundle.splits["test"]], max_lag),
                color="black", linewidth=2.5, linestyle="--", label="Held-out mean (200)")
        ax.axhline(0., color="gray", linewidth=.7)
        ax.set(xlabel="Lag (sample intervals)", ylabel="Autocorrelation (dimensionless)",
               title="Real-data ACF: per-sequence demeaning, biased normalization", xticks=lags, ylim=(-1.05, 1.05))
        ax.legend(frameon=False, ncol=2, fontsize=8)
        ax.grid(alpha=.2)
        export(fig, "dataset_autocorrelation")
    return paths
