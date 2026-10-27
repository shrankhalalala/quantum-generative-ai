"""Plot actual run losses and generated samples; no invented comparisons."""
from pathlib import Path
import numpy as np


def plot_run(history: list[dict], generated: np.ndarray, time: np.ndarray,
             directory: Path, run_id: str, model_label: str = "Classical", smoke: bool = True) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    def export(fig, stem):
        for extension in ("png","pdf"):
            path = directory/f"{run_id}_{stem}.{extension}"
            fig.savefig(path,dpi=300,bbox_inches="tight")
            paths.append(path)
        plt.close(fig)
    with plt.rc_context({"font.size":10,"pdf.fonttype":42,"axes.spines.top":False,"axes.spines.right":False}):
        fig,ax=plt.subplots(figsize=(8,4.5),layout="constrained")
        ax.plot([row['step'] for row in history], [row['train_loss'] for row in history], marker='o',label='Train: interval mean MSE')
        ax.plot([row['step'] for row in history], [row['validation_loss'] for row in history], marker='s',label='Validation: fixed noise/timesteps')
        ax.set(xlabel='Optimizer updates',ylabel='Noise-prediction MSE',title=f'{model_label} DDPM: losses' + (' (smoke test; not convergence evidence)' if smoke else ''))
        ax.legend(frameon=False); ax.grid(alpha=.2)
        export(fig,'loss')
        fig,axes=plt.subplots(3,2,figsize=(10,7),sharex=True,sharey=True,layout='constrained')
        for index,ax in enumerate(axes.flat):
            ax.plot(time,generated[index],linewidth=1.5,color='#2463a0')
            ax.set_title(f'Generated index {index}'); ax.grid(alpha=.2)
        fig.supxlabel('Time (observation windows)'); fig.supylabel('Signal amplitude (arbitrary units)')
        fig.suptitle(f'{model_label} DDPM: first six generated samples' + (' (smoke test; not a quality claim)' if smoke else ''))
        export(fig,'generated_series')
    return paths
