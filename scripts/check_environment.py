"""Phase 1 import/API checks only: no experiment circuit, model or data."""
import argparse
import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import DEFAULT_CONFIG
from utils.reproducibility import environment_metadata, save_json, seed_everything

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/environment.json")
    args = parser.parse_args()
    report = environment_metadata()
    try:
        DEFAULT_CONFIG.validate()
        import numpy as np
        import scipy
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import torch
        from qiskit.primitives import StatevectorEstimator
        from qiskit_machine_learning.neural_networks import EstimatorQNN
        from qiskit_machine_learning.connectors import TorchConnector
        from qiskit_machine_learning.gradients import ParamShiftEstimatorGradient
        assert "input_gradients" in inspect.signature(EstimatorQNN).parameters
        assert "default_precision" in inspect.signature(EstimatorQNN).parameters
        assert "default_precision" in inspect.signature(StatevectorEstimator).parameters
        seed_everything(11)
        a = torch.randn(3)
        seed_everything(11)
        assert torch.equal(a, torch.randn(3))
        x = torch.tensor([1., 2.], requires_grad=True)
        x.square().sum().backward()
        assert torch.equal(x.grad, torch.tensor([2., 4.]))
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        fig.canvas.draw()
        plt.close(fig)
        report.update(status="passed", checks=["configuration", "scientific imports", "V2 estimator/QNN/connector/gradient imports", "API signatures", "CPU autograd", "repeatable RNG", "headless plotting"],
                      torch_device="cpu", cuda_available=torch.cuda.is_available(),
                      mps_available=torch.backends.mps.is_available(), torch_threads=torch.get_num_threads(),
                      simulator="planned local StatevectorEstimator, precision=0", quantum_execution_tested=False,
                      full_qnn_gradient_compatibility="Not evaluated; Phase 4")
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        save_json(args.output, report)
        raise
    save_json(args.output, report)
    print(f"Environment checks passed; report: {args.output}")

if __name__ == "__main__":
    main()
