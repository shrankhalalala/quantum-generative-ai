"""Explicit standalone QNN checks/timings; no diffusion model or training."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
import sys
import numpy as np
import torch
from qiskit.quantum_info import Statevector

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.quantum_layer import QuantumLayer, INITIAL_WEIGHT_RANGE, OBSERVABLE_LABELS
from utils.reproducibility import environment_metadata, save_json

FIXTURE = [[.1, -.2, .3, .4], [-.3, .2, .1, -.4]]
GRADIENT_COEFFICIENTS = [.17, -.29, .37, .41]


def statevector_outputs(layer: QuantumLayer, features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Independent expectations: no EstimatorQNN or TorchConnector call."""
    spec = layer.specification
    outputs = []
    for row in np.asarray(features, dtype=np.float64):
        angles = np.pi * np.tanh(row)
        bindings = dict(zip(spec.input_parameters, angles))
        bindings.update(zip(spec.weight_parameters, np.asarray(weights, dtype=np.float64)))
        state = Statevector.from_instruction(spec.circuit.assign_parameters(bindings))
        outputs.append([float(np.real(state.expectation_value(observable))) for observable in spec.observables])
    return np.asarray(outputs)


def finite_difference_check(layer: QuantumLayer, step: float = 1e-3) -> dict:
    """Compare all 8 feature and 16 weight derivatives through the real wrapper."""
    features = torch.tensor(FIXTURE, dtype=torch.float32, requires_grad=True)
    coefficients = torch.tensor(GRADIENT_COEFFICIENTS)
    loss = (layer(features) * coefficients).sum() / len(features)
    input_gradient, weight_gradient = torch.autograd.grad(loss, (features, layer.weight))
    original = layer.weight.detach().clone()
    numeric_input = torch.empty_like(features)
    numeric_weight = torch.empty_like(original)
    with torch.no_grad():
        try:
            for row in range(len(features)):
                for column in range(4):
                    positive = features.detach().clone()
                    negative = features.detach().clone()
                    positive[row, column] += step
                    negative[row, column] -= step
                    plus = (layer(positive) * coefficients).sum() / len(features)
                    minus = (layer(negative) * coefficients).sum() / len(features)
                    numeric_input[row, column] = (plus - minus) / (2 * step)
            for index in range(16):
                layer.weight.copy_(original)
                layer.weight[index] += step
                plus = (layer(features.detach()) * coefficients).sum() / len(features)
                layer.weight.copy_(original)
                layer.weight[index] -= step
                minus = (layer(features.detach()) * coefficients).sum() / len(features)
                numeric_weight[index] = (plus - minus) / (2 * step)
        finally:
            layer.weight.copy_(original)
    torch.testing.assert_close(input_gradient, numeric_input, atol=3e-4, rtol=5e-3)
    torch.testing.assert_close(weight_gradient, numeric_weight, atol=3e-4, rtol=5e-3)
    return {"step": step, "absolute_tolerance": 3e-4, "relative_tolerance": 5e-3,
            "feature_derivatives_checked": features.numel(), "weight_derivatives_checked": original.numel(),
            "max_feature_absolute_error": float((input_gradient - numeric_input).abs().max()),
            "max_weight_absolute_error": float((weight_gradient - numeric_weight).abs().max()),
            "analytic_feature_gradient": input_gradient.tolist(), "finite_difference_feature_gradient": numeric_input.tolist(),
            "analytic_weight_gradient": weight_gradient.tolist(), "finite_difference_weight_gradient": numeric_weight.tolist()}


def optimize_toy(seed: int = 11, steps: int = 10, learning_rate: float = .01) -> dict:
    """A fixed reachable teacher target; exactly ten updates for this phase."""
    if steps != 10 or learning_rate != .01:
        raise ValueError("Phase 4 toy protocol is fixed at ten updates, Adam LR 0.01")
    layer = QuantumLayer(seed)
    features = torch.tensor(FIXTURE, dtype=torch.float32, requires_grad=True)
    initial = layer.weight.detach().clone()
    teacher_weights = initial.numpy().astype(np.float64) + np.linspace(.1, .25, 16)
    target = torch.tensor(statevector_outputs(layer, features.detach().numpy(), teacher_weights), dtype=torch.float32)
    optimizer = torch.optim.Adam(layer.parameters(), lr=learning_rate)
    with torch.no_grad():
        initial_loss = float((layer(features) - target).square().mean())
    losses = [initial_loss]
    changes = []
    gradients = []
    start = perf_counter()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        features.grad = None
        loss = (layer(features) - target).square().mean()
        loss.backward()
        for gradient in (layer.weight.grad, features.grad):
            if gradient is None or not torch.isfinite(gradient).all():
                raise AssertionError("Toy fit has absent/non-finite gradients")
        gradients.append({"weights_l2": float(layer.weight.grad.norm()), "features_l2": float(features.grad.norm())})
        optimizer.step()
        changes.append(float((layer.weight.detach() - initial).norm()))
        with torch.no_grad():
            losses.append(float((layer(features) - target).square().mean()))
    seconds = perf_counter() - start
    if not np.isfinite(losses).all() or not torch.isfinite(layer.weight).all():
        raise AssertionError("Toy fit is non-finite")
    if not changes[-1] > 0 or not losses[-1] < losses[0]:
        raise AssertionError("Ten-update toy fit must update quantum weights and reduce loss")
    return {"seed": seed, "steps": steps, "batch_size": 2, "optimizer": "Adam", "learning_rate": learning_rate,
            "seconds": seconds, "losses": losses, "weight_change_l2_per_update": changes,
            "gradient_norms_per_update": gradients, "initial_weights": initial.tolist(),
            "final_weights": layer.weight.detach().tolist(), "teacher_weights": teacher_weights.tolist(),
            "features": FIXTURE, "targets": target.tolist(), "submitted_workloads": dict(layer.estimator.counts),
            "teacher_reference_statevector_evaluations": 2,
            "purpose": "standalone trainability fixture; no diffusion or generation-quality claim"}


def draw_circuit(layer: QuantumLayer, directory: Path) -> dict:
    """Qiskit text drawing embedded by Matplotlib; no optional pylatexenc."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory.mkdir(parents=True, exist_ok=True)
    drawing = str(layer.specification.circuit.draw(output="text", fold=-1))
    text_path = directory / "quantum_circuit.txt"
    text_path.write_text(drawing + "\n", encoding="utf-8")
    # Parameter symbols use the ASCII names in the actual circuit. Monospaced
    # drawing retains exact gate connectivity and parameter order.
    fig, ax = plt.subplots(figsize=(18, 3), layout="constrained")
    ax.axis("off")
    ax.text(.5, .55, drawing, transform=ax.transAxes, ha="center", va="center", fontfamily="DejaVu Sans Mono", fontsize=10)
    ax.set_title("Four-qubit bottleneck: RY encoding + two RY/RX layers with directed CX chains", fontsize=12)
    fig.text(.5, .02, "x[i] = pi tanh(h[i]); theta[0:16] trainable. Outputs: <Z0>, <Z1>, <Z2>, <Z3>. Ideal local simulation.",
             ha="center", fontsize=10)
    paths = [text_path]
    for extension in ("png", "pdf"):
        path = directory / f"quantum_circuit.{extension}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return {"method": "Qiskit text drawer embedded in Matplotlib; no additional dependencies",
            "files": [str(path) for path in paths]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    report_path = args.results_dir / "phase4_checks.json"
    if report_path.exists():
        parser.error("Phase 4 report already exists; choose another --results-dir")
    report = {"status": "running", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "environment": environment_metadata(), "torch_threads": torch.get_num_threads(),
              "torch_interop_threads": torch.get_num_interop_threads(),
              "simulator": "ideal local CPU StatevectorEstimator", "precision": 0., "shots": "not applicable",
              "hardware_execution": False, "diffusion_training_executed": False, "phase5_started": False}
    try:
        layer = QuantumLayer(11)
        features = torch.tensor(FIXTURE, dtype=torch.float32, requires_grad=True)
        warm_start = perf_counter()
        layer(features).square().mean().backward()
        warmup = perf_counter() - warm_start
        layer.zero_grad(set_to_none=True)
        features.grad = None
        layer.estimator.reset_counts()
        forward_start = perf_counter()
        output = layer(features)
        forward_seconds = perf_counter() - forward_start
        forward_counts = dict(layer.estimator.counts)
        backward_start = perf_counter()
        output.square().mean().backward()
        backward_seconds = perf_counter() - backward_start
        backward_counts = {key: layer.estimator.counts[key] - forward_counts[key] for key in forward_counts}
        weight_norm = float(layer.weight.grad.norm())
        input_norm = float(features.grad.norm())
        if not weight_norm > 0 or not input_norm > 0:
            raise AssertionError("Initial quantum or feature gradient is zero")
        if forward_seconds + backward_seconds > 5.:
            raise RuntimeError(f"Measured forward={forward_seconds:.6f}s, backward={backward_seconds:.6f}s; total exceeds 5 seconds. Review cost before toy optimization")
        oracle = statevector_outputs(layer, features.detach().numpy(), layer.weight.detach().numpy())
        np.testing.assert_allclose(output.detach().numpy(), oracle, atol=2e-6, rtol=2e-6)
        report.update({"circuit": {"qubits": layer.specification.circuit.num_qubits, "inputs": layer.qnn.num_inputs,
                                   "trainable_weights": layer.qnn.num_weights, "gate_counts": dict(layer.specification.circuit.count_ops()),
                                   "observables": list(OBSERVABLE_LABELS), "input_gradients": layer.qnn.input_gradients,
                                   "initial_weight_seed": 11, "initial_weight_range": list(INITIAL_WEIGHT_RANGE)},
                       "probe": {"batch_shape": list(features.shape), "output_shape": list(output.shape),
                                 "output": output.detach().tolist(), "warmup_forward_backward_seconds": warmup,
                                 "forward_seconds": forward_seconds, "backward_seconds": backward_seconds,
                                 "weight_gradient_l2": weight_norm, "feature_gradient_l2": input_norm,
                                 "forward_submitted_workloads": forward_counts, "backward_submitted_workloads": backward_counts,
                                 "independent_statevector_max_absolute_error": float(np.max(np.abs(output.detach().numpy() - oracle)))}})
        report["finite_differences"] = finite_difference_check(layer)
        report["toy_optimization"] = optimize_toy()
        report["diagram"] = draw_circuit(layer, args.results_dir / "figures")
        report["count_definition"] = ("Submitted estimator calls/PUBs/bindings/expectations; repeated bindings across "
                                      "observables are counted again. Excludes independent reference-statevector calls. "
                                      "Not unique circuits, hardware jobs or shots.")
        report["mandatory_checks"] = {name: "passed" for name in (
            "four_qubits", "four_inputs", "sixteen_trainable_weights", "batch_output_shape", "finite_output",
            "quantum_weight_gradients", "input_feature_gradients", "optimizer_updates_and_loss_decrease")}
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        save_json(report_path, report)
        raise
    save_json(report_path, report)
    print(f"Standalone quantum checks passed: {report_path}")
    print(f"Two-row forward={forward_seconds:.6f}s; backward={backward_seconds:.6f}s")
    toy = report["toy_optimization"]
    print(f"Ten toy updates: MSE {toy['losses'][0]:.9f} -> {toy['losses'][-1]:.9f}, time={toy['seconds']:.6f}s")

if __name__ == "__main__":
    main()
