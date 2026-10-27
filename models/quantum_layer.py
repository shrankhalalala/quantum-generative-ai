"""Four-qubit, four-output trainable bottleneck; no execution on import."""
from dataclasses import dataclass
from typing import Iterable
import numpy as np
import torch
from torch import nn
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.primitives import StatevectorEstimator
from qiskit.primitives.containers import EstimatorPub
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.gradients import ParamShiftEstimatorGradient
from qiskit_machine_learning.neural_networks import EstimatorQNN
from config import DEFAULT_CONFIG, ModelConfig

OBSERVABLE_LABELS = ("IIIZ", "IIZI", "IZII", "ZIII")
INITIAL_WEIGHT_RANGE = (-np.pi / 4, np.pi / 4)


@dataclass(frozen=True)
class CircuitSpecification:
    circuit: QuantumCircuit
    input_parameters: tuple[Parameter, ...]
    weight_parameters: tuple[Parameter, ...]
    observables: tuple[SparsePauliOp, ...]


def build_quantum_circuit(config: ModelConfig = DEFAULT_CONFIG.model) -> CircuitSpecification:
    """Explicit parameter order: layer, qubit, RY then RX; four local Zs."""
    if (config.n_qubits != 4 or config.bottleneck_dim != 4 or config.quantum_layers != 2
            or config.rotations != ("RY", "RX")
            or config.entanglement_edges != ((0, 1), (1, 2), (2, 3))):
        raise ValueError("Circuit must match the approved four-qubit two-layer design")
    inputs = ParameterVector("x", 4)
    weights = ParameterVector("theta", 16)
    circuit = QuantumCircuit(4, name="four_qubit_bottleneck")
    for qubit in range(4):
        circuit.ry(inputs[qubit], qubit)
    cursor = 0
    for _ in range(2):
        for qubit in range(4):
            circuit.ry(weights[cursor], qubit)
            circuit.rx(weights[cursor + 1], qubit)
            cursor += 2
        for control, target in config.entanglement_edges:
            circuit.cx(control, target)
    return CircuitSpecification(circuit, tuple(inputs), tuple(weights),
                                tuple(SparsePauliOp(label) for label in OBSERVABLE_LABELS))


class CountingStatevectorEstimator(StatevectorEstimator):
    """Count submitted V2 workloads, including observable-repeated bindings.

    These are not distinct/unique circuits, hardware jobs, shots or a counter
    inside the simulator. A forward PUB can contain B parameter assignments.
    """
    def __init__(self, seed: int):
        super().__init__(default_precision=0., seed=seed)
        self.reset_counts()

    def reset_counts(self) -> None:
        self.counts = {"run_calls": 0, "pubs": 0, "submitted_parameter_bindings": 0,
                       "observable_expectation_evaluations": 0}

    def run(self, pubs: Iterable, *, precision: float | None = None):
        effective_precision = self.default_precision if precision is None else precision
        coerced = [EstimatorPub.coerce(pub, effective_precision) for pub in pubs]
        self.counts["run_calls"] += 1
        self.counts["pubs"] += len(coerced)
        for pub in coerced:
            self.counts["submitted_parameter_bindings"] += int(np.prod(pub.parameter_values.shape))
            self.counts["observable_expectation_evaluations"] += int(np.prod(pub.shape))
        return super().run(coerced, precision=precision)


class QuantumLayer(nn.Module):
    """CPU float32 features (B,4) → four local-Z expectation values (B,4)."""
    def __init__(self, seed: int = 11, config: ModelConfig = DEFAULT_CONFIG.model):
        super().__init__()
        if not isinstance(seed, int) or not 0 <= seed < 2**32:
            raise ValueError("Seed must be an integer in [0,2**32)")
        self.seed = seed
        self.specification = build_quantum_circuit(config)
        self.estimator = CountingStatevectorEstimator(seed)
        self.gradient = ParamShiftEstimatorGradient(self.estimator)
        self.qnn = EstimatorQNN(circuit=self.specification.circuit,
                                estimator=self.estimator,
                                observables=self.specification.observables,
                                input_params=self.specification.input_parameters,
                                weight_params=self.specification.weight_parameters,
                                input_gradients=True, default_precision=0., gradient=self.gradient)
        initial = np.random.default_rng(seed).uniform(*INITIAL_WEIGHT_RANGE, size=16).astype(np.float32)
        self.connector = TorchConnector(self.qnn, initial_weights=initial)

    @property
    def weight(self) -> nn.Parameter:
        return self.connector.weight

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != 4 or len(features) == 0:
            raise ValueError("Quantum input must have nonempty (B,4) shape")
        if features.device.type != "cpu" or features.dtype != torch.float32:
            raise ValueError("Quantum input must be CPU float32")
        if self.weight.device.type != "cpu" or self.weight.dtype != torch.float32:
            raise ValueError("Quantum weights must remain CPU float32")
        if not torch.isfinite(features).all() or not torch.isfinite(self.weight).all():
            raise ValueError("Quantum input/weights must be finite")
        angles = torch.pi * torch.tanh(features)
        # TorchConnector supplies its own autograd bridge to the QNN Jacobians.
        output = self.connector(angles)
        if output.shape != features.shape or not torch.isfinite(output).all():
            raise FloatingPointError("Quantum output must be finite with shape (B,4)")
        if torch.any(output.abs() > 1. + 1e-6):
            raise FloatingPointError("Local Z expectation outside numerical [-1,1] bounds")
        return output
