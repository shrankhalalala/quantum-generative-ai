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


class BatchedStatevectorEstimator(CountingStatevectorEstimator):
    """Same ideal states and parameter-shift semantics, with batched execution."""
    def _run(self, pubs):
        from qiskit.primitives.containers import DataBin, PrimitiveResult, PubResult
        from models.batched_statevector import evaluate_pub
        groups = {}
        for index, pub in enumerate(pubs):
            key = (id(pub.circuit), pub.shape, pub.precision)
            groups.setdefault(key, []).append((index, pub))
        results = [None] * len(pubs)
        for group in groups.values():
            circuit = group[0][1].circuit
            values = np.stack([np.broadcast_to(pub.parameter_values.as_array(circuit.parameters),
                                               pub.shape + (circuit.num_parameters,)) for _, pub in group])
            observables = np.stack([np.broadcast_to(np.asarray(pub.observables), pub.shape) for _, pub in group])
            merged = EstimatorPub.coerce((circuit, observables, values, group[0][1].precision))
            result = evaluate_pub(merged)
            for position, (index, pub) in enumerate(group):
                if result is None:
                    results[index] = super()._run_pub(pub)
                else:
                    evs = result.data.evs[position]
                    results[index] = PubResult(DataBin(evs=evs, stds=np.zeros_like(evs), shape=pub.shape),
                                               metadata=result.metadata)
        return PrimitiveResult(results, metadata={'version': 2})

    def _run_pub(self, pub):
        from models.batched_statevector import evaluate_pub
        result = evaluate_pub(pub)
        return super()._run_pub(pub) if result is None else result


class BatchedParamShiftEstimatorGradient(ParamShiftEstimatorGradient):
    """Same +/- pi/2 rule and Qiskit chain rule; batch shifts in each PUB."""
    def _run_unique(self, circuits, observables, parameter_values, parameters, **options):
        from qiskit_machine_learning.gradients import EstimatorGradientResult
        from qiskit.providers import Options
        if self._pass_manager is not None:
            return super()._run_unique(circuits, observables, parameter_values, parameters, **options)
        pubs, metadata, sizes = [], [], []
        for circuit, observable, values, requested in zip(circuits, observables, parameter_values, parameters):
            order = {parameter: index for index, parameter in enumerate(circuit.parameters)}
            n = len(requested)
            shifts = np.zeros((n, len(order)))
            for row, parameter in enumerate(requested):
                shifts[row, order[parameter]] = np.pi / 2
            shifted = np.concatenate((np.asarray(values) + shifts, np.asarray(values) - shifts))
            pubs.append((circuit, observable, shifted))
            metadata.append({'parameters': requested})
            sizes.append(n)
        results = self._estimator.run(pubs).result()
        gradients = [(result.data.evs[:n] - result.data.evs[n:]) / 2
                     for result, n in zip(results, sizes)]
        return EstimatorGradientResult(gradients=gradients, metadata=metadata, options=Options(**options))


class QuantumLayer(nn.Module):
    """CPU float32 features (B,4) → four local-Z expectation values (B,4)."""
    def __init__(self, seed: int = 11, config: ModelConfig = DEFAULT_CONFIG.model):
        super().__init__()
        if not isinstance(seed, int) or not 0 <= seed < 2**32:
            raise ValueError("Seed must be an integer in [0,2**32)")
        self.seed = seed
        self.specification = build_quantum_circuit(config)
        self.backend = config.quantum_backend
        estimator_type = BatchedStatevectorEstimator if self.backend == 'batched_statevector' else CountingStatevectorEstimator
        self.estimator = estimator_type(seed)
        gradient_type = BatchedParamShiftEstimatorGradient if self.backend == 'batched_statevector' else ParamShiftEstimatorGradient
        self.gradient = gradient_type(self.estimator)
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
