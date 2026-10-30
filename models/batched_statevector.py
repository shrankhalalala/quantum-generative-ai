"""Exact NumPy batching of supported Qiskit statevector PUBs, not a surrogate.

The QNN, TorchConnector and parameter-shift gradient remain Qiskit components.
Only execution of independent parameter bindings is vectorized. Unsupported
circuits/observables fall back to Qiskit's reference implementation.
"""
import numpy as np
from qiskit.circuit import Parameter
from qiskit.primitives.containers import DataBin, PubResult


def evaluate_pub(pub):
    circuit = pub.circuit
    if circuit.num_qubits != 4 or pub.precision != 0:
        return None
    observables = np.asarray(pub.observables)
    if any(set(label) - {'I', 'Z'} for observable in observables.flat for label in observable):
        return None
    parameters = tuple(circuit.parameters)
    positions = {parameter: index for index, parameter in enumerate(parameters)}
    bindings = pub.parameter_values.as_array(parameters).reshape(-1, len(parameters)) if parameters else None
    binding_shape = pub.parameter_values.shape
    size = int(np.prod(binding_shape)) if binding_shape else 1
    state = np.zeros((size, 16), dtype=np.complex128)
    state[:, 0] = 1.
    basis = np.arange(16)
    for instruction in circuit.data:
        operation = instruction.operation
        qubits = [circuit.find_bit(qubit).index for qubit in instruction.qubits]
        if operation.name == 'barrier':
            continue
        if operation.name == 'cx':
            control, target = qubits
            permutation = basis ^ (((basis >> control) & 1) << target)
            state = state[:, permutation]
            continue
        if operation.name not in ('rx', 'ry', 'rz'):
            return None
        parameter = operation.params[0]
        if isinstance(parameter, Parameter):
            angle = bindings[:, positions[parameter]]
        elif getattr(parameter, 'parameters', ()):
            return None
        else:
            angle = np.full(size, float(parameter))
        zero = basis[(basis & (1 << qubits[0])) == 0]
        one = zero | (1 << qubits[0])
        a, b = state[:, zero].copy(), state[:, one].copy()
        cosine, sine = np.cos(angle / 2)[:, None], np.sin(angle / 2)[:, None]
        if operation.name == 'ry':
            state[:, zero], state[:, one] = cosine*a - sine*b, sine*a + cosine*b
        elif operation.name == 'rx':
            state[:, zero], state[:, one] = cosine*a - 1j*sine*b, -1j*sine*a + cosine*b
        else:
            state[:, zero] = np.exp(-.5j*angle)[:, None]*a
            state[:, one] = np.exp(.5j*angle)[:, None]*b
    probabilities = state.real**2 + state.imag**2
    evs = np.zeros(pub.shape, dtype=np.float64)
    # Broadcast parameter assignments against observable axes exactly as a PUB.
    broadcast_observables = np.broadcast_to(observables, pub.shape)
    unique = {tuple(sorted(observable.items())) for observable in observables.flat}
    for observable in unique:
        expectation = np.zeros(size)
        for label, coefficient in observable:
            signs = np.ones(16)
            for qubit, pauli in enumerate(reversed(label)):
                if pauli == 'Z':
                    signs *= 1 - 2*((basis >> qubit) & 1)
            expectation += coefficient*(probabilities*signs).sum(axis=1)
        broadcast_values = np.broadcast_to(expectation.reshape(binding_shape), pub.shape)
        mask = np.fromiter((tuple(sorted(obs.items())) == observable for obs in broadcast_observables.flat),
                           dtype=bool).reshape(pub.shape)
        evs[mask] = broadcast_values[mask]
    return PubResult(DataBin(evs=evs, stds=np.zeros_like(evs), shape=evs.shape),
                     metadata={'target_precision': 0., 'circuit_metadata': circuit.metadata})
