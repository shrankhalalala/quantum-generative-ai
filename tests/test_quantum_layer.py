"""Independent structure, expectation, autograd and toy-optimizer checks."""
from dataclasses import replace
import unittest
import numpy as np
import torch
from torch import nn
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from config import DEFAULT_CONFIG
from models.quantum_layer import QuantumLayer, build_quantum_circuit, OBSERVABLE_LABELS
from scripts.check_quantum_layer import FIXTURE, finite_difference_check, optimize_toy, statevector_outputs


class QuantumLayerTests(unittest.TestCase):
    def setUp(self):
        self.layer = QuantumLayer(seed=11)
        self.features = torch.tensor(FIXTURE, dtype=torch.float32)

    def test_exact_circuit_and_parameter_order(self):
        spec = self.layer.specification
        self.assertEqual(spec.circuit.num_qubits, 4)
        self.assertEqual(self.layer.qnn.num_inputs, 4)
        self.assertEqual(self.layer.qnn.num_weights, 16)
        self.assertEqual(len(spec.circuit.parameters), 20)
        self.assertTrue(set(spec.input_parameters).isdisjoint(spec.weight_parameters))
        self.assertEqual(dict(spec.circuit.count_ops()), {"ry": 12, "rx": 8, "cx": 6})
        operations = list(spec.circuit.data)
        for qubit in range(4):
            self.assertEqual(operations[qubit].operation.name, "ry")
            self.assertEqual(operations[qubit].operation.params, [spec.input_parameters[qubit]])
            self.assertEqual(spec.circuit.find_bit(operations[qubit].qubits[0]).index, qubit)
        cursor = 4
        weight_index = 0
        for _ in range(2):
            for qubit in range(4):
                for name in ("ry", "rx"):
                    instruction = operations[cursor]
                    self.assertEqual(instruction.operation.name, name)
                    self.assertEqual(instruction.operation.params, [spec.weight_parameters[weight_index]])
                    self.assertEqual(spec.circuit.find_bit(instruction.qubits[0]).index, qubit)
                    cursor += 1
                    weight_index += 1
            for edge in ((0, 1), (1, 2), (2, 3)):
                instruction = operations[cursor]
                self.assertEqual(instruction.operation.name, "cx")
                self.assertEqual(tuple(spec.circuit.find_bit(q).index for q in instruction.qubits), edge)
                cursor += 1
        self.assertEqual(cursor, len(operations))
        self.assertEqual(sum(p.numel() for p in self.layer.parameters()), 16)

    def test_precision_and_gradient_configuration(self):
        self.assertEqual(self.layer.estimator.default_precision, 0.)
        self.assertEqual(self.layer.qnn.default_precision, 0.)
        self.assertTrue(self.layer.qnn.input_gradients)
        self.assertIs(self.layer.qnn.estimator, self.layer.estimator)
        self.assertIs(self.layer.qnn.gradient, self.layer.gradient)

    def test_shapes_finite_bounds_and_batched_individual_agreement(self):
        with torch.no_grad():
            output = self.layer(self.features)
            individual = torch.cat([self.layer(row[None, :]) for row in self.features])
        self.assertEqual(output.shape, (2, 4))
        self.assertEqual(output.dtype, torch.float32)
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.all(output.abs() <= 1 + 1e-6))
        torch.testing.assert_close(output, individual, rtol=0, atol=0)
        with torch.no_grad():
            more = self.layer(torch.cat([self.features, self.features[:1]]))
        self.assertEqual(more.shape, (3, 4))

    def test_local_z_observable_order_with_known_basis_states(self):
        self.assertEqual(tuple(op.to_list()[0][0] for op in self.layer.specification.observables), OBSERVABLE_LABELS)
        for qubit in range(4):
            circuit = QuantumCircuit(4)
            circuit.x(qubit)
            state = Statevector.from_instruction(circuit)
            measured = [state.expectation_value(op).real for op in self.layer.specification.observables]
            expected = np.ones(4)
            expected[qubit] = -1
            np.testing.assert_array_equal(measured, expected)

    def test_qnn_matches_independent_statevector(self):
        expected = statevector_outputs(self.layer, self.features.numpy(), self.layer.weight.detach().numpy())
        with torch.no_grad():
            actual = self.layer(self.features).numpy()
        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)

    def test_same_seed_repeatability_and_global_rng_independence(self):
        np.random.seed(21)
        expected_numpy = np.random.rand(3)
        np.random.seed(21)
        torch.manual_seed(21)
        before = torch.get_rng_state().clone()
        same = QuantumLayer(11)
        np.testing.assert_array_equal(np.random.rand(3), expected_numpy)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertTrue(torch.equal(same.weight, self.layer.weight))
        self.assertFalse(torch.equal(QuantumLayer(22).weight, self.layer.weight))
        with torch.no_grad():
            first = self.layer(self.features)
            second = self.layer(self.features)
            third = same(self.features)
        torch.testing.assert_close(first, second, atol=0, rtol=0)
        torch.testing.assert_close(first, third, atol=0, rtol=0)

    def test_input_and_weight_gradients(self):
        features = self.features.clone().requires_grad_()
        self.layer(features).square().mean().backward()
        for gradient in (features.grad, self.layer.weight.grad):
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.norm()), 1e-6)

    def test_gradients_reach_preceding_classical_layer(self):
        # A standalone toy encoder, not a diffusion denoiser.
        encoder = nn.Linear(3, 4)
        with torch.no_grad():
            encoder.weight.copy_(torch.arange(12).reshape(4, 3) / 30 - .2)
            encoder.bias.copy_(torch.tensor([.1, -.2, .3, -.1]))
        encoded = encoder(torch.tensor([[.2, -.4, .3], [-.1, .3, .2]]))
        self.layer(encoded).square().mean().backward()
        for gradient in (encoder.weight.grad, encoder.bias.grad, self.layer.weight.grad):
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.norm()), 1e-6)

    def test_all_feature_and_weight_derivatives_against_finite_differences(self):
        original = self.layer.weight.detach().clone()
        check = finite_difference_check(self.layer)
        self.assertEqual(check["feature_derivatives_checked"], 8)
        self.assertEqual(check["weight_derivatives_checked"], 16)
        self.assertLess(check["max_feature_absolute_error"], 3e-4)
        self.assertLess(check["max_weight_absolute_error"], 3e-4)
        self.assertTrue(torch.equal(original, self.layer.weight))

    def test_ten_optimizer_updates_reduce_reachable_target_loss(self):
        result = optimize_toy()
        self.assertEqual(result["steps"], 10)
        self.assertEqual(len(result["losses"]), 11)
        self.assertLess(result["losses"][-1], result["losses"][0])
        self.assertGreater(result["weight_change_l2_per_update"][-1], 0.)
        self.assertTrue(np.isfinite(result["final_weights"]).all())

    def test_submitted_workload_counter_tracks_forward_backward(self):
        self.layer.estimator.reset_counts()
        features = self.features.clone().requires_grad_()
        output = self.layer(features)
        self.assertEqual(self.layer.estimator.counts, {
            "run_calls": 1, "pubs": 4, "submitted_parameter_bindings": 8,
            "observable_expectation_evaluations": 8})
        output.square().mean().backward()
        # Installed parameter shift: 2 shifts × 20 parameters × 2 rows × 4 outputs.
        self.assertEqual(self.layer.estimator.counts, {
            "run_calls": 2, "pubs": 324, "submitted_parameter_bindings": 328,
            "observable_expectation_evaluations": 328})

    def test_invalid_inputs_weights_and_design_rejected(self):
        for values in (torch.zeros(4), torch.zeros(2, 3), torch.zeros(0, 4), torch.zeros(1, 2, 4),
                       torch.full((2, 4), float("nan")), torch.full((2, 4), float("inf")),
                       torch.ones(2, 4, dtype=torch.float64), torch.ones(2, 4, dtype=torch.int64)):
            with self.subTest(shape=values.shape), self.assertRaises(ValueError):
                self.layer(values)
        with torch.no_grad():
            self.layer.weight[0] = float("nan")
        with self.assertRaises(ValueError):
            self.layer(self.features)
        with self.assertRaises(ValueError):
            build_quantum_circuit(replace(DEFAULT_CONFIG.model, n_qubits=5))
        for seed in (-1, 2**32, 1.5):
            with self.assertRaises(ValueError):
                QuantumLayer(seed)

    def test_state_dict_reload_restores_expectations(self):
        restored = QuantumLayer(22)
        restored.load_state_dict(self.layer.state_dict())
        with torch.no_grad():
            torch.testing.assert_close(self.layer(self.features), restored(self.features), atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
