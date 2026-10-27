# Phase 4 plan — standalone four-qubit quantum layer

**Paper:** Quantum Generative AI: A Survey of Hybrid Architectures and Emerging Applications

Prepared 27 October 2026. This is a plan, not evidence of quantum execution. Phase 4 implementation has not started. Its approval does not imply approval for hybrid integration or long training.

## Objective and boundaries

Implement and independently validate a differentiable PyTorch module with input/output shape `(B,4)`. Preserve the already specified circuit and surrounding-model interface. Do not change the dataset, DDPM schedule, classical architecture or experiment budgets. Do not create the hybrid denoiser or train a diffusion model in this phase.

The key acceptance condition is that a loss computed from the four measured outputs can update quantum weights and propagate gradients into preceding classical features through TorchConnector. Correct circuit construction alone is insufficient.

## 1. Confirm the pinned environment

Use the existing project-local Python 3.12 environment and installed pins: Qiskit 2.5.2, Qiskit Machine Learning 0.9.1, PyTorch 2.14.1 and NumPy 2.5.3. Recheck pip consistency and installed source/API signatures before implementing. No dependency upgrade is planned; investigate failures within the pinned environment first and record any proposed version change before making it.

Use explicit V2 StatevectorEstimator, EstimatorQNN, ParamShiftEstimatorGradient and TorchConnector. No IBM account/token, cloud call, Aer package, shot-based sampling or real hardware is needed. Import compatibility has been checked; actual QNN forward/backward compatibility remains Not evaluated.

## 2. Build the documented circuit

In `models/quantum_layer.py`, create a circuit builder returning the circuit, an explicit ordered list of four input parameters, an explicit ordered list of 16 trainable parameters and four observables. Use direct gates rather than deprecated circuit-library classes. Keep construction separate from the PyTorch wrapper for inspection/testing.

1. Initialize exactly four qubits in |0000>.
2. Convert feature h_i to angle a_i = pi*tanh(h_i) in Torch, preserving the autograd graph. Features must not be detached or manually converted to NumPy before TorchConnector.
3. Apply RY(a_i) to qubit i, i=0,1,2,3.
4. Repeat two layers: RY(theta[layer,i,0]) then RX(theta[layer,i,1]) on each qubit; then CX(0,1), CX(1,2), CX(2,3), in that order.
5. Return local Z expectations in qubit order 0,1,2,3, using Pauli labels IIIZ, IIZI, IZII, ZIII. Do not substitute one global ZZZZ expectation.

Counts: four external feature/encoding inputs; 2 layers × 4 qubits × 2 rotations = 16 trainable weights; six CX gates; four encoding RY gates and 16 trainable rotations. There are 20 distinct circuit parameters in total, of which only 16 are optimizer weights.

The angle map has derivative pi*(1-tanh(h_i)^2). Very large features can saturate Tanh; nonzero-gradient checks must use moderate features and nondegenerate nonzero circuit initialization.

## 3. Wrap the circuit for joint differentiation

Instantiate an explicit local StatevectorEstimator with default_precision=0 and a recorded seed. Pass the ordered inputs, weights and all four observables to EstimatorQNN with input_gradients=True and default_precision=0. Supply ParamShiftEstimatorGradient using the same exact estimator. Its installed constructor has no precision keyword; inspect its execution path rather than inventing an argument. Both estimator and QNN precision settings matter because a QNN precision request can override the estimator default.

TorchConnector must register the 16 weights as ordinary Torch parameters. Use a dedicated seeded initializer with nonzero angles, and record its seed/range. Keep the external interface CPU float32, compatible with the existing MLP; check connector conversions and numerical accuracy explicitly. The wrapper must accept a nonempty finite `(B,4)` tensor and return a finite `(B,4)` tensor. Numerical expectation values should remain within [-1,1] up to floating-point tolerance.

## 4. Validate structure, outputs and gradients before optimization

Create `tests/test_quantum_layer.py` with meaningful checks:

| Area | Acceptance checks |
|---|---|
| Structure | Exactly four qubits, four input parameters, 16 trainable parameters, documented gate counts/order and directed CX pattern |
| Batching | Batch sizes 1 and several examples produce `(B,4)`; batching agrees with evaluating each row independently |
| Observables | Local-Z order verified against a known basis state and independent statevector expectations |
| Numerical output | Finite outputs in expectation bounds; deterministic repeated exact-simulator calls |
| Quantum gradients | Finite, nonzero aggregate gradient on registered circuit weights for a nondegenerate test loss |
| Feature gradients | Finite, nonzero aggregate gradient on input features through the pi*Tanh map |
| Preceding layer | A small standalone Linear-to-quantum toy module receives encoder-weight gradients; this is not the hybrid DDPM |
| Gradient accuracy | Compare representative feature/weight derivatives against central finite differences, with step/tolerances appropriate to the connector's actual float precision; record maximum discrepancies |
| Invalid inputs | Reject wrong width/rank, empty batches and NaN/Inf without silently clipping or detaching values |

Do not demand that every scalar derivative be nonzero at every state; circuit symmetries, particular losses and angle saturation can legitimately produce zero entries. Conversely, a graph with all-zero/absent input or quantum-weight gradients does not pass.

## 5. Demonstrate a small optimizer update

Use a deterministic batch of two four-feature examples and reachable targets generated from the same circuit with fixed teacher weights. Run an initial ten-update Adam toy fit with an explicitly recorded small learning rate (proposed 0.01). Assert that quantum weights change, gradients are finite and final loss is below initial loss for the fixed test fixture. Save every loss and parameter-change norm. This demonstrates trainability, not generative quality or quantum advantage.

First measure one forward/backward call. If the ten-update fixture is unexpectedly expensive, stop and report the measured cost rather than silently skipping tests or launching an extensive search. Do not increase optimization steps automatically to force a passing result. Tests that fail require diagnosis and repair before progression.

## 6. Measure and report standalone cost

Measure warm-up, a batched forward call, backward call and the small optimizer test separately with CPU wall-clock timing. Record library versions, device, thread count, simulator, zero precision, batch size, parameter/observable counts and seeds. Where observable, distinguish estimator submission calls, parameter-bound circuit workloads and observable evaluations. If a count cannot be measured reliably, write Not measured; do not invent circuit counts from elapsed time or call them hardware shots.

These timings describe ideal classical statevector simulation only. They are preliminary inputs to a later hybrid-training pilot, not estimates of quantum hardware speedup or the full comparison's runtime.

## 7. Deliverables, documentation and stop condition

Only create files needed for standalone quantum validation:

- `models/quantum_layer.py`: circuit builder and Torch module.
- `tests/test_quantum_layer.py`: circuit, output, gradient and optimizer checks.
- `scripts/check_quantum_layer.py`: explicit inspection/diagnostic command; no work on import.
- `results/phase4_checks.json` and execution logs: actual measured outcomes and provenance.
- `results/figures/quantum_circuit.png` and `.pdf` where supported without unnecessary packages. Use Qiskit's text drawer as a fallback if graphical export requires an extra dependency; record that limitation rather than adding dependencies silently.
- Updates to README.md, EXPERIMENTAL_METHODOLOGY.md, IMPLEMENTATION_EXPLANATION.md and RESULTS.md: equations, API decisions, commands, measured timings, test outcomes and limitations.

Run the new standalone suite first, then all existing regression tests. Record the actual test count; do not predeclare success or experimental values. No long job is authorized by this plan.

Completion requires all eight original mandatory quantum checks (qubits, input count, weight count, output shape, finite output, quantum gradients, input gradients, optimizer updates), the independent expectation/gradient checks above, and a documented finite-cost toy optimization. Report failures honestly and keep diffusion integration blocked until they are resolved.

Suggested implementation commit: `feat: add four-qubit quantum bottleneck and standalone gradient validation`.

Stop after Phase 4 and request approval for Phase 5. Phase 5 will replace only the classical bottleneck, verify end-to-end diffusion gradients and run a measured hybrid smoke test. Full comparative experiments remain Phase 6.
