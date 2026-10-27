# Experimental methodology

**Approved survey title: Quantum Generative AI: A Survey of Hybrid Architectures and Emerging Applications**

This experiment is a focused extension of the existing survey of 25 previously selected papers (15 reviewed at mid-evaluation). It does not replace the literature review. Phase 1 specified the experiment; Phase 2 implements and validates the dataset. Phase 3 implements the classical DDPM and validates it with a five-update smoke test. Phase 4 implements and validates the standalone quantum module. Phase 5 integrates and smoke-tests the hybrid denoiser with explicit paired initialization. Full comparative training and formal quality evaluation remain future work.

## Objective and hypotheses

Compare a small classical DDPM with a hybrid DDPM that replaces only a four-dimensional bottleneck with a trainable four-qubit circuit. Assess marginal generation quality, temporal correlation, simulation cost, and limited-data behavior. Hypotheses: (H1) the hybrid can learn a useful denoiser; (H2) its generation/ACF errors may differ from the baseline; (H3) the difference may depend on training-set size. These are empirical questions, not assumed improvements. Three repetitions permit descriptive variability estimates, not strong significance claims.

## Dataset and persistence (Phase 2 implemented)

For each of exactly 1,400 sequences, independently draw

\[
x(t)=A\sin(2\pi ft+\phi)+B\sin(4\pi ft+\psi)+\epsilon(t).
\]

| Quantity | Distribution |
|---|---|
| A | Uniform[0.6,1.2] |
| f | Uniform[1.0,3.0] cycles/window |
| phi, psi | independent Uniform[0,2pi] |
| B | Uniform[0.0,0.3] |
| epsilon(t) | independent Normal(0,0.05²) |
| t | arange(32)/32, over [0,1), endpoint=False |

Shape is (1400,32), each sequence (32,). Use NumPy Generator seed 2026 for synthesis and a separate seed 2027 for a permutation split: first 1,000 train, next 200 validation, last 200 held-out reference. Save raw sequences, time grid, per-sequence A/f/phi/B/psi, split indices, equation, distribution parameters, seeds, package versions, dtype, file hashes and summary statistics. Saving raw sequences preserves observation-noise realizations; exact regeneration also requires the recorded software environment.

For repetition seed 11, 22 or 33, permute only the 1,000 training indices once; take prefixes of length 100, 250, 500 and 1,000. This ensures nested subsets within each repetition, identical subsets between models, and fixed validation/test sets across all runs. Never reshuffle validation/test membership. Use independent RNG streams for subset selection, model initialization, training noise/timesteps, validation noise/timesteps and reverse sampling.

Fit a single scalar mean and population standard deviation (ddof=0) across all entries of the selected training subset. Apply (x-mu)/sigma to its training data, fixed validation and fixed test sequences. Save mu/sigma per repetition/fraction and reuse across models; reject non-finite or near-zero sigma. Thus the 100-sample condition uses no information from the other 900 training sequences. Inverse-transform generated sequences before computing physical-scale metrics. The raw reference remains fixed; transformed validation/test values can differ across fractions because the training-only transform differs.

Phase 2 plots the first six entries of a fixed held-out-index order (no visual cherry-picking), a pooled-value histogram and their individual ACFs; report mean, population std, min, max and shape. Report training-only normalization statistics separately from descriptive full-dataset summaries.

## Shared architecture specification (Phases 3–5)

Tensor notation B is batch size. Timestep encoding is a learned Embedding(20,16); code index 0 represents mathematical diffusion time 1, and index 19 represents time 20.

1. Concatenate noisy sequence (B,32) and timestep embedding (B,16), yielding (B,48).
2. Encoder: Linear(48,16), SiLU, Linear(16,4), yielding features h of shape (B,4).
3. Baseline bottleneck: Linear(4,4), Tanh, yielding (B,4).
4. Hybrid bottleneck: the four-qubit layer below, yielding (B,4).
5. Shared decoder: Linear(4,16), SiLU, Linear(16,32), yielding predicted Gaussian noise (B,32). No output bounding, as target noise is unbounded.

The classical affine bottleneck is a normal trainable nonlinear layer. The common four-dimensional bottleneck is restrictive for both models and is a threat to fidelity. Pair identical encoder/decoder/embedding initial state dicts between models, rather than relying on the same global seed with different constructor RNG consumption.

Analytical parameter counts (to verify against instantiated models later): embedding 320, encoder 784+68, decoder 80+544, shared total 1,796. Baseline bottleneck has 20 weights/biases, total 1,816. Quantum bottleneck has 16 trainable weights, total 1,812. Similar surrounding architectures are not evidence of matched capacity. An optional third control, implemented in Phase 5, is a bias-free Linear(4,4)+Tanh bottleneck (16 weights, total 1,812); parameter-count matching does not imply equal expressiveness.

## Quantum circuit specification (Phase 4)

Use exactly four qubits initialized to |0000>. Bound each input differentiably as angle_i = pi*tanh(h_i), then apply RY(angle_i) to qubit i. There are four distinct input parameters.

In each of two layers apply RY(theta[layer,i,0]) followed by RX(theta[layer,i,1]) on every qubit, then CX(0,1), CX(1,2), CX(2,3) in that order. Total trainable parameters: 2*4*2 = 16. RX is chosen for the second rotation so the final rotation is not an RZ directly before commuting Z measurements. This design does not guarantee every parameter has a nonzero gradient at every state.

Measure four separate local Z expectation values in qubit order 0,1,2,3, not one global ZZZZ observable. Qiskit Pauli labels are little-endian: observables in this order are IIIZ, IIZI, IZII, ZIII. Expectation outputs lie in [-1,1]. Build gates directly, avoiding deprecated circuit-library convenience classes.

Implemented APIs: explicitly supplied V2 StatevectorEstimator(default_precision=0, seed=run_seed); EstimatorQNN(default_precision=0, input_gradients=True); explicit ParamShiftEstimatorGradient using the same exact estimator; TorchConnector registers quantum weights as PyTorch parameters. Use ideal local CPU statevector simulation, no token/account and no shot noise. Both zero-precision settings must be supplied because a QNN's nonzero default precision could override the estimator's exact default. Installed gradient source inspection confirms its V2 path calls estimator.run without a precision override, so it inherits the estimator's zero default; its constructor has no precision keyword. Actual forward/backward compatibility passed standalone Phase 4 tests; hybrid integration remains Phase 5.

Mandatory tests: four qubits; four inputs; 16 weights; batched output (B,4); finite outputs; finite, nonzero aggregate quantum gradients and input gradients for a nondegenerate initialization; optimizer updates quantum weights; a small target-fitting loss decreases. Test observable order and deterministic exact expectations too. Gradients chain from loss through decoder, measurements, rotations, pi*tanh, encoder and timestep encoding. Do not advance if tests fail.

## DDPM equations and indexing (Phase 3)

Use mathematical time k in {1,...,20}, code index k-1. Linear beta schedule beta_1=0.0001 to beta_20=0.02; alpha_k=1-beta_k; alpha_bar_k=product_{j=1}^k alpha_j, alpha_bar_0=1.

Forward sample:
\[
x_k=\sqrt{\bar\alpha_k}x_0+\sqrt{1-\bar\alpha_k}\epsilon,\quad\epsilon\sim N(0,I).
\]

Draw timesteps uniformly and independently for each training sample. Minimize mean over batch and sequence dimensions of (epsilon - epsilon_theta(x_k,k))². There is no reconstruction-loss substitution.

Start reverse sampling x_20 ~ N(0,I), then for k=20,...,1 use
\[
\mu_\theta(x_k,k)=\alpha_k^{-1/2}\left(x_k-\frac{\beta_k}{\sqrt{1-\bar\alpha_k}}\epsilon_\theta(x_k,k)\right),
\quad \tilde\beta_k=\beta_k\frac{1-\bar\alpha_{k-1}}{1-\bar\alpha_k}.
\]

For k>1, x_{k-1}=mu_theta+sqrt(beta_tilde_k)*z, z~N(0,I); at k=1, return mu_theta with no noise. Use eval mode and no_grad; return (B,32); check finiteness and shapes at every step and record excessive-magnitude diagnostics without silently clipping outputs or discarding bad runs.

**Required schedule limitation:** with this short/small-beta schedule, alpha_bar_20 is approximately 0.817. The terminal forward distribution retains substantial signal (signal coefficient about 0.904, noise coefficient about 0.428), so it is not close to the pure Gaussian reverse initialization. This prior mismatch can limit both models' generation quality regardless of learning. Retain the supervisor's specified schedule; do not change steps or beta values without approval. Small validation MSE alone cannot establish good generation.

## Training protocol and budget controls

Shared Adam with initial LR 0.001, batch size 16, CPU float32 neural networks, shuffle per epoch, no dropped last batch, identical loss and schedule. Best checkpoint selected solely by validation noise-prediction MSE, including weights, optimizer, step/epoch, seeds, normalization and configuration. Report last and best losses; evaluate the best checkpoint. Fix validation noise/timestep draws across checkpoints and paired model comparisons, independent of training RNG. Test/reference data must not tune checkpoints, hyperparameters or budgets.

Full epochs/max_steps are unset in config until approved after measured pilot. Phase 3 smoke training and Phase 5 hybrid smoke training precede larger jobs. A proposed five-step quantum pilot includes warm-up separately, forward/backward/optimizer timing, validation timing and a small sampling-cost probe. These are future actions, not Phase 1 executions. Estimate cost using measured seconds per update times ceil(N/16)*epochs, add validation and 20-step reverse sampling for 200 sequences, and show uncertainty. Measure all fractions where batching/overheads matter. Request confirmation before a long run.

Primary comparison uses equal approved optimizer-update budgets per model within each fraction, preserving the same number of training-example exposures for paired models. To interpret scarcity across fractions, prefer one common max_steps across fractions; explicitly report effective epochs/exposures and the ceiling batch effects. If an epoch budget is selected instead, document that fewer data imply fewer updates and confound scarcity with optimization. Any alternate budget policy requires discussion. Train all three seeds if feasible; otherwise report measured cost and stop for a decision, without substituting a smaller unreported protocol.

## Evaluation protocol (Phase 6)

Initially generate exactly 200 sequences per model/seed/fraction, equal to held-out reference size. Use the same initial Gaussian arrays and per-step noise arrays in paired runs, with a separate sampling seed derived from the run seed. Increase to 1,000 only if measured sampling cost supports an approved identical setting. Inverse-normalize all outputs first. Record all raw metrics and generated arrays, including failed runs/status and explanations. Do not replace non-finite generated samples with hand-picked successful samples.

- Marginal Wasserstein-1 distance: scipy.stats.wasserstein_distance on flattened real and generated values. It measures pooled amplitude marginals, not full joint or temporal fidelity.
- Mean/std/variance differences: signed generated minus reference values, plus their absolute errors; pooled entries, ddof=0. Units are amplitude, amplitude and amplitude² respectively.
- ACF: for each length-32 sequence subtract its own mean; r_i(lag)=sum_{t=0}^{31-lag} y_i(t)y_i(t+lag)/sum_{t=0}^{31} y_i(t)², for lags 0..10 (biased finite-window normalization). Average r_i across sequences equally. Do not concatenate sequences or introduce correlations across sequence boundaries. Near-zero-denominator sequences are undefined: report their count and fail/flag temporal evaluation instead of silently dropping them. Define ACF error as mean_{lag=1..10}|mean_r_generated(lag)-mean_r_reference(lag)|. Report individual ACFs as well as aggregate curves to expose cancellation. ACF is dimensionless.
- Timing: perf_counter wall time, separate training (including validation/checkpoint overhead) and optimizer-loop time, seconds/epoch and seconds/update, and reverse-sampling time. Device synchronization if GPU timing is introduced later. Record simulator, precision/shots, CPU model/platform/thread count and device. All quantum costs are classical simulation costs, not hardware execution times.
- Counts: count trainable parameters from model.parameters; instrument submitted parameter-bound circuit/observable workloads where possible, distinguishing batched primitive calls, circuit bindings and observable evaluations. Do not call an unmeasured count a hardware shot count. If unavailable record Not measured.

Save one raw result record per model/fraction/seed and descriptive arithmetic means and sample standard deviations (ddof=1 across three repetitions). A single seed has undefined cross-seed std: record null/Not evaluated rather than zero. Plot every data fraction with the same settings and fixed sample-index choices, never select favorable runs. Pilot timings are not trained-model quality results.

## Figures and research limitations

Later phases will produce real/classical/hybrid examples, overlaid ACFs and amplitude histograms, training/validation loss, quality and ACF error versus fraction, training time, architecture and circuit diagrams. Use Matplotlib readable labels/units, fixed indices and PNG (300 dpi) plus PDF. Each figure needs provenance (run identifiers, seed, fraction, checkpoint and metric definition). No figure contains invented results.

Limitations: short terminal schedule/prior mismatch; 32-point synthetic-only sequences; constrained four-dimensional bottleneck; simulator overhead and parameter-shift costs; unequal parameter counts and expressiveness; seed/sample uncertainty; noisy denoising validation estimates; pooling/ACF omit higher-order structure; per-subset normalization changes coordinates across fractions; long training may be infeasible. The experiment cannot establish quantum hardware speedup or general quantum advantage. A null/negative result remains valid survey evidence.

## API references consulted in Phase 1

- [EstimatorQNN API, Qiskit ML 0.9.1](https://qiskit-community.github.io/qiskit-machine-learning/stubs/qiskit_machine_learning.neural_networks.EstimatorQNN.html): explicit input gradients and estimator/precision arguments.
- [TorchConnector tutorial](https://qiskit-community.github.io/qiskit-machine-learning/tutorials/05_torch_connector.html): joint Torch optimization.
- [StatevectorEstimator](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.primitives.StatevectorEstimator): V2 local statevector primitive.

Consulted 9 October 2026. Installed versions and actual environment check results are authoritative in results/environment.json; import checks alone do not establish full circuit-gradient compatibility.

## Phase 2 implementation evidence

Generation uses independent `np.random.default_rng` instances for synthesis (2026), split selection (2027) and each repetition's training permutation (11/22/33), without consuming global NumPy RNG state. Draw order is A, f, phi, B, psi, then the full (1400,32) Gaussian-noise array. Raw values and parameter arrays are float64. Save the noise explicitly to allow exact equation reconstruction; normalization is a scalar global z-score per selected subset, not per sequence or time position.

Exactly 1,400 sequences were generated and saved; the raw reference and validation splits each contain 200 sequences. The six plotted original-row indices are 825, 1024, 641, 499, 1171 and 16 (first six fixed test indices). The distribution histogram is descriptive pooled data plus the held-out reference; full-data histogram/statistics do not enter normalization. Individual ACFs and the equal-weight mean of all 200 held-out ACFs use the shared implementation in `utils/time_series.py`.

Dataset validation and all 19 Phase 1/2 tests passed. Saved artifact loading verifies SHA-256, full equation including stored noise, exact split and nested-subset seed selection, and recomputed training-only statistics. A leakage test sets all excluded rows (validation, test and unselected training rows) to 1,000,000 and verifies that fitting the selected training subset is unchanged. Inverse transformation restores original raw values to floating-point tolerance. Near-constant ACF rows raise a descriptive error, rather than silently disappearing from an aggregate.

Each saved condition shares its exact indices and scalar statistics across future model comparisons; normalized arrays are computed on demand and never refitted to validation/test. Full-data descriptive statistics and the measured observation-noise moments are reported in RESULTS.md; no trained-model outcome has been evaluated.

## Phase 3 implementation and execution evidence

The baseline has the specified shared timestep-conditioned architecture and an instantiated trainable count of 1,816. `NoiseSchedule` computes beta/alpha/cumulative-alpha/posterior-variance buffers in float64 before casting to float32 for the network. Tensor code time 0 means mathematical time 1. Forward/reverse equations and zero final-step noise are tested against known calculations. Noise prediction shape is explicitly checked to prevent accidental broadcasting into MSE.

`train_model` is model-agnostic for compatible (B,32) denoisers and requires exactly one positive epoch/update budget. Shuffling, per-example uniform timestep sampling and Gaussian training noise use distinct local Torch Generators derived by NumPy SeedSequence from the run seed and stream ID. Fixed validation noise/timesteps use a separate seed 2028 and are element-weighted over batches, including a final short batch. Training loss records interval/epoch means weighted by number of elements. Validation frequency is recorded and does not change training randomness for these deterministic denoisers. Initial model validation is included as checkpoint candidate step 0. Ties retain the earlier checkpoint.

Checkpoints contain model/optimizer/schedule state, config, seed streams, normalization, raw-data hashes, subset/validation indices, step/epoch/cursor and RNG states. They are written through a temporary file and loaded with Torch weights_only=True for inference. Exact interrupted-training resume is not claimed (current epoch order/cursor replay is not implemented). Sampling reloads the best checkpoint; test/reference samples never select it.

A single CPU smoke run (seed 11, 1,000 training rows, five optimizer updates, 80 training-example exposures, 0.08 effective epochs) completed. All 37 tests passed, including dataset regressions, component/input gradients, timestep dependence, DDPM formulas, sampling reproducibility, fixed validation, short last batches, repeatable training, best-checkpoint reload, invalid input and shape/corruption rejection. The saved checkpoint regenerates the exact saved 200 physical-scale samples; none exactly equals a raw dataset row. This is not a nearest-neighbor memorization analysis, which would require a separate metric.

Reverse sampling draws an (N,32) Gaussian tensor, then full-row noise at each non-final step before chunked denoiser evaluation. Thus random draws are independent of inference chunk size and model construction. Optional explicit tensors support later common-random-number pairing. Every step checks finite values/shapes and normalized max absolute amplitude against a diagnostic limit 100, raising rather than clipping if unstable. It runs no_grad/eval and restores the caller's mode. Timing measures CPU wall-clock training and sampling separately; warm-up is recorded outside training, which includes validation/checkpoint overhead. Actual figures and timings are in RESULTS.md and the per-run JSON.

No full training has run. Five-update timing extrapolation is limited: first-update Adam overhead is visible, small CPU kernels and validation/checkpoint cost affect estimates, and fractions with short final batches may differ from a full-batch estimate. Classical update cost cannot predict parameter-shift quantum cost. The visibly noisy smoke samples and modest validation-MSE change do not establish convergence or useful generation. Required schedule/prior mismatch remains unchanged. Before any long run, discuss budget/convergence and later quantum pilot cost; keep paired-model budgets identical.


## Phase 4 implementation and scientific interpretation

The approved circuit is implemented directly with four encoding RY gates, 12 total RY gates, eight RX gates and six directed CX gates. Four input parameters and 16 trainable weights are explicitly ordered, without relying on lexical sorting. The layer computes

\[
a_i=\pi\tanh(h_i),\qquad
z_j=\langle\psi(a,\theta)|Z_j|\psi(a,\theta)\rangle,
\quad j=0,1,2,3.
\]

The 16 weights use an independent NumPy default_rng with seed 11 and Uniform[-pi/4,pi/4), converted to float32 for TorchConnector. The inputs remain CPU float32 tensors with their autograd history. Qiskit's statevector arithmetic is higher precision internally; the installed connector returns float32 outputs/Jacobians. We do not claim float64 Torch gradcheck compatibility.

For a single-use RY/RX parameter the parameter-shift derivative is half the difference in expectations at angle +/-pi/2. The input encoding further requires the chain-rule factor pi*(1-tanh(h_i)^2). TorchConnector contracts the QNN input/weight Jacobians with the downstream loss gradient. Its internal detached NumPy interface is part of a custom Torch autograd function; the wrapper must not manually detach its features before invoking it. Aggregate nonzero input and weight gradients and a preceding Linear module test verify this chain empirically. Gradients can vanish for symmetric/saturated states, so nondegenerate moderate features and nonzero weights are used for acceptance fixtures.

Both StatevectorEstimator.default_precision and EstimatorQNN.default_precision are zero. The explicit ParamShiftEstimatorGradient shares the same estimator; inspected installed source confirms the gradient path inherits estimator precision without a shot-noise override. All results are ideal CPU simulation, with no hardware account, API token, shots or cloud execution.

Independent Statevector binding/expectation calculations check the QNN's outputs. Single-qubit-X basis-state tests check local-Z order IIIZ, IIZI, IZII, ZIII. Central finite differences use step 0.001 and absolute/relative tolerances 0.0003/0.005, comparing all eight input-feature derivatives in a two-row batch and all 16 quantum-weight derivatives for a fixed weighted-output loss. Tests also verify batched/individual consistency, exact repeatability, registered parameter count and state-dict reload.

The toy optimization uses two fixed feature vectors. A teacher circuit uses the student's initial weights plus a fixed linearly spaced offset from 0.1 to 0.25 radians, producing reachable targets via an independent Statevector calculation. Only the student's 16 circuit weights are optimized: ten Adam updates at learning rate 0.01, recording all 11 loss values, input/weight gradient norms and parameter displacement. This is a controlled trainability check, not a dataset experiment, classical-vs-quantum comparison or generative-quality result.

Measured two-row forward/backward submitted-workload instrumentation counts one forward primitive call with four PUBs/eight expectations, and one backward call with 320 PUBs/320 expectations (two parameter shifts × 20 circuit parameters × two rows × four observables). Counts describe submitted binding instances and observable evaluations; bindings repeated across observables count again. They are not counts of unique circuits, physical hardware jobs/shots, or an instrument inside the statevector engine. Instrumentation overhead is included in reported simulator timings. Independent reference-Statevector work is excluded from estimator counters and identified separately.

The standalone probe and ten-update toy fit passed. All 13 new standalone tests and all 50 regression tests passed. Numerical errors, timing breakdowns and toy loss history are reported in RESULTS.md and results/phase4_checks.json. Quantum gradient evaluation is more expensive than forward evaluation; this supports measuring the actual hybrid-training/sampling workload in Phase 5 before choosing a full experiment budget. No hybrid DDPM, generation-quality test or quantum advantage result has been produced in Phase 4.

## Phase 5 integration, pairing and pilot protocol

`HybridDenoiser` subclasses the existing Denoiser and supplies QuantumLayer as the sole bottleneck replacement. The timestep embedding, 48→16→4 encoder, 4→16→32 decoder, DDPM buffers/formulas, noise-prediction MSE, Adam learning rate 0.001 and batch size 16 remain unchanged. Instantiated counts are 1,816 classical, 1,812 hybrid and 1,812 optional parameter-matched classical. The optional control removes the bottleneck's four biases; it is not an equal-expressiveness claim and does not modify the baseline.

The new model factory uses a seeded reference baseline as the canonical initial state. It copies only embedding/encoder/decoder values into the hybrid/control, avoiding the different Torch RNG consumption of different constructors. Copies use independent storage. The control additionally copies the baseline's 4×4 bottleneck weight matrix. Initialization uses a local Torch RNG context, so building an arm does not perturb training randomness. Tests verify matching shared states for all three repetition seeds and preservation of the original baseline initialization. The initial shared-state hash is saved in each pilot and checkpoint provenance.

The shared trainer now logs per-update gradient norms for timestep encoding, encoder, bottleneck and decoder, and parameter displacement from initialization for last/best models. The hybrid diffusion loss produced finite nonzero gradients and actual parameter changes in all four groups. The same optimizer owns all registered parameters; the quantum weights are not trained through a separate circuit-only objective. Checkpoint reload, repeated hybrid reverse sampling and no-gradient inference passed integration tests.

The executed Phase 5 pilot uses seed 11 and the original 1,000-row training condition. Classical and hybrid runs receive identical normalized train/validation arrays, independent RNG stream seeds, shuffle/noise/timestep generation policy, update budget five, fixed validation corruption, initial shared values and reverse-sampling seed 10011. Both retain the required 20-step schedule and generate exactly 200 sequences from Gaussian noise. The existing Phase 3/4 artifacts were preserved; new run IDs distinguish these pilots. No data regeneration or schedule change occurred.

Warm-up timing is separate; total training includes initial/final and per-update validation plus checkpoint writes. Quantum submitted workloads are recorded separately for warm-up, training and reverse sampling. Per-component gradient logging is included in update timings. The five-update pilot validates joint optimization and sampling only. Small differences in validation MSE between two pilots do not establish generation superiority, data efficiency or statistical significance. Both sets of smoke outputs remain noise-like; no formal generation-quality metrics have been computed yet.

Budget estimates use the measured median hybrid update time and actual validation/sample timings. For common optimizer budget K in every condition, the two-model grid has 4 fractions × 3 seeds × 2 models = 24 runs. Its optimizer-only estimate is 12*K*(classical median update seconds + hybrid median update seconds). A separate one-epoch-per-condition estimate is only a runtime illustration; it is not the proposed equal-update scarcity protocol or a claim that one epoch suffices. Add initial/periodic/final validation, checkpoint I/O, warm-up and equal-size reverse sampling. Estimates assume one seed/full-batch timing generalizes and therefore do not capture convergence, shorter final batches or system variation. The optional control adds another 12 runs if included later; its full training has not been measured.

All eight new hybrid tests and 58 full regression tests passed. Exact regeneration from both best checkpoints and equality of data/config/normalization/RNG settings were verified. The measured two-row/QNN and batch-16 hybrid costs justify requiring a reviewed full-training budget before launching Phase 6. Phase 5 does not select a budget or start long training.
