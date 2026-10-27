# Experimental methodology

This experiment extends the survey described in [README.md](README.md). Implementation and short pilots are complete; the full comparative study is pending.

## Objective and hypotheses

Compare classical and hybrid DDPMs on generation quality, temporal correlation, runtime and limited-data performance. Test whether replacing the bottleneck changes these outcomes and whether the difference depends on training-set size. No quantum improvement is assumed.

## Dataset

Generate 1,400 independent length-32 sequences:

\[
x(t)=A\sin(2\pi ft+\phi)+B\sin(4\pi ft+\psi)+\epsilon_{\mathrm{obs}}(t).
\]

| Quantity | Distribution |
|---|---|
| A | Uniform[0.6,1.2) |
| f | Uniform[1.0,3.0) cycles/window |
| phi, psi | Independent Uniform[0,2pi) |
| B | Uniform[0.0,0.3) |
| Observation noise | Independent Normal(0,0.05²) |
| Time grid | arange(32)/32; [0,1), endpoint=False |

Use NumPy default_rng seed 2026, drawing A, f, phi, B, psi, then observation noise. Split with a separate permutation seed 2027: 1,000 training, 200 validation, 200 held-out reference sequences. Save raw float64 arrays, parameters, noise, indices, seeds, versions and checksums.

For repetition seeds 11, 22 and 33, permute the training pool once and take nested prefixes of 1,000, 500, 250 and 100 rows. Validation/reference membership stays fixed. Fit a scalar mean and population standard deviation (ddof=0) using each selected training subset only. Apply its z-score to all splits and inverse-transform generated samples before evaluation. Paired models use identical indices and statistics.

## Denoiser architecture

Input is a noisy (B,32) tensor and a diffusion timestep (B,). A learned Embedding(20,16) supplies timestep conditioning.

| Component | Operation |
|---|---|
| Combined input | Concatenate 32 sequence values and 16 embedding values |
| Encoder | Linear(48,16), SiLU, Linear(16,4) |
| Classical bottleneck | Linear(4,4), Tanh |
| Hybrid bottleneck | Four-qubit layer below |
| Decoder | Linear(4,16), SiLU, Linear(16,32) |

The final output predicts unbounded Gaussian noise. Measured trainable counts: classical 1,816; hybrid 1,812; optional bias-free classical bottleneck control 1,812. Equal counts do not imply equal expressiveness.

## Quantum layer

Initialize four qubits in |0000>. Map features to angles a_i=pi*tanh(h_i) in Torch, then apply RY(a_i). Repeat two layers of RY then RX on each qubit, followed by CX(0,1), CX(1,2), CX(2,3) in order. There are four inputs and 16 trainable weights, initialized with local seed 11 for the pilot and Uniform[-pi/4,pi/4).

Return local Z expectations in qubit order 0–3: Pauli labels IIIZ, IIZI, IZII, ZIII. Outputs have shape (B,4) and lie in [-1,1]. Use StatevectorEstimator and EstimatorQNN with default_precision=0, ParamShiftEstimatorGradient, input_gradients=True and TorchConnector. Torch inputs/weights are CPU float32; Qiskit uses ideal statevector simulation without shots or hardware access.

Gradients reach preceding features through the connector and the angle-map derivative pi*(1-tanh(h_i)²). Verification covers circuit/parameter counts, observable order, finite batched outputs, independent Statevector agreement, input/weight gradients, finite differences, optimizer updates and state reload. The standalone toy check uses two reachable teacher-target examples and ten Adam updates at LR 0.01; it is separate from diffusion training.

## DDPM

Use 20 steps with linearly spaced beta values from 0.0001 to 0.02. Define alpha_k=1-beta_k and cumulative alpha_bar_k, with alpha_bar_0=1. Mathematical time k=1–20 maps to code index k-1.

Forward corruption and noise-prediction loss:

\[
x_k=\sqrt{\bar\alpha_k}x_0+\sqrt{1-\bar\alpha_k}\epsilon,
\quad\epsilon\sim\mathcal N(0,I),\qquad
\mathcal L=\operatorname{mean}(\epsilon-\epsilon_\theta(x_k,k))^2.
\]

Draw timesteps independently and uniformly for each training example. Reverse sampling starts from x_20~N(0,I), using

\[
\mu_\theta=\alpha_k^{-1/2}\left(x_k-\frac{\beta_k}{\sqrt{1-\bar\alpha_k}}\epsilon_\theta(x_k,k)\right),
\qquad
\tilde\beta_k=\beta_k\frac{1-\bar\alpha_{k-1}}{1-\bar\alpha_k}.
\]

For k>1, sample x_{k-1}=mu_theta+sqrt(beta_tilde_k)*z with z~N(0,I); at k=1 return mu_theta without added noise. Check shape, finiteness and normalized maximum absolute amplitude (limit 100); raise on instability rather than clipping.

## Training and comparison controls

Use Adam, LR 0.001, batch size 16, CPU float32 and shuffled epochs retaining the last short batch. Copy identical initial embedding/encoder/decoder values into each model using independent parameter storage. Use separate RNG streams for initialization, selection, shuffling, timesteps, training noise and sampling. Fixed validation corruption uses seed 2028.

Select the best checkpoint solely by validation MSE, including the initial model as a candidate. Save weights, optimizer/schedule state, configuration, seeds, normalization, step/epoch and data provenance. Checkpoint inference reload is supported; exact interrupted-training resume is not implemented.

The executed pilots use five updates and 200 generated sequences per model. Full runs will use an approved common update budget across paired models and data fractions, with effective epochs/exposures reported. Long jobs require explicit approval. Use three repetition seeds if feasible; report measured costs and missing runs honestly if the design is unaffordable. Reference data must not select checkpoints or hyperparameters.

## Planned evaluation

Use the same 200 held-out reference sequences and initially 200 generated sequences per run. Pair initial/per-step Gaussian sampling draws across models. Keep raw per-run records and failures; summarize repeated runs with means and sample standard deviations (ddof=1). A single seed has no estimated cross-seed standard deviation.

| Metric | Definition |
|---|---|
| Marginal Wasserstein-1 | Distance between flattened generated/reference values; does not measure complete temporal fidelity |
| Mean/std/variance differences | Generated minus reference pooled moments, plus absolute errors; population variance/std |
| ACF error | Mean absolute difference between aggregate ACFs over lags 1–10 |
| Computation | Training/update/validation/sampling wall time, parameter counts, hardware/simulator and submitted quantum workloads |

For ACF, demean each sequence separately: r_i(l)=sum_t y_i(t)y_i(t+l)/sum_t y_i(t)², using valid overlapping pairs and the full-window denominator. Average curves equally across sequences; never concatenate sequence boundaries. Report/flag near-constant rows rather than dropping them. Estimator calls, parameter-binding instances and observable evaluations are distinct counts; repeated bindings are not unique circuits or hardware shots.

Figures use fixed sample indices, readable labels, 300-dpi PNG and PDF. Cover real/generated series, ACFs, distributions, losses, scarcity metrics, timing, architecture and circuit diagrams using actual data only.

## Limitations

The required schedule gives alpha_bar_20≈0.817, retaining substantial signal at the final forward step; this mismatches Gaussian reverse initialization. The four-feature bottleneck, synthetic-only 32-point sequences, small pilot budget and limited repetition also restrict conclusions. Marginals/ACFs omit higher-order structure, and matching parameter count does not match capacity. Simulator timing establishes neither hardware speedup nor quantum advantage. Small denoising-MSE improvements do not establish realistic generation.

Dependencies are pinned in requirements-lock.txt. Measured evidence is summarized in [RESULTS.md](RESULTS.md). API references: [EstimatorQNN](https://qiskit-community.github.io/qiskit-machine-learning/stubs/qiskit_machine_learning.neural_networks.EstimatorQNN.html), [StatevectorEstimator](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.primitives.StatevectorEstimator).
