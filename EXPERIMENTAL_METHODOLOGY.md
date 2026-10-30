# Experimental methodology

This experiment extends the survey described in [README.md](README.md). The three-configuration implementation and evaluation tools are available. Phase 7 completed the 36-run cost-bounded grid; Phase 8 aggregation/reporting remains pending. Historical default settings below are overridden for Phase 7 by the frozen protocol described at the end.

## Objective and hypotheses

Compare (1) classical DDPM with noise MSE, (2) hybrid quantum DDPM with noise MSE, and (3) the identical hybrid with noise MSE plus autocorrelation guidance, on generation quality, temporal correlation, runtime and limited-data performance. Test whether replacing the bottleneck changes these outcomes and whether the difference depends on training-set size. No quantum improvement is assumed.

## Dataset

Generate 1,400 independent length-32 sequences:

$$
x(t)=A\sin(2\pi ft+\phi)+B\sin(4\pi ft+\psi)+\epsilon_{\mathrm{obs}}(t).
$$

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

Initialize four qubits in $\lvert 0000\rangle$. Map features to angles $a_i=\pi\tanh(h_i)$ in Torch, then apply $R_Y(a_i)$. Repeat two layers of $R_Y$ then $R_X$ on each qubit, followed by CX(0,1), CX(1,2), CX(2,3) in order. There are four inputs and 16 trainable weights, initialized with local seed 11 for the pilot and Uniform[-pi/4,pi/4).

Return local Z expectations in qubit order 0–3: Pauli labels IIIZ, IIZI, IZII, ZIII. Outputs have shape (B,4) and lie in [-1,1]. Use StatevectorEstimator and EstimatorQNN with default_precision=0, ParamShiftEstimatorGradient, input_gradients=True and TorchConnector. Torch inputs/weights are CPU float32; Qiskit uses ideal statevector simulation without shots or hardware access.

Gradients reach preceding features through the connector and the angle-map derivative $\pi\left(1-\tanh^2(h_i)\right)$. Verification covers circuit/parameter counts, observable order, finite batched outputs, independent Statevector agreement, input/weight gradients, finite differences, optimizer updates and state reload. The standalone toy check uses two reachable teacher-target examples and ten Adam updates at LR 0.01; it is separate from diffusion training.

## DDPM

The historical default uses $T=20$ steps with linearly spaced beta values from 0.0001 to 0.02. Phase 7 uses $T=100$ and the revised beta range documented below. Define:

$$
\alpha_k=1-\beta_k,
\qquad
\bar{\alpha}_k=\prod_{s=1}^{k}\alpha_s,
\qquad
\bar{\alpha}_0=1.
$$

Mathematical timestep $k=1,\ldots,T$ maps to code index $k-1$.

Forward corruption:

$$
x_k=\sqrt{\bar{\alpha}_k}\,x_0
    +\sqrt{1-\bar{\alpha}_k}\,\epsilon,
\qquad \epsilon\sim\mathcal{N}(0,I).
$$

Noise-prediction loss (mean over all batch and sequence entries):

$$
\mathcal{L}_{\mathrm{noise}}
=\operatorname{mean}\!\left[\left(\epsilon-\epsilon_\theta(x_k,k)\right)^2\right].
$$

Draw timesteps independently and uniformly for each training example. Reverse sampling starts from $x_T\sim\mathcal{N}(0,I)$. Its predicted mean is:

$$
\mu_\theta(x_k,k)
=\frac{1}{\sqrt{\alpha_k}}
\left(x_k-\frac{\beta_k}{\sqrt{1-\bar{\alpha}_k}}\,\epsilon_\theta(x_k,k)\right).
$$

The reverse-step variance is:

$$
\tilde{\beta}_k
=\beta_k\frac{1-\bar{\alpha}_{k-1}}{1-\bar{\alpha}_k}.
$$

For $k>1$, sample:

$$
x_{k-1}=\mu_\theta(x_k,k)+\sqrt{\tilde{\beta}_k}\,z,
\qquad z\sim\mathcal{N}(0,I).
$$

At $k=1$, return $\mu_\theta(x_1,1)$ without added noise. Check shape, finiteness and normalized maximum absolute amplitude (limit 100); raise on instability rather than clipping.

## Autocorrelation-guided configuration (Phase 6)

The guided arm uses the same architecture, parameter count, initial full model state and sampling process as the unguided hybrid. Only its optimization objective changes:

Reconstruct the estimated clean signal from the predicted noise:

$$
\hat{x}_0
=\frac{x_k-\sqrt{1-\bar{\alpha}_k}\,\epsilon_\theta(x_k,k)}
       {\sqrt{\bar{\alpha}_k}}.
$$

Compare the reconstructed and target autocorrelations, averaging over batch size $B$ and $K=10$ lags:

$$
\mathcal{L}_{\mathrm{ACF}}
=\frac{1}{BK}\sum_{i=1}^{B}\sum_{\ell=1}^{K}
\left[\operatorname{ACF}(\hat{x}_{0,i},\ell)
     -\operatorname{ACF}(x_{0,i},\ell)\right]^2.
$$

The guided training objective is:

$$
\mathcal{L}_{\mathrm{total}}
=\mathcal{L}_{\mathrm{noise}}+\lambda_{\mathrm{ACF}}\,\mathcal{L}_{\mathrm{ACF}}.
$$

The target is each paired clean training signal, not a reference/test ACF. Demean each row; use valid overlapping products and the full-window energy denominator. Clamp that denominator below at 1e-12 in the differentiable training loss, making constant-row training diagnostics finite. Evaluation instead reports undefined ACFs and row indices without silently dropping rows. ACF is invariant to the scalar offset/scale used by normalization, apart from the numerical guard. All operations remain in Torch so gradients flow through clean reconstruction, decoder, quantum circuit, encoder and timestep embedding. No guidance is applied during sampling.

The pilot weight 1.0 is provisional. Tune a small predeclared set of weights using fixed-corruption validation ACF/noise diagnostics only, while keeping budgets matched; do not choose weights based on test results. Setting lambda to zero is verified to reproduce ordinary hybrid training exactly. Record noise, ACF and total losses separately. For consistency, **all three arms still select checkpoints by validation noise MSE**, including step zero; the guided total loss is reported but does not choose the checkpoint. This selection rule may favor noise prediction over ACF fidelity and must be disclosed when interpreting results.

Run `python experiments/compare.py --smoke --lambda-acf 1.0 --run-id NAME` for a paired pilot. Replace `--smoke` with an explicit common `--max-steps` or `--epochs` budget for later experiments. Repeat with seeds 11/22/33 and subset sizes 1000/500/250/100 after fixing settings. The optional parameter-matched classical control is outside the teacher's three required configurations.

## Training and comparison controls

Use Adam, LR 0.001, batch size 16, CPU float32 and shuffled epochs retaining the last short batch. Copy identical initial embedding/encoder/decoder values into each model using independent parameter storage. Use separate RNG streams for initialization, selection, shuffling, timesteps, training noise and sampling. Fixed validation corruption uses seed 2028.

Select the best checkpoint solely by validation MSE, including the initial model as a candidate. Save weights, optimizer/schedule state, configuration, seeds, normalization, step/epoch and data provenance. Checkpoint inference reload is supported; exact interrupted-training resume is not implemented.

The executed pilots use five updates and 200 generated sequences per model. Full runs will use an approved common update budget across paired models and data fractions, with effective epochs/exposures reported. Long jobs require explicit approval. Use three repetition seeds if feasible; report measured costs and missing runs honestly if the design is unaffordable. Reference data must not select checkpoints or hyperparameters.

## Evaluation implementation and planned full study

The implemented evaluator reports the metrics below for each run; pilot values remain functional diagnostics. Use the same 200 held-out reference sequences and initially 200 generated sequences per run. Pair initial/per-step Gaussian sampling draws across models. Keep raw per-run records and failures; summarize repeated runs with means and sample standard deviations (ddof=1). A single seed has no estimated cross-seed standard deviation.

| Metric | Definition |
|---|---|
| Marginal Wasserstein-1 | Distance between flattened generated/reference values; does not measure complete temporal fidelity |
| Mean/std/variance differences | Generated minus reference pooled moments, plus absolute errors; population variance/std |
| ACF error | Mean absolute difference between aggregate ACFs over lags 1–10 |
| Computation | Training/update/validation/sampling wall time, parameter counts, hardware/simulator and submitted quantum workloads |

For ACF, demean each length-$L$ sequence separately, then use valid overlapping pairs and the full-window denominator:

$$
y_{i,j}=x_{i,j}-\frac{1}{L}\sum_{s=0}^{L-1}x_{i,s},
\qquad
r_i(\ell)=\frac{\sum_{j=0}^{L-\ell-1}y_{i,j}\,y_{i,j+\ell}}
                   {\sum_{j=0}^{L-1}y_{i,j}^{2}}.
$$

Average curves equally across sequences; never concatenate sequence boundaries. Report/flag near-constant rows rather than dropping them. Estimator calls, parameter-binding instances and observable evaluations are distinct counts; repeated bindings are not unique circuits or hardware shots.

Figures use fixed sample indices, readable labels, 300-dpi PNG and PDF. Cover real/generated series, ACFs, distributions, losses, scarcity metrics, timing, architecture and circuit diagrams using actual data only.

## Limitations

The historical default schedule gives $\bar{\alpha}_{20}\approx0.817$, retaining substantial signal at the final forward step; this mismatches Gaussian reverse initialization. The four-feature bottleneck, synthetic-only 32-point sequences, small pilot budget and limited repetition also restrict conclusions. Marginals/ACFs omit higher-order structure, and matching parameter count does not match capacity. Simulator timing establishes neither hardware speedup nor quantum advantage. Small denoising-MSE improvements do not establish realistic generation.

Dependencies are pinned in requirements-lock.txt. Measured evidence is summarized in [RESULTS.md](RESULTS.md). API references: [EstimatorQNN](https://qiskit-community.github.io/qiskit-machine-learning/stubs/qiskit_machine_learning.neural_networks.EstimatorQNN.html), [StatevectorEstimator](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.primitives.StatevectorEstimator).

## Executed Phase 7 protocol

See [PHASE7_PROTOCOL.md](PHASE7_PROTOCOL.md) for the predeclared selection rules and [PHASE7_RESULTS.md](PHASE7_RESULTS.md) for outcomes. All 36 final runs used 100 linear-beta steps from 0.0001 to 0.1, a 100-entry timestep embedding, LR 0.001, batch size 16, 1000 common optimizer updates, and 200 generated sequences. Retained terminal signal variance is 0.00562. This analytic schedule revision was shared across all arms, with no test-quality schedule search. Parameter counts become 3096 classical and 3092 hybrid.

A 1000-update hybrid pilot selected the common budget by the frozen validation-improvement rule. A validation-only sweep over lambda 0, 0.1, 1 and 10 selected lambda 1.0 using generated-validation ACF error with a noise-MSE guard. The selection was frozen before final Phase 7 held-out evaluation. The original reference split had been inspected in earlier functional pilots; it is not a fresh blind test set.

Phase 7's optional batched backend evaluates the same Qiskit circuit and exact +/- pi/2 shifts using vectorized complex128 statevectors. Qiskit QNN/Torch integration and gradient preprocessing/postprocessing remain in place. Tests verify reference outputs, gradients and optimizer updates; unsupported cases fall back to StatevectorEstimator. Workload call/PUB counts change with batching and must not be confused with unique circuits, shots or hardware jobs. All final runs were sequential with one Torch thread.

All 12 paired groups and all 36 checkpoint/sample/metric reproductions passed. Generated signals nevertheless have poorly matched amplitudes and remain noise-like. The fixed budget does not establish convergence; phase completion establishes executed, reproducible experiments only. Phase 8 must report these limitations and distinguish denoising MSE from generation quality.
