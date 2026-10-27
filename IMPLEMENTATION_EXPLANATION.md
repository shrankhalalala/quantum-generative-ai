# Implementation explanation

**Quantum Generative AI: A Survey of Hybrid Architectures and Emerging Applications**

This implementation supports the survey, rather than replacing its literature review. Phases 1–2 define the experiment, verify the environment and implement the reproducible dataset. Phase 3 now implements and smoke-tests the classical DDPM; the standalone quantum layer is now validated in Phase 4; hybrid models and full training remain future work.

## How to explain the experiment

We draw short noisy sinusoids with variable shape. A diffusion process adds known Gaussian noise. A timestep-conditioned network learns to predict that noise. Sampling starts from random noise and uses the prediction in an iterative reverse process. The comparison asks what changes when only a four-feature bottleneck becomes a trainable quantum circuit. Circuit expectations become ordinary decoder inputs, and the loss gradient updates both quantum angles and classical weights.

The required schedule only weakly corrupts the data by its final step, which makes sampling from Gaussian noise a modeling limitation. We must explain this even if denoising validation loss improves. Simulation runtime cannot be presented as quantum hardware runtime or speedup.

## Files implemented in Phase 1

| File | Purpose, inputs and outputs | Important operations and connections |
|---|---|---|
| config.py | Immutable dataclass specification; input is explicit constructor overrides; output is validated configuration/dictionary | Dataset/model/diffusion/training/evaluation sections; validate rejects inconsistent settings; quantum_weight_count computes qubits × layers × rotations = 16; to_dict produces JSON-compatible data. Imported by checks and future pipelines. |
| utils/reproducibility.py | seed_everything(seed) seeds Python/NumPy/Torch and returns a dedicated Generator; environment_metadata() returns platform/package dictionary; save_json(path,payload) writes metadata | Separate Generator avoids consuming global NumPy RNG; Torch deterministic algorithms enabled; hash seed must be set before Python starts. Strict JSON refuses NaN/Inf. No global changes on import. Future runs must save config, independent stream seeds, normalization, hardware and version metadata. |
| scripts/check_environment.py | CLI accepts optional --output path; writes results/environment.json | Checks config and scientific/API imports, repeatability, a simple CPU autograd derivative (d(x²)/dx=2x), and plotting without a display. Records GPU availability but runs CPU. Does not create the experimental QNN or verify its gradients. |
| tests/test_phase1.py | unittest suite, no dataset/training inputs | Verifies finalized defaults, invalid config rejection, exact RNG repeatability, invalid seeds and JSON persistence/finiteness. All arrays here are tiny random draws for checks, not generated research time series. |
| requirements.txt / requirements-lock.txt | Dependency inputs to pip; output is the isolated scientific environment | Direct version pins versus complete transitive snapshot. pip check verifies metadata consistency, not mathematical correctness. |
| package __init__.py files | Package names; no inputs or computation | Empty aside from descriptions. Importing them cannot launch experiments. |
| README.md | Setup/execution guide | Provides only commands for programs that exist, phase boundaries and next steps. |
| EXPERIMENTAL_METHODOLOGY.md | Scientific specification | Contains dataset, models, forward/reverse equations, metric definitions, fair comparisons and validity threats. |
| RESULTS.md / results/environment.json | Evidence/status | Distinguishes executed environment checks from unexecuted scientific outcomes; never substitutes pilot checks for model results. |
| .gitignore | Local Git ignore policy | Excludes environment/cache and bulk dataset/checkpoints; retain these separately as experimental artifacts. Results/provenance files remain trackable. |

## Files implemented in Phase 2

| File | Inputs → outputs | Functions, mathematics and connections |
|---|---|---|
| data/dataset.py | DatasetConfig → SyntheticDataset and integer indices | `generate_dataset` draws the signal parameters and noise, `clean_signal` broadcasts per-row parameters over t=arange(32)/32, `make_splits` permutes all 1,400 row IDs, `nested_subsets` permutes train IDs once per seed and takes prefixes. Validators check dimensions, parameter ranges, equation reconstruction and split coverage. |
| data/normalization.py | Selected training array → scalar NormalizationStats; arrays + saved stats → normalized arrays | `fit_normalization` computes pooled mean and population std across the selected subset. `transform`: (x-mu)/sigma, `inverse_transform`: z*sigma+mu. `prepare_splits` applies one recorded transform to train/validation/test; rejects non-finite arrays and zero/near-zero scales. |
| data/artifacts.py | Dataset/config/output directory → four saved NPZ/JSON artifacts and DatasetBundle | `save_bundle` stores raw signal parameters/noise, split/subset indices, stats and provenance; overwrite requires an explicit flag. `load_bundle` rejects hash/config/selection/statistics mismatches and disables pickle. `normalized(seed,size)` prepares paired-model inputs from saved stats. `summarize` computes shape/mean/std/min/max; `sha256` identifies files. |
| utils/time_series.py | Finite (N,L) arrays and max lag → (N,max_lag+1) ACFs or mean curve | `sequence_acf` demeans each row, uses the full sum of squared deviations as denominator for every lag, and sums overlapping products. `mean_acf` weights rows equally; never concatenates sequence boundaries. Near-constant rows have undefined ACF and trigger an error. The same functions will be used in Phase 6. |
| data/plotting.py | Verified DatasetBundle → three PNG/PDF figure pairs | `plot_dataset` uses first six fixed held-out IDs; plots signals, pooled amplitude histogram, individual ACFs and 200-row held-out mean ACF. No favorable-example selection. Matplotlib headless backend, 300 dpi and embedded TrueType PDF fonts. |
| data/generate_data.py | CLI/default configuration → saved dataset, summaries, figures and validation records | Orchestrates generation → persistence/verified reload → plots → result/provenance writes. `--verify-only` only reads/checks existing data; `--overwrite` explicitly regenerates. The `__main__` guard prevents execution on import. |
| tests/test_phase2.py | Fixed specification and isolated temporary artifact directories → 15 dataset test results | Tests exact repeats and seed changes, valid parameters/noise, global RNG independence, disjoint fixed splits, nested prefixes for all seeds, per-subset stats, inverse transforms, leakage perturbations, hand-calculated ACF and checksum/config failure handling. Full suite also includes five Phase 1 tests. |

The NPZ file stores raw unnormalized values; training will call `bundle.normalized(seed,size)` and convert those arrays to float32 Torch tensors. Storing indices instead of twelve duplicate datasets keeps subset membership explicit and prevents stale copies. Float64 raw storage avoids extra rounding in equation validation.

## Files implemented in Phase 3

| File | Inputs → outputs | Key functions, mathematics and connections |
|---|---|---|
| models/classical_denoiser.py | Noisy CPU float32 (B,32), int64 code times (B,) → noise prediction (B,32) | `Denoiser` defines separately named time embedding, encoder, replaceable bottleneck and decoder. `ClassicalDenoiser` supplies affine 4→4 + Tanh. `trainable_parameters` measures the count 1,816; all components and input tensors receive gradients. |
| diffusion/noise_schedule.py | DiffusionConfig → fixed Torch buffers; clean/noisy sequences + time/noise → forward/reverse tensors | `add_noise` computes sqrt(alpha_bar)*x0 + sqrt(1-alpha_bar)*epsilon. `reverse_mean` uses predicted epsilon; `reverse_step` adds sqrt(posterior_variance)*z, with variance exactly zero at time 0. Buffers are saved with checkpoints, not optimized. |
| diffusion/sampling.py | Model, schedule, N, seed and optional explicit noise tensors → SamplingResult | `sample` begins from Gaussian noise and iterates all 20 reverse steps under no_grad/eval. Chunked inference separates memory/batch cost from random draws. Return includes (N,32) samples, wall time and per-step max magnitudes; invalid/unstable results raise errors rather than clipping. |
| training/train.py | Compatible model, normalized train/validation tensors, explicit config budget and provenance → TrainingResult, best checkpoint; CLI also outputs samples/plots/JSON | `stream_seeds` separates RNG domains; `validation_draws` fixes validation times/noise. `validation_loss` sums squared errors then divides by total entries. `train_model` minimizes mean squared predicted-noise error with Adam, retains short batches, records complete/partial epochs and chooses best validation checkpoint. `restore_checkpoint` loads safely for inference. `main` verifies saved data, casts normalized arrays to float32, runs only the supplied budget, restores best model, samples and inverse-normalizes. |
| training/plotting.py | Actual run loss history and inverse-normalized outputs → two PNG/PDF figure pairs | `plot_run` plots interval-average training loss and fixed-draw validation loss, plus generated indices 0–5. Smoke titles state that these are not convergence/quality evidence. |
| tests/test_phase3.py | Small deterministic fixtures/temp checkpoints → 17 Phase 3 tests | Verifies gradient/timestep behavior, analytic DDPM buffers/formulas, final-step noise exclusion, sampler stability/reproducibility, exact optional-noise reverse calculation, validation weighting/RNG isolation, optimizer updates, short batches, repeated training, safe best-checkpoint reload, budget/overwrite/input/shape rejection. Total suite: 37. |

Presentation explanation: the network predicts the known added noise, not the original clean signal directly. The timestep embedding tells it how much noise the forward schedule added. Forward noising is a direct formula, but generation needs 20 iterative reverse steps. Adam updates every registered network parameter together. Validation uses a fixed corruption draw so checkpoint comparisons are less noisy; the held-out reference does not pick checkpoints.

The five-step run is a functional smoke test: its outputs are visibly noisy. A small denoising-MSE improvement cannot be presented as realistic generation. The imposed schedule ends far from pure noise, so its Gaussian initialization mismatch remains a validity limitation even after more training. Training and sample timings are CPU classical-model timings; standalone quantum simulation timings are now measured in Phase 4; the full hybrid workload still needs a Phase 5 pilot.

## Future file map (specified, not implemented)

- models/hybrid_denoiser.py (Phase 5): same inputs/outputs as baseline, swaps only bottleneck and reuses common architecture. Shared parameter initialization must be explicitly copied for paired trials.
- evaluation/metrics.py (Phase 6): inverse-transformed generated/reference arrays → marginal distribution metrics and mean per-sequence biased ACF errors. Correlations never cross sequence boundaries.
- evaluation/compare_models.py (Phase 6): raw per-run records → repeated-run means/sample standard deviations and provenance-preserving comparisons.
- experiments/run_experiments.py (Phase 6): approved config/budget → model × fraction × seed runs with shared training/evaluation procedures; no automatic execution at import.
- evaluation/plotting.py (Phase 7 or earlier phase-specific plots): actual data/losses/metrics → Matplotlib PNG/PDF; fixed sample selection, labels, no invented values.

The methodology specifies complete formulas and tensor dimensions so these implementations can be defended during evaluation. Explanations will be updated with actual functions and test evidence as each phase is approved and completed.

## Files implemented in Phase 4

| File | Inputs → outputs | Main functions and connections |
|---|---|---|
| models/quantum_layer.py | Finite CPU float32 features (B,4) → local-Z expectations (B,4) | `build_quantum_circuit` returns the four-qubit circuit, explicitly ordered inputs/weights and observables. `QuantumLayer` applies pi*Tanh, then TorchConnector backed by an exact EstimatorQNN. `weight` exposes the same 16 registered optimizer parameters. `CountingStatevectorEstimator` records submitted workload counts without changing ideal expectations. |
| scripts/check_quantum_layer.py | Explicit CLI invocation → diagnostic JSON/log, toy loss history and circuit files | `statevector_outputs` independently binds the circuit and computes exact observables. `finite_difference_check` compares all fixture-feature and weight derivatives to autograd, restoring perturbed weights afterward. `optimize_toy` performs exactly ten Adam updates against a reachable teacher target. `draw_circuit` exports the actual Qiskit text diagram using existing Matplotlib. `main` records warm-up, forward/backward timing, counters and failures; it refuses to overwrite an existing report. |
| tests/test_quantum_layer.py | Deterministic fixtures → 13 standalone tests | Checks structure/gate/parameter order, observable endianness, ideal precision settings, batched/individual outputs, independent Statevector agreement, finite-difference gradients, preceding-classical-layer gradients, optimizer updates/loss decrease, invalid input rejection, workload counters, reproducible initialization/outputs and state-dict reload. |
| results/phase4_checks.json and logs | Actual executions → evidence | Contains all eight mandatory-check statuses, concrete outputs/gradient norms, all derivative discrepancies, eleven toy losses, parameter changes, CPU/thread/package metadata and submitted-workload counters. New test counts/timings and artifact hashes are appended after the full suite and visual QA. |
| results/figures/quantum_circuit.* | Actual circuit → PNG/PDF/TXT | Shows the exact encoding and trainable parameter placement with directed CX connections. Text-drawer embedding avoids an extra optional drawing dependency. |

For presentation: four classical numbers become four bounded angles. Each RY gate places its qubit into a superposition; trainable RY/RX gates and CX interactions transform the joint four-qubit state. The outputs are expectation values, not sampled bitstrings. A loss on these four values produces gradients for the 16 rotation angles and for the original features. Thus an earlier classical encoder can learn along with the circuit in a future hybrid model.

The parameter-shift rule evaluates circuits at shifted angles to form derivatives. TorchConnector uses these derivatives in its custom backward function. Its internal NumPy conversion does not itself break autograd because the connector supplies the backward mapping; manually converting features to NumPy before the connector would break the preceding classical graph. The code avoids that mistake and tests the full feature-angle-circuit chain.

The independent toy fit is a reachable-target test, not sinusoidal generation. The ten updates reduced its recorded MSE and moved the quantum parameters. This validates optimization wiring. It does not imply that the hybrid diffusion model will beat the classical model or require less data. Phase 5 will integrate only the bottleneck and measure the actual combined workload; Phase 6 will perform fair comparative experiments.
