# Implementation explanation

**Quantum Generative AI: A Survey of Hybrid Architectures and Emerging Applications**

This implementation supports the survey, rather than replacing its literature review. Phase 1 only defines a reproducible experiment and verifies the environment.

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
| package __init__.py files | Package scaffolding; no inputs or computation | Empty aside from descriptions. Importing them cannot launch experiments. |
| README.md | Setup/execution guide | Provides only commands for programs that exist, phase boundaries and next steps. |
| EXPERIMENTAL_METHODOLOGY.md | Scientific specification | Contains dataset, models, forward/reverse equations, metric definitions, fair comparisons and validity threats. |
| RESULTS.md / results/environment.json | Evidence/status | Distinguishes executed environment checks from unexecuted scientific outcomes; never substitutes pilot checks for model results. |
| .gitignore | Local Git ignore policy | Excludes environment/cache and bulk dataset/checkpoints; retain these separately as experimental artifacts. Results/provenance files remain trackable. |

## Future file map (specified, not implemented)

- data/generate_data.py (Phase 2): config and seed → raw (1400,32) arrays, sampled signal parameters, split indices, nested subsets, normalization statistics, metadata and plots. Sin/cos arguments encode cycles over a window; training-only z-score and inverse transform prevent leakage.
- models/classical_denoiser.py (Phase 3): noisy (B,32), timestep (B,) → predicted noise (B,32), using learned time embedding, shared encoder, affine/Tanh bottleneck and decoder.
- diffusion/noise_schedule.py (Phase 3): beta bounds and T → beta/alpha/cumulative-alpha arrays; schedule buffers drive both forward noising and sampling.
- diffusion/sampling.py (Phase 3): trained model, schedule and independent RNG → newly generated (B,32) arrays through reverse means and posterior variances. Never copies dataset rows.
- training/train.py (Phase 3, extended Phase 5): model type, split/subset config and approved budget → losses, timings and best-validation checkpoint. Shared loss is mean squared Gaussian-noise prediction error; validation draws fixed for comparisons.
- models/quantum_layer.py (Phase 4): four encoder features → pi*tanh angles → four RY inputs → two trainable entangling layers → four local Z expectations. QNN and TorchConnector carry input/weight gradients; exactly 16 trainable circuit parameters.
- models/hybrid_denoiser.py (Phase 5): same inputs/outputs as baseline, swaps only bottleneck and reuses common architecture. Shared parameter initialization must be explicitly copied for paired trials.
- evaluation/metrics.py (Phase 6): inverse-transformed generated/reference arrays → marginal distribution metrics and mean per-sequence biased ACF errors. Correlations never cross sequence boundaries.
- evaluation/compare_models.py (Phase 6): raw per-run records → repeated-run means/sample standard deviations and provenance-preserving comparisons.
- experiments/run_experiments.py (Phase 6): approved config/budget → model × fraction × seed runs with shared training/evaluation procedures; no automatic execution at import.
- evaluation/plotting.py (Phase 7 or earlier phase-specific plots): actual data/losses/metrics → Matplotlib PNG/PDF; fixed sample selection, labels, no invented values.

The methodology specifies complete formulas and tensor dimensions so these implementations can be defended during evaluation. Explanations will be updated with actual functions and test evidence as each phase is approved and completed.
