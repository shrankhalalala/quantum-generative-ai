# Results and execution status

**Quantum Generative AI: A Survey of Hybrid Architectures and Emerging Applications**

## Phase 1

Environment installation/checks: **Passed** on 9 October 2026. Python 3.12.3, macOS ARM64, CPU execution. Direct installed versions: NumPy 2.5.3, SciPy 1.18.1, Matplotlib 3.11.2, PyTorch 2.14.1, Qiskit 2.5.2, Qiskit Machine Learning 0.9.1, pytest 9.1.1. Complete snapshot: requirements-lock.txt.

`pip check`: **No broken requirements found.**
Environment checker: **Passed** configuration/import/API, CPU autograd, repeatable RNG and headless rendering checks. Machine-readable evidence: results/environment.json. The first Matplotlib font-cache build printed cache-path warnings inside the sandbox but completed successfully; README uses writable temporary cache paths.
Configuration/reproducibility tests: **5 passed**, zero failures, using `python -m unittest discover -s tests -v` (15.515 seconds including first imports). Evidence: results/phase1_checks.json.

Phase 1 recorded in the initial local Git commit. GitHub origin configured; no push performed.
Schedule diagnostic (analytic computation only): alpha_bar_20 = 0.8167771026789972, signal coefficient = 0.9037572144547434, noise coefficient = 0.4280454383835936. Saved in results/schedule_diagnostic.json. This documents terminal-prior mismatch; it is not an experimental quality result.
At the end of Phase 1, no dataset or model had been executed. Dataset generation is now completed in Phase 2; experimental models, quantum execution and training remain unexecuted.

## Scientific outcomes

| Outcome | Status |
|---|---|
| Dataset summaries and validation plots | Computed and verified in Phase 2 |
| Classical training and generation quality | Not evaluated |
| Standalone quantum forward/gradient/update tests | Not evaluated |
| Hybrid training and generation quality | Not evaluated |
| Wasserstein, mean/std/variance differences | Not evaluated |
| ACF preservation/errors | Not evaluated |
| Training, pilot and reverse-sampling timings | Not evaluated |
| Limited-data comparisons across three seeds | Not evaluated |
| Quantum circuit evaluation counts | Not evaluated |
| Instantiated model parameter counts | Not evaluated |

Analytical architecture counts and schedule diagnostics in the methodology are design calculations, not measured model outcomes. Raw computed results will be saved under results/ before being reported here.

## Phase 2 — computed dataset validation

Executed on 9 October 2026. Exactly 1,400 sequences generated with seed 2026; fixed split seed 2027. All 20 tests passed (15 new Phase 2 tests plus five Phase 1 regression tests), zero failures/errors, in 1.704 seconds. Raw output: results/phase2_tests.log. The generation CLI saved and reloaded all artifacts successfully; a separate `--verify-only` check passed. All three PNGs were visually inspected for readable labels, layout and correct fixed examples; matching vector PDFs were exported.

Raw physical-scale statistics (pooled entries, population std, ddof=0):

| Split | Shape | Mean | Std | Min | Max |
|---|---|---|---|---|---|
| all | 1400 × 32 | -0.002746375 | 0.669162346 | -1.507732958 | 1.571020196 |
| train | 1000 × 32 | -0.002543960 | 0.669347798 | -1.507732958 | 1.505788168 |
| validation | 200 × 32 | -0.001659977 | 0.668995108 | -1.489091838 | 1.571020196 |
| test | 200 × 32 | -0.004844845 | 0.668397417 | -1.431009339 | 1.521335152 |

Measured observation-noise mean: -0.000112292; std: 0.049888525. These are finite-sample realizations, not exact theoretical moments.

All 12 seed/subset conditions have saved indices and subset-only normalization records. Training data normalize to pooled mean approximately 0/std approximately 1; inverse transforms recover raw values within 1e-14 test tolerance. A perturbation test verifies that excluded training rows and held-out/validation values cannot affect the fit. Split counts/disjointness/coverage, exact RNG repeats, nested subsets, parameter ranges, non-finite/constant input rejection, and corruption detection all passed. No unresolved Phase 2 implementation issue was observed.

Representative indices (first six fixed held-out rows): 825, 1024, 641, 499, 1171, 16.

Computed artifacts: results/tables/dataset_summary.csv and .json; results/tables/normalization.json; results/dataset_metadata.json; results/dataset_validation.json. Saved data: data/datasets/dataset.npz, subsets.npz, normalization.json, metadata.json (locally retained; bulk artifacts ignored by Git).

Figures (each PNG/PDF): results/figures/dataset_representative_series, dataset_amplitude_distribution, dataset_autocorrelation. Their checksums and the held-out mean ACF values are recorded in results/dataset_validation.json.

No denoiser, quantum model, training, reverse generation, generation-quality comparison or timing experiment has run. Phase 2 is recorded in a separate local Git commit with message: `feat: add reproducible sinusoidal dataset pipeline and validation`. Phase 3 awaits explicit approval.
