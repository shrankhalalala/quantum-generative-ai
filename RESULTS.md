# Results and execution status

**Quantum Generative AI: A Survey of Hybrid Architectures and Emerging Applications**

## Phase 1

Environment installation/checks: **Passed** on 9 October 2026. Python 3.12.3, macOS ARM64, CPU execution. Direct installed versions: NumPy 2.5.3, SciPy 1.18.1, Matplotlib 3.11.2, PyTorch 2.14.1, Qiskit 2.5.2, Qiskit Machine Learning 0.9.1, pytest 9.1.1. Complete snapshot: requirements-lock.txt.

`pip check`: **No broken requirements found.**
Environment checker: **Passed** configuration/import/API, CPU autograd, repeatable RNG and headless rendering checks. Machine-readable evidence: results/environment.json. The first Matplotlib font-cache build printed cache-path warnings inside the sandbox but completed successfully; README uses writable temporary cache paths.
Configuration/reproducibility tests: **5 passed**, zero failures, using `python -m unittest discover -s tests -v` (15.515 seconds including first imports). Evidence: results/phase1_checks.json.

Phase 1 recorded in the initial local Git commit. GitHub origin configured; Phases 1 and 2 subsequently pushed to origin/master on 9 October 2026.
Schedule diagnostic (analytic computation only): alpha_bar_20 = 0.8167771026789972, signal coefficient = 0.9037572144547434, noise coefficient = 0.4280454383835936. Saved in results/schedule_diagnostic.json. This documents terminal-prior mismatch; it is not an experimental quality result.
At the end of Phase 1, no dataset or model had been executed. Dataset generation completed in Phase 2; the classical model and smoke training completed in Phase 3. Quantum execution/full comparisons remain unexecuted.

## Scientific outcomes

| Outcome | Status |
|---|---|
| Dataset summaries and validation plots | Computed and verified in Phase 2 |
| Classical training and generation quality | Five-update smoke completed; converged generation quality Not evaluated |
| Standalone quantum forward/gradient/update tests | Passed in Phase 4 |
| Hybrid training and generation quality | Not evaluated |
| Wasserstein, mean/std/variance differences | Not evaluated |
| ACF preservation/errors | Not evaluated |
| Training, pilot and reverse-sampling timings | Classical smoke and standalone quantum timings measured; hybrid/full timings Not evaluated |
| Limited-data comparisons across three seeds | Not evaluated |
| Quantum circuit evaluation counts | Submitted estimator workloads measured in Phase 4; hardware execution Not evaluated |
| Instantiated model parameter counts | Classical: 1,816; standalone quantum: 16; hybrid/control Not evaluated |

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

At the end of Phase 2, no model/training/reverse-generation experiment had run. Phase 3 evidence follows below. Phase 2 is recorded in a separate local Git commit with message: `feat: add reproducible sinusoidal dataset pipeline and validation`. Phase 3 was subsequently approved and implemented.

## Phase 3 — executed classical smoke validation

Commits created before Phase 3: Phase 1 `f0417c2`; Phase 2 `84ea5d3`. Both pushed to origin/master on 9 October 2026. Phase 3 is recorded in a separate commit for publication on 27 October 2026.

All **37 tests passed**, zero failures/errors, in 1.047 seconds (five Phase 1 + 15 Phase 2 + 17 Phase 3). Evidence: results/phase3_tests.log and phase3_checks.json. Checkpoint hash verification, exact regeneration of saved outputs and no exact raw-row copying passed. Loss and generated-series PNGs were visually inspected; matching PDFs exported.

Single measured smoke condition: classical model, seed 11, 1,000-row training subset, five updates, batch 16, Adam LR 0.001, 80 training-example exposures (0.08 effective epochs), CPU float32, required 20-step schedule. This is not full/converged training.

| Computed smoke outcome | Value |
|---|---:|
| Trainable parameters | 1816 |
| Initial validation MSE | 1.020604596 |
| Best/final validation MSE | 1.017361593 |
| Best checkpoint update | 5 |
| Warm-up (seconds; outside training) | 0.057706 |
| Total training, including validation/checkpoints (seconds) | 0.028980 |
| Optimizer-loop total (seconds) | 0.008032 |
| Mean update time (seconds) | 0.001606 |
| Median update time (seconds) | 0.000720 |
| Validation total (seconds) | 0.011091 |
| Checkpoint-write total (seconds) | 0.009432 |
| Reverse-sampling 200 sequences (seconds) | 0.023875 |
| Maximum absolute generated amplitude (raw units) | 3.375194 |

Generated shape (200,32), all finite; sampling seed 10011; maximum normalized sample magnitude 5.051081, below diagnostic threshold 100. These samples are visibly noise-like; no useful time-series generation claim follows from this smoke test. Marginal Wasserstein/ACF error and formal held-out quality remain Not evaluated.

Classical runtime extrapolations below are **estimates**, not measured training runs at those fractions: multiply median full-batch update time by ceil(N/16). They exclude validation/checkpoint overhead, use only five updates, do not determine a convergence budget, and say nothing about quantum runtime.

| Training rows | Updates/epoch | Estimated optimizer-only seconds/epoch |
|---|---:|---:|
| 1000 | 63 | 0.045347 |
| 500 | 32 | 0.023033 |
| 250 | 16 | 0.011517 |
| 100 | 7 | 0.005039 |

Raw evidence/config/history/timing/normalization/hardware/hash records: results/classical_smoke_seed11_n1000.json. Local best checkpoint: results/checkpoints/classical_smoke_seed11_n1000_best.pt (ignored by Git). Saved generated physical-scale samples: results/samples/classical_smoke_seed11_n1000.npy. Figures: results/figures/classical_smoke_seed11_n1000_loss.png/.pdf and _generated_series.png/.pdf.

No unresolved functional Phase 3 test failure remains. At Phase 3 completion, convergence, realistic generation, comparative metrics and quantum compatibility remained unestablished. Standalone quantum compatibility has since passed Phase 4 validation; the other outcomes remain unestablished. Full training budget is still unset and requires discussion/approval; no long job has run. Phase 4 was subsequently approved; its computed evidence follows below. Suggested Phase 3 commit: `feat: implement classical DDPM training sampling and smoke validation`.

## Publication and next-phase planning — 27 October 2026

Phase 3 prepared for publication as its own commit, preserving the measured 9 October smoke results. PHASE4_PLAN.md specifies standalone circuit/QNN construction, gradient verification and a short toy optimization. At planning time, Phase 4 implementation and quantum forward/backward execution had not started; subsequent Phase 4 execution is documented below. Publication verification is reported in the chat; preparing this commit is not evidence that a remote push has already succeeded.

Pre-publication checks on 27 October 2026: all 37 regression tests passed in 2.046 seconds; pip check reported no broken requirements. Installed StatevectorEstimator, EstimatorQNN and ParamShiftEstimatorGradient signatures were inspected for the Phase 4 plan; no quantum circuit was executed.

## Phase 4 — standalone quantum validation, 27 October 2026

Phase 3 is published as commit cda3611. Phase 4 is recorded in its own implementation commit for publication on 27 October 2026. The approved four-qubit architecture is unchanged. No dataset regeneration, hybrid-model implementation or diffusion training occurred in this phase.

All eight mandatory quantum checks passed: four qubits, four inputs, 16 trainable weights, batched four-dimensional outputs, finite outputs, quantum-weight gradients, input-feature gradients, and optimizer updates/loss decrease. The preceding-classical-Linear gradient check passed too. Standalone tests: **13 passed** in 3.275 s. Full regression suite: **50 passed** in 3.664 s, zero failures/errors. Raw evidence: phase4_quantum_tests.log and phase4_regression_tests.log.

| Measured standalone outcome | Value |
|---|---:|
| Qubits / input parameters / trainable weights | 4 / 4 / 16 |
| RY / RX / CX gate counts | 12 / 8 / 6 |
| Probe batch / output shape | (2,4) / (2,4) |
| Warm-up forward+backward (seconds) | 0.174878 |
| Probe forward (seconds) | 0.004612 |
| Probe backward (seconds) | 0.144488 |
| Quantum-weight gradient L2 norm | 0.425402731 |
| Feature-gradient L2 norm | 0.680278778 |
| Independent Statevector maximum output error | 1.86841003e-08 |
| Maximum absolute feature-derivative discrepancy | 1.85370445e-05 |
| Maximum absolute weight-derivative discrepancy | 2.25454569e-05 |
| Toy fit: initial MSE | 0.018879812 |
| Toy fit: final MSE after ten Adam updates | 0.003244146 |
| Final weight-displacement L2 norm | 0.361921400 |
| Ten-update toy-loop time (seconds) | 1.575515 |

Finite differences used step 0.001, atol 0.0003 and rtol 0.005, checking all eight fixture-feature entries and 16 weights. The independent expectation error compares against direct Statevector calculations, not a second QNN invocation. Toy targets were generated by the same circuit with initial weights plus a fixed 0.1–0.25-radian offset; batch size two, Adam LR 0.01, exactly ten updates, not a diffusion dataset experiment. All 11 losses, input/weight gradient norms, weights and displacement history are saved.

Measured submitted workloads for the two-row probe:

| Pass | Estimator run calls | PUBs | Parameter-binding instances | Observable evaluations |
|---|---:|---:|---:|---:|
| Forward | 1 | 4 | 8 | 8 |
| Backward | 1 | 320 | 320 | 320 |

Binding instances repeated across observables count again. These are instrumented submitted workloads, not unique circuits, hardware jobs/shots or a count internal to the simulator. The direct reference-Statevector work is excluded from estimator counts. Timings include instrumentation, and the ten-update time excludes initial target/optimizer construction; its loop includes post-update loss evaluations. All timings are ideal local CPU statevector simulation using pinned Qiskit 2.5.2 / Qiskit ML 0.9.1 / Torch 2.14.1. Both estimator and QNN precision are zero; no shots or hardware were used.

Circuit figures: results/figures/quantum_circuit.png, .pdf and .txt, exported from Qiskit's text drawer through Matplotlib without new dependencies. The PNG was visually inspected. Source/figure hashes, CPU thread counts and environment versions are in results/phase4_checks.json. No dependency version changed; pip check passed.

No unresolved functional Phase 4 failure remains. A small teacher-target fit demonstrates differentiability/trainability only. Hybrid DDPM gradients, training/sampling cost, generation quality, data scarcity and quantum advantage remain Not evaluated. Phase 5 awaits explicit approval. Suggested commit: `feat: add four-qubit quantum bottleneck and standalone gradient validation`.
