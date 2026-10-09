# Results and execution status

**Quantum Generative AI: A Survey of Hybrid Architectures and Emerging Applications**

## Phase 1

Environment installation/checks: **Passed** on 9 October 2026. Python 3.12.3, macOS ARM64, CPU execution. Direct installed versions: NumPy 2.5.3, SciPy 1.18.1, Matplotlib 3.11.2, PyTorch 2.14.1, Qiskit 2.5.2, Qiskit Machine Learning 0.9.1, pytest 9.1.1. Complete snapshot: requirements-lock.txt.

`pip check`: **No broken requirements found.**
Environment checker: **Passed** configuration/import/API, CPU autograd, repeatable RNG and headless rendering checks. Machine-readable evidence: results/environment.json. The first Matplotlib font-cache build printed cache-path warnings inside the sandbox but completed successfully; README uses writable temporary cache paths.
Configuration/reproducibility tests: **5 passed**, zero failures, using `python -m unittest discover -s tests -v` (15.515 seconds including first imports). Evidence: results/phase1_checks.json.

Phase 1 recorded in the initial local Git commit. GitHub origin configured; no push performed.
Schedule diagnostic (analytic computation only): alpha_bar_20 = 0.8167771026789972, signal coefficient = 0.9037572144547434, noise coefficient = 0.4280454383835936. Saved in results/schedule_diagnostic.json. This documents terminal-prior mismatch; it is not an experimental quality result.
No dataset, experimental model, quantum circuit, training, or generation has been executed.

## Scientific outcomes

| Outcome | Status |
|---|---|
| Dataset summaries and validation plots | Not evaluated |
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
