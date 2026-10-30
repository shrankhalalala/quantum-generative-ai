# Results

Phases 1–7 are complete under the [eight-phase roadmap](PROJECT_PHASES.md). The latest full suite passed **79 tests** on 30 October 2026. Phase 7 executed **36 final runs**, selected lambda/budget using validation data, and verified exact checkpoint/sample/metric reproduction for all 12 groups. **Generation quality remains poor; convergence and quantum advantage have not been demonstrated.** See [Phase 7 results](PHASE7_RESULTS.md) and the [raw final-run CSV](results/phase7/final_runs.csv). Phase 8 aggregation and reporting remain pending.

The sections below preserve the historical Phase 1–6 pilot measurements. They used the original 20-step schedule and should not be combined with Phase 7's revised 100-step study as one protocol.

## Dataset

Generated 1,400 sequences of length 32, split 1,000/200/200. Nested subsets and training-only normalization passed validation.

| Mean | Population std | Minimum | Maximum |
|---:|---:|---:|---:|
| -0.002746 | 0.669162 | -1.507733 | 1.571020 |

Evidence: [dataset summary](results/tables/dataset_summary.json), [validation record](results/dataset_validation.json).

## Standalone quantum layer

Verified four qubits, four inputs, 16 trainable weights and finite (B,4) outputs. Input/weight gradients, independent Statevector expectations, finite differences and state reload passed.

| Check | Measured result |
|---|---:|
| Ten-update toy MSE, initial → final | 0.018880 → 0.003244 |
| Maximum feature/weight derivative discrepancies | 1.85e-05 / 2.25e-05 |
| Two-row forward / backward time | 0.0046 s / 0.1445 s |

This reachable-target toy fit verifies trainability only. Evidence: [quantum checks](results/phase4_checks.json).

## Paired diffusion pilots

Both models used seed 11, 1,000 training rows, five Adam updates, batch size 16, LR 0.001, identical shared initial weights and corruption streams, and 200 generated sequences. Each processed 80 training examples, or 0.08 effective epochs.

| Measurement | Classical | Hybrid |
|---|---:|---:|
| Trainable parameters | 1,816 | 1,812 |
| Initial validation MSE | 1.020605 | 1.020298 |
| Best validation MSE | 1.017362 | 1.017026 |
| Training, including validation/checkpoints | 0.0437 s | 8.0859 s |
| Median optimizer update | 0.0010 s | 1.1562 s |
| Sampling 200 sequences | 0.0262 s | 7.1279 s |

Joint classical/quantum gradients and parameter updates passed. Both best checkpoints reproduced their saved samples exactly. Outputs were finite but remain noise-like; the small MSE difference establishes neither realistic generation nor superiority. The 1,812-parameter classical control is implemented/tested but has not undergone a comparative quality experiment.

Timings are local CPU/statevector simulation measurements. Warm-up is excluded from training time. The full training budget is unset; five updates do not establish convergence or quantum advantage.

Evidence: [classical pilot](results/phase5_classical_seed11_n1000.json), [hybrid pilot](results/phase5_hybrid_seed11_n1000.json), [paired verification and cost estimates](results/phase5_checks.json). Raw logs, figures and generated arrays remain in results/; local checkpoints are excluded from Git.

## Phase 6: teacher's three configurations

All arms used seed 11, the same 1,000-row subset, five updates, matched shared initialization and noise streams, and 200 generated samples. Both hybrid arms started with identical full parameter states. Guided lambda was 1.0 (provisional, not tuned). All checkpoints were selected by validation noise MSE, never test metrics.

| Configuration | Validation noise MSE | Marginal W1 | Held-out ACF MAE | Training seconds | Sampling seconds |
|---|---:|---:|---:|---:|---:|
| classical | 1.017362 | 0.119016 | 0.378347 | 0.0430 | 0.0244 |
| hybrid | 1.017026 | 0.116828 | 0.379179 | 8.2479 | 7.0440 |
| hybrid_acf | 1.017031 | 0.116829 | 0.379177 | 8.0426 | 7.1070 |

These five-update outputs remain pilot diagnostics. The tiny differences do not establish improved generation, temporal fidelity or quantum advantage. Local timings include loss diagnostics and vary across runs.

New checks cover agreement with the existing NumPy ACF definition, clean reconstruction, finite-difference autograd checks, stable constant-row behavior, ACF-only gradients through the quantum/classical components, identical hybrid initialization, zero-weight training equivalence, checkpoint settings/reload and held-out metric sanity checks.

Evidence: [70-test log](results/phase6_tests.log), [paired comparison](results/phase6_three_config_seed11_n1000_comparison.json), [CSV table](results/tables/phase6_three_config_seed11_n1000_comparison.csv), [pilot execution log](results/phase6_pilot.log), and [verification](results/phase6_checks.json). Lambda tuning and the 36-run budgeted grid were subsequently completed in Phase 7; aggregation and reporting remain Phase 8.
