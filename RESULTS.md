# Results

Phases 1–5 are implemented and published. The latest full test suite passed **58 tests** on 27 October 2026. Only short pilots have run; full training, held-out quality metrics, scarcity comparisons and repeated-seed variability are **Not evaluated**.

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
