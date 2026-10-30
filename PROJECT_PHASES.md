# Project phases

Updated 30 October 2026 for the teacher's required three-configuration comparison.
This expanded eight-phase roadmap preserves the existing completed Phases 1–5
and defines the new implementation and remaining research work explicitly.

**7 phases complete; 1 phase remaining (Phase 8).** Implementation completion is separate
from evidence of model quality or quantum advantage.

| Phase | Work | Status / evidence |
|---|---|---|
| 1 | Environment, dependencies, scientific configuration and reproducibility | Complete — results/phase1_checks.json |
| 2 | Synthetic dataset, splits, nested training subsets and leakage-safe normalization | Complete — results/phase2_checks.json |
| 3 | Classical diffusion model, noise schedule, training and reverse sampling | Complete — results/phase3_checks.json |
| 4 | Trainable four-qubit layer and independent gradient/circuit verification | Complete — results/phase4_checks.json |
| 5 | Hybrid integration, paired initialization and classical/hybrid pilots | Complete — results/phase5_checks.json |
| 6 | Autocorrelation-guided hybrid, separate losses, held-out metrics, paired three-mode runner and pilots | Complete — results/phase6_tests.log, results/phase6_checks.json and results/phase6_three_config_seed11_n1000_comparison.json |
| 7 | Validation-only ACF-weight tuning; fix a common training budget; run full three-configuration experiments across seeds and training sizes | Complete — 36 final runs; 12 verified groups; PHASE7_RESULTS.md and results/phase7/completion.json |
| 8 | Aggregate repeated-run results, uncertainty, comparative figures/tables, interpretation and final paper/report | Remaining |

## What Phase 6 establishes

- Modes: `classical`, `hybrid`, `hybrid_acf`. The parameter-matched classical control remains optional.
- Guided loss: noise MSE + lambda × reconstructed-clean-signal ACF MSE, over lags 1–10.
- Gradients reach the quantum circuit; lambda zero reproduces ordinary hybrid training.
- Shared initial layers, full hybrid initial states, data, budgets and corruption/sampling streams are paired.
- Every arm selects its checkpoint by fixed-corruption validation noise MSE.
- Held-out Wasserstein, ACF and moment metrics are computed after selection, in original signal units.
- Five-update, seed-11 pilots generated 200 samples per arm. These are functional diagnostics, not a converged comparative study.

## Before drawing research conclusions

Phase 7 selected lambda 1.0 and 1000 common updates using validation data only,
then executed all 36 runs across the three seeds and four training sizes. See
[PHASE7_RESULTS.md](PHASE7_RESULTS.md) for the frozen selection, raw records,
verification and generation-quality limitations.

Phase 7 revised the shared schedule to 100 steps with beta 0.0001–0.1, reducing
terminal retained signal variance to 0.00562. This approximates a Gaussian prior
more closely, but generated amplitude distributions remain poorly matched.
Execution completion is not a convergence or realistic-generation claim.

Phase 8 must distinguish denoising loss, generated-signal temporal fidelity,
marginal fidelity and simulator cost. Report incomplete runs honestly; neither
these pilots nor simulator timings establish quantum advantage.
