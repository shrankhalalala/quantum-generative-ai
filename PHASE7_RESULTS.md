# Phase 7 results

Completed 30 October 2026. **All 36 final experiments and all 12 comparison-group reproduction checks passed.** The full execution took approximately 33 minutes, including validation selection and verification. Historical Phase 1–6 artifacts were preserved.

Execution completion does not mean successful signal generation: the generated signals remain noise-like and have much larger amplitudes than the reference data.

## What ran

| Item | Executed setting |
|---|---|
| Required configurations | Classical; hybrid without extra loss; hybrid with ACF guidance |
| Final seeds | 11, 22, 33 |
| Training-set sizes | 1000, 500, 250, 100 |
| Final experiments | 3 × 3 × 4 = 36 |
| Updates per final run | 1000, selected by the predeclared validation rule |
| Generated sequences per final run | 200; 7200 in total |
| Additional selection runs | One 1000-update budget pilot and four 1000-update lambda candidates |
| Guided weight | lambda = 1.0, selected on validation data only |
| Checkpoint selection | Fixed-corruption validation noise MSE for every arm |
| Shared diffusion schedule | 100 linear-beta steps, 0.0001 to 0.1 |
| Terminal retained signal variance | 0.00561876 (historical schedule: 0.816777) |
| Trainable parameter counts | Classical 3096; each hybrid 3092 |
| Simulator | Exact batched ideal statevectors and parameter-shift gradients, checked against Qiskit |

All runs used the same LR 0.001 and batch size 16, paired corruption/sampling streams, identical initial shared layers, and identical full initial states for the two hybrid arms. Runs were sequential, using one Torch thread. The optional parameter-matched classical control was not part of the required grid.

## Validation-only selection

The budget pilot's best validation noise MSE decreased from 0.939467 by step 500 to 0.924274 by step 1000, a 1.62% relative improvement. The frozen 0.5% rule therefore selected 1000 updates.

| Lambda candidate | Best validation noise MSE | Generated-validation ACF MAE | Generated-validation W1 |
|---:|---:|---:|---:|
| 0 | 0.924274 | 0.394330 | 8.958082 |
| 0.1 | 0.924079 | 0.394322 | 8.955830 |
| 1 | 0.922786 | 0.394258 | 8.945300 |
| 10 | 0.992770 | 0.397694 | 9.383384 |

Lambda 1.0 had the lowest generated-validation ACF MAE among positive weights satisfying the noise-MSE guard. The difference was tiny. This limited, one-seed search does not establish a statistically meaningful improvement. The zero-weight control reproduced the budget pilot's validation history exactly. Both selections were frozen before Phase 7 final held-out metrics were computed.

## Verification and observed limitations

- The full test suite passed **79 tests**. New tests cover JSON configuration reload, the terminal schedule, reference/batched outputs, input/weight gradients, optimizer updates, supported/fallback circuit execution, and selection rules.
- All 12 final groups passed data/configuration/initial-state/stream pairing checks.
- All 36 checkpoints reproduced their saved generated arrays and held-out metrics exactly.
- Every final run completed 1000 updates and saved losses, timings, configuration, environment, data provenance, weights, generated arrays and figures. There are no missing final runs.
- All evaluated ACFs were defined; no near-constant rows were silently removed.
- The source snapshot remained unchanged during execution.

The generated population standard deviations range from **11.38 to 12.05**, compared with **0.6684** for the held-out reference. Marginal Wasserstein distances range from **8.46 to 9.06**. These discrepancies and the generated-series figures show that realistic generation has not been achieved. The stronger schedule reduces the prior mismatch but does not solve the model's generation problem. Do not interpret lower noise-prediction loss as successful generation or claim quantum advantage.

The common update budget corresponds to 15.88, 31.256, 62.512 and 142.96 effective epochs for training sizes 1000, 500, 250 and 100 respectively. Final short batches were retained, so actual example counts differ across subset sizes. Paired configurations within a condition have identical exposures. These are cost-bounded experiments, not a convergence guarantee.

The reference split is the same one used in earlier functional pilots, not a newly collected blind test set. Phase 7's schedule choice was analytic and its budget/lambda selection used validation data only. Disclose the earlier reference exposure and the limited tuning search in the final report.

## Saved evidence

Checkpoints remain local under the project's Git ignore policy. The published source snapshot, fixed dataset generator, protocol, sample arrays and raw records support reproduction; a fresh clone should use `python experiments/phase7.py --results-dir results/phase7_reproduction` after generating the dataset.

- [Raw 36-run table](results/phase7/final_runs.csv)
- [Completion record](results/phase7/completion.json)
- [Frozen protocol](results/phase7/protocol.json) and [readable protocol](PHASE7_PROTOCOL.md)
- [Budget selection](results/phase7/budget_selection.json) and [frozen lambda selection](results/phase7/frozen_selection.json)
- [All-group verification](results/phase7/verification.json) and [quality sanity record](results/phase7/quality_sanity.json)
- [Execution log](results/phase7/execution.log) and [79-test log](results/phase7_tests.log)
- [Source snapshot](results/phase7/source_snapshot.zip) and [source checksums](results/phase7/source_sha256.json)
- [Representative guided samples](results/phase7/final/figures/phase7_seed11_n1000_hybrid_acf_generated_series.png)

**7 phases complete; 1 phase remaining.** Phase 8 will aggregate repeated-seed results, quantify variability, produce comparative figures/tables and write the findings and limitations. It must report the generation failure honestly; the completed grid does not establish model superiority.
