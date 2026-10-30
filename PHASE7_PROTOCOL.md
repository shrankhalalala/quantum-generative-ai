# Phase 7 execution protocol

The user authorized Phase 7 on 30 October 2026. The machine-readable protocol
was written to results/phase7/protocol.json before selection runs began.
Historical Phase 1–6 measurements remain unchanged.

## Shared schedule revision

Use 100 linear-beta steps from 0.0001 to 0.1 in every arm. This reduces terminal
retained signal variance from approximately 0.8168 to 0.00562. The schedule was
chosen analytically to leave less than 1% signal variance, without searching
generated test quality. It still approximates, rather than exactly equals, a
Gaussian terminal prior. All models use the same 100-entry timestep embedding;
this changes parameter counts to 3,096 classical and 3,092 hybrid. The dataset,
encoder/decoder dimensions, quantum circuit, optimizer and batch size are unchanged.

## Exact batched quantum execution

Phase 7 uses the optional `batched_statevector` backend. Qiskit still specifies
the circuit; EstimatorQNN and TorchConnector still connect it to PyTorch. Independent
statevectors and +/- pi/2 parameter shifts are evaluated in batches using NumPy
complex128 arithmetic. This implements the same four-qubit gates, observables and
parameter-shift derivatives; it does not substitute a classical learned layer.

Tests compare outputs, input/weight gradients and optimizer steps against the
original Qiskit StatevectorEstimator. Unsupported circuits or observables fall
back to the reference implementation. The original `qiskit` backend remains the
default for historical commands. Batching changes primitive/PUB call counts, so
compare submitted parameter bindings and observable evaluations with their
documented meanings. These timings describe local classical simulation only.

## Budget selection (validation only)

Train a seed-11 hybrid pilot on the 1,000-row training subset for 1,000 updates.
Evaluate fixed-corruption validation noise MSE every 100 updates. Compare the
best validation MSE by step 500 with the best by step 1,000. Select 1,000 updates
for all final arms if the relative improvement is at least 0.5%; otherwise select
500. This is a predeclared cost-bounded rule, not a convergence guarantee.

## Guidance selection (validation only)

At the selected budget, train identically initialized hybrids with lambda values
0, 0.1, 1 and 10 using seed 11 and 1,000 training rows. Checkpoints in all cases
are selected solely by fixed-corruption validation noise MSE. Generate 200 samples
from each selected checkpoint with the same validation sampling seed 2029.

Among positive lambdas whose best validation noise MSE is at most 110% of the
zero-weight control's value, choose the lowest generated-validation ACF MAE.
Break ties in favor of the smaller lambda. Fail explicitly if no positive lambda
qualifies. The zero-weight arm is a tuning control, not an eligible replacement
for the required guided configuration. Write and freeze the selection before
computing final held-out quality metrics.

## Final grid and verification

Execute the three required configurations at each seed (11, 22, 33) and nested
training size (1000, 500, 250, 100): **36 final runs**. Use the selected common
update budget and guided weight. Keep training data, shared initial layers,
full hybrid initial states, corruption streams and sampling noise paired.
Training retains the final short batch; record actual examples and effective
epochs rather than assuming equal epochs at different training sizes.

Run sequentially with one Torch thread. Record warm-up separately and preserve
environment, simulator, timing, per-component losses, data hashes, checkpoints,
generated arrays, and physical-unit held-out metrics. Save 200 generated sequences
per run. Verify pairing and independently reproduce every final checkpoint's
saved samples and metrics exactly. Near-constant ACF rows are flagged explicitly.

## Progress and recovery

Run `python experiments/phase7.py`. Inspect results/phase7/status.json and the
current run's progress JSON. Completed training/results are reused only after
configuration/checksum checks. If training itself was interrupted, preserve that
attempt's checkpoint and restart the case deterministically in a fresh attempt;
exact optimizer resume is not claimed. If sampling/plotting was interrupted after
completed training, reuse the completed training checkpoint.

Completion requires 36 final runs, 12 verified comparison groups, and a completed
status record. Results are stored in results/phase7/final_runs.csv. Phase 8 will
aggregate across seeds, create comparison figures and write the interpretation.
Neither execution completion nor an improved validation loss demonstrates
convergence, realistic generation or quantum advantage.
