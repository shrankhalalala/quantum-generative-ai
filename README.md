# Quantum Generative AI: A Survey of Hybrid Architectures and Emerging Applications

A focused synthetic time-series diffusion experiment extending the existing survey of 25 selected research papers. **Current stage: Phase 3 complete — classical DDPM implementation and five-update smoke validation.** Full training, quantum circuit and hybrid comparisons remain future work. Phase 4 awaits explicit approval.

## Setup

Use Python 3.12 on the current macOS ARM64 system. The project-local `.venv` is already created. To recreate it from this directory:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
```

`requirements.txt` pins direct dependencies; `requirements-lock.txt` freezes the verified complete environment. For a closer reproduction on the same platform, install the lock instead. CPU is the common execution device for initial classical/hybrid runs. No IBM account, API token, Aer installation, or cloud service is required for the planned Qiskit StatevectorEstimator.

## Phase 1 commands

Run from `quantum-generative-ai/`:

```bash
XDG_CACHE_HOME=/tmp/qgai-cache MPLCONFIGDIR=/tmp/qgai-matplotlib PYTHONHASHSEED=0 .venv/bin/python scripts/check_environment.py
PYTHONHASHSEED=0 .venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m pip check
```

The check saves `results/environment.json` and checks imports, V2 API signatures, CPU autograd, repeatable random draws and plotting. It does not construct an experimental quantum circuit. Mandatory quantum forward/backward/optimization tests belong to Phase 4. The Phase 1 test suite validates configuration, seed repeatability and strict JSON provenance.

## Project structure and phase ownership

```text
config.py                         central specification, implemented
utils/reproducibility.py          explicit seeding/provenance, implemented
scripts/check_environment.py      Phase 1 checks, implemented
requirements.txt                  direct version pins
requirements-lock.txt             complete installed package snapshot
data/                             Phase 2 implemented; saved datasets/ artifacts
models/classical_denoiser.py       shared architecture and classical bottleneck
diffusion/                        implemented noise schedule and reverse sampler
training/                         implemented shared trainer and smoke plots
evaluation/                       Phase 6; metrics/comparison not yet created
experiments/                      Phase 6; run_experiments not yet created
results/figures, tables, checkpoints, samples/  computed data/smoke artifacts
tests/test_phase1.py, test_phase2.py, test_phase3.py  37 tests
EXPERIMENTAL_METHODOLOGY.md        scientific design
IMPLEMENTATION_EXPLANATION.md      file guide and presentation explanation
RESULTS.md                        only measured outcomes/status
```

Package initializers have no execution side effects. Models, diffusion and training programs are now implemented for the classical baseline. Future `models/quantum_layer.py`, `models/hybrid_denoiser.py`, `evaluation/metrics.py`, `evaluation/compare_models.py` and `experiments/run_experiments.py` will be created in their approved phases.

## Scientific decisions

- Exactly 1,400 sequences, split 1,000/200/200; deterministic nested training prefixes 1,000/500/250/100.
- Fit scalar z-score statistics separately on each training subset, reuse across paired models, evaluate inverse-transformed outputs.
- Shared timestep-conditioned 48→16→4 encoder and 4→16→32 decoder; replace only the 4→4 bottleneck.
- Four-qubit RY input encoding, two RY/RX ansatz layers, directed CX chain, four local Z outputs; 16 quantum weights.
- Required 20-step linear-beta DDPM. Its weak terminal noise creates a documented Gaussian-prior mismatch; schedule changes require approval.
- Full training budget is unset; approve it after a measured pilot. Three-seed full comparisons remain planned. The Phase 3 smoke run uses one seed, five updates and 200 generated samples.

See the methodology for equations, ACF definition, comparison controls and limitations. Dataset validation and classical smoke losses/timings are computed; converged generation quality and model comparisons remain **Not evaluated**.

## Phase 2 commands and outputs

Generate from the fixed central configuration (refuses to overwrite existing artifacts):

```bash
XDG_CACHE_HOME=/tmp/qgai-cache MPLCONFIGDIR=/tmp/qgai-matplotlib PYTHONHASHSEED=0 .venv/bin/python data/generate_data.py
.venv/bin/python data/generate_data.py --verify-only
PYTHONHASHSEED=0 .venv/bin/python -m unittest discover -s tests -v
```

For intentional regeneration use `--overwrite`. `--output-dir` and `--results-dir` allow alternate locations. Verification checks hashes, equation reconstruction using stored observation noise, exact seeded split/subset indices and training-only normalization. Reload compatibility checks dataset settings and repetition seeds; later training-budget settings do not invalidate saved data. It creates no output files. Importing modules never generates a dataset or launches training.

Local artifacts:

- `data/datasets/dataset.npz`: raw float64 sequences, time grid, per-row amplitude/frequency/phases/harmonic amplitude, observation noise and three split-index arrays; load with `allow_pickle=False`.
- `data/datasets/subsets.npz`: twelve original-row index arrays, keyed `seed_11_n_1000`, etc.; one permutation and nested prefixes per repetition.
- `data/datasets/normalization.json`: twelve pooled scalar mean/std records fitted only on the relevant training subset; metadata includes sequence count/length and ddof=0.
- `data/datasets/metadata.json`: equation, distributions/config, RNG draw order, versions, summaries, sample selection, timestamp and SHA-256 hashes.
- `results/dataset_metadata.json`, `results/dataset_validation.json`: small provenance copies and saved-artifact checks.
- `results/tables/dataset_summary.csv`, `.json` and `normalization.json`: summaries and per-condition normalization records.
- `results/figures/dataset_representative_series.*`, `dataset_amplitude_distribution.*`, `dataset_autocorrelation.*`: three figures, PNG at 300 dpi and vector PDF.
- `results/phase2_tests.log`, `phase2_generation.log`, `phase2_checks.json`: executed test/generation evidence.

Raw arrays use float64 for accurate reproducibility; Phase 3 will explicitly convert normalized arrays to PyTorch float32. The saved arrays remain in raw physical units. Normalized tensors are prepared on demand using `DatasetBundle.normalized(seed, size)`; held-out/validation membership is never changed. Training-normalized data have pooled mean 0/std 1; evaluation data are not forced to have those moments.

Bulk datasets remain ignored by Git under the Phase 1 policy, but are saved locally. Track code and the small results/provenance records; regenerate the raw artifacts with the pinned environment or retain the dataset files separately. SHA-256 identifies this saved artifact; JSON/PDF timestamps may differ on regeneration even when all numerical data are identical. Phase 2 has its own Git commit; Phases 1 and 2 have been pushed to `origin/master`.

## Phase 3 commands and outputs

Run the classical five-update smoke test (uses seed 11 and the 1,000-row subset):

```bash
XDG_CACHE_HOME=/tmp/qgai-cache MPLCONFIGDIR=/tmp/qgai-matplotlib PYTHONHASHSEED=0 .venv/bin/python training/train.py --smoke
PYTHONHASHSEED=0 .venv/bin/python -m unittest discover -s tests -v
```

The smoke artifacts already exist; the command intentionally refuses to overwrite them. To reproduce numerical behavior into a new run, add `--run-id classical_smoke_repeat`. CPU wall times may differ. Use `--help` for the three allowed seeds, four subset sizes, data/results directories and validation interval. Without `--smoke`, exactly one explicit `--max-steps N` or `--epochs N` is required. A long-training budget must first be discussed and approved; none has been selected or executed. Full epochs/max_steps remain unset in DEFAULT_CONFIG; each actual run records its own budget.

The default smoke validates after every update. An explicitly budgeted run defaults to validation every 63 updates and at the final update; `--validation-every` can override it. Every epoch shuffles once and keeps the final short batch. The validation noise/timesteps are fixed across checkpoints and model/seed comparisons. Separate Torch Generators isolate shuffling, training timesteps, training noise and reverse-sampling randomness.

Implemented architecture: learned Embedding(20,16) + noisy 32-vector → concat 48 → Linear/SiLU/Linear encoder 48→16→4 → Linear(4,4)/Tanh classical bottleneck → Linear/SiLU/Linear decoder 4→16→32. Baseline count is 1,816. `Denoiser` exposes shared components for Phase 5; later paired models must explicitly copy initial shared states. No hybrid/quantum module is implemented yet.

Outputs of the existing smoke run:

- `results/classical_smoke_seed11_n1000.json`: full configuration, independent RNG seeds, losses, example/update/epoch counts, timing splits, normalization, hardware/package metadata, artifact hashes and explicitly estimated classical update costs.
- `results/checkpoints/classical_smoke_seed11_n1000_best.pt`: local best checkpoint, selected only by fixed-draw validation MSE, with model/optimizer/schedule, config, step/epoch/cursor, RNG states, normalization and dataset provenance. Step 0 is eligible if no update improves validation. Restore with `training.train.restore_checkpoint`, using `weights_only=True`.
- `results/samples/classical_smoke_seed11_n1000.npy`: 200 new inverse-normalized sequences, shape (200,32), loaded with `allow_pickle=False`.
- `results/figures/classical_smoke_seed11_n1000_loss.*` and `_generated_series.*`: actual losses and first six generated rows, clearly labeled as smoke output; PNG/PDF.
- `results/phase3_tests.log`, `phase3_smoke.log`, `phase3_checks.json`: executed test, CLI and saved-checkpoint/sample-reproduction evidence.

Sampling starts from Gaussian noise, iterates code times 19→0 with DDPM posterior variance and adds no noise at code time 0. The API accepts explicit initial/per-step noise for later paired comparisons; outputs must remain finite with max absolute normalized amplitude ≤100 (diagnostic threshold, no clipping). It preserves model mode and returns per-step magnitude diagnostics. The sample API never reads dataset rows. Exact checkpoint-to-saved-sample regeneration and absence of exact copied dataset rows were checked.

Checkpoint inference reload is supported; exact interrupted-training resume is not yet implemented. Epoch counts distinguish complete and partial passes. Recorded update-time extrapolations come from only five updates and exclude validation/checkpoint overhead; they do not establish convergence or quantum runtime. Smoke outputs remain visibly noise-like and cannot demonstrate time-series fidelity.

## Phase 4, after explicit approval

Implement only the standalone four-qubit RY-encoded, two-layer RY/RX/CX circuit and EstimatorQNN/TorchConnector module. Verify four inputs, 16 trainable circuit parameters, four local Z outputs, exact simulator behavior, input/weight gradients and optimizer updates before integrating it. Hybrid integration remains Phase 5; comparative experiments remain Phase 6.

Phase 3 is recorded in a separate commit, `feat: implement classical DDPM training sampling and smoke validation`. Phases 1 and 2 were published previously; Phase 3 is being published on 27 October 2026. The detailed standalone quantum-layer plan is in [PHASE4_PLAN.md](PHASE4_PLAN.md). Phase 4 implementation requires explicit approval; stop after each phase.
