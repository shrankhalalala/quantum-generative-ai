# Quantum Generative AI: A Survey of Hybrid Architectures and Emerging Applications

A focused synthetic time-series diffusion experiment extending the existing survey of 25 selected research papers. **Current stage: Phase 2 complete — dataset pipeline, artifacts and validation.** Models and training have not been implemented. Phase 3 awaits explicit approval.

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
models/                           Phases 3–5; model files not yet created
diffusion/                        Phase 3; schedule/sampling not yet created
training/                         Phase 3; shared training not yet created
evaluation/                       Phase 6; metrics/comparison not yet created
experiments/                      Phase 6; run_experiments not yet created
results/figures, tables, checkpoints/  future computed artifacts
tests/test_phase1.py, test_phase2.py  environment/config and dataset tests
EXPERIMENTAL_METHODOLOGY.md        scientific design
IMPLEMENTATION_EXPLANATION.md      file guide and presentation explanation
RESULTS.md                        only measured outcomes/status
```

Package initializers are empty scaffolding, not model implementations. `data/generate_data.py` is implemented. Future `models/classical_denoiser.py`, `models/quantum_layer.py`, `models/hybrid_denoiser.py`, `diffusion/noise_schedule.py`, `diffusion/sampling.py`, `training/train.py`, `evaluation/metrics.py`, `evaluation/compare_models.py` and `experiments/run_experiments.py` will be created only in their approved phases. Their execution commands will be added when the programs exist.

## Scientific decisions

- Exactly 1,400 sequences, split 1,000/200/200; deterministic nested training prefixes 1,000/500/250/100.
- Fit scalar z-score statistics separately on each training subset, reuse across paired models, evaluate inverse-transformed outputs.
- Shared timestep-conditioned 48→16→4 encoder and 4→16→32 decoder; replace only the 4→4 bottleneck.
- Four-qubit RY input encoding, two RY/RX ansatz layers, directed CX chain, four local Z outputs; 16 quantum weights.
- Required 20-step linear-beta DDPM. Its weak terminal noise creates a documented Gaussian-prior mismatch; schedule changes require approval.
- Full training budget is unset; approve it after a measured pilot. Three seeds and at least 200 generated sequences per run are planned, not executed.

See the methodology for equations, ACF definition, comparison controls and limitations. Dataset validation is computed; trained-model quality/timing results remain **Not evaluated**.

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

Bulk datasets remain ignored by Git under the Phase 1 policy, but are saved locally. Track code and the small results/provenance records; regenerate the raw artifacts with the pinned environment or retain the dataset files separately. SHA-256 identifies this saved artifact; JSON/PDF timestamps may differ on regeneration even when all numerical data are identical. Phase 2 is recorded in a separate local Git commit; no push has been performed.

## Phase 3, after explicit approval

Implement the timestep-conditioned classical denoiser, 20-step linear schedule/forward noising, shared Adam training pipeline, fixed-noise validation and best-checkpoint saving, and correct reverse sampling with shape/finiteness checks. Test equations, timestep conditioning/gradients, deterministic streams, checkpoint reload and final-step sampling behavior. Run only a short smoke test first and estimate cost before requesting any long-training budget. Document actual outcomes and classical loss/sample plots. Quantum circuit implementation remains Phase 4, hybrid integration Phase 5 and formal comparisons Phase 6.

The workflow stops after every phase for approval. Suggested Phase 2 commit message: `feat: add reproducible sinusoidal dataset pipeline and validation`. Phase 1 is recorded in the initial local Git commit. The GitHub repository is configured as `origin`; no push has been performed.
