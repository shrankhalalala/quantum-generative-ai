# Quantum Generative AI: A Survey of Hybrid Architectures and Emerging Applications

A focused synthetic time-series diffusion experiment extending the existing survey of 25 selected research papers. **Current stage: Phase 1 — specification and environment only.** No data generation, model implementation or training has run.

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
data/                             Phase 2; datasets/ currently empty
models/                           Phases 3–5; model files not yet created
diffusion/                        Phase 3; schedule/sampling not yet created
training/                         Phase 3; shared training not yet created
evaluation/                       Phase 6; metrics/comparison not yet created
experiments/                      Phase 6; run_experiments not yet created
results/figures, tables, checkpoints/  future computed artifacts
tests/test_phase1.py               implemented environment/config tests
EXPERIMENTAL_METHODOLOGY.md        scientific design
IMPLEMENTATION_EXPLANATION.md      file guide and presentation explanation
RESULTS.md                        only measured outcomes/status
```

Package initializers are empty scaffolding, not model implementations. Future `data/generate_data.py`, `models/classical_denoiser.py`, `models/quantum_layer.py`, `models/hybrid_denoiser.py`, `diffusion/noise_schedule.py`, `diffusion/sampling.py`, `training/train.py`, `evaluation/metrics.py`, `evaluation/compare_models.py` and `experiments/run_experiments.py` will be created only in their approved phases. Their execution commands will be added when the programs exist.

## Scientific decisions

- Exactly 1,400 sequences, split 1,000/200/200; deterministic nested training prefixes 1,000/500/250/100.
- Fit scalar z-score statistics separately on each training subset, reuse across paired models, evaluate inverse-transformed outputs.
- Shared timestep-conditioned 48→16→4 encoder and 4→16→32 decoder; replace only the 4→4 bottleneck.
- Four-qubit RY input encoding, two RY/RX ansatz layers, directed CX chain, four local Z outputs; 16 quantum weights.
- Required 20-step linear-beta DDPM. Its weak terminal noise creates a documented Gaussian-prior mismatch; schedule changes require approval.
- Full training budget is unset; approve it after a measured pilot. Three seeds and at least 200 generated sequences per run are planned, not executed.

See the methodology for equations, ACF definition, comparison controls and limitations. All experimental quality/timing results remain **Not evaluated**.

## Phase 2, after explicit approval

Implement data generation, saved split indices and signal parameters, deterministic nested subset construction, training-only normalization and persistence, summaries and PNG/PDF visualizations. Test shapes, exact counts, parameter ranges, split disjointness/coverage, subset nesting, deterministic generation, normalization leakage prevention and inverse transforms. Do not begin a denoiser or training in Phase 2.

The workflow stops after every phase for approval. Suggested Phase 1 commit message: `chore: scaffold reproducible quantum diffusion experiment and specify protocol`. Phase 1 is recorded in the initial local Git commit. The GitHub repository is configured as `origin`; no push has been performed.
