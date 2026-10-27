# Implementation guide

The pipeline loads normalized training sequences, adds known diffusion noise, trains a timestep-conditioned noise predictor, then generates new sequences by reversing the noise process. The hybrid uses the same pipeline with a quantum bottleneck. Equations and comparison controls are in [EXPERIMENTAL_METHODOLOGY.md](EXPERIMENTAL_METHODOLOGY.md).

## Configuration and data

| File | Inputs → outputs; main functions |
|---|---|
| config.py | Immutable dataset/model/training/evaluation settings; `validate` checks consistency and `to_dict` serializes them. |
| utils/reproducibility.py | Seed → reproducible Python/NumPy/Torch streams; `environment_metadata` records versions and `save_json` rejects non-finite metadata. |
| data/dataset.py | Config → raw (1400,32) sequences and parameters. `generate_dataset` evaluates the sinusoid equation; `make_splits` and `nested_subsets` select deterministic row indices. |
| data/normalization.py | Selected training entries → scalar mean/std. `transform` computes (x-mu)/sigma; `inverse_transform` restores raw units; `prepare_splits` reuses the training fit. |
| data/artifacts.py | Arrays/config ↔ NPZ/JSON files. `save_bundle` persists data, indices and provenance; `load_bundle` verifies hashes/equation/selections/statistics; `normalized` prepares a chosen condition. |
| data/plotting.py | Dataset → representative series, histogram and ACF PNG/PDF figures. `plot_dataset` uses fixed held-out examples. |
| data/generate_data.py | CLI → saved dataset, summaries and figures; `--verify-only` checks existing artifacts. |
| utils/time_series.py | (N,L) sequences → per-sequence or mean ACF. `sequence_acf` uses row-wise demeaning and a full-window variance denominator; `mean_acf` averages rows. |

Raw arrays remain float64; training converts normalized arrays to float32. Validation/reference rows never fit normalization.

## Models and diffusion

| File | Inputs → outputs; main functions |
|---|---|
| models/classical_denoiser.py | Noisy (B,32) + timestep (B,) → predicted noise (B,32). `Denoiser` separates embedding, encoder, bottleneck and decoder; subclasses provide the baseline or bias-free control. |
| models/quantum_layer.py | Features (B,4) → four local-Z expectations. `build_quantum_circuit` orders inputs, 16 weights and observables; `QuantumLayer` applies pi*Tanh and TorchConnector. `CountingStatevectorEstimator` counts submitted workloads. |
| models/hybrid_denoiser.py | Same inputs/outputs as the baseline; `HybridDenoiser` replaces only its bottleneck with QuantumLayer. |
| models/factory.py | Model choice/seed → initialized model. `make_model` and `copy_shared_state` pair common weights without sharing storage; `shared_state_sha256` identifies those values. |
| diffusion/noise_schedule.py | Config → fixed schedule buffers. `add_noise` implements forward corruption; `reverse_mean` and `reverse_step` implement the DDPM reverse update. |
| diffusion/sampling.py | Model/schedule/seed → generated (N,32) tensor, timing and magnitude diagnostics. `sample` iterates all 20 reverse steps without gradients and adds no noise at the final step. |

The timestep embedding identifies the diffusion noise level, not observation time. Four learned features become bounded circuit angles; the circuit returns expectations rather than sampled bitstrings. TorchConnector supplies a custom backward mapping, letting the loss gradient reach circuit weights and earlier classical layers. Features must remain Torch tensors before the connector.

## Training and checks

| File | Inputs → outputs; main functions |
|---|---|
| training/train.py | Model/data/explicit budget → losses, timings and best checkpoint. `train_model` uses Adam and noise MSE; `validation_draws` fixes corruption; `validation_loss` weights all entries equally; `restore_checkpoint` reloads inference state. The CLI also samples, inverse-normalizes and saves results. |
| training/plotting.py | Recorded losses/generated arrays → model-labeled PNG/PDF figures through `plot_run`. |
| scripts/check_environment.py | Installed environment → import/API, RNG, autograd and rendering checks. |
| scripts/check_quantum_layer.py | Standalone layer → output/gradient/timing report and diagram. `statevector_outputs` is an independent reference; `finite_difference_check` checks derivatives; `optimize_toy` verifies ten optimizer updates. |
| scripts/check_hybrid_model.py | Existing paired pilot records → reproducibility and cost report. `verify_pilots` checks matching settings, joint updates, workloads and exact checkpoint-to-sample regeneration. |
| tests/ | Configuration, data/leakage, DDPM mathematics, quantum gradients and hybrid integration tests. Latest full suite: 58 passed. |

One optimizer updates all classical and quantum parameters. Separate RNG streams keep validation independent of training draws. Per-component gradient norms and parameter changes verify joint optimization; a shared seed alone cannot guarantee paired initial weights, so the factory copies them explicitly.

Imports never launch experiments. Full budgets remain unset, and all training commands require an explicit smoke/step/epoch budget. Current results are short functional pilots; formal quality and scarcity evaluation are pending. See [RESULTS.md](RESULTS.md).
