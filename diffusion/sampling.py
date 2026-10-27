"""Gaussian-initialized iterative DDPM sampling with explicit diagnostics."""
from dataclasses import dataclass
from time import perf_counter
import torch
from diffusion.noise_schedule import NoiseSchedule

@dataclass
class SamplingResult:
    samples: torch.Tensor
    seconds: float
    step_max_abs: list[float]
    seed: int


@torch.no_grad()
def sample(model: torch.nn.Module, schedule: NoiseSchedule, n_sequences: int, seed: int,
           batch_size: int = 16, initial_noise: torch.Tensor | None = None,
           reverse_noise: torch.Tensor | None = None, max_abs: float = 100.) -> SamplingResult:
    """Code timesteps T-1,...,0; optional noises enable paired model comparisons.

    reverse_noise shape is (T-1,N,32); entry index t-1 belongs to code time t>0.
    Generate all rows' noise per step before chunked inference, so RNG draws do
    not depend on inference batch size or model parameter initialization.
    """
    if n_sequences <= 0 or batch_size <= 0 or not max_abs > 0:
        raise ValueError("Sampling counts and diagnostic threshold must be positive")
    device = schedule.beta.device
    if device.type != "cpu":
        raise ValueError("Initial experiment sampling uses CPU")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    shape = (n_sequences,32)
    start = perf_counter()
    # Consume initial draws even when explicitly supplied, preserving stream alignment.
    random_initial = torch.randn(shape, generator=generator, device=device)
    values = random_initial if initial_noise is None else initial_noise.detach().clone()
    if values.shape != shape or values.device != device or values.dtype != torch.float32 or not torch.isfinite(values).all():
        raise ValueError("Initial noise must be finite CPU float32 with shape (N,32)")
    if reverse_noise is not None and (reverse_noise.shape != (schedule.steps-1,*shape)
            or reverse_noise.device != device or reverse_noise.dtype != values.dtype or not torch.isfinite(reverse_noise).all()):
        raise ValueError("Reverse noise must be finite CPU float32 with shape (T-1,N,32)")
    was_training = model.training
    model.eval()
    diagnostics = []
    try:
        for time in range(schedule.steps-1,-1,-1):
            predictions = []
            for begin in range(0,n_sequences,batch_size):
                chunk = values[begin:begin+batch_size]
                timestep = torch.full((len(chunk),), time, dtype=torch.long, device=device)
                predictions.append(model(chunk, timestep))
            prediction = torch.cat(predictions)
            timestep = torch.full((n_sequences,), time, dtype=torch.long, device=device)
            if time == 0:
                noise = torch.zeros_like(values)
            else:
                random_noise = torch.randn(shape, generator=generator, device=device)
                noise = random_noise if reverse_noise is None else reverse_noise[time-1]
            values = schedule.reverse_step(values, timestep, prediction, noise)
            if not torch.isfinite(values).all():
                raise FloatingPointError(f"Non-finite reverse sample at code time {time}")
            magnitude = float(values.abs().max())
            diagnostics.append(magnitude)
            if magnitude > max_abs:
                raise FloatingPointError(f"Unstable reverse sample at code time {time}: max abs={magnitude}, limit={max_abs}")
    finally:
        model.train(was_training)
    return SamplingResult(values, perf_counter()-start, diagnostics, seed)
