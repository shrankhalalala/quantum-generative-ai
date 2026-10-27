"""DDPM buffers and exact forward/reverse formulas; code time 0 means k=1."""
import torch
from torch import nn
from config import DEFAULT_CONFIG, DiffusionConfig


class NoiseSchedule(nn.Module):
    def __init__(self, config: DiffusionConfig = DEFAULT_CONFIG.diffusion):
        super().__init__()
        if config.steps <= 0 or not 0 < config.beta_start <= config.beta_end < 1:
            raise ValueError("Invalid beta schedule")
        self.steps = config.steps
        # Compute cumulative products accurately, then use float32 with the MLP.
        beta = torch.linspace(config.beta_start, config.beta_end, config.steps, dtype=torch.float64)
        alpha = 1-beta
        alpha_bar = alpha.cumprod(0)
        alpha_bar_previous = torch.cat((torch.ones(1, dtype=torch.float64), alpha_bar[:-1]))
        variance = beta*(1-alpha_bar_previous)/(1-alpha_bar)
        for name, value in (("beta", beta), ("alpha", alpha), ("alpha_bar", alpha_bar),
                            ("alpha_bar_previous", alpha_bar_previous), ("posterior_variance", variance)):
            self.register_buffer(name, value.float())

    def _validate(self, values: torch.Tensor, timestep: torch.Tensor) -> None:
        if values.ndim != 2 or values.shape[1] != 32 or len(values) == 0:
            raise ValueError("Expected nonempty (B,32) sequence tensors")
        if not values.is_floating_point() or not torch.isfinite(values).all():
            raise ValueError("Expected finite floating-point sequences")
        if timestep.shape != (len(values),) or timestep.dtype != torch.long:
            raise ValueError("Timestep must be int64 with shape (B,)")
        if values.device != self.beta.device or timestep.device != values.device:
            raise ValueError("Sequence, timestep and schedule devices must agree")
        if torch.any(timestep < 0) or torch.any(timestep >= self.steps):
            raise ValueError("Timestep index outside schedule")

    def _at(self, buffer: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        return buffer[timestep].unsqueeze(1)

    def add_noise(self, clean: torch.Tensor, timestep: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        self._validate(clean, timestep)
        if noise.shape != clean.shape or not torch.isfinite(noise).all():
            raise ValueError("Noise must match clean tensor shape and be finite")
        if noise.device != clean.device or noise.dtype != clean.dtype:
            raise ValueError("Noise dtype/device must match clean tensor")
        cumulative = self._at(self.alpha_bar, timestep)
        return cumulative.sqrt()*clean + (1-cumulative).sqrt()*noise

    def reverse_mean(self, noisy: torch.Tensor, timestep: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        self._validate(noisy, timestep)
        if prediction.shape != noisy.shape or prediction.device != noisy.device or prediction.dtype != noisy.dtype or not torch.isfinite(prediction).all():
            raise ValueError("Predicted noise must be finite and match the noisy tensor")
        alpha = self._at(self.alpha, timestep)
        beta = self._at(self.beta, timestep)
        cumulative = self._at(self.alpha_bar, timestep)
        return (noisy-beta*prediction/(1-cumulative).sqrt())/alpha.sqrt()

    def reverse_step(self, noisy: torch.Tensor, timestep: torch.Tensor,
                     prediction: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        if noise.shape != noisy.shape or noise.device != noisy.device or noise.dtype != noisy.dtype or not torch.isfinite(noise).all():
            raise ValueError("Reverse noise must be finite and match the noisy tensor")
        mean = self.reverse_mean(noisy, timestep, prediction)
        variance = self._at(self.posterior_variance, timestep)
        # variance[0] is exactly zero: the final step adds no random noise.
        return mean + variance.sqrt()*noise
