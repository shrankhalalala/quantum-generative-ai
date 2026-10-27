"""Shared timestep-conditioned denoiser with a replaceable 4→4 module."""
import torch
from torch import nn
from config import DEFAULT_CONFIG, ModelConfig


class Denoiser(nn.Module):
    """Shared encoder/decoder; subclasses select only the bottleneck."""
    def __init__(self, bottleneck: nn.Module, config: ModelConfig = DEFAULT_CONFIG.model,
                 diffusion_steps: int = DEFAULT_CONFIG.diffusion.steps):
        super().__init__()
        self.sequence_length = DEFAULT_CONFIG.dataset.sequence_length
        self.diffusion_steps = diffusion_steps
        self.timestep_encoding = nn.Embedding(diffusion_steps, config.timestep_dim)
        self.encoder = nn.Sequential(nn.Linear(self.sequence_length + config.timestep_dim, config.hidden_dim),
                                     nn.SiLU(), nn.Linear(config.hidden_dim, config.bottleneck_dim))
        self.bottleneck = bottleneck
        self.decoder = nn.Sequential(nn.Linear(config.bottleneck_dim, config.hidden_dim), nn.SiLU(),
                                     nn.Linear(config.hidden_dim, self.sequence_length))

    def forward(self, noisy: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        if noisy.ndim != 2 or noisy.shape[1] != self.sequence_length or noisy.shape[0] == 0:
            raise ValueError("Denoiser input must have shape (B,32), B>0")
        if not noisy.is_floating_point() or not torch.isfinite(noisy).all():
            raise ValueError("Denoiser input must be finite floating-point values")
        if timestep.shape != (len(noisy),) or timestep.dtype != torch.long:
            raise ValueError("Timestep must be an int64 vector of shape (B,)")
        if timestep.device != noisy.device or torch.any(timestep < 0) or torch.any(timestep >= self.diffusion_steps):
            raise ValueError("Timestep device/index outside schedule")
        features = self.encoder(torch.cat((noisy, self.timestep_encoding(timestep)), dim=1))
        reduced = self.bottleneck(features)
        if reduced.shape != features.shape or not torch.isfinite(reduced).all():
            raise ValueError("Bottleneck must return finite (B,4) values")
        predicted = self.decoder(reduced)
        if not torch.isfinite(predicted).all():
            raise FloatingPointError("Non-finite predicted noise")
        return predicted


class ClassicalDenoiser(Denoiser):
    def __init__(self, config: ModelConfig = DEFAULT_CONFIG.model,
                 diffusion_steps: int = DEFAULT_CONFIG.diffusion.steps):
        super().__init__(nn.Sequential(nn.Linear(config.bottleneck_dim, config.bottleneck_dim), nn.Tanh()),
                         config, diffusion_steps)


def trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


class ParameterMatchedDenoiser(Denoiser):
    """Optional 16-weight classical bottleneck; count matching is not capacity matching."""
    def __init__(self, config: ModelConfig = DEFAULT_CONFIG.model,
                 diffusion_steps: int = DEFAULT_CONFIG.diffusion.steps):
        super().__init__(nn.Sequential(nn.Linear(config.bottleneck_dim, config.bottleneck_dim, bias=False),
                                      nn.Tanh()), config, diffusion_steps)
