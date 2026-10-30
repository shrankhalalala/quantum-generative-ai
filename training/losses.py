"""Differentiable clean-signal reconstruction and per-sequence ACF guidance."""
import torch
from config import LossConfig
from diffusion.noise_schedule import NoiseSchedule


def reconstruct_clean(noisy: torch.Tensor, prediction: torch.Tensor,
                      timestep: torch.Tensor, schedule: NoiseSchedule) -> torch.Tensor:
    schedule._validate(noisy, timestep)
    if (prediction.shape != noisy.shape or prediction.dtype != noisy.dtype
            or prediction.device != noisy.device or not torch.isfinite(prediction).all()):
        raise ValueError("Noise prediction must be finite and match noisy sequences")
    cumulative = schedule.alpha_bar[timestep, None]
    return (noisy - (1 - cumulative).sqrt() * prediction) / cumulative.sqrt()


def sequence_acf(values: torch.Tensor, max_lag: int = 10,
                 epsilon: float = 1e-12) -> torch.Tensor:
    """Lags 1..K; row demeaning, full-window denominator, no lag correction.

    Constant rows return zero with a clamped denominator. Evaluation instead
    flags undefined rows explicitly. Nothing is detached from the Torch graph.
    """
    if (values.ndim != 2 or 0 in values.shape or not values.is_floating_point()
            or not torch.isfinite(values).all()):
        raise ValueError("Expected nonempty finite floating-point sequences")
    if not isinstance(max_lag, int) or not 1 <= max_lag < values.shape[1]:
        raise ValueError("ACF lag must be an integer in [1,L)")
    if not 0 < epsilon < float('inf'):
        raise ValueError("ACF epsilon must be finite and positive")
    centered = values - values.mean(dim=1, keepdim=True)
    denominator = centered.square().sum(dim=1).clamp_min(epsilon)
    return torch.stack([(centered[:, :-lag] * centered[:, lag:]).sum(dim=1) / denominator
                        for lag in range(1, max_lag + 1)], dim=1)


def autocorrelation_loss(predicted: torch.Tensor, target: torch.Tensor,
                         max_lag: int = 10, epsilon: float = 1e-12) -> torch.Tensor:
    if predicted.shape != target.shape or predicted.device != target.device or predicted.dtype != target.dtype:
        raise ValueError("ACF prediction and target must have identical shape/device/dtype")
    return (sequence_acf(predicted, max_lag, epsilon)
            - sequence_acf(target, max_lag, epsilon)).square().mean()


def diffusion_losses(prediction: torch.Tensor, target_noise: torch.Tensor,
                     noisy: torch.Tensor, clean: torch.Tensor, timestep: torch.Tensor,
                     schedule: NoiseSchedule, config: LossConfig) -> dict[str, torch.Tensor]:
    noise = (prediction - target_noise).square().mean()
    estimated_clean = reconstruct_clean(noisy, prediction, timestep, schedule)
    acf = autocorrelation_loss(estimated_clean, clean, config.max_lag, config.epsilon)
    # Preserve the original objective and exact gradients when guidance is off.
    total = noise if config.objective == "noise" or config.lambda_acf == 0 else noise + config.lambda_acf * acf
    return {"noise_loss": noise, "acf_loss": acf, "total_loss": total}
