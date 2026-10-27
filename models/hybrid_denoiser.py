"""Hybrid denoiser: only the baseline's four-dimensional bottleneck changes."""
from config import DEFAULT_CONFIG, ModelConfig
from models.classical_denoiser import Denoiser
from models.quantum_layer import QuantumLayer


class HybridDenoiser(Denoiser):
    def __init__(self, config: ModelConfig = DEFAULT_CONFIG.model,
                 diffusion_steps: int = DEFAULT_CONFIG.diffusion.steps, seed: int = 11):
        super().__init__(QuantumLayer(seed=seed, config=config), config, diffusion_steps)
