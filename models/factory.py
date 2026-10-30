"""Paired initial states across model types, independent of constructor RNG use."""
import hashlib
import torch
from config import DEFAULT_CONFIG, ExperimentConfig
from models.classical_denoiser import ClassicalDenoiser, ParameterMatchedDenoiser

MODEL_TYPES = ("classical", "hybrid", "hybrid_acf", "classical_matched")
SHARED_COMPONENTS = ("timestep_encoding", "encoder", "decoder")


def copy_shared_state(source: torch.nn.Module, target: torch.nn.Module) -> None:
    """Copy values without sharing parameter storage or touching bottlenecks."""
    for name in SHARED_COMPONENTS:
        getattr(target, name).load_state_dict(getattr(source, name).state_dict())


def shared_state_sha256(model: torch.nn.Module) -> str:
    """Identify initial embedding/encoder/decoder tensor values and names."""
    digest = hashlib.sha256()
    for component in SHARED_COMPONENTS:
        for name, value in sorted(getattr(model, component).state_dict().items()):
            tensor = value.detach().cpu().contiguous()
            digest.update(f"{component}.{name}:{tensor.dtype}:{tuple(tensor.shape)}".encode())
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def model_state_sha256(model: torch.nn.Module) -> str:
    """Identify all initial parameters, including the quantum bottleneck."""
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(f"{name}:{tensor.dtype}:{tuple(tensor.shape)}".encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def make_model(model_type: str, seed: int = 11, config: ExperimentConfig = DEFAULT_CONFIG):
    """Reference baseline supplies identical shared initial states in every arm."""
    config.validate()
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unknown model type: {model_type}")
    if not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("Model seed must be an integer in [0,2**32)")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        reference = ClassicalDenoiser(config.model, config.diffusion.steps)
        if model_type == "classical":
            return reference
        if model_type in ("hybrid", "hybrid_acf"):
            # Keep classical-only imports/usage independent of the Qiskit stack.
            from models.hybrid_denoiser import HybridDenoiser
            model = HybridDenoiser(config.model, config.diffusion.steps, seed=seed)
        else:
            model = ParameterMatchedDenoiser(config.model, config.diffusion.steps)
            with torch.no_grad():
                model.bottleneck[0].weight.copy_(reference.bottleneck[0].weight)
        copy_shared_state(reference, model)
        return model
