"""Central scientific specification; importing this file starts no work."""
from dataclasses import asdict, dataclass, field

PAPER_TITLE = "Quantum Generative AI: A Survey of Hybrid Architectures and Emerging Applications"

@dataclass(frozen=True)
class DatasetConfig:
    n_sequences: int = 1400
    sequence_length: int = 32
    split_sizes: tuple[int, int, int] = (1000, 200, 200)
    subset_sizes: tuple[int, ...] = (1000, 500, 250, 100)
    generation_seed: int = 2026
    split_seed: int = 2027
    amplitude_range: tuple[float, float] = (0.6, 1.2)
    frequency_range: tuple[float, float] = (1.0, 3.0)
    harmonic_range: tuple[float, float] = (0.0, 0.3)
    noise_std: float = 0.05
    normalization: str = "per_subset_global_zscore"

@dataclass(frozen=True)
class ModelConfig:
    hidden_dim: int = 16
    timestep_dim: int = 16
    bottleneck_dim: int = 4
    n_qubits: int = 4
    quantum_layers: int = 2
    rotations: tuple[str, ...] = ("RY", "RX")
    entanglement_edges: tuple[tuple[int, int], ...] = ((0, 1), (1, 2), (2, 3))
    @property
    def quantum_weight_count(self) -> int:
        return self.n_qubits * self.quantum_layers * len(self.rotations)

@dataclass(frozen=True)
class DiffusionConfig:
    steps: int = 20
    beta_start: float = 0.0001
    beta_end: float = 0.02

@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 0.001
    batch_size: int = 16
    # Full training budget is deliberately unset until a measured pilot.
    epochs: int | None = None
    max_steps: int | None = None
    pilot_steps: int = 5
    seeds: tuple[int, ...] = (11, 22, 33)
    device: str = "cpu"

@dataclass(frozen=True)
class EvaluationConfig:
    generated_sequences: int = 200
    max_acf_lag: int = 10
    ddof: int = 0
    validation_seed: int = 2028
    sampling_seed_offset: int = 10000

@dataclass(frozen=True)
class ExperimentConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def validate(self) -> None:
        d, m, q, t, e = self.dataset, self.model, self.diffusion, self.training, self.evaluation
        if sum(d.split_sizes) != d.n_sequences or any(n <= 0 for n in d.split_sizes):
            raise ValueError("Split sizes must be positive and sum to dataset size")
        if not d.subset_sizes or tuple(sorted(set(d.subset_sizes), reverse=True)) != d.subset_sizes:
            raise ValueError("Subset sizes must be distinct, positive, descending")
        if min(d.subset_sizes) <= 0 or max(d.subset_sizes) > d.split_sizes[0]:
            raise ValueError("Subsets must fit the training pool")
        if d.sequence_length != 32 or m.bottleneck_dim != 4 or m.n_qubits != 4:
            raise ValueError("Approved sequence/bottleneck/qubit dimensions are 32/4/4")
        if m.hidden_dim != 16 or m.timestep_dim <= 0 or m.quantum_layers != 2:
            raise ValueError("Invalid approved hidden dimension or circuit depth")
        if m.rotations != ("RY", "RX") or m.entanglement_edges != ((0, 1), (1, 2), (2, 3)):
            raise ValueError("Circuit design differs from the documented specification")
        for low, high in (d.amplitude_range, d.frequency_range, d.harmonic_range):
            if not low < high:
                raise ValueError("Parameter range must increase")
        if not d.noise_std > 0 or d.normalization != "per_subset_global_zscore":
            raise ValueError("Invalid noise or normalization policy")
        if q.steps != 20 or not 0 < q.beta_start < q.beta_end < 1:
            raise ValueError("Invalid diffusion schedule")
        if not t.learning_rate > 0 or t.batch_size <= 0 or t.pilot_steps <= 0:
            raise ValueError("Training settings must be positive")
        if any(budget is not None and budget <= 0 for budget in (t.epochs, t.max_steps)):
            raise ValueError("Explicit training budgets must be positive")
        if not t.seeds or len(set(t.seeds)) != len(t.seeds):
            raise ValueError("Repetition seeds must be nonempty and unique")
        if t.device != "cpu":
            raise ValueError("Initial shared simulator experiments use CPU")
        if e.generated_sequences < 200 or not 1 <= e.max_acf_lag < d.sequence_length or e.ddof != 0:
            raise ValueError("Invalid evaluation settings")

    def to_dict(self) -> dict:
        self.validate()
        return asdict(self)

DEFAULT_CONFIG = ExperimentConfig()
