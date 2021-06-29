import dataclasses
from typing import Optional, List


@dataclasses.dataclass(frozen=True)
class AugmentationConfig:
    train: Optional[str]
    val: Optional[str]


@dataclasses.dataclass(frozen=True)
class LrSchedulerConfig:
    name: str
    base_lr: float
    step_size: Optional[int]
    step_gamma: Optional[float]
    warmup: Optional[str]
    warmup_epochs: Optional[int]
    warmup_factor: Optional[float] = 0.01


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    name: str
    input_size: int
    options: List[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(frozen=True)
class OptimizerConfig:
    name: str = 'sgd'
    momentum: float = 0.9
    weight_decay: float = 1e-5


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    """Used by mitorch-agent to prepare a training environment."""
    train: str
    val: Optional[str]


@dataclasses.dataclass(frozen=True)
class TrainingConfig:
    task_type: str
    batch_size: int
    max_epochs: int
    model: ModelConfig = None
    augmentation: AugmentationConfig = None
    lr_scheduler: LrSchedulerConfig = None
    optimizer: OptimizerConfig = None
    dataset: Optional[DatasetConfig] = None
    num_processes: int = -1
    accumulate_grad_batches: int = 1
