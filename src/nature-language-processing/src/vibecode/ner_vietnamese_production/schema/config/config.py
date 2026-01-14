from dataclasses import dataclass
from model_config import ModelConfig
from data_config import DataConfig
from training_config import TrainingConfig


@dataclass
class Config:
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    seed: int
    output_dir: str
    log_dir: str
    cache_dir: str
    use_cuda: bool