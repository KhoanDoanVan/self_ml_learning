from dataclasses import dataclass




@dataclass
class ModelConfig:
    name: str
    pretrained_model: str
    hidden_size: str
    num_labels: int
    dropout: float
    learning_rate: float
    max_seq_length: int