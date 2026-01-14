from dataclasses import dataclass


@dataclass
class TrainingConfig:
    batch_size: int
    num_epochs: int
    gradient_acculumation_steps: int
    warmup_steps: int
    weight_decay: float
    save_steps: int
    eval_steps: int
    logging_steps: int