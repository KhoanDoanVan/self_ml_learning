from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class DataConfig:
    train_path: str
    valid_path: str
    test_path: str
    label_list: List[str]
    max_samples: Optional[int] = None