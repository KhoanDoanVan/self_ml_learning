import yaml
from pathlib import Path
import os
from schema.config import (
    config as cf,
    data_config,
    model_config,
    training_config
)
from typing import Optional


def load_yaml_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)
    


def get_config(config_path: Optional[str] = None) -> cf:
    """Get configuration object."""
    if config_path is None:
        config_path = os.path.join(
            Path(__file__).parent,
            'model_config.yaml'
        )
    
    yaml_config = load_yaml_config(config_path)
    
    # Create nested config objects
    model_cfg = model_config.ModelConfig(**yaml_config['model'])
    training_cfg = training_config.TrainingConfig(**yaml_config['training'])
    data_cfg = data_config(**yaml_config['data'])
    
    # Create main config
    config = cf.Config(
        model=model_cfg,
        training=training_cfg,
        data=data_cfg,
        seed=yaml_config.get('seed', 42),
        output_dir=yaml_config.get('output_dir', './outputs'),
        log_dir=yaml_config.get('log_dir', './logs'),
        cache_dir=yaml_config.get('cache_dir', './cache'),
        use_cuda=yaml_config.get('use_cuda', True)
    )
    
    # Create directories if they don't exist
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.log_dir).mkdir(parents=True, exist_ok=True)
    Path(config.cache_dir).mkdir(parents=True, exist_ok=True)
    
    return config