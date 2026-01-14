"""Base model class for Vietnamese NER."""

from abc import ABC, abstractmethod
from typing import Dict, Optional

import torch
import torch.nn as nn


class BaseNERModel(ABC, nn.Module):
    """Abstract base class for NER models."""
    
    def __init__(self, num_labels: int):
        super().__init__()
        self.num_labels = num_labels
    
    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        pass
    
    @abstractmethod
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Predict labels for input."""
        pass
    
    def save_pretrained(self, save_path: str) -> None:
        """Save model weights."""
        torch.save(self.state_dict(), save_path)
    
    def load_pretrained(self, load_path: str) -> None:
        """Load model weights."""
        state_dict = torch.load(load_path, map_location='cpu')
        self.load_state_dict(state_dict)
    
    def freeze_encoder(self) -> None:
        """Freeze encoder layers for transfer learning."""
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self) -> None:
        """Unfreeze encoder layers."""
        for param in self.encoder.parameters():
            param.requires_grad = True
    
    def get_num_parameters(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_trainable_parameters(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)