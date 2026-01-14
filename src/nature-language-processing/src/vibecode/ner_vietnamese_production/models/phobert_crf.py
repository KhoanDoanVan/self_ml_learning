from .base import BaseNERModel
import torch.nn as nn
from transformers import AutoModel
import torch
from typing import Optional, Dict, List


class PhoBERTWithCRF(BaseNERModel):
    """
    PhoBERT with CRF Layer for better sequence tagging
    """

    def __init__(
            self,
            model_name: str,
            num_labels: int,
            dropout: float = 0.1,
            hidden_size: int = 768
    ):
        super().__init__(num_labels)

        # PhoBERT encoder
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)

        # Hidden layer
        self.hidden = nn.Linear(hidden_size, hidden_size // 2)
        self.activation = nn.ReLU()

        # Emission scores
        self.emission = nn.Linear(hidden_size // 2, num_labels)

        # CRF transition parameters
        self.transitions = nn.Parameter(
            torch.rand(num_labels, num_labels)
        )

        self._init_weights()


    def _init_weights(self) -> None:
        """Initialize weights."""
        nn.init.xavier_uniform_(self.hidden.weight)
        nn.init.xavier_uniform_(self.emission.weight)
        nn.init.zeros_(self.hidden.bias)
        nn.init.zeros_(self.emission.bias)


    def _compute_emissions(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute emission scores"""

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)

        hidden = self.activation(self.hidden(sequence_output))
        emissions = self.emission(hidden)

        return emissions
    

    def _viterbi_decode(
            self,
            emissions: torch.Tensor,
            mask: torch.Tensor
    ) -> torch.Tensor:
        """Viterbi decoding for CRF"""
        batch_size, seq_len, num_labels = emissions.shape

        # Initialize
        score = emissions[:, 0]
        history = []

        # Forward pass
        for i in range(1, seq_len):
            broadcast_score = score.unsqueeze(2)
            broadcast_emission = emissions[:, i].unsqueeze(1)
            next_score = broadcast_score + self.transitions + broadcast_emission

            next_score, indices = next_score.max(dim=1)
            score = torch.where(
                mask[:, i].unsqueeze(1),
                next_score,
                score
            )
            history.append(indices)

        # Backward pass
        best_tags = [score.argmax(dim=1)]
        for idx in reversed(history):
            best_tags.append(idx.gather(1, best_tags[-1].unsqueeze(1)).squeeze(1))
        
        best_tags.reverse()
        return torch.stack(best_tags, dim=1)
    

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with CRF."""
        emissions = self._compute_emissions(input_ids, attention_mask)
        
        loss = None
        if labels is not None:
            # CRF loss computation (simplified)
            loss = self._crf_loss(emissions, labels, attention_mask)
        
        return {
            'loss': loss,
            'emissions': emissions
        }
    

    def _crf_loss(
        self,
        emissions: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute CRF loss (simplified version)."""
        # This is a simplified version
        # Full CRF implementation would include forward algorithm
        batch_size, seq_len, _ = emissions.shape
        
        score = torch.gather(
            emissions,
            2,
            labels.unsqueeze(2)
        ).squeeze(2)
        
        score = score * mask.float()
        loss = -score.sum() / mask.float().sum()
        
        return loss
    

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Predict with Viterbi decoding."""
        with torch.no_grad():
            emissions = self._compute_emissions(input_ids, attention_mask)
            predictions = self._viterbi_decode(emissions, attention_mask)
        
        return predictions