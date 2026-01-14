from .base import BaseNERModel
from typing import Dict, Optional
import torch
import torch.nn as nn
from transformers import AutoModel


class PhoBERTForNER(BaseNERModel):
    """
    PhoBERT model for Named Entity Recognition
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        dropout: float = 0.1,
        hidden_size: int = 768
    ):
        super.__init__(num_labels)
        
        # Load PhoBERT encoder
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)

        # Classification head
        self.classifier = nn.Linear(hidden_size, num_labels)

        # Initialize weights
        self._init_weights()


    def _init_weights(self) -> None:
        """
        Initialize classifier weights
        """
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)


    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Encode input
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)

        # Classify tokens
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                logits.view(-1, self.num_labels),
                labels.view(-1)
            )

        return {
            'loss': loss,
            'logits': logits
        }
    

    def predict(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Predict Labels
        """
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs['logits']
            predictions = torch.argmax(logits, dim=-1)

        return predictions