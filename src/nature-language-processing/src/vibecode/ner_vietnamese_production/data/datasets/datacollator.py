import torch
from typing import Dict, List


class DataCollator:
    """
    Custom data collator for NER
    """

    def __init__(
            self,
            pad_token_id: int = 0,
            label_pad_id: int = -100
    ):
        self.pad_token_id = pad_token_id
        self.label_pad_id = label_pad_id

    
    def __call__(
            self,
            features: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        # Stack all features
        batch = {
            'input_ids': torch.stack(
                [f['input_ids'] for f in features]
            )
        }

        # Add labels if present
        if 'labels' in features[0]:
            batch['labels'] = torch.stack(
                [f['labels'] for f in features]
            )

        return batch