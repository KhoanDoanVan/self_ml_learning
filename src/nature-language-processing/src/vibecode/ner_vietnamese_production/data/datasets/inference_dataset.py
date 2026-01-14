from transformers import PreTrainedTokenizer
from torch.utils.data import Dataset
from typing import Dict, List
import torch


class InferenceDataset(Dataset):

    def __init__(
            self,
            texts: List[str],
            tokenizer: PreTrainedTokenizer,
            max_length: int = 256
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length


    def __len__(self) -> int:
        return len(self.texts)
    

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:

        text = self.texts[index]

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }