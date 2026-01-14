from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from utils.helpers import align_labels_with_tokens



class NERDataset(Dataset):
    """
    Dataset class for NER task.
    """

    def __init__(
            self,
            sentences: List[Tuple[List[str], List[str]]],
            tokenizer: PreTrainedTokenizer,
            label2id: Dict[str, int],
            max_length: int = 256
    ):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    
    def __len__(self) -> int:
        return len(self.sentences)
    

    def __getitem__(
            self, 
            index: int
    ) -> Dict[str, torch.Tensor]:
        tokens, labels = self.sentences[index]

        # Convert labels to IDs
        label_ids = [
            self.label2id[label] for label in labels
        ]

        # Tokenize
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Align labels with tokens
        word_ids = encoding.word_ids(batch_index=0)
        aligned_labels = align_labels_with_tokens(label_ids, word_ids)

        # Convert to tensor
        labels_tensor = torch.tensor(aligned_labels, dtype=torch.long)

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': labels_tensor
        }