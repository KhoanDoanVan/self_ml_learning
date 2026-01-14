"""Inference predictor for Vietnamese NER."""

from typing import Dict, List, Tuple

import torch
from transformers import AutoTokenizer

from models.base import BaseNERModel
from utils.helpers import bio_to_entities


class NERPredictor:
    """Predictor for NER inference."""
    
    def __init__(
        self,
        model: BaseNERModel,
        tokenizer: AutoTokenizer,
        id2label: Dict[int, str],
        device: str = 'cuda',
        max_length: int = 256
    ):
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.id2label = id2label
        self.device = device
        self.max_length = max_length
    
    def predict_text(
        self,
        text: str
    ) -> List[Tuple[str, str, int, int]]:
        """Predict entities in text."""
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_offsets_mapping=True
        )
        
        # Move to device
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        offset_mapping = encoding['offset_mapping'][0]
        
        # Predict
        with torch.no_grad():
            predictions = self.model.predict(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        # Convert to labels
        predictions = predictions[0].cpu().numpy()
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        # Filter special tokens
        pred_labels = []
        pred_tokens = []
        for token, pred, (start, end) in zip(
            tokens, predictions, offset_mapping
        ):
            if token not in [self.tokenizer.pad_token, 
                           self.tokenizer.cls_token,
                           self.tokenizer.sep_token]:
                pred_labels.append(self.id2label[pred])
                pred_tokens.append(token)
        
        # Extract entities
        entities = bio_to_entities(pred_tokens, pred_labels)
        
        return entities
    
    def predict_batch(
        self,
        texts: List[str],
        batch_size: int = 16
    ) -> List[List[Tuple[str, str, int, int]]]:
        """Predict entities for multiple texts."""
        all_entities = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            encodings = self.tokenizer(
                batch_texts,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Move to device
            input_ids = encodings['input_ids'].to(self.device)
            attention_mask = encodings['attention_mask'].to(self.device)
            
            # Predict
            with torch.no_grad():
                predictions = self.model.predict(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            
            # Process each text in batch
            for j, text in enumerate(batch_texts):
                entities = self._extract_entities_from_predictions(
                    text,
                    predictions[j].cpu().numpy(),
                    input_ids[j].cpu().numpy()
                )
                all_entities.append(entities)
        
        return all_entities
    
    def _extract_entities_from_predictions(
        self,
        text: str,
        predictions: List[int],
        input_ids: List[int]
    ) -> List[Tuple[str, str, int, int]]:
        """Extract entities from predictions."""
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        
        # Filter predictions
        pred_labels = []
        pred_tokens = []
        for token, pred in zip(tokens, predictions):
            if token not in [self.tokenizer.pad_token,
                           self.tokenizer.cls_token,
                           self.tokenizer.sep_token]:
                pred_labels.append(self.id2label[pred])
                pred_tokens.append(token)
        
        # Extract entities
        entities = bio_to_entities(pred_tokens, pred_labels)
        
        return entities
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        model_class: type,
        tokenizer_name: str,
        id2label: Dict[int, str],
        device: str = 'cuda'
    ) -> 'NERPredictor':
        """Load predictor from pretrained model."""
        # Load model
        model = model_class(
            model_name=tokenizer_name,
            num_labels=len(id2label)
        )
        
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        return cls(
            model=model,
            tokenizer=tokenizer,
            id2label=id2label,
            device=device
        )