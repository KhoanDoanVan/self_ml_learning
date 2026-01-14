"""Helper utilities for Vietnamese NER project."""

import random
import re
from typing import List, Tuple

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_vietnamese_text(text: str) -> str:
    """Normalize Vietnamese text."""
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing whitespaces
    text = text.strip()
    return text


def split_vietnamese_sentences(text: str) -> List[str]:
    """Split Vietnamese text into sentences."""
    # Vietnamese sentence delimiters
    delimiters = r'[.!?;]+'
    sentences = re.split(delimiters, text)
    return [s.strip() for s in sentences if s.strip()]


def align_labels_with_tokens(
    labels: List[str],
    word_ids: List[int]
) -> List[int]:
    """Align labels with tokenized words."""
    aligned_labels = []
    previous_word_id = None
    
    for word_id in word_ids:
        if word_id is None:
            # Special token
            aligned_labels.append(-100)
        elif word_id != previous_word_id:
            # First token of a word
            aligned_labels.append(labels[word_id])
        else:
            # Subsequent token of a word
            label = labels[word_id]
            if label % 2 == 1:  # B- tag
                aligned_labels.append(label + 1)  # Convert to I- tag
            else:
                aligned_labels.append(label)
        
        previous_word_id = word_id
    
    return aligned_labels


def bio_to_entities(
    tokens: List[str],
    labels: List[str]
) -> List[Tuple[str, str, int, int]]:
    """Convert BIO tags to entity spans."""
    entities = []
    current_entity = None
    start_idx = 0
    
    for idx, (token, label) in enumerate(zip(tokens, labels)):
        if label.startswith('B-'):
            # Save previous entity
            if current_entity:
                entities.append(current_entity)
            # Start new entity
            entity_type = label[2:]
            current_entity = (token, entity_type, idx, idx + 1)
            start_idx = idx
        elif label.startswith('I-') and current_entity:
            # Continue current entity
            entity_type = label[2:]
            if entity_type == current_entity[1]:
                text = current_entity[0] + ' ' + token
                current_entity = (text, entity_type, start_idx, idx + 1)
        else:
            # O tag - end current entity
            if current_entity:
                entities.append(current_entity)
                current_entity = None
    
    # Add last entity
    if current_entity:
        entities.append(current_entity)
    
    return entities


def calculate_f1_score(
    precision: float,
    recall: float
) -> float:
    """Calculate F1 score from precision and recall."""
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)