from typing import Dict, List

import numpy as np
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score
)


class NERMetrics:
    """Metrics calculator for NER task."""
    
    def __init__(self, id2label: Dict[int, str]):
        self.id2label = id2label
    
    def compute_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """Compute precision, recall, and F1 score."""
        # Convert predictions and labels to label strings
        pred_labels = self._convert_to_labels(predictions, labels)
        true_labels = self._convert_to_labels(labels, labels)
        
        # Calculate metrics using seqeval
        precision = precision_score(true_labels, pred_labels)
        recall = recall_score(true_labels, pred_labels)
        f1 = f1_score(true_labels, pred_labels)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def _convert_to_labels(
        self,
        predictions: np.ndarray,
        labels: np.ndarray
    ) -> List[List[str]]:
        """Convert numeric predictions to label strings."""
        label_list = []
        
        for pred_seq, label_seq in zip(predictions, labels):
            pred_labels = []
            for pred, label in zip(pred_seq, label_seq):
                if label != -100:  # Ignore padding
                    pred_labels.append(self.id2label[pred])
            label_list.append(pred_labels)
        
        return label_list
    
    def get_entity_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """Get metrics per entity type."""
        pred_labels = self._convert_to_labels(predictions, labels)
        true_labels = self._convert_to_labels(labels, labels)
        
        # Get classification report
        report = classification_report(
            true_labels,
            pred_labels,
            output_dict=True
        )
        
        return report
    
    def compute_confusion_matrix(
        self,
        predictions: np.ndarray,
        labels: np.ndarray
    ) -> np.ndarray:
        """Compute confusion matrix for NER predictions."""
        num_labels = len(self.id2label)
        confusion_matrix = np.zeros((num_labels, num_labels), dtype=np.int64)
        
        # Flatten predictions and labels
        pred_flat = predictions.flatten()
        label_flat = labels.flatten()
        
        # Filter out padding
        mask = label_flat != -100
        pred_flat = pred_flat[mask]
        label_flat = label_flat[mask]
        
        # Build confusion matrix
        for pred, label in zip(pred_flat, label_flat):
            confusion_matrix[label, pred] += 1
        
        return confusion_matrix