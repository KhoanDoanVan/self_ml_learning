from typing import Dict

class MetricsTracker:
    """Track metrics across training steps."""
    
    def __init__(self):
        self.history = {
            'train_loss': [],
            'eval_loss': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
    
    def update(self, metrics: Dict[str, float], step: int) -> None:
        """Update metrics history."""
        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append((step, value))
    
    def get_best_f1(self) -> float:
        """Get best F1 score."""
        if not self.history['f1']:
            return 0.0
        return max(score for _, score in self.history['f1'])
    
    def get_latest_metrics(self) -> Dict[str, float]:
        """Get latest metrics."""
        latest = {}
        for key, values in self.history.items():
            if values:
                latest[key] = values[-1][1]
        return latest