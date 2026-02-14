from dataclasses import dataclass, field



@dataclass
class BenchmarkResult:
    layout: str
    operation: str
    mean_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float
    throughput_fps: float
    memory_footprint_mb: float
    num_runs: int

    def speedup_vs(self, other: 'BenchmarkResult') -> float:
        """Calculate speedup compared to another result."""
        if other.mean_time_ms == 0:
            return float('inf')
        return other.mean_time_ms / self.mean_time_ms

    def __repr__(self) -> str:
        return (
            f"{self.layout:6s} | {self.operation:30s} | "
            f"{self.mean_time_ms:8.2f}ms ± {self.std_time_ms:6.2f}ms | "
            f"{self.throughput_fps:6.1f} FPS"
        )