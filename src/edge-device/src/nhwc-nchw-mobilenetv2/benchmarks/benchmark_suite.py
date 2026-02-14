from dataclasses import dataclass, field
from typing import List, Tuple
from benchmarks.benchmark_result import BenchmarkResult
import time
from tensor import TensorLayout, EdgeTensor
import numpy as np
from blocks.inverted_residual import InvertedResidual
from model import EdgeMobileNet


@dataclass
class BenchmarkSuite:
    """
    Comprehesive benchmark suite for layout performance analysis

    Tests:
    1. Basic operations (conv, transpose)
    2. Single block inference
    3. Full model inference
    4. Memory access patterns
    """

    batch_size: int = 1
    input_size: int = 224
    num_warmup: int = 3
    num_runs: int = 10
    results: List[BenchmarkResult] = field(default_factory=list)

    def _timer(self, func, *args, **kwargs) -> Tuple[float, any]:
        """
        Time a function execution

        Returns:
            (elapsed_time_ms, result)
        """
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        elapsed_ms = (end - start) * 1000
        return elapsed_ms, result
    

    def _run_benchmark(
            self,
            operation_name: str,
            operation_func,
            layout: TensorLayout,
            *args,
            **kwargs
    ) -> BenchmarkResult:
        """
        Run benchmark for a single operation
        """

        # Warmup runs
        for _ in range(self.num_warmup):
            _ = operation_func(*args, **kwargs)

        # Timed runs
        times = []
        for _ in range(self.num_runs):
            elapsed_ms, _ = self._timer(operation_func, *args, **kwargs)
            times.append(elapsed_ms)

        times = np.array(times)

        # Calculate statistics
        mean_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        

        # Calculate throughput (images/sec)
        throughput_fps = 1000.0 / mean_time if mean_time > 0 else 0

        # Estimate memory footprint (simplified)
        memory_mb = 0.0 # will be updated by specific benchmarks

        result = BenchmarkResult(
            layout=layout.value,
            operation=operation_name,
            mean_time_ms=mean_time,
            std_time_ms=std_time,
            min_time_ms=min_time,
            max_time_ms=max_time,
            throughput_fps=throughput_fps,
            memory_footprint_mb=memory_mb,
            num_runs=self.num_runs
        )

        self.results.append(result)
        return result
    

    def benchmark_layout_conversion(self):
        """
        Benchmark NCHW <-> NHWC conversion overhead
        Critical for understanding mixed-layout performance
        """

        print("\n=== Layout Conversion Benchmark ===")

        # Create test tensor in NCHW
        nchw_data = np.random.randn(
            self.batch_size, 3, self.input_size, self.input_size
        ).astype(np.float32)

        tensor_nchw = EdgeTensor(nchw_data, layout=TensorLayout.NCHW)

        # Benchmark NCHW -> NHWC
        def nchw_to_nhwc():
            return tensor_nchw.to_layout(TensorLayout.NHWC)
        
        result = self._run_benchmark(
            "NCHW -> NHWC conversion",
            nchw_to_nhwc,
            TensorLayout.NCHW
        )
        print(result)

        # Create test tensor in NHWC
        nhwc_data = np.random.randn(
            self.batch_size, self.input_size, self.input_size, 3
        )

        tensor_nhwc = EdgeTensor(nhwc_data, layout=TensorLayout.NHWC)

        # Benchmark NHWC -> NCHW
        def nhwc_to_nchw():
            return tensor_nchw.to_layout(TensorLayout.NCHW)
        
        result = self._run_benchmark(
            "NHWC -> NCHW conversion",
            nhwc_to_nchw,
            TensorLayout.NHWC
        )

        print(result)


    def benchmark_inverted_residual_block(self):
        """
        Benchmark Single inverted residual block
        Core component of MobileNet - representative of real workload
        """

        print("\n=== Inverted Residual Block Benchmark ===")

        # Test configuration
        in_channels = 32
        out_channels = 64
        spatial_size = 56 # Typical intermediate feature map size

        # NHWC block
        block_nhwc = InvertedResidual(
            in_channels,
            out_channels,
            stride=1,
            expand_ratio=6,
            layout=TensorLayout.NHWC
        )

        # Create NHWC input
        input_nhwc = EdgeTensor(
            np.random.randn(self.batch_size, spatial_size, spatial_size, in_channels).astype(np.float32),
            TensorLayout.NHWC
        )

        # Benchmark NHWC
        result_nhwc = self._run_benchmark(
            "InvertedResidual (56x56, C=32 -> 64)",
            block_nhwc.forward,
            TensorLayout.NHWC,
            input_nhwc
        )
        print(result_nhwc)

        # NCHW block
        block_nchw = InvertedResidual(
            in_channels,
            out_channels,
            stride=1,
            expand_ratio=6,
            layout=TensorLayout.NCHW
        )

        # Create NCHW input
        input_nchw = EdgeTensor(
            np.random.randn(self.batch_size, in_channels, spatial_size, spatial_size).astype(np.float32),
            layout=TensorLayout.NCHW
        )

        # Benchmark NCHW
        result_nchw = self._run_benchmark(
            "InvertedResidual (56x56, C=32 -> 64)",
            block_nchw,
            TensorLayout.NCHW,
            input_nchw
        )
        print(result_nchw)


        # Calculate speedup
        speedup = result_nchw.speedup_vs(result_nhwc)
        print(f"\nNHWC speedup vs NCHW: {speedup:.2f}x")


    
    def benchmark_full_model(self):
        """
        Benchmark full mobilenet inference
        End-to-end performance - most importance metric
        """

        print("\n=== Full Model Inference Benchmark ===")

        # Reduced model for faster benchmarking
        width_mult = 0.5

        # NHWC model
        model_nhwc = EdgeMobileNet(
            num_classes=1000,
            width_multiplier=width_mult,
            layout=TensorLayout.NHWC,
            input_size=self.input_size
        )

        # Create NHWC input
        input_nhwc = EdgeTensor(
            np.random.randn(self.batch_size, self.input_size, self.input_size, 3).astype(np.float32),
            TensorLayout.NHWC
        )

        # Benchmark NHWC
        result_nhwc = self._run_benchmark(
            f"MobileNet-{width_mult} Full Forward",
            model_nhwc.forward,
            TensorLayout.NHWC,
            input_nhwc
        )
        print(result_nhwc)

        # NCHW model
        model_nchw = EdgeMobileNet(
            num_classes=1000,
            width_multiplier=width_mult,
            layout=TensorLayout.NCHW,
            input_size=self.input_size
        )

        # Create NCHW input
        input_nchw = EdgeTensor(
            np.random.randn(self.batch_size, 3, self.input_size, self.input_size).astype(np.float32),
            TensorLayout.NCHW
        )

        # Benchmark NCHW
        result_nchw = self._run_benchmark(
            f"MobileNet-{width_mult} Full Forward",
            model_nchw.forward,
            TensorLayout.NCHW,
            input_nchw
        )
        print(result_nchw)

        # Calculate speedup
        speedup = result_nchw.speedup_vs(result_nhwc)
        print(f"\nNHWC speedup vs NCHW: {speedup:.2f}x")
        
        # Print model info
        print(f"\nModel info:")
        print(f"  Parameters: {model_nhwc.count_parameters():,}")



    def benchmark_memory_access_patterns(self):
        """
        Benchmark memory access patterns.
        Demonstrates cache behavior differences
        """

        print("\n=== Memory Access Pattern Benchmark ===")

        # Small feature maps to fit in L1/L2 cache
        small_size = 32
        channels = 64

        # NHWC: Spatial locality
        nhwc_data = np.random.randn(
            self.batch_size, small_size, small_size, channels
        ).astype(np.float32)

        tensor_nhwc = EdgeTensor(nhwc_data, TensorLayout.NHWC)

        def spatial_scan_nhwc():
            """
            Scan by spatial - good for NHWC
            """
            result = 0.0
            for h in range(small_size):
                for w in range(small_size):
                    # access all channels (h,w) - contigous in NHWC
                    result += np.sum(tensor_nhwc.data[0, h, w, :])

            return result
        
        result = self._run_benchmark(
            "Spatial scan (cache - friendly)",
            spatial_scan_nhwc,
            TensorLayout.NHWC
        )
        print(result)

        # NCHW: Channel locality
        nchw_data = np.random.randn(
            self.batch_size, channels, small_size, small_size
        ).astype(np.float32)

        tensor_nchw = EdgeTensor(nchw_data, TensorLayout.NCHW)

        def channel_scan_nchw():
            """
            Scan by channel - good for NCHW
            """
            result = 0.0
            for c in range(channels):
                # Access entire spatial extent of channel - contiguous in NCHW
                result += np.sum(tensor_nchw.data[0, c, :, :])
            return result
        

        result = self._run_benchmark(
            "Channel scan (cache - friendly)",
            channel_scan_nchw,
            TensorLayout.NCHW
        )
        print(result)


    def run_all(self):
        """Run complete benchmark suite."""
        print("=" * 80)
        print("Edge AI Layout Performance Benchmark")
        print("=" * 80)
        print(f"Configuration:")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Input size: {self.input_size}x{self.input_size}")
        print(f"  Warmup runs: {self.num_warmup}")
        print(f"  Benchmark runs: {self.num_runs}")
        
        # Run all benchmarks
        self.benchmark_layout_conversion()
        self.benchmark_inverted_residual_block()
        self.benchmark_full_model()
        self.benchmark_memory_access_patterns()
        
        # Summary
        self._print_summary()
    
    def _print_summary(self):
        """Print benchmark summary and analysis."""
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        
        # Group results by operation type
        nhwc_results = [r for r in self.results if r.layout == "NHWC"]
        nchw_results = [r for r in self.results if r.layout == "NCHW"]
        
        print(f"\nNHWC Results ({len(nhwc_results)} operations):")
        for r in nhwc_results:
            print(f"  {r}")
        
        print(f"\nNCHW Results ({len(nchw_results)} operations):")
        for r in nchw_results:
            print(f"  {r}")
        
        print("\n" + "=" * 80)


    def export_results(self, filepath: str):
        """Export results to CSV."""
        import csv
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Layout', 'Operation', 'Mean (ms)', 'Std (ms)',
                'Min (ms)', 'Max (ms)', 'Throughput (FPS)', 'Runs'
            ])
            
            for r in self.results:
                writer.writerow([
                    r.layout, r.operation, r.mean_time_ms, r.std_time_ms,
                    r.min_time_ms, r.max_time_ms, r.throughput_fps, r.num_runs
                ])
        
        print(f"\nResults exported to: {filepath}")
