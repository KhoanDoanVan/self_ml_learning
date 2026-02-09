## Level 0: Pre-edge
1. ML/DL fundamentals
- Regression / Classification
- CNN basics
- Loss, backprop, overfitting
2. Inference vs Training
- Forward-only
- Latency vs throughput
3. Basic system concepts
- CPU vs GPU
- Memory vs compute bottleneck
- Cache, batch size, parallelism

## Level 1: Model-level
4. Lightweight model architectures
- MobileNetV1/V2/V3
- EfficientNet / EfficientNet-Lite
- Depthwise separable convolution
5. Model compression - basic
- Post-training quantization (INT8, FP16)
- Static vs dynamic quantization
6. Benchmark basics
- FPS
- Latency medium
- Model size / RAM usage

## Level 2: Optimization core
7. Quantization - advanced
- Quantization-aware training (QAT)
- Per-channel vs per-tensor
- Symmetric vs asymmetric
8. Pruning
- Magnitude-based pruning
- Structured vs unstructured
- Channel / filter pruning
9. Trade-off analysis
- Accuracy ↔ latency ↔ memory
- Prune + quant interaction

## Level 3: Compiler & Graph Optimization
10. Computation graph
- Operator graph
- Data layout (NCHW / NHWC)
11. Graph-level optimization
- Operator fusion
- Constant folding
- Dead node elimination
12. Compiler / runtime
- TFLite converter internals
- CoreML tools
- TensorRT / TVM (conceptual)

## Level 4: Hardware - Aware Edge AI
13. Hardware fundamentals
- SIMD / NEON
- Cache line & memory access
- DMA / zero-copy
14. Device-specific acceleration
- Apple ANE
- Android NNAPI
- Edge GPU / NPU
15. Latency source analysis
- Memory-bound vs compute-bound
- Kernel launch overhead

## Level 5: Inference Runtime System
16. Inference pipeline
- Preprocess → inference → postprocess
17. Runtime optimization
- Batch = 1 optimization
- Async inference
- Thread pinning
18. Latency metrics
- p50 / p95 / p99
- Cold start vs warm start
19. Power & thermal
- Throttling
- Performance degradation over time

## Level 6: Edge model design (ADVANCED)
20. Edge-specific architectures
- NAS for latency (MnasNet, FBNet)
- MobileViT / TinyViT
21. Transformer on edge
- Attention memory cost
- Token length constraints
22. Multi-model strategy
- Model switching theo latency budget

## Level 7: End-to-End Edge AI System
23. On-device pipeline
- Camera → tensor (zero-copy)
- Sensor fusion
24. Deployment
- Model update not make kill app
- Versioning & rollback
25. Privacy & reliability
- On-device only inference
- Fallback strategies

## Level 8: Edge AI Advanced / Research-Level (Optional)
26. AutoML for Edge
- Hardware-aware NAS
27. Bayesian Optimization
- Tuning quant/prune config
28. Gaussian Process
- Latency modeling
- Performance prediction
29. Learning-based scheduling
- Power / performance control