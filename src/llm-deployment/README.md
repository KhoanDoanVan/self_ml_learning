## LLM SELF-HOST DEPLOYMENT (PRODUCTION-READY)

### 0. Core Concepts
- Transformer Architecture
- Decoder-only LLM
- Tokenization (BPE, SentencePiece)
- KV Cache
- Autoregressive Decoding
- Attention Complexity
- Context Window

### 1. Decision Layer (Size & Models)
- Model zoo (Qwen, LLaMA, Mistral)
- Instruct vs Base Model
- Parameter scaling laws
- Multilingual training
- Reasoning vs Chat Tradeoff
- Model card analysis

Related:
- Qwen2.5-Instruct
- LLaMA 3 Instruct
- Mistral Instruct

### 2. Inference Fundamentals
- Greedy vs Sampling
- Temperature
- Top-p/Top-k
- Repetition penalty
- Max tokens
- Stop sequences

### 3. Prompt Engineering
- System prompt
- Instruction hierarchy
- Prompt injection
- Output constraints
- JSON schema enforcement 
- Function calling
- Tool calling

### 4. Fine-tuning & Apptation
- Supervised fine-tuning (SFT)
- Instruction tuning
- Domain adaptation
- Dataset curation
- Data leakage
- Ovefitting LLM

### 5. PEFT / LoRA ecosystem
- PEFT
- LoRA
- QLoRA
- Rank (r)
- Alpha
- Target modules
- Adapter stacking
- Merge LoRA weights

### 6. Quantization (critial for self-host)
- Quantization theory
- INT8 / INT4
- GPTQ
- AWQ
- GGUF
- Quantization error
- Per-channel vs per-tensor

### 7. Inference Engine / Runtime
- vLLM
- llama.cpp
- Text Generation Inference (TGI)
- TensorRT-LLM
- FlashAttention
- PagedAttention
- Continuous batching

### 8. Hardware & System Optimization
- GPU memory fragmentation
- VRAM vs RAM
- CPU offloading
- NUMA
- PCle bottleneck
- Batch size tuning
- Throughput vs latency

### 9. API Layer vs Serving
- FastAPI
- OpenAI-compatible API
- Streaming response
- SSE (Server-Sent Events)
- Request queue
- Rate limiting
- Load shedding

### 10. Observability & Logging
- Token usage logging
- Latency P50/P95/P99
- Time to first token (TTFT)
- Prompt/output logging
- Trace ID
- Structured logging

### 11. Evaluation & Regression Testing
- LLM evaluation
- Task-specific metrics
- LLM-as-Judge
- Hallucination detection
- Faithfulness
- Relevance scoring
- Golden dataset
- Offline batch eval

### 12. Safety & Guardrails
- Content moderation
- Prompt injection defense
- Jailbreak detection
- Output filtering
- Regex / rule-based guardrail
- Schema validation
- Refusal correctness

### 13. RAG (Optional)
- Embedding models
- Vector database
- Chunking strategy
- Retrieval recall
- Context compression
- RAG hallucination
- Hybird search
- Re-ranking

### 14. Agent & Tool System (Advanced)
- Tool calling
- Function schema
- Agent loop
- Planner-executor
- ReAct
- State management
- Tool hallucination

### 15. Cost & Scaling
- Tokens per request
- Cost per task
- Autoscaling
- Horizontal vs Vertical scaling
- GPU sharing
- Admission control

### 16. CI/CD for LLM
- Model versioning
- Prompt versioning
- Canary deployment
- A/B testing LLM
- Rollback strategy
- Model registry

### 17. Production failure modes (critical)
- Hallucination spikes
- Latency degradation
- OOM errors
- Context explosion
- Prompt drift
- Silent failure
- Degenerate outputs