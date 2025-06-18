# GP-nvidia-genai-algo-demonstration
Building Complete RAG + NVIDIA PDFs with NeMo, ONNX/TensorRT &amp; Triton

## Benchmarking Performance on NVIDIA A100

I ran RAG pipeline on an NVIDIA GPU A100 and measured performance benchmark for latency:

| Question                                              | Latency (ms) |
|-------------------------------------------------------|-------------:|
| “What is the primary innovation in NVIDIA’s Ampere architecture?” |        32.3 |
| “How many CUDA cores does the A100 have?”             |        35.6 |
| “Explain the Multi-Instance GPU feature.”             |        31.5 |
| “What is the significance of structural sparsity?”    |        31.0 |
| **Average**                                           |        32.6 |
