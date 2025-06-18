# GP-nvidia-genai-algo-demonstration

Building Complete RAG + NVIDIA PDFs with NeMo, ONNX/TensorRT & Triton

## Project Description

GPT with LLM powered by a large language model is revolutionizing industries by efficiently processing vast text corpora and extracting actionable insights from heterogeneous data sources across the spectrum. This versatility enables businesses to transcend simple automation by deploying AI agents that solve complex, multi-step problems and unlock significant productivity gains.

The first wave of generative AI applications, including chatbots, offered valuable one-off solutions through natural language processing. Agentic AI will transform the future by enabling autonomous systems, AI factories, and robotics to use advanced reasoning and iterative planning to solve complex challenges. These next-generation AI agents promise to unlock unprecedented levels of efficiency and foster groundbreaking innovation across diverse industries. Embracing this evolution is not just an opportunity, but it is essential for organizations aiming to thrive in an increasingly competitive landscape.

This repository demonstrates a full-stack Retrieval-Augmented Generation (RAG) pipeline built with NVIDIA AI infrastructure. PDFs are ingested and chunked locally via a Python virtual environment, indexed with FAISS, and served through a NeMo LLM wrapper. For high-performance inference, the model is exported to ONNX, converted into a TensorRT engine on an NVIDIA GPU A100 (using a cloud IDE or DGX Cloud), and deployed within Triton Inference Server (via Docker). An interactive Gradio UI ties the components together for easy experimentation and benchmarking.

## Architecture

```mermaid
flowchart LR
  subgraph Retrieval
    A[PDFs (data/)] --> B[chunker.py -> chunks.json]
    B --> C[build_index.py -> FAISS index]
  end

  subgraph Generation
    C --> D[generator/llm_model.py]
    D --> E[RAG answer]
    E --> F[app.py (Gradio UI)]
  end

  subgraph Inference
    G[export_onnx.py -> onnx/model.onnx]
    G --> H[build_trt_engine.py -> trt/model.plan]
    H --> I[triton_models/rag/1/ + config.pbtxt]
    I --> J[Triton Server]
  end

## Quickstart

1. **Clone and install**

   ```bash
   git clone git@github.com:GPSingularity/GP-nvidia-genai-algo-demonstration.git
   cd GP-nvidia-genai-algo-demonstration
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Ingest & index**

   ```bash
   mkdir -p data && cp /path/to/nvidia-ampere-architecture-whitepaper.pdf data/
   python retriever/chunker.py
   python retriever/build_index.py
   ```

3. **Run the demo**

   ```bash
   python app.py
   ```

## AI Infrastructure

Our demo leverages the following NVIDIA AI stack:

- **Local Development**: Python 3 venv on macOS or Linux for chunking, embedding, and Gradio UI.
- **Vector Search**: FAISS running on CPU for efficient retrieval of document embeddings.
- **Model Serving**:
  - **NeMo LLM**: Wrapped in Python for initial local testing.
  - **ONNX Export**: Converts HF model to ONNX format for compatibility with NVIDIA runtimes.
  - **TensorRT Engine**: Optimized engine built on NVIDIA A100 GPU (SXM4 or PCIe) using HBM2e memory (>2 TB/s bandwidth).
  - **Inference Server**: Triton Inference Server Docker container (`nvcr.io/nvidia/tritonserver:24.03-py3`) to host the TensorRT engine with HTTP/gRPC endpoints.

### Hardware Acquisition (On-Premises or Co-Loc)

| Component                  | Notes                                       |
| -------------------------- | ------------------------------------------- |
| NVIDIA A100 80 GB SXM4     | Typical list; discounts for volume orders   |
| Server chassis / CPU / RAM | Dual-socket CPU, 512 GB+ DDR4, NVMe storage |
| Networking & Switches      | 100 GbE or InfiniBand to feed GPUs          |
| Power & Cooling Upgrade    | Rack-power capacity, PDUs, CRAC adjustments |

### Cloud-Hosted GPU Instances (OPEX)

This demo prefers **Blackwell B200** GPUs on NVIDIA Lepton AI where available. NVIDIA DGX Cloud uses A100 GPUs; other cloud providers offer A100 as fallback.

| Provider         | Instance Type    | GPU Config             |
| ---------------- | ---------------- | ---------------------- |
| NVIDIA Lepton AI | GPU Cloud B200   | 1 × B200 (Blackwell\*) |
| NVIDIA DGX Cloud | DGX A100 service | Multi × A100 nodes     |
| AWS              | p4d.24xlarge     | 8 × A100‑80GB          |
| Google Cloud     | A2 MegaGPU       | 16 × A100‑80GB         |
| Azure            | ND A100 v4       | 8 × A100‑80GB          |
| Oracle Cloud     | BM.GPU4.8        | 8 × A100‑80GB          |

## Benchmarking Performance on NVIDIA GPU A100

We ran our RAG pipeline on an NVIDIA A100 and measured end-to-end latency:

| Question                                                          | Latency (ms) |
| ----------------------------------------------------------------- | ------------ |
| “What is the primary innovation in NVIDIA’s Ampere architecture?” | 32.3         |
| “How many CUDA cores does the A100 have?”                         | 35.6         |
| “Explain the Multi-Instance GPU feature.”                         | 31.5         |
| “What is the significance of structural sparsity?”                | 31.0         |
| **Average (mean over sample questions)**                          | 32.6         |

## ONNX & TensorRT Export

Export the HF model to ONNX and build a TensorRT engine:

```bash
python inference/export_onnx.py
python inference/build_trt_engine.py
```

## Triton Deployment

Fetch the TensorRT plan and launch Triton:

```bash
mkdir -p triton_models/rag/1
curl -L -o triton_models/rag/1/model.plan \
  https://github.com/GPSingularity/GP-nvidia-genai-algo-demonstration/releases/download/v1.0/model.plan

docker run --gpus all --rm \
  -p8000:8000 -p8001:8001 -p8002:8002 \
  -v $(pwd)/triton_models:/models \
  nvcr.io/nvidia/tritonserver:24.03-py3 \
  tritonserver --model-repository=/models
```

## Code Structure

- **data/**: Raw PDF source files
- **retriever/**: PDF ingestion & FAISS index scripts
- **generator/**: NeMo LLM wrapper for RAG
- **inference/**: ONNX export & TRT engine build scripts
- **benchmarks/**: Latency benchmarking scripts
- **triton\_models/**: Triton model repo (config + plan placeholder)

## References

Facebook AI. (2019). *FAISS: A library for efficient similarity search*. Retrieved June 18, 2025, from [https://github.com/facebookresearch/faiss](https://github.com/facebookresearch/faiss)

NVIDIA Corporation. (2020). *NVIDIA Ampere Architecture Whitepaper*. Retrieved June 18, 2025, from [https://www.nvidia.com/content/dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf](https://www.nvidia.com/content/dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf)

NVIDIA Corporation. (2024). *Triton Inference Server Documentation*. NVIDIA. Retrieved June 18, 2025, from [https://docs.nvidia.com/deeplearning/triton-inference-server/](https://docs.nvidia.com/deeplearning/triton-inference-server/)

Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. arXiv. Retrieved June 18, 2025, from [https://arxiv.org/abs/1908.10084](https://arxiv.org/abs/1908.10084)

Saranová, I. (2023). *Advanced Evolutionary Image Filtering* [PDF]. CORE. Retrieved June 18, 2025, from [https://core.ac.uk/download/574570613.pdf](https://core.ac.uk/download/574570613.pdf)

Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., Cistac, P., Rault, T., Louf, R., Funtowicz, M., & Brew, J. (2020). *Transformers: State-of-the-art natural language processing*. EMNLP: System Demonstrations. Retrieved June 18, 2025, from [https://doi.org/10.18653/v1/2020.emnlp-demos.6](https://doi.org/10.18653/v1/2020.emnlp-demos.6)

## License

This project uses the Apache License 2.0.

## Contributing

Feel free to open issues or submit pull requests for improvements!

