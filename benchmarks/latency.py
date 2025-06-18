import time
from app import rag_answer

# A few representative questions
questions = [
    "What is the primary innovation in NVIDIA’s Ampere architecture?",
    "How many CUDA cores does the A100 have?",
    "Explain the Multi-Instance GPU feature.",
    "What is the significance of structural sparsity?"
]

def benchmark(qs, **kwargs):
    # Warm-up
    for q in qs:
        _ = rag_answer(q, **kwargs)
    # Measure
    latencies = []
    for q in qs:
        start = time.time()
        _ = rag_answer(q, **kwargs)
        latencies.append(time.time() - start)
    return latencies

if __name__ == "__main__":
    lats = benchmark(questions, top_k=5, max_new_tokens=128)
    for q, lat in zip(questions, lats):
        print(f"Q: {q!r}\\n  → {lat*1000:.1f} ms\\n")
    print(f"Average latency: {sum(lats)/len(lats)*1000:.1f} ms")
