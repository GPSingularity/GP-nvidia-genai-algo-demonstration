import os
import faiss
import pickle
import torch
from sentence_transformers import SentenceTransformer
from generator.llm_model import NeMoLLM
import gradio as gr

# Configuration
INDEX_PATH = "faiss.index"
META_PATH = "chunks_meta.pkl"
EMBED_MODEL = "all-MiniLM-L6-v2"

# Ensure index and metadata exist
if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
    raise FileNotFoundError(
        "Index or metadata not found. Run retriever/chunker.py and retriever/build_index.py first."
    )

# Load FAISS index and metadata
index = faiss.read_index(INDEX_PATH)
with open(META_PATH, "rb") as f:
    chunks = pickle.load(f)

# Select device for embeddings & model
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize embedding model on GPU if available
embedder = SentenceTransformer(EMBED_MODEL, device=device)

# Initialize LLM wrapper (uses GPU)
llm = NeMoLLM()

# RAG inference function
def rag_answer(query: str, top_k: int = 5, max_new_tokens: int = 128):
    try:
        # Embed the query
        q_emb = embedder.encode([query]).astype("float32")
        # Retrieve top_k chunks
        _, indices = index.search(q_emb, top_k)
        retrieved = [chunks[i]["text"] for i in indices[0]]
        # Build context and prompt
        context = "\n\n---\n\n".join(retrieved)
        prompt = (
            "Answer the question based on the following context:\n"
            f"{context}\n\nQ: {query}\nA:"
        )
        # Generate response
        response = llm.generate([prompt], max_new_tokens=max_new_tokens)[0]
        return response, retrieved
    except Exception as e:
        return str(e), []

# Build the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown(
        """
        <div style="text-align: center; margin-bottom: 1em;">
          <span style="font-size: 2.5em; color: #c00; font-weight: bold;">
            GP’s NVIDIA RAG Demo on GPU A100 (CUDA)
          </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    with gr.Row():
        inp = gr.Textbox(lines=2, placeholder="Enter your question here...", label="Question")
    out = gr.Textbox(lines=5, label="Answer")
    src = gr.Dataframe(headers=["Retrieved Chunks"], datatype="str", label="Sources")
    btn = gr.Button("Ask")
    btn.click(fn=rag_answer, inputs=[inp], outputs=[out, src])

# Launch with host and port for container/VM with sharing enabled
demo.launch(server_name="0.0.0.0", server_port=7860, share=True)