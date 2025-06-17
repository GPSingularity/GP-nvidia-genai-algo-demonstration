import faiss
import pickle
from retriever.chunker import load_and_chunk  # if you want dynamic ingestion
from retriever.build_index import build_faiss_index
from sentence_transformers import SentenceTransformer
from generator.llm_model import NeMoLLM
import gradio as gr
import os

# Paths
INDEX_PATH = "faiss.index"
META_PATH = "chunks_meta.pkl"
EMBED_MODEL = "all-MiniLM-L6-v2"

# Load FAISS index and metadata
index = faiss.read_index(INDEX_PATH)
with open(META_PATH, "rb") as f:
    chunks = pickle.load(f)

# Embedding model (for queries)
embedder = SentenceTransformer(EMBED_MODEL)

# NeMo LLM
nemo_llm = NeMoLLM(model_name="gpt-j-6B")

# RAG function

def rag_answer(query: str, top_k: int = 5, max_length: int = 256):
    # Embed query
    q_emb = embedder.encode([query]).astype("float32")
    distances, indices = index.search(q_emb, top_k)
    retrieved = [chunks[i]["text"] for i in indices[0]]
    context = "\n\n---\n\n".join(retrieved)
    prompt = f"Answer the question based on the following context:\n{context}\n\nQ: {query}\nA:"

    # Generate with NeMo
    response = nemo_llm.generate([prompt], max_length=max_length)[0]
    return response, retrieved

# Gradio UI
def launch_app():
    with gr.Blocks() as demo:
        gr.Markdown("# NVIDIA RAG Demo with NeMo & FAISS")
        with gr.Row():
            inp = gr.Textbox(lines=2, placeholder="Enter your question here...", label="Question")
        out = gr.Textbox(lines=5, label="Answer")
        src = gr.Dataframe(headers=["Retrieved Chunks"], datatype="str", label="Sources")
        btn = gr.Button("Ask")
        btn.click(fn=rag_answer, inputs=[inp], outputs=[out, src])
    demo.launch()

if __name__ == "__main__":
    # Optionally, re-index if new PDFs added:
    # if not os.path.exists(INDEX_PATH):
    #     build_faiss_index()
    launch_app()