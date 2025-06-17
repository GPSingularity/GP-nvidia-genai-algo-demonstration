import json
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

def build_faiss_index(
    chunks_path: str = "chunks.json",
    index_path: str = "faiss.index",
    meta_path: str = "chunks_meta.pkl",
    embed_model: str = "all-MiniLM-L6-v2"
):
    # Load text chunks
    with open(chunks_path, "r") as fh:
        chunks = json.load(fh)
    texts = [c["text"] for c in chunks]

    # Compute embeddings
    model = SentenceTransformer(embed_model)
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    # Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, index_path)

    # Save metadata
    with open(meta_path, "wb") as fh:
        pickle.dump(chunks, fh)

    print(f"Indexed {index.ntotal} vectors → {index_path}")
    print(f"Metadata saved → {meta_path}")

if __name__ == "__main__":
    build_faiss_index()
