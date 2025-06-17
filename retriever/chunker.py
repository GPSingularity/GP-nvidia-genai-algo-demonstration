import os
import fitz  # PyMuPDF
import json
from typing import List, Dict

def load_and_chunk(
    pdf_paths: List[str], chunk_size: int = 500, overlap: int = 50
) -> List[Dict]:
    """
    Read PDFs from `pdf_paths`, extract text, and split into overlapping word chunks.
    Returns a list of dicts with keys: text, source, chunk_id.
    """
    docs = []
    for path in pdf_paths:
        basename = os.path.basename(path)
        doc = fitz.open(path)
        text = "".join(page.get_text() for page in doc)
        words = text.split()
        step = chunk_size - overlap
        for idx in range(0, len(words), step):
            chunk = " ".join(words[idx : idx + chunk_size])
            docs.append({
                "text": chunk,
                "source": basename,
                "chunk_id": idx // step,
            })
    return docs

if __name__ == "__main__":
    pdf_dir = "data"
    pdf_paths = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
    chunks = load_and_chunk(pdf_paths)
    print(f"Extracted {len(chunks)} chunks from {len(pdf_paths)} PDFs.")
    with open("chunks.json", "w") as fh:
        json.dump(chunks, fh, indent=2)
    print("Wrote chunks.json")
