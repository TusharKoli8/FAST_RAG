import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ── Model ────────────────────────────────────────────────────────────────────
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# all-MiniLM-L6-v2 produces 384-dimensional vectors (was wrongly set to 256)
DIMENSION = 384

# ── Persistence paths ────────────────────────────────────────────────────────
STORAGE_DIR   = "storage"
INDEX_PATH    = os.path.join(STORAGE_DIR, "faiss.index")
CHUNKS_PATH   = os.path.join(STORAGE_DIR, "chunks.json")

os.makedirs(STORAGE_DIR, exist_ok=True)

# ── Load or create index + chunk list ────────────────────────────────────────
def _load_index():
    if os.path.exists(INDEX_PATH):
        return faiss.read_index(INDEX_PATH)
    return faiss.IndexFlatL2(DIMENSION)

def _load_chunks():
    if os.path.exists(CHUNKS_PATH):
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

index          = _load_index()
document_chunks: list[str] = _load_chunks()

# ── Public API ────────────────────────────────────────────────────────────────
def create_embeddings(text_chunks: list[str]) -> np.ndarray:
    embeddings = embedding_model.encode(text_chunks, show_progress_bar=True)
    return np.array(embeddings).astype("float32")


def store_embeddings(text_chunks: list[str], embeddings: np.ndarray) -> None:
    """Add chunks + embeddings to the in-memory store and persist to disk."""
    global index, document_chunks

    index.add(embeddings)
    document_chunks.extend(text_chunks)

    # ── Persist ──────────────────────────────────────────────────────────────
    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(document_chunks, f, ensure_ascii=False, indent=2)

    print(f"Stored {len(text_chunks)} new chunks. "
          f"Total in index: {len(document_chunks)}")


def search_similar_chunks(question: str, k: int = 3) -> list[str]:
    """Return the k most relevant chunks for the given question."""
    if index.ntotal == 0:
        return []

    question_embedding = embedding_model.encode([question])
    distances, indices = index.search(
        np.array(question_embedding).astype("float32"), k
    )

    retrieved = []
    for i in indices[0]:
        if 0 <= i < len(document_chunks):
            retrieved.append(document_chunks[i])
    return retrieved


def get_stats() -> dict:
    """Return basic stats about the current index."""
    return {
        "total_chunks": len(document_chunks),
        "index_size":   index.ntotal,
        "dimension":    DIMENSION,
    }


def clear_store() -> None:
    """Wipe the index and chunk list (both in-memory and on disk)."""
    global index, document_chunks
    index = faiss.IndexFlatL2(DIMENSION)
    document_chunks = []

    if os.path.exists(INDEX_PATH):
        os.remove(INDEX_PATH)
    if os.path.exists(CHUNKS_PATH):
        os.remove(CHUNKS_PATH)
    print("Vector store cleared.")
