from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Load the model once and reuse it for the whole process (singleton)."""
    return SentenceTransformer(MODEL_NAME)


def embed(text: str) -> list[float]:
    """Return an L2-normalised embedding as a JSON-serialisable float list.

    Normalising here means cosine similarity later is just a dot product.
    """
    vector = get_model().encode(text, normalize_embeddings=True)
    return vector.astype(float).tolist()


def rank(
    query_embedding: list[float],
    candidates: list[tuple[int, list[float]]],
    top_k: int,
) -> list[tuple[int, float]]:
    """Cosine-rank candidates against the query; return top_k (id, score)."""
    if not candidates:
        return []

    ids = [doc_id for doc_id, _ in candidates]
    matrix = np.array([vec for _, vec in candidates], dtype=np.float32)  # (N, 384)
    query = np.asarray(query_embedding, dtype=np.float32)                # (384,)

    # Stored vectors and the query are already unit length, so the dot product
    # equals cosine similarity. One matrix-vector product scores all docs.
    scores = matrix @ query  # (N,)

    top = np.argsort(-scores)[:top_k]
    return [(ids[i], float(scores[i])) for i in top]
