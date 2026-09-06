"""Embedding wrapper.

Anthropic has no embeddings API, so we use a local sentence-transformers
model. This avoids per-chunk API cost while iterating on chunking strategy
and keeps ingestion fully offline.
"""
from functools import lru_cache

import numpy as np

_MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _get_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(_MODEL_NAME)


def embed_texts(texts: list[str]) -> np.ndarray:
    return _get_model().encode(texts, show_progress_bar=False)


def embed_text(text: str) -> np.ndarray:
    return embed_texts([text])[0]
