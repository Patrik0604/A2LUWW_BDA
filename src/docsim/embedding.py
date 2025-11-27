from __future__ import annotations

from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

_model: Optional[SentenceTransformer] = None


def get_model() -> SentenceTransformer:
    """
    Lazy-load and return the global SentenceTransformer model instance.

    The model is instantiated only once per process. Subsequent calls reuse
    the same model instance, which is important for performance in services.
    """
    global _model
    if _model is None:
        model = SentenceTransformer(_MODEL_NAME)
        # Növeljük a max_seq_length-et, hogy ne dobjon figyelmeztetést hosszabb szövegeknél
        # (az underlying transformer default 128 token, itt engedünk többet).
        model.max_seq_length = 256
        _model = model
    return _model


def embed_text(text: str) -> np.ndarray:
    """
    Get a vector embedding for an entire document.

    Uses a multilingual Sentence-BERT model suitable for Hungarian business texts.

    Parameters
    ----------
    text:
        Preprocessed document text as a single string.

    Returns
    -------
    np.ndarray
        A 1D numpy array representing the document embedding.
    """
    model = get_model()
    embedding = model.encode(text, convert_to_numpy=True)
    return np.asarray(embedding).ravel()
