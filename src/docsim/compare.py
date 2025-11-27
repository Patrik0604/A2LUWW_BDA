from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .embedding import embed_text
from .metrics import (
    compare_numeric_tokens,
    cosine_similarity_score,
    extract_keywords,
)
from .preprocessing import preprocess_text, tokenize


def compare_documents(ref_text: str, gen_text: str) -> Dict[str, Any]:
    """
    Compare a reference document and a generated document.

    Steps:
    1. Preprocess both texts (whitespace normalization, lowercasing).
    2. Tokenize both texts with a simple regex-based tokenizer.
    3. Build embeddings with Sentence-BERT for both preprocessed texts.
    4. Compute cosine similarity between the embeddings (textual similarity).
    5. Extract top textual keywords from each document based on simple frequency.
    6. Compute textual keyword overlaps (common/missing/added).
    7. Extract and compare numeric values (amounts, rates, years, other)
       directly from the preprocessed text.

    Returns
    -------
    Dict[str, Any]
        {
            "similarity": float,
            "common_keywords": list[str],
            "missing_keywords": list[str],
            "added_keywords": list[str],
            "numeric": {
                "numeric_similarity": float,
                "ref_numeric_values": list[float],
                "gen_numeric_values": list[float],
                "numeric_common": list[float],
                "numeric_missing": list[float],
                "numeric_added": list[float],
                "by_type": {
                    "amount": {...},
                    "percent": {...},
                    "year": {...},
                    "other": {...},
                },
            },
        }
    """
    # 1. Preprocess texts
    ref_clean: str = preprocess_text(ref_text)
    gen_clean: str = preprocess_text(gen_text)

    # 2. Tokenize (regex-based, human-friendly tokens)
    ref_tokens: List[str] = tokenize(ref_clean)
    gen_tokens: List[str] = tokenize(gen_clean)

    # 3. Embeddings
    ref_vec = embed_text(ref_clean)
    gen_vec = embed_text(gen_clean)

    # 4. Textual similarity (cosine)
    similarity_raw = cosine_similarity_score(ref_vec, gen_vec)
    similarity: float = float(similarity_raw)

    # 5. Keywords
    ref_kw: List[str] = extract_keywords(ref_tokens, top_n=20)
    gen_kw: List[str] = extract_keywords(gen_tokens, top_n=20)

    # 6. Keyword set operations
    ref_set = set(ref_kw)
    gen_set = set(gen_kw)

    common = sorted(ref_set.intersection(gen_set))
    missing = sorted(ref_set - gen_set)
    added = sorted(gen_set - ref_set)

    # 7. Numeric comparison – works on full preprocessed text
    numeric_stats = compare_numeric_tokens(ref_clean, gen_clean)

    result: Dict[str, Any] = {
        "similarity": similarity,
        "common_keywords": common,
        "missing_keywords": missing,
        "added_keywords": added,
        "numeric": numeric_stats,
    }

    return result


def compare_document_files(ref_path: str | Path, gen_path: str | Path) -> Dict[str, Any]:
    """
    Convenience wrapper: compare two documents provided as text files (.txt).
    """
    ref_path = Path(ref_path)
    gen_path = Path(gen_path)

    ref_text = ref_path.read_text(encoding="utf-8")
    gen_text = gen_path.read_text(encoding="utf-8")

    return compare_documents(ref_text, gen_text)
