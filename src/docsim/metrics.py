from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, Iterable, List

import numpy as np


# -------------------------
# Szöveges hasonlóság (cosine)
# -------------------------
def cosine_similarity_score(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two 1D numpy vectors.

    Returns
    -------
    float
        Cosine similarity value between -1 and 1.
        In this use-case it will typically be between 0 and 1.

    Notes
    -----
    If either vector has zero norm, the similarity is defined as 0.0.
    """
    v1 = np.asarray(vec1, dtype=float).ravel()
    v2 = np.asarray(vec2, dtype=float).ravel()

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    similarity = float(np.dot(v1, v2) / (norm1 * norm2))
    return similarity


# -------------------------
# Kulcsszó kinyerés
# -------------------------
def extract_keywords(tokens: Iterable[str], top_n: int = 20) -> List[str]:
    """
    Very simple keyword extraction based on term frequency.

    Parameters
    ----------
    tokens:
        A sequence of preprocessed tokens.
    top_n:
        Return at most this many keywords.

    Behavior
    --------
    - Count frequencies with Counter.
    - Ignore purely numeric tokens (e.g., token.isdigit()) because future
      releases may handle numerics separately.
    - Sort by frequency (descending), then alphabetically for ties.
    - Return list of the most common tokens, as strings.
    """
    filtered_tokens: List[str] = [t for t in tokens if not t.isdigit()]
    counter = Counter(filtered_tokens)

    sorted_items = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))

    keywords = [token for token, _ in sorted_items[:top_n]]
    return keywords


# -------------------------
# Okosabb numerikus parser
# -------------------------
def _normalize_number_string(raw: str) -> float | None:
    """
    Normalize a raw number string to float.

    Steps:
    - remove spaces and non-breaking spaces between digit groups
    - replace ',' with '.'
    """
    cleaned = raw.replace(" ", "").replace("\u00A0", "").replace(",", ".")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _extract_numbers_by_type(text: str) -> Dict[str, List[float]]:
    """
    Extract numeric values from text and categorize them.

    Types:
    - "amount": numbers followed by currency markers (huf, ft, eur, usd)
    - "percent": numbers followed by '%'
    - "year": 4-digit years (1900-2100)
    - "other": any remaining numbers

    The function ensures that overlapping matches are not double-counted.
    """
    result: Dict[str, List[float]] = {
        "amount": [],
        "percent": [],
        "year": [],
        "other": [],
    }
    used_spans: List[tuple[int, int]] = []

    def span_overlaps(start: int, end: int) -> bool:
        for s, e in used_spans:
            if not (end <= s or start >= e):
                return True
        return False

    def add_match(kind: str, match: re.Match) -> None:
        start, end = match.span("num")
        if span_overlaps(start, end):
            return

        raw = match.group("num")
        value = _normalize_number_string(raw)
        if value is None:
            return

        used_spans.append((start, end))
        result.setdefault(kind, []).append(value)

    # 1) Percents: 5,5%, 1.14 %, stb.
    percent_pattern = re.compile(
        r"(?P<num>[-+]?\d[\d \u00A0]*(?:[.,]\d+)?)[ ]*%",
        flags=re.IGNORECASE,
    )
    for m in percent_pattern.finditer(text):
        add_match("percent", m)

    # 2) Amounts: 100 000 000 HUF, 5 000 ft, stb.
    amount_pattern = re.compile(
        r"(?P<num>[-+]?\d[\d \u00A0]*(?:[.,]\d+)?)[ ]*(huf|ft|eur|usd)\b",
        flags=re.IGNORECASE,
    )
    for m in amount_pattern.finditer(text):
        add_match("amount", m)

    # 3) Years: 4 digit (1900–2100)
    year_pattern = re.compile(
        r"\b(?P<num>(19\d{2}|20\d{2}|2100))\b",
        flags=re.IGNORECASE,
    )
    for m in year_pattern.finditer(text):
        add_match("year", m)

    # 4) Generic numbers (other)
    generic_pattern = re.compile(
        r"(?P<num>[-+]?\d[\d \u00A0]*(?:[.,]\d+)?)",
        flags=re.IGNORECASE,
    )
    for m in generic_pattern.finditer(text):
        start, end = m.span("num")
        if span_overlaps(start, end):
            continue

        # Ha a szám előtt közvetlenül betű van (pl. 'q1'), akkor hagyjuk ki
        if start > 0 and text[start - 1].isalpha():
            continue

        add_match("other", m)

    return result


def _compute_numeric_stats(ref_values: List[float], gen_values: List[float]) -> Dict[str, Any]:
    """
    Compute Jaccard-like similarity and overlaps for numeric values.
    """
    def to_rounded_set(values: List[float]) -> set[float]:
        return {round(v, 6) for v in values}

    ref_set = to_rounded_set(ref_values)
    gen_set = to_rounded_set(gen_values)

    union = ref_set | gen_set
    intersection = ref_set & gen_set

    if not union:
        numeric_similarity = 1.0
    else:
        numeric_similarity = float(len(intersection) / len(union))

    numeric_common = sorted(intersection)
    numeric_missing = sorted(ref_set - gen_set)
    numeric_added = sorted(gen_set - ref_set)

    return {
        "numeric_similarity": numeric_similarity,
        "ref_numeric_values": sorted(ref_set),
        "gen_numeric_values": sorted(gen_set),
        "numeric_common": numeric_common,
        "numeric_missing": numeric_missing,
        "numeric_added": numeric_added,
    }


def compare_numeric_tokens(ref_text: str, gen_text: str, tolerance: float = 1e-6) -> Dict[str, Any]:
    """
    Compare numeric content between two documents (smarter parser).

    Parameters
    ----------
    ref_text:
        Preprocessed reference document text.
    gen_text:
        Preprocessed generated document text.
    tolerance:
        Reserved for future use (e.g. value-based tolerance).
        Currently we use simple rounding-based matching.

    Returns
    -------
    Dict[str, Any]
        {
          "numeric_similarity": float,            # overall similarity
          "ref_numeric_values": list[float],
          "gen_numeric_values": list[float],
          "numeric_common": list[float],
          "numeric_missing": list[float],
          "numeric_added": list[float],
          "by_type": {
              "amount": {...},   # same structure as above
              "percent": {...},
              "year": {...},
              "other": {...},
          }
        }
    """
    ref_by_type = _extract_numbers_by_type(ref_text)
    gen_by_type = _extract_numbers_by_type(gen_text)

    # Per-type stats
    all_types = sorted(set(ref_by_type.keys()) | set(gen_by_type.keys()))
    by_type_stats: Dict[str, Any] = {}
    all_ref_values: List[float] = []
    all_gen_values: List[float] = []

    for kind in all_types:
        ref_values = ref_by_type.get(kind, [])
        gen_values = gen_by_type.get(kind, [])

        all_ref_values.extend(ref_values)
        all_gen_values.extend(gen_values)

        by_type_stats[kind] = _compute_numeric_stats(ref_values, gen_values)

    # Overall stats (összes típus együtt)
    overall = _compute_numeric_stats(all_ref_values, all_gen_values)

    overall["by_type"] = by_type_stats
    return overall
