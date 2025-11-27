from __future__ import annotations

import re
from typing import List


_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)


def preprocess_text(text: str) -> str:
    """
    Basic preprocessing for Hungarian business texts.

    Steps:
    - strip leading/trailing whitespace
    - normalize inner whitespace to single spaces
    - lowercase

    NOTE:
    - This is intentionally simple for an MVP.
    - In later releases we may extend this with more sophisticated logic
      (e.g., handling punctuation, domain-specific cleaning, etc.).
    """
    stripped = text.strip()
    if not stripped:
        return ""
    normalized = " ".join(stripped.split())
    return normalized.lower()


def tokenize(text: str) -> List[str]:
    """
    Simple regex-based tokenization AFTER preprocess_text.

    Behavior:
    - splits on non-word characters
    - keeps Hungarian letters and digits
    - strips punctuation (e.g. 'huf,' -> 'huf')
    """
    if not text:
        return []
    # \w+ a UNICODE flaggel: betűk, számok, aláhúzás – magyar ékezetes karakterekkel
    return _WORD_RE.findall(text)
