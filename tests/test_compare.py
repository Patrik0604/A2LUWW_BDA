from __future__ import annotations

import math

from src.docsim.compare import compare_documents


def test_reference_pair_detection() -> None:
    ref = "Ez egy vállalati hitel kockázatelemzési riport."
    gen = "Ez egy vállalati hitel kockázatelemzése, rövidített formában."

    result = compare_documents(ref, gen)

    # basic assertions:
    assert "similarity" in result
    assert "common_keywords" in result
    assert "missing_keywords" in result
    assert "added_keywords" in result

    # similarity should be reasonably high for these similar texts
    assert isinstance(result["similarity"], float)
    assert 0.0 <= result["similarity"] <= 1.0
    assert result["similarity"] > 0.5

    # the keyword 'hitel' should appear in common keywords (after preprocessing)
    assert any(tok == "hitel" for tok in result["common_keywords"])


def test_empty_strings_smoke() -> None:
    """Smoke test to ensure empty inputs do not crash."""
    result = compare_documents("", "")

    assert "similarity" in result
    assert isinstance(result["similarity"], float)
    assert not math.isnan(result["similarity"])

    assert result["common_keywords"] == []
    assert result["missing_keywords"] == []
    assert result["added_keywords"] == []
