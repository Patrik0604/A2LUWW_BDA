from __future__ import annotations

from typing import Any, Dict, List


def _format_number(value: float) -> str:
    """
    Egyszerű számformázó a riporthoz.

    - Nagy összegek esetén próbál "M" jelölést használni (millió).
    - Kisebb számoknál maximum 2 tizedesjegyet mutat.
    """
    abs_val = abs(value)
    if abs_val >= 1_000_000:
        millions = value / 1_000_000
        return f"{millions:.1f} M"
    if abs_val.is_integer():
        return f"{int(value)}"
    return f"{value:.2f}"


def _format_list(values: List[Any], max_items: int = 10) -> str:
    """
    Lista rövid, egy soros reprezentációja a riporthoz.
    """
    if not values:
        return "-"
    if len(values) > max_items:
        shown = values[:max_items]
        return f"{', '.join(map(str, shown))} ... (+{len(values) - max_items} további)"
    return ", ".join(map(str, values))


def _format_similarity_block(result: Dict[str, Any]) -> str:
    sim = result.get("similarity", None)
    numeric = result.get("numeric", {})
    num_sim = numeric.get("numeric_similarity", None)

    lines: List[str] = []
    lines.append("szöveges és numerikus értékek")

    if sim is not None:
        lines.append(f"- Szöveges hasonlóság (cosine): {sim:.4f}")

    if num_sim is not None:
        lines.append(f"- Numerikus hasonlóság (Jaccard-szerű): {num_sim:.4f}")

    return "\n".join(lines)


def _format_keyword_block(result: Dict[str, Any], max_items: int = 15) -> str:
    common = result.get("common_keywords", [])
    missing = result.get("missing_keywords", [])
    added = result.get("added_keywords", [])

    lines: List[str] = []
    lines.append("Kulcsszó összehasonlítás (Ez nem olyan fontos)")
    lines.append(f"- Közös kulcsszavak: { _format_list(common, max_items=max_items) }")
    lines.append(f"- Hiányzó (csak referencia): { _format_list(missing, max_items=max_items) }")
    lines.append(f"- Hozzáadott (csak generált): { _format_list(added, max_items=max_items) }")

    return "\n".join(lines)


def _format_numeric_type_block(kind: str, stats: Dict[str, Any]) -> str:
    """
    Egy adott numerikus típus (amount / percent / year / other) rövid összefoglalása.
    """
    ref_vals = stats.get("ref_numeric_values", [])
    gen_vals = stats.get("gen_numeric_values", [])
    common = stats.get("numeric_common", [])
    missing = stats.get("numeric_missing", [])
    added = stats.get("numeric_added", [])
    sim = stats.get("numeric_similarity", None)

    # Számok formázása
    ref_fmt = [_format_number(v) for v in ref_vals]
    gen_fmt = [_format_number(v) for v in gen_vals]
    common_fmt = [_format_number(v) for v in common]
    missing_fmt = [_format_number(v) for v in missing]
    added_fmt = [_format_number(v) for v in added]

    kind_label_map = {
        "amount": "Összegek",
        "percent": "Százalékos értékek",
        "year": "Évszámok",
        "other": "Egyéb numerikus értékek",
    }
    label = kind_label_map.get(kind, kind)

    lines: List[str] = []
    lines.append(f"- {label}:")
    if sim is not None:
        lines.append(f"    - Hasonlóság: {sim:.4f}")
    if ref_vals or gen_vals:
        lines.append(f"    - Referencia értékek: { _format_list(ref_fmt) }")
        lines.append(f"    - Generált értékek:   { _format_list(gen_fmt) }")
        lines.append(f"    - Közös:              { _format_list(common_fmt) }")
        lines.append(f"    - Csak referencia:    { _format_list(missing_fmt) }")
        lines.append(f"    - Csak generált:      { _format_list(added_fmt) }")
    else:
        lines.append("    * Nincs ilyen típusú numerikus érték egyik dokumentumban sem.")

    return "\n".join(lines)


def _format_numeric_block(result: Dict[str, Any]) -> str:
    numeric = result.get("numeric", {})
    if not numeric:
        return "Numerikus összehasonlítás ( na ez itt a lényeg)\nNincs numerikus információ."

    lines: List[str] = []
    lines.append("Numerikus összehasonlítás ( na ez itt a lényeg)")

    overall_sim = numeric.get("numeric_similarity", None)
    if overall_sim is not None:
        lines.append(f"- Összesített numerikus hasonlóság: {overall_sim:.4f}")

    by_type = numeric.get("by_type", {})
    # Rendezett sorrend: amount, percent, year, other
    for kind in ["amount", "percent", "year", "other"]:
        stats = by_type.get(kind)
        if not stats:
            continue

        # Csak akkor írjuk ki, ha van legalább egy érték a ref/gen oldalon
        if stats.get("ref_numeric_values") or stats.get("gen_numeric_values"):
            lines.append("")
            lines.append(_format_numeric_type_block(kind, stats))

    return "\n".join(lines)


def _format_short_conclusion(result: Dict[str, Any]) -> str:
    """
    Egy rövid szöveges konklúzió, ami emberi nyelven összefoglalja az eredményt.
    Nem túl okoskodó, csak pár mondatos összegzés.
    """
    sim = result.get("similarity", None)
    numeric = result.get("numeric", {})
    num_sim = numeric.get("numeric_similarity", None)

    lines: List[str] = []
    lines.append("Szöveges összefoglalás")

    text_part = ""
    if sim is not None:
        if sim > 0.9:
            text_part = "A két dokumentum szövegesen nagyon hasonló."
        elif sim > 0.7:
            text_part = "A két dokumentum szövegesen hasonló, de vannak eltérések."
        else:
            text_part = "A két dokumentum szövegesen jelentősen eltér."
        lines.append(f"- Szövegesen: {text_part} (cosine={sim:.4f})")

    if num_sim is not None:
        if num_sim > 0.8:
            num_part = "A numerikus tartalom szinte teljesen megegyezik."
        elif num_sim > 0.5:
            num_part = "A numerikus tartalom részben egyezik, de vannak lényeges eltérések."
        else:
            num_part = "A numerikus tartalom jelentősen eltér."
        lines.append(f"- Számok alapján: {num_part} (numeric_similarity={num_sim:.4f})")

    if not lines:
        return "=== RÖVID ÖSSZEFOGLALÓ ===\nNincs értékelhető információ."

    return "\n".join(lines)


def format_report(result: Dict[str, Any]) -> str:
    """
    Emberi olvasásra optimalizált, tömör riport formázása a nyers eredmény dict-ből.

    A riport blokkokra tagolva mutatja:
    - szöveges és numerikus hasonlóságot,
    - kulcsszavas egyezéseket és eltéréseket,
    - numerikus értékek (összegek, százalékok, évszámok) főbb különbségeit,
    - egy rövid, interpretálható összefoglalót.
    """
    blocks: List[str] = []

    blocks.append(_format_similarity_block(result))
    blocks.append("")
    blocks.append(_format_keyword_block(result))
    blocks.append("")
    blocks.append(_format_numeric_block(result))
    blocks.append("")
    blocks.append(_format_short_conclusion(result))

    return "\n".join(blocks)
