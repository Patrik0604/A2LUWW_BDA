from __future__ import annotations

from pathlib import Path

from src.docsim import compare_document_files
from src.docsim.report_formatter import format_report


def main() -> None:
    """
    Small demo that compares two .txt files and prints a human-friendly report.

    Elvárt fájlok (a projekt gyökerében):
    - reference.txt
    - generated.txt
    """
    ref_path = Path("reference.txt")
    gen_path = Path("generated.txt")

    if not ref_path.exists() or not gen_path.exists():
        print("Hiányzik a reference.txt vagy generated.txt a projekt gyökerében.")
        print("Hozz létre két .txt fájlt a következő nevekkel ugyanebben a mappában:")
        print("  - reference.txt  (referencia riport)")
        print("  - generated.txt  (generált / módosított riport)")
        return

    result = compare_document_files(ref_path, gen_path)
    report = format_report(result)

    print(report)


if __name__ == "__main__":
    main()
