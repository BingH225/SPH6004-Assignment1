from pathlib import Path

import pypdf

BASE_DIR = Path(__file__).resolve().parents[1]
PDF_PATH = BASE_DIR / "docs" / "AY2526_SPH6004_Assignment_release.pdf"
OUTPUT_PATH = BASE_DIR / "docs" / "assignment_task.txt"


def main() -> None:
    reader = pypdf.PdfReader(str(PDF_PATH))
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for i, page in enumerate(reader.pages):
            f.write(f"--- Page {i + 1} ---\n")
            f.write((page.extract_text() or "") + "\n")


if __name__ == "__main__":
    main()
