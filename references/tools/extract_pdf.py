# -*- coding: utf-8 -*-
"""Extract references/papers PDFs into references/extracted_text/*.txt."""
import sys
from pathlib import Path

import fitz  # PyMuPDF

REFERENCE_ROOT = Path(__file__).resolve().parents[1]
PAPERS = REFERENCE_ROOT / "papers"
OUT = REFERENCE_ROOT / "extracted_text"
OUT.mkdir(exist_ok=True)

for pdf in sorted(PAPERS.glob("*.pdf")):
    print(f"=== {pdf.name} ===")
    try:
        doc = fitz.open(pdf)
    except Exception as e:
        print(f"  FAILED to open: {e}")
        continue
    print(f"  pages: {doc.page_count}")
    print(f"  metadata: {doc.metadata.get('title', '')} | {doc.metadata.get('author', '')}")
    parts = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        parts.append(f"\n\n===== PAGE {i + 1} =====\n{text}")
    out_file = OUT / (pdf.stem + ".txt")
    out_file.write_text("".join(parts), encoding="utf-8")
    n_chars = sum(len(p) for p in parts)
    print(f"  extracted {n_chars} chars -> {out_file.name}")
    doc.close()
