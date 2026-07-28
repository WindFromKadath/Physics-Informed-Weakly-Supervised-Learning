# -*- coding: utf-8 -*-
"""Dump text spans (with font size / superscript flags) around the simulation
equations to reconstruct their exact algebraic form."""
from pathlib import Path
import fitz

ROOT = Path(__file__).resolve().parents[1] / "papers"

def dump(pdf_name, page_idx, y0=None, y1=None, out=print):
    doc = fitz.open(ROOT / pdf_name)
    page = doc[page_idx]
    d = page.get_text("dict")
    out(f"### {pdf_name} page {page_idx+1} (region y={y0}..{y1})")
    for block in d["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            ly = line["bbox"][1]
            if y0 is not None and not (y0 <= ly <= y1):
                continue
            parts = []
            for span in line["spans"]:
                flags = span["flags"]
                sup = "^" if (flags & 1) else ""
                txt = span["text"]
                if txt.strip():
                    parts.append(f"{sup}[{txt}]{span['size']:.0f}")
            if parts:
                out(f"  y={ly:7.1f}  " + " ".join(parts))
    doc.close()

# IEEE paper: find page with "functional forms"
ieee = next(ROOT.glob("Alenezi*2025*.pdf"))
doc = fitz.open(ROOT / ieee)
for i in range(doc.page_count):
    t = doc[i].get_text().lower()
    if "functional forms" in t:
        print(f"IEEE: 'functional forms' found on PDF page {i+1}")
doc.close()

# Dissertation: find page with Eq (15) eta_true
dis = "ALENEZI-DISSERTATION-2024.pdf"
doc = fitz.open(ROOT / dis)
for i in range(doc.page_count):
    t = doc[i].get_text().lower()
    if "functional forms" in t:
        print(f"Dissertation: 'functional forms' found on PDF page {i+1}")
doc.close()

def dump2(pdf_name, page_idx, y0, y1, out):
    """span-level dump with per-span y0 to distinguish super/subscripts"""
    doc = fitz.open(ROOT / pdf_name)
    page = doc[page_idx]
    d = page.get_text("dict")
    out(f"### {pdf_name} page {page_idx+1} spans y={y0}..{y1}")
    for block in d["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                sy0, sy1 = span["bbox"][1], span["bbox"][3]
                if y0 <= sy0 <= y1 and span["text"].strip():
                    out(f"  x0={span['bbox'][0]:6.1f} y0={sy0:6.1f} y1={sy1:6.1f} sz={span['size']:.1f} fl={span['flags']} [{span['text']}]")
    doc.close()

out_lines = []
out = out_lines.append
dump2(ieee.name if hasattr(ieee, 'name') else ieee, 7, 680, 750, out)   # IEEE Eq 9-10
out("=" * 60)
dump2(ieee.name if hasattr(ieee, 'name') else ieee, 8, 45, 75, out)     # IEEE Eq 11 (next page)
out("=" * 60)
dump2(dis, 79, 235, 395, out)                                           # dissertation Eq 15-17
(Path(__file__).parent / "eq_dump.txt").write_text("\n".join(out_lines), encoding="utf-8")
print("written eq_dump.txt, lines:", len(out_lines))
