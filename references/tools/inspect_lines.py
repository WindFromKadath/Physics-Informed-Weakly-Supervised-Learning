# -*- coding: utf-8 -*-
"""Check horizontal rule(s) (fraction bars) inside Eq (9)-(11) to determine
whether it's one big fraction or two separate fractions."""
from pathlib import Path
import fitz

ROOT = Path(__file__).resolve().parents[1] / "papers"
ieee = next(ROOT.glob("Alenezi*2025*.pdf"))
doc = fitz.open(ROOT / ieee)
page = doc[7]  # PDF page 8 with Eq (9),(10)
lines = []
for d in page.get_drawings():
    for item in d["items"]:
        if item[0] == "l":  # line
            p1, p2 = item[1], item[2]
            if abs(p1.y - p2.y) < 0.5:  # horizontal
                lines.append((p1.x, p2.x, p1.y))
        elif item[0] == "re":  # rectangle
            r = item[1]
            if r.height < 1.0 and r.width > 3:  # thin horizontal rect = rule
                lines.append((r.x0, r.x1, (r.y0 + r.y1) / 2))

# fraction bars of Eq (9): x in [170, 300], y in [680, 760]
print("Horizontal rules on IEEE PDF page 8 (x in 150-300, y in 680-760):")
for x0, x1, y in sorted(lines, key=lambda t: (t[2], t[0])):
    if 150 < x0 < 310 and 680 < y < 760:
        print(f"  y={y:7.2f}  x: {x0:6.1f} -> {x1:6.1f}  (len={x1-x0:5.1f})")

# also Eq (11) is on page 9 (index 8)
page9 = doc[8]
lines9 = []
for d in page9.get_drawings():
    for item in d["items"]:
        if item[0] == "l":
            p1, p2 = item[1], item[2]
            if abs(p1.y - p2.y) < 0.5:
                lines9.append((p1.x, p2.x, p1.y))
        elif item[0] == "re":
            r = item[1]
            if r.height < 1.0 and r.width > 3:
                lines9.append((r.x0, r.x1, (r.y0 + r.y1) / 2))
print("Horizontal rules on IEEE PDF page 9 (y in 40-75):")
for x0, x1, y in sorted(lines9, key=lambda t: (t[2], t[0])):
    if 40 < y < 75:
        print(f"  y={y:7.2f}  x: {x0:6.1f} -> {x1:6.1f}  (len={x1-x0:5.1f})")
doc.close()
