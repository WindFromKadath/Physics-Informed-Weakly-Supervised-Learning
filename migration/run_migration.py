"""Run the migration CLI without installing the packages first."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent / "reproduction" / "src"))

from pwl_migration.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
