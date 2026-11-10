"""Repository paths. The package is installed editable from src/, so the repository root is two levels up."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "paper_results"
PAPER = ROOT / "paper"
WEB = ROOT / "docs" / "paper"
