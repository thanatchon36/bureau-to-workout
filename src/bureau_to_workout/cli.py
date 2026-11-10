"""Command line: python -m bureau_to_workout.cli {bureau,workout,experiments,tables,figures,paper,web}."""
from __future__ import annotations

import argparse
import runpy
import shutil
import subprocess
import sys

from .paths import PAPER, WEB


def build_pdf() -> None:
    """paper/main.tex -> paper/paper.pdf with tectonic (runs bibtex and reruns LaTeX as needed)."""
    if shutil.which("tectonic") is None:
        raise SystemExit("tectonic not found; install it (brew install tectonic) or run pdflatex + bibtex by hand")
    subprocess.run(["tectonic", "-X", "compile", "main.tex", "--keep-logs"], cwd=PAPER, check=True)
    (PAPER / "main.pdf").replace(PAPER / "paper.pdf")
    print("wrote", PAPER / "paper.pdf")


def build_web() -> None:
    from .paper_web import build_web as _build
    print("wrote", _build(PAPER, WEB))


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in {"-h", "--help"}:
        print(__doc__.strip())
        print("bureau and workout take their own options, e.g. bureau --exp e2 --seed 8; see bureau --help")
        return
    cmd, rest = argv[0], argv[1:]
    if cmd == "bureau":
        from . import bureau
        bureau.main(rest)
    elif cmd == "workout":
        from . import workout
        workout.main(rest)
    elif cmd == "experiments":
        from . import experiments
        experiments.main()
    elif cmd == "tables":
        runpy.run_module("bureau_to_workout.tables", run_name="__main__")
    elif cmd == "figures":
        from . import figures
        figures.main(rest)
    elif cmd == "paper":
        build_pdf()
        build_web()
    elif cmd == "web":
        build_web()
    else:
        raise SystemExit(f"unknown command {cmd!r}; see python -m bureau_to_workout.cli --help")


if __name__ == "__main__":
    main()
