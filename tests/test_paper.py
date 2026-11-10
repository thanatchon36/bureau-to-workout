"""The paper's tables and numbers are generated from paper_results/, and the web reader resolves them."""
import re
import runpy
import shutil

import pytest

from bureau_to_workout.paths import PAPER


def test_tables_and_macros_regenerate_identically():
    files = sorted((PAPER / "tables").glob("*.tex")) + [PAPER / "results_macros.tex"]
    before = {f: f.read_bytes() for f in files}
    try:
        runpy.run_module("bureau_to_workout.tables", run_name="__main__")
        changed = [f.name for f in files if f.read_bytes() != before[f]]
    finally:
        for f, data in before.items():
            f.write_bytes(data)
    assert not changed, f"committed files differ from what make tables writes: {changed}"


@pytest.mark.skipif(shutil.which("pandoc") is None, reason="pandoc not installed")
def test_web_reader_builds(tmp_path):
    from bureau_to_workout.paper_web import build_web

    page = build_web(PAPER, tmp_path).read_text()
    assert "From Bureau to Workout" in page
    assert page.count('class="r r-') == 6
    assert "Appendix\u00a0A" in page and 'id="references"' in page and 'id="alg:budget"' in page
    targets = set(re.findall(r'id="([^"]+)"', page))
    assert set(re.findall(r'href="#([^"]+)"', page)) <= targets, "a cross-reference points nowhere"
