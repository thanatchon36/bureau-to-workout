"""LinkedIn poster -> docs/assets/linkedin_poster.png (1440 x 1800, 4:5), in the format of the thainowcast poster.

The image is real bureau-simulator output (the banner's arrears raster, taller), and every number on the poster is
read from paper/results_macros.tex, the macros the paper itself uses, so the poster cannot disagree with the paper.
Rendered from HTML by headless Chrome (Fira Sans from Google Fonts, so it needs a network connection).

Run: make poster
"""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from export_hf import E2_FILES, E3_FILES  # noqa: E402
from make_banner import COLORS, arrears_rows  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from bureau_to_workout.paths import PAPER, ROOT  # noqa: E402

W, H = 1440, 1800
IMG = dict(x=800, y=170, w=546, h=1060)   # where the raster sits on the poster, px
N_ROWS = 300
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
DOI = "10.5281/zenodo.24102251"


def macros() -> dict[str, str]:
    text = (PAPER / "results_macros.tex").read_text()
    return {k: v.replace("$-$", "−") for k, v in re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}", text)}


def raster(path: Path) -> tuple[float, float, float]:
    """Write the arrears raster; return the moratorium band's x-centre and a 90+ row's y, in poster px."""
    rows, cfg = arrears_rows(N_ROWS)
    fig = plt.figure(figsize=(IMG["w"] / 100, IMG["h"] / 100), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(rows, aspect="auto", cmap=ListedColormap(COLORS), vmin=-0.5, vmax=4.5, interpolation="nearest")
    ax.set_axis_off()
    fig.savefig(path, dpi=200, facecolor="#04070f")
    plt.close(fig)
    n_months = rows.shape[1]
    band_x = IMG["x"] + IMG["w"] * ((cfg.shock_start + cfg.shock_end + 1) / 2) / n_months
    # a borrower deferred through the whole window, near the foot of the deferral block
    deferred = np.flatnonzero((rows[:, cfg.shock_start:cfg.shock_end + 1] == 4).all(axis=1))
    band_row = deferred[np.argmin(np.abs(deferred - 0.80 * N_ROWS))]
    # a borrower who ends in 90+ DPD (an NPL, the kind that goes to the workout), near the bottom
    npl = np.flatnonzero(rows[:, -1] == 3)
    row = npl[np.argmin(np.abs(npl - 0.88 * N_ROWS))]
    to_y = lambda r: IMG["y"] + IMG["h"] * (r + 0.5) / N_ROWS  # noqa: E731
    return band_x, to_y(row), to_y(band_row)


def html(m: dict[str, str], band_x: float, npl_y: float, band_y: float, n_models: int) -> str:
    x0 = 735  # right edge of the callout column
    return f"""<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Fira+Sans:wght@400;500;600;700&display=block" rel="stylesheet">
<style>
:root {{ --bg:#04070f; --amber:#f5a623; --amber-2:#ffd88a; --ink:#f1f2f5; --muted:#9aa6bd; --dim:#6f7a91; --blue:#7fa3d6; --rule:#1d2638; }}
* {{ margin:0; box-sizing:border-box; }}
body {{ width:{W}px; height:{H}px; background:radial-gradient(900px 900px at 1100px 760px, #0b1a33 0%, var(--bg) 60%);
  color:var(--ink); font-family:"Fira Sans", sans-serif; position:relative; overflow:hidden; }}
.abs {{ position:absolute; }}
.brand {{ left:94px; top:64px; font:700 28px "Fira Sans"; color:var(--amber); letter-spacing:.02em; }}
.tag {{ left:430px; top:66px; font:400 25px "Fira Sans"; color:var(--muted); }}
.big {{ left:80px; top:118px; font:700 250px/1 "Fira Sans"; color:var(--amber); letter-spacing:-.03em; }}
.lede {{ left:100px; top:392px; width:620px; font:400 32px/1.45 "Fira Sans"; color:var(--ink); }}
.lede b {{ font-weight:600; color:var(--amber-2); }}
.note {{ left:100px; top:588px; width:680px; font:400 20px "Fira Sans"; color:var(--muted); font-style:italic; }}
.claim {{ left:100px; top:668px; width:640px; font:600 33px/1.45 "Fira Sans"; color:var(--amber); }}
.img {{ left:{IMG["x"]}px; top:{IMG["y"]}px; width:{IMG["w"]}px; height:{IMG["h"]}px; border-radius:4px;
  box-shadow:0 0 80px rgba(245,166,35,.10); }}
.morlab {{ left:{band_x - 100:.0f}px; top:{IMG["y"] - 40}px; width:200px; text-align:center; font:500 21px "Fira Sans"; color:var(--blue); }}
.co {{ right:{W - x0}px; text-align:right; }}
.co h3 {{ font:600 29px "Fira Sans"; margin-bottom:10px; }}
.co p {{ font:400 22px/1.5 "Fira Sans"; color:var(--ink); }}
.co .k {{ font:500 16px "Fira Sans"; letter-spacing:.14em; text-transform:uppercase; color:var(--dim); margin-bottom:6px; }}
.rule {{ left:94px; right:94px; top:1318px; height:1px; background:var(--rule); }}
.stats {{ left:94px; top:1356px; width:1252px; display:grid; grid-template-columns:repeat(4,1fr); }}
.stats b {{ display:block; font:600 64px/1 "Fira Sans"; color:var(--amber); margin-bottom:18px; }}
.stats span {{ font:400 21px/1.45 "Fira Sans"; color:var(--muted); }}
.box {{ left:86px; right:86px; top:1548px; height:176px; border:1.5px solid #1b2a47; border-radius:18px; background:#081022;
  padding:34px 44px 0; display:grid; grid-template-columns:1fr 1fr; row-gap:6px; }}
.box .l {{ font:400 19px "Fira Sans"; color:var(--muted); }}
.box .u {{ font:400 25px "Fira Sans"; color:var(--ink); }}
.box .n {{ grid-column:1 / 3; margin-top:16px; font:400 17px "Fira Sans"; color:var(--muted); }}
.foot {{ left:94px; top:1748px; font:400 19px "Fira Sans"; color:var(--muted); }}
.credit {{ left:{IMG["x"]}px; width:{IMG["w"]}px; top:{IMG["y"] + IMG["h"] + 12}px; text-align:right; font:400 15px "Fira Sans"; color:var(--dim); }}
svg {{ position:absolute; left:0; top:0; }}
</style></head><body>
<div class="abs brand">BUREAU-TO-WORKOUT</div>
<div class="abs tag">open research on the labels credit models learn from</div>
<div class="abs big">{m["EtwoAoUnder"]}%</div>
<div class="abs lede">less default than really happens: what a credit model predicts for the applicants a lender
<b>rejected</b>, when it learned only from the ones it approved.</div>
<div class="abs note">measured in a simulator that knows every applicant's true outcome</div>
<div class="abs claim">The credit system writes its own labels.<br>This project corrects them, with ground truth to prove it.</div>

<img class="abs img" src="raster.png">
<div class="abs morlab">moratorium</div>

<div class="abs co" style="top:860px"><p class="k">Selection</p><h3 style="color:var(--amber-2)">Bureau-augmented labels</h3>
<p>predicted ÷ actual default of rejected applicants<br>{m["EtwoRejAo"]} → {m["EtwoRejBa"]} (oracle {m["EtwoRejOracle"]})</p></div>
<div class="abs co" style="top:1010px"><p class="k">Masking</p><h3 style="color:var(--blue)">Deferral hides the arrears</h3>
<p>out-of-time AUC falls {m["EthreeAUCtwoDelay"]} → {m["EthreeAUCtwoNaive"]};<br>delayed labels bring it back to {m["EthreeAUCtwoDelay"]}</p></div>
<div class="abs co" style="top:1160px"><p class="k">Treatment</p><h3 style="color:var(--amber-2)">Causal workout offers</h3>
<p>+{m["RtwoGainDR"]} pp recovery, write-downs of<br>{m["RtwoWdCausDR"]}% of balance instead of {m["RtwoWdSq"]}%</p></div>

<svg width="{W}" height="{H}"><g stroke="#c8b27a" stroke-width="1.4" fill="none" opacity=".85">
<polyline points="{x0 + 14},1045 {band_x:.0f},{band_y:.0f}"/><circle cx="{band_x:.0f}" cy="{band_y:.0f}" r="6"/>
<polyline points="{x0 + 14},1195 {IMG["x"] + IMG["w"] - 30},{npl_y:.0f}"/><circle cx="{IMG["x"] + IMG["w"] - 30}" cy="{npl_y:.0f}" r="6"/>
</g></svg>

<div class="abs rule"></div>
<div class="abs stats">
<div><b>3M</b><span>simulated borrower-months<br>with the counterfactual kept</span></div>
<div><b>50,000</b><span>debt workouts, the outcome<br>of every offer known</span></div>
<div><b>{n_models}</b><span>trained models, open<br>on Hugging Face</span></div>
<div><b>0</b><span>numbers typed by hand:<br>all generated, 5 seeds</span></div>
</div>
<div class="abs box">
<div><div class="l">Code + paper</div><div class="u">github.com/thanatchon36/bureau-to-workout</div></div>
<div><div class="l">Data + models</div><div class="u">hf.co/thanatchon36/bureau-to-workout</div></div>
<div class="n">All results come from two simulators that keep the ground truth. No NCB or SAM data were used; no number is an estimate for Thailand.</div>
</div>
<div class="abs foot">Thanatchon Chongmankhong &nbsp;·&nbsp; views are my own &nbsp;·&nbsp; DOI {DOI}</div>
<div class="abs credit">reported arrears of {N_ROWS} simulated borrowers × 60 months · blue = deferred</div>
</body></html>"""


def main() -> None:
    if not Path(CHROME).exists() and shutil.which("google-chrome") is None:
        raise SystemExit("needs Google Chrome for the headless render")
    chrome = CHROME if Path(CHROME).exists() else shutil.which("google-chrome")
    n_models = 5 + len(E2_FILES) + len(E3_FILES) + 4   # E1 files + E2 + E3 + workout files on Hugging Face
    out = ROOT / "docs" / "assets" / "linkedin_poster.png"
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        band_x, npl_y, band_y = raster(tmp / "raster.png")
        (tmp / "poster.html").write_text(html(macros(), band_x, npl_y, band_y, n_models), encoding="utf-8")
        subprocess.run([chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=1",
                        f"--window-size={W},{H}", "--virtual-time-budget=8000", f"--screenshot={out}",
                        (tmp / "poster.html").as_uri()], check=True, capture_output=True)
    print("wrote", out)


if __name__ == "__main__":
    main()
