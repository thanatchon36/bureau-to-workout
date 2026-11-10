"""Build the web reader docs/paper/ from paper/main.tex (same text and numbers as the PDF).

pandoc renders the LaTeX body with paper/reader.html: masthead, abstract, a results strip whose numbers
come from paper/results_macros.tex (the same macros the PDF uses), contents, numbered sections, references
and the appendices. pandoc does not number figures, tables, equations or appendices the way LaTeX does,
so this module numbers them in source order (as LaTeX does) and rewrites \\cref, \\Cref and \\eqref to
linked text first. The PDF and the PNG previews of the figures are copied next to the page; GitHub Pages
serves docs/ at https://thanatchon36.github.io/bureau-to-workout/. Requires pandoc on PATH.
"""
from __future__ import annotations

import html
import re
import shutil
import string
import subprocess
from pathlib import Path

REPO_URL = "https://github.com/thanatchon36/bureau-to-workout"
PAPER_URL = "https://thanatchon36.github.io/bureau-to-workout/paper/"
DATA_URL = "https://huggingface.co/datasets/thanatchon36/bureau-to-workout"
MODEL_URL = "https://huggingface.co/thanatchon36/bureau-to-workout"
CONCEPT_DOI = "10.5281/zenodo.24102251"

# (label, style, finding, detail); \Macro names are resolved from results_macros.tex.
# style: "ref" = simulation result, "null" = pending.
RESULTS = [
    (
        "Selection · E2",
        "ref",
        r"Bureau-augmented labels: predicted ÷ observed default \EtwoRejBa{} on rejected applicants",
        r"approved-only \EtwoRejAo{} (under-prediction by \EtwoAoUnder%); oracle \EtwoRejOracle{}; fuzzy augmentation \EtwoRejFuzzy{}",
    ),
    (
        "Masking · E3",
        "ref",
        r"Delayed labels restore out-of-time AUC to \EthreeAUCDelay{} after a moratorium",
        r"reported labels \EthreeAUCNaive{}; oracle \EthreeAUCOracle{}; the naive model reads deferral as safety",
    ),
    (
        "Treatment · R2",
        "ref",
        r"Causal targeting: recovery \RtwoRecCausDR% of balance, write-downs \RtwoWdCausDR%",
        r"logged policy \RtwoRecSq% and \RtwoWdSq%; \RtwoShareDR% of the attainable gain against \RtwoSharePred% for predictive targeting",
    ),
    (
        "Benchmark · E1",
        "ref",
        r"Monotone GBM out-of-time AUC \EoneAUCGbm{} against \EoneAUCSc{} for the WoE scorecard",
        r"GBM + SSL embedding \EoneAUCGbmssl{} but PSI \EonePSIGbmssl{} against \EonePSIGbm{}",
    ),
    (
        "Off-policy · R4",
        "ref",
        r"Doubly robust OPE error \RfourDr{} pp against \RfourDm{} (DM) and \RfourIps{} (IPS)",
        r"mean absolute error over \RfourNpoints{} policy-budget points",
    ),
    (
        "NCB × SAM data",
        "null",
        "Real-data results pending",
        "every number here comes from the simulators; the estimators run unchanged on the linked data",
    ),
]

CREF = {"fig": "Figure", "tab": "Table", "sec": "Section", "app": "Appendix", "eq": "Equation", "alg": "Algorithm"}


def _macros(paper_dir: Path) -> dict[str, str]:
    text = (paper_dir / "results_macros.tex").read_text(encoding="utf-8")
    return {m.group(1): m.group(2) for m in re.finditer(r"\\newcommand\{\\(\w+)\}\{(.*)\}\s*$", text, re.M)}


def _resolve(s: str, macros: dict[str, str]) -> str:
    """\\Macro{} or \\Macro -> its value, with LaTeX math minus signs made plain."""
    def one(m: re.Match) -> str:
        if m.group(1) not in macros:
            raise RuntimeError(f"results strip uses unknown macro \\{m.group(1)}")
        return macros[m.group(1)].replace("$-$", "−")
    return re.sub(r"\\(\w+)(?:\{\})?", one, s)


def _numbers(tex: str) -> dict[str, str]:
    """label -> number as LaTeX prints it: sections 1.., appendices A.., floats and equations in source order."""
    out: dict[str, str] = {}
    body, _, appendix = tex.partition("\\appendix")
    for k, m in enumerate(re.finditer(r"\\section\{[^}]*\}\\label\{([^}]+)\}", body), 1):
        out[m.group(1)] = str(k)
    for k, m in enumerate(re.finditer(r"\\section\{[^}]*\}\\label\{([^}]+)\}", appendix)):
        out[m.group(1)] = string.ascii_uppercase[k]
    for env in ("figure", "table", "algorithm", "equation"):
        blocks = re.findall(rf"\\begin\{{{env}\}}(.*?)\\end\{{{env}\}}", tex, re.S)
        for k, block in enumerate(blocks, 1):
            label = re.search(r"\\label\{([^}]+)\}", block)
            if label:
                out[label.group(1)] = str(k)
    return out


def _algorithmic(block: str) -> str:
    """algorithm environment -> a captioned block of plain LaTeX lines that pandoc keeps (it drops algorithmic)."""
    caption = re.search(r"\\caption\{(.*?)\}\\label", block, re.S).group(1)
    lines, depth = [], 0
    body = re.search(r"\\begin\{algorithmic\}(?:\[\d\])?(.*?)\\end\{algorithmic\}", block, re.S).group(1)
    for raw in body.strip().splitlines():
        line = raw.strip()
        line = re.sub(r"\\Comment\{(.*)\}$", r"\\quad\\emph{// \1}", line)
        if line.startswith(("\\EndFor", "\\EndIf")):
            depth -= 1
            continue
        for cmd, word in (("\\Require", r"\textbf{Require} "), ("\\State", ""), ("\\Return", r"\textbf{return} ")):
            if line.startswith(cmd):
                text = word + line[len(cmd):].strip()
                break
        else:
            m = re.match(r"\\(For|If)\{(.*)\}$", line)
            if not m:
                raise RuntimeError(f"unhandled algorithmic line: {line}")
            word = "for" if m.group(1) == "For" else "if"
            text = rf"\textbf{{{word}}} {m.group(2)} \textbf{{{'do' if word == 'for' else 'then'}}}"
        lines.append("~" * (4 * depth) + text)
        if line.startswith(("\\For", "\\If")):
            depth += 1
    return (
        "\\begin{quote}\n\\textbf{Algorithm ALGNUM.} " + caption.strip() + "\n\n"
        + "\n\n".join(lines) + "\n\\end{quote}\n"
    )


def _prepare(tex: str, numbers: dict[str, str]) -> tuple[str, str, list[str]]:
    """main.tex -> (body for pandoc, abstract LaTeX, keywords)."""
    abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S).group(1).strip()
    kw = re.search(r"\\textbf\{\\textit\{Keywords\}\}\\enspace (.*?)\}\s*$", tex, re.M).group(1)
    keywords = [k.strip() for k in kw.split("$\\cdot$")]
    body = tex[tex.index("\\section{"):tex.index("\\end{document}")]
    body = body.replace("\\appendix", "")
    body = re.sub(r"\\bibliographystyle\{[^}]*\}\s*\{\\small\\bibliography\{[^}]*\}\}", "BIBLIOGRAPHY-HERE", body)
    body = re.sub(r"(\\includegraphics(?:\[[^]]*\])?\{figures/[\w-]+)\.pdf\}", r"\1.png}", body)

    def ref(m: re.Match) -> str:
        cmd, label = m.group(1), m.group(2)
        n = numbers[label]
        if cmd == "eqref":
            return rf"\hyperref[{label}]{{({n})}}"
        return rf"\hyperref[{label}]{{{CREF[label.split(':')[0]]}~{n}}}"
    body = re.sub(r"\\(cref|Cref|eqref)\{([^},]+)\}", ref, body)
    if re.search(r"\\(cref|Cref|eqref)\{", body):
        raise RuntimeError("a \\cref with several labels is not handled")

    body = re.sub(r"\\begin\{algorithm\}.*?\\end\{algorithm\}",
                  lambda m: _algorithmic(m.group(0)).replace("ALGNUM", numbers["alg:budget"]), body, flags=re.S)
    return body, abstract, keywords


def _pandoc(text: str, paper_dir: Path, *extra: str) -> str:
    return subprocess.run(
        ["pandoc", "--from=latex", "--to=html5", "--mathml", "--wrap=none", *extra],
        input=text, capture_output=True, text=True, check=True, cwd=str(paper_dir),
    ).stdout


def _number_captions(page: str) -> str:
    """Table and figure numbers in captions, in document order (the PDF numbers them the same way)."""
    count = {"caption": 0, "figcaption": 0}

    def caption(m: re.Match) -> str:
        tag, attrs, para = m.group(1), m.group(2), m.group(3) or ""
        count[tag] += 1
        word = "Table" if tag == "caption" else "Figure"
        return f"<{tag}{attrs}>{para}<b>{word} {count[tag]}.</b> "

    return re.sub(r"<(caption|figcaption)(\b[^>]*)>(\s*<p>)?", caption, page)


def _web_images(page: str) -> str:
    """Drop the PDF sizing (width=100% etc.) and load lazily."""
    def img(m: re.Match) -> str:
        tag = re.sub(r'\s(?:style|width|height)="[^"]*"', "", m.group(0))
        return tag.replace("<img ", '<img loading="lazy" ', 1)

    return re.sub(r"<img\b[^>]*>", img, page)


def _equations(page: str, numbers: dict[str, str]) -> str:
    """Give each labelled display equation its anchor and its number, as in the PDF."""
    def eq(m: re.Match) -> str:
        label = re.search(r"\\label\{([^}]+)\}", m.group(0))
        if not label:
            return m.group(0)
        n = numbers[label.group(1)]
        return (f'<span class="eq" id="{label.group(1)}">{m.group(0)}'
                f'<span class="eqno">({n})</span></span>')
    return re.sub(r'<math display="block".*?</math>', eq, page, flags=re.S)


def _appendix_letters(page: str, numbers: dict[str, str]) -> str:
    """pandoc numbers appendices on from the last section; LaTeX letters them A, B, ..."""
    for label, letter in numbers.items():
        if not label.startswith("app:"):
            continue
        page = re.sub(
            rf'(<h1 data-number=")\d+(" id="{re.escape(label)}"><span class="header-section-number">)\d+',
            rf"\g<1>{letter}\g<2>{letter}", page,
        )
        page = re.sub(
            rf'(<a href="#{re.escape(label)}"[^>]*><span class="toc-section-number">)\d+', rf"\g<1>{letter}", page
        )
    return page


def build_web(paper_dir: Path, out_dir: Path) -> Path:
    """main.tex -> out_dir/index.html (pandoc, reader.html), plus paper.pdf and the figure previews."""
    if shutil.which("pandoc") is None:
        raise RuntimeError("pandoc not found; install pandoc to build the web reader")
    tex = (paper_dir / "main.tex").read_text(encoding="utf-8")
    macros = _macros(paper_dir)
    numbers = _numbers(tex)
    body, abstract, keywords = _prepare(tex, numbers)
    preamble = (paper_dir / "results_macros.tex").read_text(encoding="utf-8") + "\n" + "\n".join(
        re.findall(r"^\\newcommand\{\\(?:Exp|Prob|ind)\}.*$", tex, re.M)
    ) + "\n"
    title = re.search(r"\\title\{(.*?)\}\n", tex).group(1)
    author = re.search(r"\\author\{(.*?)\}", tex).group(1)
    date = re.search(r"\{\\normalsize (.*?)\\par\}", tex).group(1)

    page = _pandoc(
        preamble + body, paper_dir,
        "--standalone", "--template=reader.html", "--number-sections", "--toc", "--toc-depth=1",
        "--citeproc", "--bibliography=references.bib", "--metadata=link-citations:true",
        f"--metadata=title:{title}", f"--metadata=author:{author}", f"--metadata=date:{date}",
        *[f"--metadata=keywords:{k}" for k in keywords],
        "--metadata=pagetitle:From Bureau to Workout",
        "--metadata=description:Preprint: credit scoring and causal debt resolution under policy-shaped labels, "
        "with two simulators that keep the ground truth.",
    )
    abstract_html = _pandoc(preamble + abstract, paper_dir)

    # The bibliography goes where the PDF prints it: after the conclusion, before the appendices.
    refs = re.search(r'<div id="refs".*?</div>\s*</div>\s*(?=</article>)', page, re.S)
    if not refs or "BIBLIOGRAPHY-HERE" not in page:
        raise RuntimeError("could not place the bibliography")
    page = page.replace(refs.group(0), "")
    page = re.sub(r"<p>BIBLIOGRAPHY-HERE</p>", lambda _: '<h1 class="unnumbered" id="references">References</h1>\n'
                  + refs.group(0), page)

    # the algorithm is a block quote in the reader; give it the anchor that "Algorithm 1" links to
    alg = f"<blockquote>\n<p><strong>Algorithm {numbers['alg:budget']}.</strong>"
    if alg not in page:
        raise RuntimeError("could not find the algorithm block")
    page = page.replace(alg, alg.replace("<blockquote>", '<blockquote class="algorithm" id="alg:budget">'))
    page = _number_captions(page)
    page = _web_images(page)
    page = _equations(page, numbers)
    page = _appendix_letters(page, numbers)
    # The caption sits above the table, outside its sideways scroll, so it wraps at the text width.
    page = re.sub(
        r"<table\b[^>]*>\s*<caption>(.*?)</caption>(.*?)</table>",
        r'<div class="tbl"><p class="tcap">\1</p><div class="tscroll"><table>\2</table></div></div>',
        page, flags=re.DOTALL,
    )
    # Column widths are sized for the PDF page; in the browser the tables size to their content and scroll.
    page = re.sub(r"<colgroup>.*?</colgroup>\s*", "", page, flags=re.DOTALL)
    text = re.sub(r"<annotation\b.*?</annotation>", "", page.split("<article", 1)[1], flags=re.S)  # TeX on purpose
    leftover = re.findall(r'data-reference="[^"]*">\[[^]]*\]|\\[A-Za-z]+\{', text)
    if leftover:
        raise RuntimeError(f"unresolved LaTeX in the web reader: {leftover[:5]}")

    items = "\n".join(
        f'<li class="r r-{style}"><span class="r-id">{html.escape(label)}</span>'
        f'<span class="r-text">{html.escape(_resolve(finding, macros))}'
        f'<small>{html.escape(_resolve(detail, macros))}</small></span></li>'
        for label, style, finding, detail in RESULTS
    )
    actions = (
        '<li><a href="paper.pdf">Download PDF</a></li>'
        f'<li><a href="{REPO_URL}">Code and replication</a></li>'
        f'<li><a href="{DATA_URL}">Data (Hugging Face)</a></li>'
        f'<li><a href="{MODEL_URL}">Models (Hugging Face)</a></li>'
        f'<li><a href="https://doi.org/{CONCEPT_DOI}">Zenodo archive</a></li>'
    )
    bib = (
        "@misc{chongmankhong2026bureau,\n"
        "  author = {Chongmankhong, Thanatchon},\n"
        f"  title  = {{{title}}},\n"
        "  year   = {2026},\n"
        "  note   = {Preprint},\n"
        f"  doi    = {{{CONCEPT_DOI}}},\n"
        f"  url    = {{{PAPER_URL}}}\n"
        "}"
    )
    cite = f"<h2>How to cite</h2><pre><code>{html.escape(bib)}</code></pre>"
    colophon = (
        "<p>Rendered from <code>paper/main.tex</code> by <code>bureau_to_workout.paper_web</code> "
        "(<code>make web</code>). Every number comes from <code>paper/results_macros.tex</code>, the macros the "
        "PDF uses, which <code>make tables</code> generates from <code>paper_results/</code>. The PDF is "
        "authoritative. All data are synthetic.</p>"
    )
    for marker, value in (
        ("<!--ABSTRACT-->", abstract_html),
        ("<!--ACTIONS-->", actions),
        ("<!--RESULTS-NOTE-->", (
            "Headline numbers, read from the macros the paper uses. Simulation results, mean of five seeds; "
            "they show what each estimator recovers, not estimates for Thailand. Amber: simulation; grey: pending."
        )),
        ("<!--RESULTS-->", items),
        ("<!--CITE-->", cite),
        ("<!--COLOPHON-->", colophon),
    ):
        if marker not in page:
            raise RuntimeError(f"reader.html lacks {marker}")
        page = page.replace(marker, value)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "index.html").write_text(page, encoding="utf-8")
    pdf = paper_dir / "paper.pdf"
    if pdf.exists():
        shutil.copyfile(pdf, out_dir / "paper.pdf")
    (out_dir / "figures").mkdir(exist_ok=True)
    for name in sorted(set(re.findall(r'src="figures/([\w.-]+)"', page))):
        shutil.copyfile(paper_dir / "figures" / name, out_dir / "figures" / name)
    return out_dir / "index.html"
