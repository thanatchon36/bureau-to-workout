# From Bureau to Workout

Credit scoring and causal debt resolution under policy-shaped labels: the paper, two simulators with ground truth, and the code that produces every number in the paper.

**Paper:** [`paper/paper.pdf`](paper/paper.pdf) · Thanantchon Chongmankhong · November 2026 · preprint

![The debt lifecycle seen by a credit bureau linked to an asset-management company](paper/figures/fig1_lifecycle.png)

Credit-risk models learn from labels that the credit system itself produces. Lenders see repayment only for applicants they approve, debt-relief programmes suspend the reporting of arrears, and collectors decide which distressed borrowers receive concessions. This repository studies these three mechanisms, **selection**, **masking** and **treatment**, along the debt lifecycle that a national credit bureau (NCB) linked to a public asset-management company (SAM) would reveal, and proposes an estimator for each.

> **All data here are synthetic.** No NCB or SAM data were used. Every number is a property of the simulators, whose parameters were chosen to make the mechanisms visible at plausible magnitudes. None is an estimate for Thailand.

## Key results

Simulation results, mean of five seeds.

| Mechanism | Metric | Standard practice | Proposed | Oracle |
| --- | --- | --- | --- | --- |
| Selection | Predicted ÷ observed default, rejected applicants | 0.55 (approved-only) | 0.86 (bureau-augmented) | 0.89 |
| Masking | Out-of-time AUC after a moratorium | 0.625 (reported labels) | 0.831 (delayed labels) | 0.838 |
| Treatment | Recovery value, % of balance | 55.55 (logged policy) | 57.22 (causal targeting) | 57.97 |
| Treatment | Write-downs, % of balance | 10.4 (logged policy) | 4.4 (causal targeting) | 2.9 |

## Repository layout

```
.
├── ncb_scoring_lab.py       bureau simulator, WoE scorecard, monotone GBM, SSL encoder; experiments E1–E5
├── sam_resolution_lab.py    workout simulator, DR-learner, budgeted offer policies, off-policy evaluation
├── paper_experiments.py     runs every experiment over five seeds and writes paper_results/
├── paper_tables.py          builds paper/tables/ and paper/results_macros.tex from paper_results/
├── paper_figures.py         builds paper/figures/ (PDF for LaTeX, PNG previews)
├── paper_results/           experiment outputs used by the tables and figures
├── paper/
│   ├── paper.pdf            compiled preprint
│   ├── main.tex             LaTeX source in arXiv preprint style
│   ├── references.bib, main.bbl, results_macros.tex
│   ├── tables/
│   └── figures/
├── requirements.txt
└── CITATION.cff
```

## Quick start

Requires Python 3.10 or later.

```bash
pip install -r requirements.txt
python ncb_scoring_lab.py --exp all     # one seed of E1–E5, about a minute on one CPU core
python sam_resolution_lab.py            # one seed of the workout experiments, a few seconds
```

## Reproduce the paper

```bash
python paper_experiments.py             # five seeds, about four minutes on one CPU core
python paper_tables.py
python paper_figures.py
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Every number in the paper's text comes from `paper/results_macros.tex`, which `paper_tables.py` generates, so no number in the paper is typed by hand.

## Experiments

| ID | Question | Code | Paper |
| --- | --- | --- | --- |
| E1 | Do self-supervised sequence embeddings beat a monotone GBM out of time? | `run_e1` in `ncb_scoring_lab.py` | Section 6.1 |
| E2 | How biased is approved-only training, and how much do bureau-observed outcomes repair? | `run_e2` | Section 6.2 |
| E3 | What does a moratorium do to labels, and which labelling strategy repairs it? | `run_e3` | Section 6.3 |
| E4 | Does calibration hold across groups after a regime shift? | `run_e4` | Section 6.4 |
| E5 | Scorecard points and adverse-action reason codes | `run_e5` | Appendix C |
| R1–R4 | Offer effects, budgeted offer policies, budget frontier, off-policy evaluation | `run_all` in `sam_resolution_lab.py` | Section 6.5 |

## Moving to real data

The estimators do not depend on the simulators. Replace `simulate_bureau()` with a loader that fills the same `BureauPanel` arrays from an account-month table, and `simulate_workouts()` with a loader for the workout file. Do this only inside an approved secure environment, with a lawful purpose under the Credit Information Business Act B.E. 2545 and the Personal Data Protection Act B.E. 2562, a data-protection impact assessment and ethics review. Section 7 of the paper sets out identification strategies for the linked data.

## Citation

```bibtex
@misc{chongmankhong2026bureau,
  author = {Chongmankhong, Thanantchon},
  title  = {From Bureau to Workout: Credit Scoring and Causal Debt Resolution under Policy-Shaped Labels},
  year   = {2026},
  note   = {Preprint}
}
```

## สรุปภาษาไทย

Repository นี้รวม paper และโค้ดอ้างอิงของงานวิจัย credit scoring บนข้อมูลแบบ NCB และโมเดลแก้หนี้ NPL บนข้อมูลแบบ บสส. งานศึกษา 3 กลไกที่ทำให้ label ในระบบสินเชื่อถูกบิด ได้แก่ การคัดเลือกผู้ได้รับอนุมัติ การพักหนี้ที่บดบังการค้างชำระ และข้อเสนอแก้หนี้ที่เปลี่ยนผลลัพธ์เอง พร้อมเสนอวิธีแก้สำหรับแต่ละกลไก ทุกตัวเลขมาจากข้อมูลจำลอง ไม่ได้ใช้ข้อมูลจริงของ NCB หรือ บสส. และไม่ใช่ค่าประมาณของประเทศไทย เริ่มอ่านได้ที่ `paper/paper.pdf` และรันซ้ำทุกตัวเลขได้ด้วยคำสั่งในหัวข้อ Reproduce the paper ซึ่งใช้เวลาราว 4 นาที
