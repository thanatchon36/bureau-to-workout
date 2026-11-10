---
license: cc-by-4.0
pretty_name: "bureau-to-workout: simulated credit-bureau panels and debt workouts with ground truth"
language:
  - en
size_categories:
  - 1M<n<10M
task_categories:
  - tabular-classification
  - tabular-regression
tags:
  - credit-scoring
  - credit-risk
  - reject-inference
  - selective-labels
  - forbearance
  - causal-inference
  - uplift-modeling
  - off-policy-evaluation
  - debt-restructuring
  - thailand
  - synthetic
configs:
  - config_name: scoring_e1
    default: true
    data_files:
      - split: train
        path: scoring/e1_dev.parquet
      - split: test
        path: scoring/e1_oot.parquet
  - config_name: scoring_e3_moratorium
    data_files:
      - split: train
        path: scoring/e3_dev.parquet
      - split: test
        path: scoring/e3_oot.parquet
  - config_name: workout_cases
    data_files: workout/cases.parquet
  - config_name: panel_clean
    data_files: bureau/panel_clean.parquet
  - config_name: panel_moratorium
    data_files: bureau/panel_moratorium.parquet
  - config_name: borrowers_clean
    data_files: bureau/borrowers_clean.parquet
  - config_name: borrowers_moratorium
    data_files: bureau/borrowers_moratorium.parquet
  - config_name: applications_clean
    data_files: bureau/applications_clean.parquet
  - config_name: applications_moratorium
    data_files: bureau/applications_moratorium.parquet
---

# bureau-to-workout: simulated credit-bureau panels and debt workouts with ground truth

> **Every file in this dataset is synthetic.** It comes from the two simulators of the paper, not from the
> National Credit Bureau (NCB) or Sukhumvit Asset Management (SAM). The simulators' parameters make the mechanisms
> visible at plausible magnitudes; nothing here is an estimate for Thailand. What the data offer that real data
> cannot is the ground truth: the outcomes of rejected applicants, the arrears a moratorium hides, and every
> borrower's response to every workout offer.

Data for the preprint *From Bureau to Workout: Credit Scoring and Causal Debt Resolution under Policy-Shaped
Labels* by Thanatchon Chongmankhong ([read online](https://thanatchon36.github.io/bureau-to-workout/paper/) ·
[PDF](https://github.com/thanatchon36/bureau-to-workout/blob/main/paper/paper.pdf)). The paper studies three ways
the credit system writes its own labels, **selection** at approval, **masking** by deferral and **treatment** in
the workout, and proposes an estimator for each. Code: <https://github.com/thanatchon36/bureau-to-workout>;
archived on Zenodo: {{DOI_LINK}}. Models trained on these data:
[`thanatchon36/bureau-to-workout`](https://huggingface.co/thanatchon36/bureau-to-workout).

*Independent personal research. Does not represent the views of any employer or institution.*

```python
from datasets import load_dataset

scoring = load_dataset("thanatchon36/bureau-to-workout", "scoring_e1")          # train / test, out of time
workout = load_dataset("thanatchon36/bureau-to-workout", "workout_cases", split="train")
```

## Files

All draws are the paper's first seed, as `make experiments` makes them: the bureau without a moratorium with
seed 7, the bureau with a moratorium with seed 1007, the workouts with seed 107. Settings are in `configs.json`.

| path | content |
|---|---|
| `scoring/e1_dev.parquet`, `scoring/e1_oot.parquet` | the E1 modelling tables from the bureau without a moratorium: one row per scorable borrower (active, not yet 90+ DPD) at observation months 12–24 (train, {{N_E1_DEV}} rows) and 36–45 (test, after the shock; {{N_E1_OOT}} rows). 16 bureau attributes, `y_obs` (reported 90+ DPD within 12 months), `y_true` (the same without deferral), group columns and `thin_file` |
| `scoring/e3_dev.parquet`, `scoring/e3_oot.parquet` | the E3 tables from the bureau with a moratorium: training through it (months 15–24, {{N_E3_DEV}} rows) and testing after it (months 39–45, {{N_E3_OOT}} rows). Adds `y_delayed` (windows touching a deferral extended to six months after exit) and `mor_in_window` |
| `bureau/panel_{clean,moratorium}.parquet` | the account-month panels, long format: 50,000 borrowers × 60 months. `s_obs` reported delinquency state (0 current, 1 30 DPD, 2 60 DPD, 3 90+ DPD, absorbing), `s_true` the state without deferral, `mor` deferred this month, `util`, `n_acc`, `n_lenders`, `limit` (THB), `inq`, `new_acc` |
| `bureau/borrowers_*.parquet` | static attributes: age at month 0, gender, region, occupation, `new_to_credit_at_start`; `ability` and `soft` are **latent** (lenders' soft information), for evaluation only |
| `bureau/applications_*.parquet` | every credit application: borrower, month, lender (0 is the lender studied in E2, 5 the lenient lender), approved ({{N_APPS_CLEAN}} and {{N_APPS_MOR}} rows) |
| `workout/cases.parquet` | 50,000 NPL cases: 13 covariates from the AMC file and the bureau, `balance`, the logged `offer` (0 standard, 1 term extension, 2 write-down, 3 both), realised `recovery` (present value ÷ balance), `officer`, and the truth for every offer k: `propensity_k`, `value_k` (expected recovery), `p_complete_k`, `spill_k` (new 90+ DPD elsewhere within 24 months); latent `capacity` and `willingness`; `split` as in the paper's R1–R4 |
| `results/` | every result file of the paper, five seeds (`make experiments`): the CSVs behind each table and figure |
| `configs.json`, `versions.json` | simulator settings, and the library versions used to build this upload |
| `SHA256SUMS` | checksums of every file above |

## Tasks these data support

- **Reject inference / selective labels.** Train on approved applicants only and score everyone: the panel and
  applications tables reveal rejected applicants' outcomes at other lenders and, through `s_true`, the truth.
- **Label masking.** In `scoring_e3_moratorium`, compare models trained on `y_obs`, `y_delayed` and `y_true`, and
  test after the moratorium.
- **Heterogeneous treatment effects and policy learning.** In `workout_cases`, estimate offer effects from the
  logged offers, then check them against `value_k`; learn a write-down-budgeted policy and evaluate it exactly.
- **Off-policy evaluation.** Estimate a policy's value from logged data and compare with its true value, using
  `propensity_k` (true) or your own estimate.

## Licence

CC BY 4.0 (see `LICENSE`). The code that generates the data is MIT-licensed.

## Citation

```bibtex
@misc{chongmankhong2026bureau,
  author = {Chongmankhong, Thanatchon},
  title  = {From Bureau to Workout: Credit Scoring and Causal Debt Resolution under Policy-Shaped Labels},
  year   = {2026},{{DOI_BIB}}
  url    = {https://thanatchon36.github.io/bureau-to-workout/paper/},
  note   = {Preprint; all data are synthetic}
}
```
