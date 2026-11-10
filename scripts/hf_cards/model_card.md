---
license: mit
library_name: sklearn
pipeline_tag: tabular-classification
datasets:
  - thanatchon36/bureau-to-workout
tags:
  - credit-scoring
  - scorecard
  - reject-inference
  - causal-inference
  - uplift-modeling
  - dr-learner
  - off-policy-evaluation
  - debt-restructuring
  - skops
  - synthetic
---

# bureau-to-workout: credit-scoring and debt-workout models trained on simulated data

> **Trained on synthetic data only.** These models were fitted to the paper's simulators, not to NCB or SAM data.
> They reproduce the paper's experiments and serve as baselines for the
> [dataset](https://huggingface.co/datasets/thanatchon36/bureau-to-workout). **They are not credit models to
> score real people.** They have not been validated on real data, and the simulators' group effects are
> arbitrary design choices.

Models for the preprint *From Bureau to Workout: Credit Scoring and Causal Debt Resolution under Policy-Shaped
Labels* by Thanatchon Chongmankhong ([read online](https://thanatchon36.github.io/bureau-to-workout/paper/)).
Code: <https://github.com/thanatchon36/bureau-to-workout>; archived on Zenodo: {{DOI_LINK}}.

*Independent personal research. Does not represent the views of any employer or institution.*

## Files

All models come from the paper's first seed (bureau seeds 7 and 1007, workouts seed 107), fitted by the same
code as `make experiments`. scikit-learn models are saved with [skops](https://skops.readthedocs.io) (no pickle);
the scorecard is plain JSON.

| path | model | paper |
|---|---|---|
| `bureau/woe_scorecard.json`, `bureau/woe_points_table.csv` | weight-of-evidence scorecard: monotone bins, WoE, logistic fit, points (600 at odds 50:1, 20 points to double) | E1, E5 |
| `bureau/monotone_gbm.skops` | gradient-boosted classifier with monotone constraints on the 15 bureau attributes that vary without a moratorium | E1 |
| `bureau/ssl_encoder.skops` | masked self-supervised encoder of 24-month bureau sequences (scaler + MLP; the embedding is the bottleneck layer) | E1 |
| `bureau/ssl_linear_probe.skops`, `bureau/gbm_ssl.skops` | logistic probe on the embedding alone; GBM on attributes + embedding | E1 |
| `bureau/e2_{approved_only,approved_only_ipw,fuzzy,bureau_augmented,bureau_augmented_ipw,oracle}.skops` | monotone GBMs for reject inference at lender 0: approved applicants only, with IPW, classic fuzzy augmentation, bureau-augmented labels (rejects whose outcome the bureau sees), with IPW for unobserved rejects, and the oracle on every applicant's true outcome | E2 |
| `bureau/e3_{naive,exclude,delayed,oracle}.skops` | logistic models trained through a moratorium on reported labels, without deferred windows, on delayed labels, and on the no-deferral truth | E3 |
| `bureau/features.json` | feature lists, monotone constraints, embedding columns | |
| `workout/outcome_models.skops` | expected recovery under each of the four offers (list of 4 regressors; T-learner) | R1 |
| `workout/propensity.skops` | multinomial propensity model of the logged offers (scaler + logistic) | R1, R4 |
| `workout/dr_learner.skops` | cross-fitted DR-learner: effect of offers 1–3 against standard collection (dict of 3 regressors) | R1–R3 |
| `workout/predictive_score.skops` | P(recovery ≥ 60% under standard collection), the predictive-targeting baseline | R2 |
| `metrics.json`, `versions.json` | the metrics below in full; library versions | |

## Usage

```python
import json
import pandas as pd
import skops.io as sio
from huggingface_hub import snapshot_download
from bureau_to_workout.bureau import WoEScorecard   # pip install git+https://github.com/thanatchon36/bureau-to-workout

path = snapshot_download("thanatchon36/bureau-to-workout")
oot = pd.read_parquet(
    snapshot_download("thanatchon36/bureau-to-workout", repo_type="dataset") + "/scoring/e1_oot.parquet")
feats = json.load(open(f"{path}/bureau/features.json"))["e1_features"]

def load(file):
    untrusted = sio.get_untrusted_types(file=file)   # inspect this list before trusting it
    return sio.load(file, trusted=untrusted)

gbm = load(f"{path}/bureau/monotone_gbm.skops")
pd_gbm = gbm.predict_proba(oot[feats])[:, 1]

e2 = pd.read_parquet(
    snapshot_download("thanatchon36/bureau-to-workout", repo_type="dataset") + "/scoring/e2_test.parquet")
ba = load(f"{path}/bureau/e2_bureau_augmented.skops")
pd_ba = ba.predict_proba(e2[json.load(open(f"{path}/bureau/features.json"))["e2_features"]])[:, 1]

scorecard = WoEScorecard.from_dict(json.load(open(f"{path}/bureau/woe_scorecard.json")))
points = scorecard.score(oot[feats])
reasons = scorecard.reason_codes(oot[feats].head(3))

dr = load(f"{path}/workout/dr_learner.skops")        # {"1": ..., "2": ..., "3": ...}
```

## Metrics of these exact models

Out-of-time test (E1): months 36–45, after the shock. The paper column is the paper's seed-7 result. These
uploads were refitted with newer library versions ({{VERSIONS}}), which moves some numbers in the third decimal.

| model | OOT AUC | paper, seed 7 | PSI dev → OOT |
|---|---|---|---|
{{E1_ROWS}}

Reject inference (E2), test half of lender 0's applicants. Predicted ÷ observed default among **rejected**
applicants shows how far each model under-predicts the risk it never saw (ideal 1):

| training labels | AUC, all applicants | predicted ÷ observed, rejected | paper, seed 7 |
|---|---|---|---|
{{E2_ROWS}}

After a moratorium (E3), test months 39–45:

| labels used in training | OOT AUC | relative calibration of deferred borrowers (ideal 1) |
|---|---|---|
{{E3_ROWS}}

Workout policies at the status-quo write-down budget (R2), true values on the held-out half:

| policy | recovery, % of balance | write-downs, % of balance |
|---|---|---|
{{R2_ROWS}}

## Limitations

The simulators decide what these models can learn: in the bureau, repayment at other lenders is a perfect proxy
for the new loan; in the workouts, unconfoundedness and overlap hold by construction. The SSL encoder is a small
CPU stand-in for sequence models trained at bureau scale. See Section 8 of the paper.

## Citation

```bibtex
@misc{chongmankhong2026bureau,
  author = {Chongmankhong, Thanatchon},
  title  = {From Bureau to Workout: Credit Scoring and Causal Debt Resolution under Policy-Shaped Labels},
  year   = {2026},{{DOI_BIB}}
  url    = {https://thanatchon36.github.io/bureau-to-workout/paper/},
  note   = {Preprint; models trained on synthetic data}
}
```
