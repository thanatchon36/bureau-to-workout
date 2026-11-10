# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-11-10

### Added
- Bureau simulator (`bureau_to_workout.bureau`): 50,000 borrowers over 60 months with six lenders, a macro shock
  and an optional moratorium, keeping the counterfactual no-deferral world and lenders' soft information so every
  method is scored against the truth.
- Scoring models: weight-of-evidence scorecard with monotone binning and points, monotone gradient boosting, and a
  masked self-supervised sequence encoder (a CPU stand-in for a bureau foundation model).
- Experiments E1 to E5: out-of-time benchmark, selective labels and reject inference (approved-only, IPW, fuzzy
  augmentation, bureau-augmented labels, oracle), label masking by a moratorium (naive, exclude, delayed labels,
  oracle), group-wise calibration, and scorecard points with adverse-action reason codes.
- Workout simulator (`bureau_to_workout.workout`): 50,000 NPL cases and four offers whose effects run through
  liquidity and debt overhang, with officer-driven logged offers and every potential outcome kept.
- Experiments R1 to R4: T-learner against cross-fitted DR-learner, write-down-budgeted offer policies (status quo,
  uniform, predictive, causal, oracle), the budget frontier, and off-policy evaluation (DM, IPS, DR).
- Five-seed protocol (`make experiments`) writing `paper_results/`; generated tables, number macros and figures
  (`make tables`, `make figures`), so no number in the paper is typed by hand.
- The preprint *From Bureau to Workout: Credit Scoring and Causal Debt Resolution under Policy-Shaped Labels*
  (`paper/paper.pdf`, LaTeX source `paper/main.tex`) and a web reader in `docs/paper/`, served by GitHub Pages
  (`make web`).
- Package layout with `pyproject.toml`, a Makefile, tests for the simulators, the budgeted policy, off-policy
  evaluation and the regeneration of the paper's tables, and CI.
- README banner (`docs/assets/banner.png`) drawn from bureau-simulator output (`make banner`).
- Dataset on Hugging Face, [`thanatchon36/bureau-to-workout`](https://huggingface.co/datasets/thanatchon36/bureau-to-workout): the simulated bureau panels, the E1 and
  E3 modelling tables, the workout cases with every potential outcome, and every result file (CC BY 4.0).
- Models on Hugging Face, [`thanatchon36/bureau-to-workout`](https://huggingface.co/thanatchon36/bureau-to-workout): the scorecard as JSON and the scikit-learn
  models as skops files, with a card that reports their metrics (`make hf`, `make hf-push`).
- `WoEScorecard.to_dict()` and `WoEScorecard.from_dict()`, so the scorecard travels without pickle.
