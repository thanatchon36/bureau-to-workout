"""
bureau_to_workout.bureau
========================
Reference research code for Track A of the NCB x SAM research agenda:
credit scoring on bureau data whose labels are shaped by lender decisions and by policy.

Experiments (all on synthetic data generated in this file):
  E1  Out-of-time benchmark: WoE scorecard vs monotone GBM vs GBM + self-supervised sequence
      embeddings (a CPU stand-in for the bureau foundation model, topic A1)
  E2  Selective labels / reject inference: approved-only, IPW, fuzzy augmentation,
      bureau-augmented labels, oracle (topic A2)
  E3  Label masking by a debt moratorium: naive, exclude, delayed labels, oracle (topic A3)
  E4  Group-wise calibration and approval report (topic A4)
  E5  Scorecard points table and adverse-action reason codes

Why a simulator: it knows the truth (latent defaults, lenders' soft information, counterfactual
labels without the moratorium), so every method is scored against ground truth, which real data
never allows. Group effects inside the simulator are arbitrary design choices, not claims about
real Thai borrowers. No real NCB or SAM data is used or needed. To move to real data, replace
simulate_bureau() with a loader that fills the same BureauPanel arrays from the bureau's
account-month table.

Requires numpy, pandas, scikit-learn >= 1.4.
Run:  make bureau   (python -m bureau_to_workout.cli bureau --exp all --n 50000 --seed 7 --out lab_outputs)
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# =============================================================================================
# 1. Synthetic bureau
# =============================================================================================

@dataclass
class SimConfig:
    n_borrowers: int = 50_000
    n_months: int = 60
    n_lenders: int = 6          # lender 0 = "our" lender in E2 (30% share), last lender = lenient lender
    burn_in: int = 12           # months simulated before the panel starts, so month 0 is not a cold start
    shock_start: int = 24       # first month of a macro shock (COVID-like)
    shock_end: int = 35         # last month of the shock
    moratorium: bool = True     # deferral that freezes reported delinquency for enrolled borrowers
    exit_window: int = 6        # months of catch-up payment stress after the moratorium ends
    seed: int = 7


@dataclass
class BureauPanel:
    cfg: SimConfig
    borrowers: pd.DataFrame     # static attributes; 'ability' and 'soft' are latent (evaluation only)
    s_true: np.ndarray          # (N,T) delinquency state 0..3 in the counterfactual world without a moratorium
    s_obs: np.ndarray           # (N,T) delinquency state in the observed world (what lenders report)
    mor: np.ndarray             # (N,T) True while the borrower is reported as deferred
    util: np.ndarray            # (N,T) revolving utilisation
    n_acc: np.ndarray           # (N,T) active accounts
    n_lenders: np.ndarray       # (N,T) distinct lenders with an active account
    limit: np.ndarray           # (N,T) total credit limit, THB
    inq: np.ndarray             # (N,T) applications (= inquiries) in the month
    new_acc: np.ndarray         # (N,T) accounts opened in the month
    first_open: np.ndarray      # (N,) panel month the first account was opened (< 0 = before the panel)
    miss: np.ndarray            # (N,T) latent missed-payment draws, observed world
    cure: np.ndarray            # (N,T) latent cure draws
    applications: pd.DataFrame  # one row per application: borrower, month, lender, approved


def _step(s, miss, cure):
    """Delinquency transition. 0 current, 1 = 30 DPD, 2 = 60 DPD, 3 = 90+ DPD (NPL, absorbing)."""
    worse = np.minimum(s + 1, 3)
    cured = np.where((s > 0) & (s < 3) & cure, 0, s)
    return np.where(miss, worse, cured).astype(np.int8)


def _popcount(bits, width):
    out = np.zeros(bits.shape, dtype=np.int16)
    for k in range(width):
        out += ((bits >> k) & 1).astype(np.int16)
    return out


def simulate_bureau(cfg: SimConfig) -> BureauPanel:
    """Monthly bureau panel with multi-lender applications, soft information, a macro shock and an
    optional moratorium. Two delinquency paths share the same random draws: s_obs (observed world)
    and s_true (counterfactual world without deferral), so masking can be measured exactly."""
    rng = np.random.default_rng(cfg.seed)
    N, T, L, B = cfg.n_borrowers, cfg.n_months, cfg.n_lenders, cfg.burn_in

    # --- static attributes (group effects are arbitrary simulation choices) ------------------
    age = rng.integers(21, 66, N)
    gender = rng.choice(np.array(["F", "M"]), N)
    region = rng.choice(np.array(["BKK", "Central", "North", "Northeast", "South"]), N,
                        p=[0.25, 0.25, 0.15, 0.22, 0.13])
    occupation = rng.choice(np.array(["salaried", "self_employed", "informal"]), N, p=[0.5, 0.25, 0.25])
    is_sal, is_self = occupation == "salaried", occupation == "self_employed"
    ability = rng.normal(np.select([is_sal, is_self], [0.3, 0.0], -0.3) + 0.01 * (age - 40), 1.0)
    soft = rng.normal(0.0, 1.0, N)                        # seen by loan officers, never by the bureau
    vol = np.select([is_sal, is_self], [0.6, 0.9], 1.1)   # income volatility
    vuln = np.select([is_sal, is_self], [0.7, 1.2], 1.4)  # sensitivity to the macro shock
    new_to_credit = rng.random(N) < np.clip(0.10 + 0.25 * (age < 30) + 0.15 * (occupation == "informal"), 0, 0.8)
    thin = new_to_credit | (rng.random(N) < 0.10)

    # --- initial credit book --------------------------------------------------------------------
    n_acc = np.where(new_to_credit, 0, np.where(thin, 1, 1 + rng.poisson(1.6, N))).astype(np.int16)
    hist = np.where(thin, rng.integers(1, 13, N), rng.integers(12, 121, N))
    first_open = np.where(n_acc > 0, -B - hist, 10_000).astype(np.int32)   # 10_000 = no account yet
    lender_bits = np.zeros(N, dtype=np.int64)
    limit = np.zeros(N)
    for k in range(int(n_acc.max())):
        has_k = n_acc > k
        lender_bits |= np.where(has_k, np.left_shift(1, rng.integers(0, L, N)), 0)
        limit += np.where(has_k, np.exp(rng.normal(np.log(40_000) + 0.3 * ability, 0.6)), 0.0)
    util_target = np.clip(sigmoid(-0.5 * ability) + rng.normal(0, 0.15, N), 0.02, 1.0)
    U = np.where(n_acc > 0, np.clip(util_target + rng.normal(0, 0.1, N), 0, 1.2), 0.0)

    # --- lenders: market share, weight on soft information, approval rate --------------------
    share = np.r_[0.30, np.full(L - 2, 0.56 / (L - 2)), 0.14]
    lam = np.r_[0.9, np.full(L - 2, 0.6), 0.1]
    target_rate = np.r_[0.50, np.linspace(0.45, 0.60, L - 2), 0.85]

    macro = np.zeros(T)
    macro[cfg.shock_start: cfg.shock_end + 1] = 1.0
    macro[cfg.shock_end + 1: cfg.shock_end + 1 + cfg.exit_window] = 0.3

    s_true = np.zeros((N, T), np.int8); s_obs = np.zeros((N, T), np.int8); mor = np.zeros((N, T), bool)
    util = np.zeros((N, T), np.float32); n_acc_p = np.zeros((N, T), np.int16)
    n_len_p = np.zeros((N, T), np.int16); limit_p = np.zeros((N, T), np.float32)
    inq = np.zeros((N, T), np.int8); new_acc = np.zeros((N, T), np.int8)
    miss_p = np.zeros((N, T), bool); cure_p = np.zeros((N, T), bool)

    st = np.zeros(N, np.int8); so = np.zeros(N, np.int8)
    z = rng.normal(0, 0.5, N)
    enrolled = np.zeros(N, bool)
    hist_rep = np.zeros((N, 6), np.int8)   # last 6 reported states, what lenders see
    hist_inq = np.zeros((N, 3), np.int8)   # last 3 months of inquiries
    apps = []

    for t in range(B + T):
        tm = t - B                          # panel month; < 0 during burn-in
        mac = macro[tm] if 0 <= tm < T else 0.0
        mac_prev = macro[tm - 1] if 1 <= tm <= T else 0.0
        has = n_acc > 0

        # (1) applications, decided on information up to the end of the previous month ---------
        if t >= 1:
            w6, inq3 = hist_rep.max(axis=1), hist_inq.sum(axis=1)
            p_now = sigmoid(-3.0 - 0.7 * ability - 0.5 * soft + 1.0 * z + 0.9 * mac_prev * vuln)
            apply = (rng.random(N) < 0.015 + 0.25 * p_now + 0.03 * ~has) & (so < 3)
            lender = rng.choice(L, N, p=share)
            risk_idx = 1.0 * w6 + 1.2 * U * has + 0.3 * inq3 - 0.2 * np.minimum(n_acc, 4) + 0.4 * ~has
            score = -risk_idx + lam[lender] * soft + rng.normal(0, 0.6, N)
            approve = np.zeros(N, bool)
            for l in range(L):                 # each lender approves its target share of applicants
                sel = apply & (lender == l)
                if sel.any():
                    approve |= sel & (score >= np.quantile(score[sel], 1 - target_rate[l]))
            first = approve & ~has
            first_open = np.where(first, tm, first_open)
            st = np.where(first, 0, st).astype(np.int8)          # a new obligation starts current
            so = np.where(first, 0, so).astype(np.int8)
            old_limit = limit
            limit = limit + np.where(approve, np.exp(rng.normal(np.log(30_000) + 0.3 * ability, 0.6)), 0.0)
            U = np.where(approve & has, U * old_limit / np.maximum(limit, 1.0), U)
            U = np.where(first, np.clip(util_target + rng.normal(0, 0.1, N), 0, 1.2), U)
            n_acc = (n_acc + approve).astype(np.int16)
            lender_bits = lender_bits | np.where(approve, np.left_shift(1, lender), 0)
            hist_inq = np.roll(hist_inq, 1, axis=1)
            hist_inq[:, 0] = apply
            if tm >= 0:
                inq[:, tm], new_acc[:, tm] = apply, approve
                ids = np.flatnonzero(apply)
                apps.append(np.column_stack([ids, np.full(len(ids), tm), lender[ids], approve[ids]]))
            has = n_acc > 0

        # (2) latent repayment dynamics -----------------------------------------------------------
        z = 0.85 * z + rng.normal(0, 0.5, N) * vol
        logit = -3.0 - 0.7 * ability - 0.5 * soft + 1.0 * z + 0.9 * mac * vuln + 0.6 * (U - 0.5) * has
        if cfg.moratorium and tm == cfg.shock_start:
            elig = has & (so < 3)
            zl = (logit - logit[elig].mean()) / logit[elig].std()
            enrolled = elig & (rng.random(N) < sigmoid(-1.0 + 1.6 * zl))   # riskier borrowers enrol more
        in_mor = enrolled & (cfg.shock_start <= tm <= cfg.shock_end)
        in_exit = enrolled & (cfg.shock_end < tm <= cfg.shock_end + cfg.exit_window)
        u01 = rng.random(N)                                   # common random numbers for both worlds
        miss_cf = u01 < sigmoid(logit)                        # counterfactual: no deferral, no catch-up
        miss_ob = u01 < sigmoid(logit + 0.7 * in_exit)        # observed: catch-up stress after exit
        cure = rng.random(N) < 0.5
        st = _step(st, miss_cf, cure)
        so = np.where(in_mor, so, _step(so, miss_ob, cure)).astype(np.int8)   # deferral freezes reporting

        U = np.clip(0.85 * U + 0.15 * util_target + 0.06 * miss_ob + 0.05 * np.maximum(z, 0) + 0.04 * mac
                    + rng.normal(0, 0.04, N), 0, 1.3)
        U = np.where(has, U, 0.0)
        close = has & (so < 3) & (rng.random(N) < 0.008 * n_acc)
        limit = np.where(close, limit * (n_acc - 1) / np.maximum(n_acc, 1), limit)
        n_acc = (n_acc - close).astype(np.int16)
        lender_bits = np.where(n_acc == 0, 0, lender_bits)
        hist_rep = np.roll(hist_rep, 1, axis=1)
        hist_rep[:, 0] = np.where(n_acc > 0, so, 0)

        if tm >= 0:
            s_true[:, tm], s_obs[:, tm], mor[:, tm], util[:, tm] = st, so, in_mor, U
            n_acc_p[:, tm], limit_p[:, tm] = n_acc, limit
            n_len_p[:, tm] = np.minimum(_popcount(lender_bits, L), n_acc)
            miss_p[:, tm], cure_p[:, tm] = miss_ob, cure

    applications = pd.DataFrame(np.vstack(apps), columns=["borrower", "month", "lender", "approved"])
    borrowers = pd.DataFrame({"age": age, "gender": gender, "region": region, "occupation": occupation,
                              "new_to_credit_at_start": new_to_credit, "ability": ability, "soft": soft})
    return BureauPanel(cfg, borrowers, s_true, s_obs, mor, util, n_acc_p, n_len_p, limit_p, inq, new_acc,
                       first_open, miss_p, cure_p, applications)


# =============================================================================================
# 2. Bureau attributes, sequences and labels
# =============================================================================================

FEATURES = ["n_acc", "n_lenders", "months_on_file", "log_limit", "util", "util_avg_6m", "util_trend_6m",
            "worst_dpd_3m", "worst_dpd_6m", "worst_dpd_12m", "n_delinq_12m", "months_since_delinq",
            "inq_3m", "inq_6m", "new_acc_6m", "moratorium_24m"]

# Monotone constraints for the GBM: +1 risk may only rise with the feature, -1 only fall, 0 free.
# The moratorium flag is deliberately left free so that E3 can show which sign the model learns.
MONOTONE = {"months_on_file": -1, "util": 1, "util_avg_6m": 1, "util_trend_6m": 1, "worst_dpd_3m": 1,
            "worst_dpd_6m": 1, "worst_dpd_12m": 1, "n_delinq_12m": 1, "months_since_delinq": -1,
            "inq_3m": 1, "inq_6m": 1}

SEQ_LEN = 24


def _rolling(a, w, how):
    out = np.empty(a.shape, dtype=np.float32)
    for t in range(a.shape[1]):
        blk = a[:, max(0, t - w + 1): t + 1]
        out[:, t] = blk.max(axis=1) if how == "max" else (blk.sum(axis=1) if how == "sum" else blk.mean(axis=1))
    return out


def build_feature_cube(p: BureauPanel) -> dict:
    """Every bureau attribute as an (N, T) array: its value at the end of month t, using months <= t."""
    has = p.n_acc > 0
    rep = np.where(has, p.s_obs, 0).astype(np.float32)   # the bureau only sees reported states
    delinq = (rep >= 1).astype(np.float32)
    N, T = rep.shape
    since = np.empty((N, T), np.float32)
    last = np.full(N, -10_000)
    for t in range(T):
        last = np.where(delinq[:, t] > 0, t, last)
        since[:, t] = np.minimum(t - last, 36)
    util = np.where(has, p.util, 0.0).astype(np.float32)
    lag6 = np.concatenate([np.repeat(util[:, :1], 6, axis=1), util[:, :-6]], axis=1)
    inq = p.inq.astype(np.float32)
    return {
        "n_acc": p.n_acc.astype(np.float32),
        "n_lenders": p.n_lenders.astype(np.float32),
        "months_on_file": np.clip(np.arange(T)[None, :] - p.first_open[:, None], 0, 240).astype(np.float32),
        "log_limit": np.log1p(p.limit).astype(np.float32),
        "util": util,
        "util_avg_6m": _rolling(util, 6, "mean"),
        "util_trend_6m": util - lag6,
        "worst_dpd_3m": _rolling(rep, 3, "max"),
        "worst_dpd_6m": _rolling(rep, 6, "max"),
        "worst_dpd_12m": _rolling(rep, 12, "max"),
        "n_delinq_12m": _rolling(delinq, 12, "sum"),
        "months_since_delinq": since,
        "inq_3m": _rolling(inq, 3, "sum"),
        "inq_6m": _rolling(inq, 6, "sum"),
        "new_acc_6m": _rolling(p.new_acc.astype(np.float32), 6, "sum"),
        "moratorium_24m": _rolling(p.mor.astype(np.float32), 24, "max"),
    }


def sequence_matrix(p: BureauPanel, idx, t):
    """Last 24 months per borrower x 5 channels, flattened. Input for self-supervised pretraining."""
    has = p.n_acc > 0
    chans = [np.where(has, p.s_obs, 0) / 3.0, np.where(has, p.util, 0.0), np.minimum(p.n_acc, 8) / 8.0,
             np.minimum(p.inq, 3) / 3.0, p.mor.astype(np.float32)]
    months = np.asarray(t)[:, None] - np.arange(SEQ_LEN - 1, -1, -1)[None, :]
    valid = months >= 0
    m = np.clip(months, 0, None)
    rows = np.asarray(idx)[:, None]
    out = np.stack([np.where(valid, c[rows, m], 0.0) for c in chans], axis=2)
    return out.reshape(len(idx), -1).astype(np.float32)


def snapshot_dataset(p: BureauPanel, cube: dict, t0s, horizon=12, delayed_end=None) -> pd.DataFrame:
    """Scorable population at each observation month t0 (active, not yet 90+), features and labels.

    y_obs      reported 90+ within (t0, t0+horizon]            -- what a modeller normally uses
    y_true     90+ in the counterfactual world without deferral -- simulator-only oracle
    y_delayed  like y_obs, but windows that touch a deferral are extended to delayed_end
    """
    frames = []
    for t0 in t0s:
        idx = np.flatnonzero((p.n_acc[:, t0] > 0) & (p.s_obs[:, t0] < 3))
        win = slice(t0 + 1, t0 + 1 + horizon)
        y_obs = ((p.s_obs[idx, win] == 3) & (p.n_acc[idx, win] > 0)).any(axis=1)
        y_true = (p.s_true[idx, win] == 3).any(axis=1)
        mor_win = p.mor[idx, t0: t0 + 1 + horizon].any(axis=1)
        y_del = y_obs
        if delayed_end is not None:
            end = max(t0 + 1 + horizon, delayed_end + 1)
            y_del = np.where(mor_win, (p.s_obs[idx, t0 + 1:end] == 3).any(axis=1), y_obs)
        df = pd.DataFrame({f: cube[f][idx, t0] for f in FEATURES})
        df["borrower"], df["t0"] = idx, t0
        df["y_obs"], df["y_true"], df["y_delayed"] = y_obs.astype(int), y_true.astype(int), y_del.astype(int)
        df["mor_in_window"] = mor_win
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    attrs = p.borrowers.iloc[out["borrower"].to_numpy()].reset_index(drop=True)
    out["gender"], out["region"], out["occupation"] = attrs["gender"], attrs["region"], attrs["occupation"]
    age_now = attrs["age"].to_numpy() + out["t0"].to_numpy() / 12
    out["age_band"] = pd.cut(age_now, [0, 30, 45, 60, 200], right=False, labels=["<30", "30-44", "45-59", "60+"])
    out["thin_file"] = (out["n_acc"] <= 1) & (out["months_on_file"] < 24)
    return out


def potential_outcome(p: BureauPanel, borrower, month, horizon=12):
    """Y(1): would the applicant reach 90+ within `horizon` months if given a loan at `month`?
    Replays the borrower's latent repayment draws from a fresh (current) obligation."""
    b, m = np.asarray(borrower), np.asarray(month)
    s = np.zeros(len(b), np.int8)
    for k in range(horizon):
        s = _step(s, p.miss[b, m + k], p.cure[b, m + k])
    return (s == 3).astype(int)


# =============================================================================================
# 3. Models
# =============================================================================================

class WoEScorecard:
    """Classic bureau scorecard: monotone WoE binning -> logistic regression -> points.

    Binning starts from quantiles and merges small or order-violating neighbours until the bad
    rate is monotone. Features with IV < min_iv are dropped, then features whose coefficient has
    the wrong sign (a symptom of collinearity) are removed one at a time.
    Points: score = offset + factor * ln(odds_good), factor = PDO / ln 2.
    """

    def __init__(self, max_bins=8, min_frac=0.05, min_iv=0.02, pdo=20.0, base_score=600.0,
                 base_odds=50.0, C=1.0):
        self.max_bins, self.min_frac, self.min_iv = max_bins, min_frac, min_iv
        self.factor = pdo / np.log(2)
        self.offset = base_score - self.factor * np.log(base_odds)
        self.C = C

    def _edges(self, x, y):
        u = np.unique(x)
        if len(u) <= self.max_bins:
            edges = np.r_[-np.inf, (u[:-1] + u[1:]) / 2, np.inf]
        else:
            edges = np.r_[-np.inf, np.unique(np.quantile(x, np.linspace(0, 1, self.max_bins + 1)[1:-1])), np.inf]
        r = np.corrcoef(x, y)[0, 1]
        direction = 1.0 if (np.isnan(r) or r >= 0) else -1.0
        while len(edges) > 2:
            b = np.digitize(x, edges[1:-1], right=True)
            n = np.bincount(b, minlength=len(edges) - 1)
            bad = np.bincount(b, weights=y, minlength=len(edges) - 1)
            small = np.flatnonzero(n < self.min_frac * len(x))
            if len(small):
                i = small[0]
                edges = np.delete(edges, (i if i < len(n) - 1 else i - 1) + 1)
                continue
            rate = (bad + 0.5) / (n + 1.0)
            viol = np.flatnonzero(direction * np.diff(rate) < 0)
            if len(viol):
                edges = np.delete(edges, viol[0] + 1)
                continue
            break
        return edges

    def _bin_stats(self, x, y, edges):
        b = np.digitize(x, edges[1:-1], right=True)
        k = len(edges) - 1
        n = np.bincount(b, minlength=k).astype(float)
        bad = np.bincount(b, weights=y, minlength=k)
        good = n - bad
        G, B = good.sum(), bad.sum()
        woe = np.log((good + 0.5) / (G + 0.5)) - np.log((bad + 0.5) / (B + 0.5))
        iv = float(np.sum(((good + 0.5) / (G + 0.5) - (bad + 0.5) / (B + 0.5)) * woe))
        return n, bad, woe, iv

    def fit(self, X: pd.DataFrame, y):
        y = np.asarray(y, float)
        self.bins = {}
        for c in X.columns:
            x = X[c].to_numpy(float)
            if np.std(x) == 0:
                continue
            edges = self._edges(x, y)
            n, bad, woe, iv = self._bin_stats(x, y, edges)
            if len(edges) > 2 and iv >= self.min_iv:
                self.bins[c] = {"edges": edges, "woe": woe, "n": n, "bad": bad, "iv": iv}
        self.features = sorted(self.bins, key=lambda c: -self.bins[c]["iv"])
        self.dropped_wrong_sign = []
        while True:
            W = self._woe(X, self.features)
            self.lr = LogisticRegression(C=self.C, max_iter=2000).fit(W, y)
            coef = self.lr.coef_[0]
            if (coef <= 0).all():        # WoE rises with goodness, so every coefficient must be <= 0
                break
            worst = self.features[int(np.argmax(coef))]
            self.features.remove(worst)
            self.dropped_wrong_sign.append(worst)
        k = len(self.features)
        alpha = self.lr.intercept_[0]
        for j, c in enumerate(self.features):
            self.bins[c]["points"] = (-self.factor * coef[j] * self.bins[c]["woe"]
                                      + (self.offset - self.factor * alpha) / k)
        return self

    def _woe(self, X, cols):
        return np.column_stack([self.bins[c]["woe"][np.digitize(X[c].to_numpy(float),
                                self.bins[c]["edges"][1:-1], right=True)] for c in cols])

    def predict_proba(self, X):
        return self.lr.predict_proba(self._woe(X, self.features))[:, 1]

    def _points(self, X):
        return np.column_stack([self.bins[c]["points"][np.digitize(X[c].to_numpy(float),
                                self.bins[c]["edges"][1:-1], right=True)] for c in self.features])

    def score(self, X):
        return self._points(X).sum(axis=1)

    def reason_codes(self, X, k=3):
        """Adverse-action reasons: features with the largest shortfall from their maximum points."""
        pts = self._points(X)
        shortfall = np.array([self.bins[c]["points"].max() for c in self.features])[None, :] - pts
        order = np.argsort(-shortfall, axis=1)[:, :k]
        return [[self.features[j] for j in row if shortfall[i, j] > 0.5] for i, row in enumerate(order)]

    def to_dict(self) -> dict:
        """Everything needed to score, as plain JSON types (no pickle): bins, WoE, points and the logistic fit."""
        return {
            "features": list(self.features),
            "dropped_wrong_sign": list(self.dropped_wrong_sign),
            "factor": float(self.factor), "offset": float(self.offset),
            "coef": [float(c) for c in self.lr.coef_[0]], "intercept": float(self.lr.intercept_[0]),
            "bins": {c: {"edges": [float(e) for e in self.bins[c]["edges"]],
                         "woe": [float(v) for v in self.bins[c]["woe"]],
                         "points": [float(v) for v in self.bins[c]["points"]],
                         "n": [int(v) for v in self.bins[c]["n"]],
                         "bad": [float(v) for v in self.bins[c]["bad"]],
                         "iv": float(self.bins[c]["iv"])} for c in self.features},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WoEScorecard":
        sc = cls()
        sc.features, sc.dropped_wrong_sign = list(d["features"]), list(d["dropped_wrong_sign"])
        sc.factor, sc.offset = d["factor"], d["offset"]
        sc.bins = {c: {k: (np.asarray(v) if isinstance(v, list) else v) for k, v in b.items()}
                   for c, b in d["bins"].items()}
        sc.lr = LogisticRegression()
        sc.lr.classes_ = np.array([0.0, 1.0])
        sc.lr.coef_ = np.asarray([d["coef"]])
        sc.lr.intercept_ = np.asarray([d["intercept"]])
        return sc

    def points_table(self):
        rows = []
        for c in self.features:
            e, bs = self.bins[c]["edges"], self.bins[c]
            for i in range(len(e) - 1):
                rows.append({"feature": c, "iv": round(bs["iv"], 3), "bin": f"({e[i]:.3g}, {e[i + 1]:.3g}]",
                             "n": int(bs["n"][i]), "bad_rate": bs["bad"][i] / max(bs["n"][i], 1),
                             "woe": bs["woe"][i], "points": bs["points"][i]})
        return pd.DataFrame(rows)


def fit_gbm(X: pd.DataFrame, y, w=None, seed=0, min_leaf=200):
    cst = [MONOTONE.get(c, 0) for c in X.columns]
    m = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                       min_samples_leaf=min_leaf, l2_regularization=1.0, monotonic_cst=cst,
                                       early_stopping=True, validation_fraction=0.15, n_iter_no_change=25,
                                       random_state=seed)
    return m.fit(X, y, sample_weight=w)


class MaskedSequenceEncoder:
    """Self-supervised pretraining by masked-month reconstruction: a CPU stand-in for topic A1.

    A denoising MLP autoencoder rebuilds 24-month bureau sequences from copies in which a share
    of months is blanked out; the bottleneck activations become the borrower embedding.
    Production A1 swaps this for a hierarchical transformer (PyTorch, GPU) with masked-event and
    contrastive objectives. The interface stays the same: fit on unlabelled sequences, embed,
    feed a downstream model.
    """

    def __init__(self, hidden=(96, 16, 96), mask_frac=0.25, epochs=25, seed=0):
        self.hidden, self.mask_frac, self.epochs, self.seed = hidden, mask_frac, epochs, seed

    def fit(self, S):
        rng = np.random.default_rng(self.seed)
        self.scaler = StandardScaler().fit(S)
        Z = self.scaler.transform(S)
        n, d = Z.shape
        C = d // SEQ_LEN
        Zin = Z.reshape(n, SEQ_LEN, C).copy()
        Zin[rng.random((n, SEQ_LEN)) < self.mask_frac] = 0.0
        self.net = MLPRegressor(hidden_layer_sizes=self.hidden, activation="relu", alpha=1e-4, batch_size=512,
                                learning_rate_init=2e-3, max_iter=self.epochs, random_state=self.seed)
        self.net.fit(Zin.reshape(n, d), Z)
        return self

    def embed(self, S):
        h = self.scaler.transform(S)
        for layer in range(len(self.hidden) // 2 + 1):
            h = np.maximum(h @ self.net.coefs_[layer] + self.net.intercepts_[layer], 0.0)
        return h


# =============================================================================================
# 4. Metrics
# =============================================================================================

def ks_stat(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def ece(y, p, bins=10):
    order = np.argsort(p)
    err = 0.0
    for chunk in np.array_split(order, bins):
        err += abs(p[chunk].mean() - y[chunk].mean()) * len(chunk)
    return err / len(p)


def psi(expected, actual, bins=10):
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    e = np.histogram(np.clip(expected, edges[0], edges[-1]), edges)[0] / len(expected) + 1e-6
    a = np.histogram(np.clip(actual, edges[0], edges[-1]), edges)[0] / len(actual) + 1e-6
    return float(np.sum((a - e) * np.log(a / e)))


def score_report(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    auc = roc_auc_score(y, p)
    return {"AUC": auc, "Gini": 2 * auc - 1, "KS": ks_stat(y, p), "Brier": float(np.mean((p - y) ** 2)),
            "ECE": ece(y, p), "PD/actual": float(p.mean() / max(y.mean(), 1e-9))}


def bad_rate_at_approval(y, p, rate=0.7):
    keep = np.argsort(p)[: int(rate * len(p))]
    return float(np.asarray(y)[keep].mean())


def show(title, df, digits=3):
    print(f"\n=== {title} ===")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(df.round(digits).to_string())


# =============================================================================================
# 5. Experiments
# =============================================================================================

def describe_panel(p: BureauPanel, name):
    d = snapshot_dataset(p, build_feature_cube(p), [18, 39])
    a = p.applications
    return {
        "panel": name,
        "12m bad rate, t0=18": d.loc[d.t0 == 18, "y_obs"].mean(),
        "12m bad rate, t0=39": d.loc[d.t0 == 39, "y_obs"].mean(),
        "thin-file share": d["thin_file"].mean(),
        "lender-0 approval": a[a.lender == 0]["approved"].mean(),
        "lenient-lender approval": a[a.lender == p.cfg.n_lenders - 1]["approved"].mean(),
        "ever deferred (of borrowers)": p.mor.any(axis=1).mean(),
    }


def run_e1(p: BureauPanel, cube, seed):
    """E1 (A1): out-of-time benchmark. Train t0 = 12..24, test t0 = 36..45 (after the shock)."""
    dev = snapshot_dataset(p, cube, [12, 15, 18, 21, 24])
    oot = snapshot_dataset(p, cube, [36, 39, 42, 45])
    feats = [f for f in FEATURES if dev[f].std() > 0]
    y, yo = dev["y_obs"].to_numpy(), oot["y_obs"].to_numpy()

    sc = WoEScorecard().fit(dev[feats], y)
    gbm = fit_gbm(dev[feats], y, seed=seed)

    S_dev = sequence_matrix(p, dev["borrower"], dev["t0"])
    S_oot = sequence_matrix(p, oot["borrower"], oot["t0"])
    rng = np.random.default_rng(seed)
    pre = S_dev[rng.choice(len(S_dev), min(60_000, len(S_dev)), replace=False)]   # labels never used
    enc = MaskedSequenceEncoder(seed=seed).fit(pre)
    E_dev, E_oot = enc.embed(S_dev), enc.embed(S_oot)
    ecols = [f"emb_{i}" for i in range(E_dev.shape[1])]
    probe_scaler = StandardScaler().fit(E_dev)
    probe = LogisticRegression(max_iter=2000).fit(probe_scaler.transform(E_dev), y)
    Xd = pd.concat([dev[feats].reset_index(drop=True), pd.DataFrame(E_dev, columns=ecols)], axis=1)
    Xo = pd.concat([oot[feats].reset_index(drop=True), pd.DataFrame(E_oot, columns=ecols)], axis=1)
    gbm_ssl = fit_gbm(Xd, y, seed=seed)

    preds = {
        "WoE scorecard": (sc.predict_proba(dev[feats]), sc.predict_proba(oot[feats])),
        "Monotone GBM": (gbm.predict_proba(dev[feats])[:, 1], gbm.predict_proba(oot[feats])[:, 1]),
        "SSL embedding only (linear probe)": (probe.predict_proba(probe_scaler.transform(E_dev))[:, 1],
                                              probe.predict_proba(probe_scaler.transform(E_oot))[:, 1]),
        "GBM + SSL embedding": (gbm_ssl.predict_proba(Xd)[:, 1], gbm_ssl.predict_proba(Xo)[:, 1]),
    }
    thin = oot["thin_file"].to_numpy()
    rows = []
    for name, (pd_dev, pd_oot) in preds.items():
        r = score_report(yo, pd_oot)
        r["AUC thin-file"] = roc_auc_score(yo[thin], pd_oot[thin])
        r["PSI dev->OOT"] = psi(pd_dev, pd_oot)
        r["bad rate @70% approval"] = bad_rate_at_approval(yo, pd_oot)
        rows.append({"model": name, **r})
    res = pd.DataFrame(rows).set_index("model")
    info = {"n_dev": len(dev), "n_oot": len(oot), "dev_bad_rate": float(y.mean()),
            "oot_bad_rate": float(yo.mean()), "scorecard_features": sc.features,
            "scorecard_dropped_wrong_sign": sc.dropped_wrong_sign}
    art = {"sc": sc, "gbm": gbm, "oot": oot, "feats": feats, "preds": preds, "y_dev": y, "y_oot": yo,
           "dev": dev, "enc": enc, "probe": (probe_scaler, probe), "gbm_ssl": gbm_ssl, "ecols": ecols}
    return res, info, art


def run_e2(p: BureauPanel, cube, seed):
    """E2 (A2): selective labels at lender 0, scored against every applicant's true outcome Y(1)."""
    T = p.cfg.n_months
    a = p.applications
    a = a[(a.lender == 0) & (a.month >= 6) & (a.month <= T - 13)].reset_index(drop=True)
    b, m = a["borrower"].to_numpy(), a["month"].to_numpy()
    X = pd.DataFrame({f: cube[f][b, m - 1] for f in FEATURES})          # what the lender saw at t-1
    feats = [f for f in FEATURES if X[f].std() > 0]
    X = X[feats]
    approved = a["approved"].to_numpy().astype(bool)
    y1 = potential_outcome(p, b, m)                                      # oracle Y(1) for everyone
    future_acc = np.array([p.n_acc[bi, mi:mi + 12].max() for bi, mi in zip(b, m)])
    observable = approved | (future_acc > 0)                             # bureau sees other accounts

    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed).split(X, groups=b))
    Xtr, Xte = X.iloc[tr], X.iloc[te]
    app_tr, obs_tr, y_tr = approved[tr], observable[tr], y1[tr]
    rej_tr = ~app_tr

    def lr_prop(Xf, target):
        s = StandardScaler().fit(Xf)
        m_ = LogisticRegression(max_iter=2000).fit(s.transform(Xf), target)
        return m_.predict_proba(s.transform(Xf))[:, 1]

    models = {}
    models["Approved-only (AO)"] = fit_gbm(Xtr[app_tr], y_tr[app_tr], seed=seed, min_leaf=100)
    w_ipw = 1.0 / np.clip(lr_prop(Xtr, app_tr.astype(int)), 0.05, 1.0)
    models["AO + IPW on observables"] = fit_gbm(Xtr[app_tr], y_tr[app_tr], w=w_ipw[app_tr], seed=seed, min_leaf=100)
    p_rej = models["Approved-only (AO)"].predict_proba(Xtr[rej_tr])[:, 1]   # classic fuzzy augmentation
    Xf = pd.concat([Xtr[app_tr], Xtr[rej_tr], Xtr[rej_tr]])
    yf = np.r_[y_tr[app_tr], np.ones(rej_tr.sum()), np.zeros(rej_tr.sum())]
    wf = np.r_[np.ones(app_tr.sum()), p_rej, 1 - p_rej]
    models["Fuzzy augmentation"] = fit_gbm(Xf, yf, w=wf, seed=seed, min_leaf=100)
    use = app_tr | (rej_tr & obs_tr)
    models["Bureau-augmented (BA)"] = fit_gbm(Xtr[use], y_tr[use], seed=seed, min_leaf=100)
    p_obs_rej = np.ones(len(tr))
    p_obs_rej[rej_tr] = np.clip(lr_prop(Xtr[rej_tr], obs_tr[rej_tr].astype(int)), 0.05, 1.0)
    models["BA + IPW for unobserved rejects"] = fit_gbm(Xtr[use], y_tr[use], w=(1.0 / p_obs_rej)[use],
                                                        seed=seed, min_leaf=100)
    models["Oracle (all applicants, true Y)"] = fit_gbm(Xtr, y_tr, seed=seed, min_leaf=100)

    y_te, app_te = y1[te], approved[te]
    rows, preds = [], {}
    for name, mdl in models.items():
        pr = mdl.predict_proba(Xte)[:, 1]
        preds[name] = pr
        rows.append({"method": name,
                     "AUC all applicants": roc_auc_score(y_te, pr),
                     "AUC approved only (in-house view)": roc_auc_score(y_te[app_te], pr[app_te]),
                     "PD/actual all": pr.mean() / y_te.mean(),
                     "PD/actual rejected": pr[~app_te].mean() / y_te[~app_te].mean(),
                     "bad rate @70% approval": bad_rate_at_approval(y_te, pr)})
    info = {"n_applications": int(len(a)), "approval_rate": float(approved.mean()),
            "bad_rate_approved": float(y1[approved].mean()), "bad_rate_rejected": float(y1[~approved].mean()),
            "share_rejects_observable": float(observable[~approved].mean())}
    table = X.copy()
    table.insert(0, "borrower", b)
    table.insert(1, "month", m)
    table["approved"], table["observable"], table["y_true"] = approved, observable, y1.astype(int)
    split = np.full(len(table), "test", dtype=object)
    split[tr] = "train"
    table["split"] = split
    art = {"y_te": y_te, "approved_te": app_te, "preds": preds, "models": models, "feats": feats, "table": table}
    return pd.DataFrame(rows).set_index("method"), info, art


def run_e3(p: BureauPanel, cube, seed):
    """E3 (A3): train through a moratorium (t0 = 15..24), test after it ends (t0 = 39..45)."""
    cfg = p.cfg
    dev = snapshot_dataset(p, cube, [15, 18, 21, 24], delayed_end=cfg.shock_end + cfg.exit_window)
    oot = snapshot_dataset(p, cube, [39, 42, 45])
    feats = [f for f in FEATURES if dev[f].std() > 0]
    yo = oot["y_obs"].to_numpy()
    flag = oot["moratorium_24m"].to_numpy() > 0
    strategies = {
        "Naive: reported labels": (np.ones(len(dev), bool), "y_obs"),
        "Exclude deferred windows": (~dev["mor_in_window"].to_numpy(), "y_obs"),
        "Delayed labels (+6m after exit)": (np.ones(len(dev), bool), "y_delayed"),
        "Oracle: no-deferral counterfactual": (np.ones(len(dev), bool), "y_true"),
    }
    rows, preds, models = [], {}, {}
    for name, (keep, label) in strategies.items():
        d = dev[keep]
        s = StandardScaler().fit(d[feats])
        lr = LogisticRegression(max_iter=3000).fit(s.transform(d[feats]), d[label])
        pr = lr.predict_proba(s.transform(oot[feats]))[:, 1]
        preds[name] = pr
        models[name] = (s, lr)
        coef = dict(zip(feats, lr.coef_[0]))
        flag_varies = d["moratorium_24m"].std() > 0
        cal_all = pr.mean() / yo.mean()
        rows.append({"label strategy": name,
                     "train bad rate": d[label].mean(),
                     "coef: moratorium flag": coef["moratorium_24m"] if flag_varies else np.nan,
                     "OOT AUC": roc_auc_score(yo, pr),
                     "OOT PD/actual": cal_all,
                     "relative calibration, deferred (ideal 1)": (pr[flag].mean() / yo[flag].mean()) / cal_all,
                     "OOT Brier": float(np.mean((pr - yo) ** 2))})
    info = {"oot_bad_rate_flagged": float(yo[flag].mean()), "oot_bad_rate_not_flagged": float(yo[~flag].mean()),
            "share_flagged_oot": float(flag.mean()),
            "dev_true_vs_reported_bad_rate": [float(dev["y_true"].mean()), float(dev["y_obs"].mean())]}
    art = {"yo": yo, "flag": flag, "preds": preds, "models": models, "feats": feats, "dev": dev, "oot": oot}
    return pd.DataFrame(rows).set_index("label strategy"), info, art


def masking_timeseries(p: BureauPanel, cube, t0s):
    """Reported vs counterfactual 12-month bad rate by observation month (for the masking figure)."""
    cfg = p.cfg
    rows = []
    for t0 in t0s:
        d = snapshot_dataset(p, cube, [t0], delayed_end=cfg.shock_end + cfg.exit_window)
        rows.append({"t0": t0, "reported": d["y_obs"].mean(), "counterfactual": d["y_true"].mean(),
                     "delayed": d["y_delayed"].mean(), "deferred_in_window": d["mor_in_window"].mean()})
    return pd.DataFrame(rows)


def run_e4(oot: pd.DataFrame, pd_hat, approve_rate=0.7):
    """E4 (A4): discrimination, relative calibration and approval by group at one global cut-off.
    Relative calibration = (group PD / group bad rate) / (overall PD / overall bad rate); ideal 1."""
    y = oot["y_obs"].to_numpy()
    cut = np.quantile(pd_hat, approve_rate)
    approved = pd_hat <= cut
    overall = pd_hat.mean() / y.mean()
    rows = []
    for col in ["gender", "region", "occupation", "age_band", "thin_file"]:
        for g, idx in oot.groupby(col, observed=True).indices.items():
            yy, pp = y[idx], pd_hat[idx]
            rows.append({"group": f"{col}={g}", "n": len(idx), "bad rate": yy.mean(), "mean PD": pp.mean(),
                         "relative calibration": (pp.mean() / max(yy.mean(), 1e-9)) / overall,
                         "AUC": roc_auc_score(yy, pp) if 0 < yy.sum() < len(yy) else np.nan,
                         "approval rate": approved[idx].mean(),
                         "approval among goods": approved[idx][yy == 0].mean()})
    return pd.DataFrame(rows).set_index("group")


def run_e5(sc: WoEScorecard, oot: pd.DataFrame, feats, approve_rate=0.7):
    """E5: points table and reason codes for declined applicants at the 5th, 15th and 25th score percentile."""
    score = sc.score(oot[feats])
    order = np.argsort(score)
    picks = [order[int(q * len(order))] for q in (0.05, 0.15, 0.25)]
    reasons = sc.reason_codes(oot.iloc[picks][feats])
    cut = np.quantile(score, 1 - approve_rate)
    examples = pd.DataFrame({"score percentile": ["5th", "15th", "25th"], "score": np.round(score[picks]),
                             "cut-off at 70% approval": round(float(cut)),
                             "reason codes": [", ".join(r) for r in reasons]})
    return sc.points_table(), examples


# =============================================================================================
# 6. Main
# =============================================================================================

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", default="all", help="all or comma list of e1,e2,e3,e4,e5")
    ap.add_argument("--n", type=int, default=50_000, help="number of synthetic borrowers")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="lab_outputs")
    args = ap.parse_args(argv)
    exps = {"e1", "e2", "e3", "e4", "e5"} if args.exp == "all" else set(args.exp.split(","))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"args": vars(args)}
    t_start = time.time()

    clean = simulate_bureau(SimConfig(n_borrowers=args.n, moratorium=False, seed=args.seed))
    cube_clean = build_feature_cube(clean)
    sanity = [describe_panel(clean, "clean (no moratorium)")]
    if "e3" in exps:
        morat = simulate_bureau(SimConfig(n_borrowers=args.n, moratorium=True, seed=args.seed + 1))
        cube_morat = build_feature_cube(morat)
        sanity.append(describe_panel(morat, "with moratorium"))
    show("Simulator sanity checks", pd.DataFrame(sanity).set_index("panel"))
    summary["sanity"] = sanity

    if exps & {"e1", "e4", "e5"}:
        res, info, art = run_e1(clean, cube_clean, args.seed)
        if "e1" in exps:
            show("E1  Out-of-time benchmark (A1)", res)
            res.to_csv(out / "e1_benchmark.csv")
            summary["e1"] = {"table": res.round(4).reset_index().to_dict("records"), **info}
        if "e4" in exps:
            fair = run_e4(art["oot"], art["preds"]["Monotone GBM"][1])
            show("E4  Group report, monotone GBM, 70% approval (A4)", fair)
            fair.to_csv(out / "e4_group_report.csv")
            summary["e4"] = fair.round(4).reset_index().to_dict("records")
        if "e5" in exps:
            pts, ex = run_e5(art["sc"], art["oot"], art["feats"])
            show("E5  Scorecard points (top features by IV)", pts[pts.feature.isin(art["sc"].features[:5])], 2)
            show("E5  Reason codes for declined applicants", ex)
            pts.to_csv(out / "e5_points_table.csv", index=False)
            summary["e5"] = {"examples": ex.to_dict("records")}

    if "e2" in exps:
        res2, info2, _ = run_e2(clean, cube_clean, args.seed)
        show("E2  Selective labels / reject inference at lender 0 (A2)", res2)
        print("    ", {k: round(v, 4) if isinstance(v, float) else v for k, v in info2.items()})
        res2.to_csv(out / "e2_reject_inference.csv")
        summary["e2"] = {"table": res2.round(4).reset_index().to_dict("records"), **info2}

    if "e3" in exps:
        res3, info3, _ = run_e3(morat, cube_morat, args.seed)
        show("E3  Label masking by a moratorium (A3)", res3)
        print("    ", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in info3.items()})
        res3.to_csv(out / "e3_label_masking.csv")
        summary["e3"] = {"table": res3.round(4).reset_index().to_dict("records"), **info3}

    summary["runtime_sec"] = round(time.time() - t_start, 1)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nDone in {summary['runtime_sec']} s. Tables and summary.json written to {out.resolve()}")


if __name__ == "__main__":
    main()
