"""
bureau_to_workout.workout
=========================
Reference research code for Track B of the NCB x SAM research agenda: causal targeting of
debt-restructuring offers on asset-management (AMC) workout data with credit-bureau covariates.

Simulator: a portfolio of non-performing loans and four offers. Effects are heterogeneous through
two latent mechanisms: cash-flow liquidity (can the borrower afford the instalment?) and debt
overhang (can the borrower ever repay the stock of debt?). Term extensions relieve liquidity;
write-downs relieve overhang. Historical offers were chosen by officers from observables plus
their own generosity. The simulator keeps every potential outcome, so estimators and policies
are scored against the truth. All numbers are synthetic; nothing here describes real SAM data.

Experiments
  R1  CATE accuracy: T-learner vs cross-fitted DR-learner (RMSE against the simulator's truth)
  R2  Policies at the status-quo write-down budget: status quo, uniform offers, predictive
      targeting, causal (DR-learner) targeting, oracle
  R3  Budget frontier: portfolio recovery vs write-down budget
  R4  Off-policy evaluation: direct method, IPS and doubly robust estimates vs true values

Requires numpy, pandas, scikit-learn >= 1.4.
Run:  make workout   (python -m bureau_to_workout.cli workout --n 50000 --seed 11 --out resolution_outputs)
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)

ARMS = ["Standard", "Term extension", "Write-down", "Extension + write-down"]
HAIRCUT = np.array([0.0, 0.0, 0.30, 0.30])   # principal written off
EXTEND = np.array([False, True, False, True])  # longer term at a concessional rate
DISC_M = 0.08 / 12                             # AMC cost of funds, monthly
F_GRID = np.linspace(0.025, 0.975, 20)         # quadrature for the share of the plan paid before re-default
X_COLS = ["secured", "log_balance", "coll_ratio", "log_income_proxy", "free_share_proxy", "dsr_proxy", "dti_proxy",
          "prior_pay_ratio", "months_npl", "age", "n_other", "other_delinquent", "inq_6m"]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def annuity(principal, annual_rate, months):
    r = annual_rate / 12
    return principal * r / (1 - (1 + r) ** -months)


def pv_level(payment, months):
    """Present value at the AMC's cost of funds of `months` level payments."""
    return payment * (1 - (1 + DISC_M) ** -months) / DISC_M


# =============================================================================================
# 1. Simulator
# =============================================================================================

@dataclass
class WorkoutConfig:
    n_cases: int = 50_000
    n_officers: int = 60
    overhang_months: int = 48     # horizon over which a borrower believes the debt can be repaid
    overhang_coef: float = 5.0    # how sharply willingness collapses once debt exceeds that capacity
    seed: int = 11


@dataclass
class Workouts:
    X: pd.DataFrame          # observed covariates (AMC file + bureau)
    balance: np.ndarray      # outstanding balance, THB
    A: np.ndarray            # logged offer 0..3
    Y: np.ndarray            # realised recovery value (PV / balance) under the logged offer
    officer: np.ndarray
    e_true: np.ndarray       # (n,4) true assignment probabilities
    ev: np.ndarray           # (n,4) expected recovery value under each offer (truth)
    p_complete: np.ndarray   # (n,4) completion probability under each offer (truth)
    spill: np.ndarray        # (n,4) probability of a new 90+ DPD on another bureau account within 24m
    latent: pd.DataFrame     # capacity and willingness (evaluation only)


def simulate_workouts(cfg: WorkoutConfig) -> Workouts:
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_cases
    secured = rng.random(n) < 0.30
    balance = np.where(secured, np.exp(rng.normal(np.log(900_000), 0.5, n)),
                       np.exp(rng.normal(np.log(120_000), 0.7, n)))
    coll_ratio = np.where(secured, rng.uniform(0.6, 1.5, n), 0.0)       # collateral value / balance
    income = np.exp(rng.normal(np.log(15_000), 0.5, n))                # monthly income, THB
    cap_share = rng.uniform(0.10, 0.35, n)                             # share of income free for repayment
    capacity = income * cap_share                                      # monthly repayment capacity (latent)
    will = rng.normal(0.0, 1.0, n)                                     # willingness to pay (latent)

    base_rate, low_rate = np.where(secured, 0.07, 0.12), np.where(secured, 0.04, 0.06)
    base_term, long_term = np.where(secured, 120, 36), np.where(secured, 240, 96)
    inst, debt, term, full_pv = (np.zeros((n, 4)) for _ in range(4))
    for k in range(4):
        P = balance * (1 - HAIRCUT[k])
        rate = np.where(EXTEND[k], low_rate, base_rate)
        term[:, k] = np.where(EXTEND[k], long_term, base_term)
        inst[:, k] = annuity(P, rate, term[:, k])
        debt[:, k] = P
        full_pv[:, k] = pv_level(inst[:, k], term[:, k]) / balance

    # completion: liquidity (affordability of the instalment) + overhang (debt stock vs repayable capacity)
    afford = np.log(capacity[:, None] / inst)
    overhang = np.minimum(0.0, np.log(capacity[:, None] * cfg.overhang_months / debt))
    p_complete = sigmoid(0.3 + 1.8 * afford + cfg.overhang_coef * overhang + 0.8 * will[:, None])

    # re-default path: some instalments are paid (fewer when unaffordable), then legal recovery
    legal = np.where(secured, np.minimum(1.0, 0.7 * coll_ratio) * 0.9, 0.05)
    delay = np.where(secured, 30, 24)
    paid_share = np.minimum(1.0, capacity[:, None] / inst)

    def fail_value(f):
        months = f * np.minimum(term, 36) * paid_share     # re-default comes within three years
        return (pv_level(inst, months) / balance[:, None]
                + legal[:, None] * (1 + DISC_M) ** -(months + delay[:, None]))

    ev_fail = np.mean([fail_value(f) for f in F_GRID], axis=0)
    ev = p_complete * full_pv + (1 - p_complete) * ev_fail
    spill = sigmoid(-2.2 + 1.3 * np.minimum(inst / capacity[:, None], 4) - 0.4 * will[:, None])
    n_other = rng.poisson(1.2, n)
    spill = spill * (n_other > 0)[:, None]

    # observed covariates
    income_proxy = income * np.exp(rng.normal(0, 0.25, n))             # declared or bureau-inferred income
    free_share_proxy = np.clip(cap_share + rng.normal(0, 0.05, n), 0.0, 0.6)   # from a budget interview
    X = pd.DataFrame({
        "secured": secured.astype(float),
        "log_balance": np.log(balance),
        "coll_ratio": coll_ratio,
        "log_income_proxy": np.log(income_proxy),
        "free_share_proxy": free_share_proxy,
        "dsr_proxy": np.minimum(inst[:, 0] / income_proxy, 5.0),
        "dti_proxy": np.minimum(balance / (12 * income_proxy), 20.0),
        "prior_pay_ratio": sigmoid(0.8 * will + 0.8 * afford[:, 0] + rng.normal(0, 0.5, n)),
        "months_npl": rng.integers(3, 61, n).astype(float),
        "age": rng.integers(22, 66, n).astype(float),
        "n_other": n_other.astype(float),
        "other_delinquent": (rng.random(n) < sigmoid(-0.5 - 1.0 * afford[:, 0] - 0.5 * will)).astype(float),
        "inq_6m": rng.poisson(np.exp(-0.3 - 0.3 * will)).astype(float),
    })

    # historical offers: officers act on observables plus their own generosity
    officer = rng.integers(0, cfg.n_officers, n)
    gen = rng.normal(0.0, 0.7, cfg.n_officers)[officer]
    dsr_c, dti_c = np.clip(X.dsr_proxy - 0.8, -1, 3), np.clip(X.dti_proxy - 1.0, -1, 6)
    s = np.column_stack([
        np.zeros(n),
        -0.2 + 1.0 * dsr_c + 0.4 * gen - 0.3 * secured,
        -1.9 + 0.4 * dti_c + 0.6 * gen - 0.5 * secured + 0.5 * (1 - X.prior_pay_ratio),
        -2.3 + 0.5 * dsr_c + 0.3 * dti_c + 0.8 * gen,
    ])
    e_true = np.exp(s - s.max(axis=1, keepdims=True))
    e_true = 0.92 * e_true / e_true.sum(axis=1, keepdims=True) + 0.02    # overlap floor of 2% per offer
    A = (rng.random(n)[:, None] > np.cumsum(e_true, axis=1)).sum(axis=1)
    A = np.minimum(A, 3)

    # realised outcome under the logged offer (completion and re-default timing drawn once per case)
    u, f = rng.random(n), rng.choice(F_GRID, n)
    idx = np.arange(n)
    complete = u < p_complete[idx, A]
    months = f * np.minimum(term[idx, A], 36) * paid_share[idx, A]
    y_fail = (pv_level(inst[idx, A], months) / balance
              + legal * (1 + DISC_M) ** -(months + delay))
    Y = np.where(complete, full_pv[idx, A], y_fail)
    latent = pd.DataFrame({"capacity": capacity, "will": will})
    return Workouts(X, balance, A, Y, officer, e_true, ev, p_complete, spill, latent)


# =============================================================================================
# 2. Estimators
# =============================================================================================

def _reg(seed):
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                         min_samples_leaf=50, l2_regularization=1.0, early_stopping=True,
                                         validation_fraction=0.15, n_iter_no_change=20, random_state=seed)


def fit_nuisances(X, A, Y, seed=0):
    """Outcome model per offer + multinomial propensity model, fitted on (X, A, Y)."""
    mu = [_reg(seed).fit(X[A == k], Y[A == k]) for k in range(4)]
    sc = StandardScaler().fit(X)
    prop = LogisticRegression(max_iter=3000).fit(sc.transform(X), A)
    return mu, (sc, prop)


def predict_nuisances(models, X):
    mu, (sc, prop) = models
    m = np.column_stack([mk.predict(X) for mk in mu])
    e = np.clip(prop.predict_proba(sc.transform(X)), 0.02, 1.0)
    return m, e / e.sum(axis=1, keepdims=True)


def dr_learner(X, A, Y, seed=0, folds=2):
    """Cross-fitted DR-learner (Kennedy 2023): pseudo-outcomes from out-of-fold nuisances, then a
    second-stage regression per offer. Returns fitted CATE models for offers 1..3 vs offer 0."""
    n = len(Y)
    m_hat, e_hat = np.zeros((n, 4)), np.zeros((n, 4))
    for tr, te in KFold(folds, shuffle=True, random_state=seed).split(X):
        mdl = fit_nuisances(X.iloc[tr], A[tr], Y[tr], seed)
        m_hat[te], e_hat[te] = predict_nuisances(mdl, X.iloc[te])
    idx = np.arange(n)
    resid = (Y - m_hat[idx, A]) / e_hat[idx, A]
    cate = {}
    for k in (1, 2, 3):
        phi = m_hat[:, k] - m_hat[:, 0] + resid * (A == k) - resid * (A == 0)
        cate[k] = _reg(seed + k).fit(X, phi)
    return cate


def predict_cate(cate, X):
    return np.column_stack([np.zeros(len(X))] + [cate[k].predict(X) for k in (1, 2, 3)])


# =============================================================================================
# 3. Policies
# =============================================================================================

def budget_policy(tau, balance, budget):
    """Greedy allocation under a write-down budget (THB). tau: (n,4) gains vs offer 0 as a share of
    balance. Every case first gets its best offer without a write-down; write-downs then go to the
    cases with the largest extra gain per baht written off until the budget is spent."""
    free_best = np.where(tau[:, 1] > 0, 1, 0)
    free_gain = np.maximum(tau[:, 1], 0.0)
    wd_arm = np.where(tau[:, 3] > tau[:, 2], 3, 2)
    extra = np.maximum(tau[:, 2], tau[:, 3]) - free_gain
    cost = HAIRCUT[2] * balance
    ratio = extra * balance / cost
    policy = free_best.copy()
    spent = 0.0
    for i in np.argsort(-ratio):
        if extra[i] <= 0 or spent + cost[i] > budget:
            continue
        policy[i] = wd_arm[i]
        spent += cost[i]
    return policy


def predictive_policy(score, balance, budget):
    """Common practice: rank by predicted success under standard collection; the top 30% stay on
    the standard track, the middle 40% get a term extension, the bottom 30% get extension plus a
    write-down, lowest scores first, until the write-down budget is spent (the rest get extensions)."""
    n = len(score)
    order = np.argsort(score)
    policy = np.zeros(n, int)
    policy[order[: int(0.7 * n)]] = 1
    spent = 0.0
    for i in order[: int(0.3 * n)]:
        c = HAIRCUT[3] * balance[i]
        if spent + c <= budget:
            policy[i] = 3
            spent += c
    return policy


def spend(policy, balance):
    return float(np.sum(HAIRCUT[policy] * balance))


def true_value(w: Workouts, idx, policy):
    b = w.balance[idx]
    rows = np.arange(len(idx))
    return {
        "recovery (% of balance)": 100 * float(np.sum(b * w.ev[idx][rows, policy]) / b.sum()),
        "plan completion (%)": 100 * float(np.mean(w.p_complete[idx][rows, policy])),
        "new 90+ elsewhere (%)": 100 * float(np.mean(w.spill[idx][rows, policy])),
        "write-downs (% of balance)": 100 * spend(policy, b) / b.sum(),
    }


def ope(policy, m_hat, e_hat, A, Y, balance):
    """Balance-weighted direct-method, IPS and doubly robust estimates of a policy's recovery."""
    rows = np.arange(len(A))
    w = balance / balance.sum()
    match = (A == policy).astype(float)
    dm = np.sum(w * m_hat[rows, policy])
    ips = np.sum(w * match * Y / e_hat[rows, A])
    dr = dm + np.sum(w * match * (Y - m_hat[rows, A]) / e_hat[rows, A])
    return 100 * dm, 100 * ips, 100 * dr


# =============================================================================================
# 4. Experiments
# =============================================================================================

def run_all(w: Workouts, seed=0, budgets_pct=(0, 1, 2, 4, 6, 8, 10, 12, 15)):
    n = len(w.Y)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    tr, te = np.sort(perm[: n // 2]), np.sort(perm[n // 2:])
    Xtr, Xte = w.X.iloc[tr], w.X.iloc[te]
    Atr, Ytr = w.A[tr], w.Y[tr]
    b_te = w.balance[te]
    tau_true = w.ev[te] - w.ev[te][:, [0]]

    # R1: CATE accuracy -------------------------------------------------------------------------
    nuis = fit_nuisances(Xtr, Atr, Ytr, seed)
    m_te, e_te = predict_nuisances(nuis, Xte)
    tau_t = m_te - m_te[:, [0]]
    dr_models = dr_learner(Xtr, Atr, Ytr, seed)
    tau_dr = predict_cate(dr_models, Xte)
    r1 = pd.DataFrame([{"offer": ARMS[k],
                        "T-learner RMSE": float(np.sqrt(np.mean((tau_t[:, k] - tau_true[:, k]) ** 2))),
                        "DR-learner RMSE": float(np.sqrt(np.mean((tau_dr[:, k] - tau_true[:, k]) ** 2))),
                        "true mean effect": float(tau_true[:, k].mean()),
                        "sd of true effect": float(tau_true[:, k].std())}
                       for k in (1, 2, 3)]).set_index("offer")

    # predictive score: P(recover >= 60% of balance | standard track), learned from standard-track cases
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, min_samples_leaf=50,
                                         early_stopping=True, random_state=seed)
    clf.fit(Xtr[Atr == 0], (Ytr[Atr == 0] >= 0.6).astype(int))
    score = clf.predict_proba(Xte)[:, 1]

    # R2: policies at the status-quo write-down budget -------------------------------------------
    A_te = w.A[te]
    sq_budget = spend(A_te, b_te)
    policies = {
        "Status quo (logged offers)": A_te,
        "All standard": np.zeros(len(te), int),
        "All term extension": np.ones(len(te), int),
        "Predictive targeting": predictive_policy(score, b_te, sq_budget),
        "Causal targeting (T-learner)": budget_policy(tau_t, b_te, sq_budget),
        "Causal targeting (DR-learner)": budget_policy(tau_dr, b_te, sq_budget),
        "Oracle": budget_policy(tau_true, b_te, sq_budget),
    }
    r2 = pd.DataFrame([{"policy": k, **true_value(w, te, pol)} for k, pol in policies.items()]).set_index("policy")

    # R3: budget frontier -------------------------------------------------------------------------
    total = b_te.sum()
    frontier = []
    for pct in budgets_pct:
        B = pct / 100 * total
        for name, pol in [("Predictive", predictive_policy(score, b_te, B)),
                          ("Causal (DR-learner)", budget_policy(tau_dr, b_te, B)),
                          ("Oracle", budget_policy(tau_true, b_te, B))]:
            frontier.append({"budget (% of balance)": pct, "policy": name,
                             "recovery (% of balance)": true_value(w, te, pol)["recovery (% of balance)"]})
    r3 = pd.DataFrame(frontier)

    # R4: off-policy evaluation of candidate policies on the held-out half ------------------------
    cands = {k: v for k, v in policies.items() if k != "Status quo (logged offers)"}
    cands.update({"All write-down": np.full(len(te), 2), "All extension + write-down": np.full(len(te), 3)})
    r4 = []
    for name, pol in cands.items():
        dm, ips, dr = ope(pol, m_te, e_te, A_te, w.Y[te], b_te)
        r4.append({"policy": name, "true": true_value(w, te, pol)["recovery (% of balance)"],
                   "DM": dm, "IPS": ips, "DR": dr})
    r4 = pd.DataFrame(r4).set_index("policy")

    # uplift (gain) curves for the figure: rank by each method, accumulate the true gain of offer k
    curves = {}
    for k in (1, 3):
        rankers = {"DR-learner": -tau_dr[:, k], "T-learner": -tau_t[:, k], "Predictive (low score first)": score,
                   "Oracle": -tau_true[:, k]}
        frac = np.linspace(0, 1, 101)
        out = {"fraction": frac}
        for name, key in rankers.items():
            order = np.argsort(key)
            gain = np.r_[0, np.cumsum((tau_true[:, k] * b_te)[order])] / total * 100
            out[name] = np.interp(frac * len(order), np.arange(len(order) + 1), gain)
        out["Random"] = frac * float(np.sum(tau_true[:, k] * b_te)) / total * 100
        curves[ARMS[k]] = pd.DataFrame(out)

    info = {"n_test": int(len(te)), "logged offer shares": np.bincount(w.A, minlength=4).tolist(),
            "status-quo write-downs (% of balance)": 100 * sq_budget / total}
    models = {"nuisances": nuis, "dr_learner": dr_models, "predictive": clf, "train_idx": tr, "test_idx": te}
    return {"r1": r1, "r2": r2, "r3": r3, "r4": r4, "curves": curves, "info": info, "models": models}


def show(title, df, digits=3):
    print(f"\n=== {title} ===")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(df.round(digits).to_string())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default="resolution_outputs")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    w = simulate_workouts(WorkoutConfig(n_cases=args.n, seed=args.seed))
    res = run_all(w, seed=args.seed)
    print("logged offers:", dict(zip(ARMS, res["info"]["logged offer shares"])))
    show("R1  CATE accuracy vs truth (recovery as share of balance)", res["r1"])
    show("R2  Policies at the status-quo write-down budget", res["r2"], 2)
    show("R3  Budget frontier", res["r3"].pivot(index="budget (% of balance)", columns="policy",
                                               values="recovery (% of balance)"), 2)
    show("R4  Off-policy evaluation (recovery, % of balance)", res["r4"], 2)
    for k in ("r1", "r2", "r3", "r4"):
        res[k].to_csv(out / f"{k}.csv")
    (out / "summary.json").write_text(json.dumps({"args": vars(args), **res["info"],
                                                  "runtime_sec": round(time.time() - t0, 1)}, indent=2))
    print(f"\nDone in {time.time() - t0:.1f} s; tables written to {out.resolve()}")


if __name__ == "__main__":
    main()
