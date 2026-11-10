"""
bureau_to_workout.tables
Turns paper_results/ into LaTeX tables (paper/tables/*.tex) and number macros (paper/results_macros.tex),
so every number in the text is generated, not retyped.

Run: make tables (the module does its work on import, like the script it was).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .paths import PAPER, RESULTS as RES

TAB = PAPER / "tables"
TAB.mkdir(parents=True, exist_ok=True)
MACROS = []


def mk(name, value):
    assert name.isalpha(), name
    MACROS.append(f"\\newcommand{{\\{name}}}{{{value}}}")


def ms(d, key, order):
    g = d.drop(columns=["seed"]).groupby(key, sort=False)
    return g.mean().loc[order], g.std().loc[order]


def pm(m, s, f="{:.3f}"):
    return f"${f.format(m)} \\pm {f.format(s)}$"


def write(name, body):
    (TAB / f"{name}.tex").write_text(body)


# ------------------------------------------------------------------ E1
e1 = pd.read_csv(RES / "e1_all.csv")
E1 = ["WoE scorecard", "Monotone GBM", "SSL embedding only (linear probe)", "GBM + SSL embedding"]
E1n = ["WoE scorecard", "Monotone GBM", "SSL embedding, linear probe", "GBM + SSL embedding"]
m, s = ms(e1, "model", E1)
rows = []
for i, k in enumerate(E1):
    rows.append(f"{E1n[i]} & {pm(m.loc[k,'AUC'], s.loc[k,'AUC'])} & {pm(m.loc[k,'AUC thin-file'], s.loc[k,'AUC thin-file'])} & "
                f"{pm(m.loc[k,'KS'], s.loc[k,'KS'])} & {pm(m.loc[k,'Brier'], s.loc[k,'Brier'])} & "
                f"{pm(m.loc[k,'PSI dev->OOT'], s.loc[k,'PSI dev->OOT'], '{:.2f}')} & {pm(m.loc[k,'PD/actual'], s.loc[k,'PD/actual'], '{:.2f}')} \\\\")
write("tab_e1", "\\begin{tabular}{lcccccc}\n\\toprule\nModel & AUC & AUC, thin file & KS & Brier & PSI & PD/observed \\\\\n\\midrule\n"
      + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
for key, nm in zip(E1, ["Sc", "Gbm", "Ssl", "Gbmssl"]):
    mk(f"EoneAUC{nm}", f"{m.loc[key,'AUC']:.3f}")
    mk(f"EoneThin{nm}", f"{m.loc[key,'AUC thin-file']:.3f}")
    mk(f"EonePSI{nm}", f"{m.loc[key,'PSI dev->OOT']:.2f}")
    mk(f"EoneCal{nm}", f"{m.loc[key,'PD/actual']:.2f}")
i1 = pd.read_csv(RES / "info_e1.csv").mean(numeric_only=True)
mk("EoneDevBad", f"{100*i1['dev_bad_rate']:.1f}")
mk("EoneOotBad", f"{100*i1['oot_bad_rate']:.1f}")
mk("EoneNdev", f"{i1['n_dev']/1000:.0f}")
mk("EoneNoot", f"{i1['n_oot']/1000:.0f}")

# ------------------------------------------------------------------ E2
e2 = pd.read_csv(RES / "e2_all.csv")
E2 = ["Approved-only (AO)", "AO + IPW on observables", "Fuzzy augmentation", "Bureau-augmented (BA)",
      "BA + IPW for unobserved rejects", "Oracle (all applicants, true Y)"]
E2n = ["Approved-only (AO)", "AO + IPW on observables", "Fuzzy augmentation", "Bureau-augmented (BA)",
       "BA + IPW for unobserved rejects", "Oracle (all applicants, true $Y$)"]
m, s = ms(e2, "method", E2)
rows = []
for i, k in enumerate(E2):
    if i == 5:
        rows.append("\\midrule")
    rows.append(f"{E2n[i]} & {pm(m.loc[k,'AUC all applicants'], s.loc[k,'AUC all applicants'])} & "
                f"{pm(m.loc[k,'AUC approved only (in-house view)'], s.loc[k,'AUC approved only (in-house view)'])} & "
                f"{pm(m.loc[k,'PD/actual all'], s.loc[k,'PD/actual all'], '{:.2f}')} & "
                f"{pm(m.loc[k,'PD/actual rejected'], s.loc[k,'PD/actual rejected'], '{:.2f}')} & "
                f"{pm(100*m.loc[k,'bad rate @70% approval'], 100*s.loc[k,'bad rate @70% approval'], '{:.1f}')} \\\\")
write("tab_e2", "\\begin{tabular}{lccccc}\n\\toprule\n & \\multicolumn{2}{c}{AUC} & \\multicolumn{2}{c}{Predicted / observed} & Default rate at \\\\\n"
      "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\nMethod & all applicants & approved only & all & rejected & 70\\% approval (\\%) \\\\\n\\midrule\n"
      + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
for key, nm in zip(E2, ["Ao", "Aoipw", "Fuzzy", "Ba", "Baipw", "Oracle"]):
    mk(f"EtwoAUC{nm}", f"{m.loc[key,'AUC all applicants']:.3f}")
    mk(f"EtwoHouse{nm}", f"{m.loc[key,'AUC approved only (in-house view)']:.3f}")
    mk(f"EtwoRej{nm}", f"{m.loc[key,'PD/actual rejected']:.2f}")
    mk(f"EtwoAll{nm}", f"{m.loc[key,'PD/actual all']:.2f}")
    mk(f"EtwoBad{nm}", f"{100*m.loc[key,'bad rate @70% approval']:.1f}")
mk("EtwoAoUnder", f"{100*(1-m.loc['Approved-only (AO)','PD/actual rejected']):.0f}")
house = m["AUC approved only (in-house view)"]
mk("EtwoHouseMin", f"{house.min():.3f}"); mk("EtwoHouseMax", f"{house.max():.3f}")
i2 = pd.read_csv(RES / "info_e2.csv").mean(numeric_only=True)
mk("EtwoNapps", f"{i2['n_applications']/1000:.1f}")
mk("EtwoApprove", f"{100*i2['approval_rate']:.0f}")
mk("EtwoBadApproved", f"{100*i2['bad_rate_approved']:.1f}")
mk("EtwoBadRejected", f"{100*i2['bad_rate_rejected']:.1f}")
mk("EtwoObservable", f"{100*i2['share_rejects_observable']:.0f}")
mk("EtwoUnobserved", f"{100*(1-i2['share_rejects_observable']):.0f}")

# ------------------------------------------------------------------ E3
e3 = pd.read_csv(RES / "e3_all.csv")
E3 = ["Naive: reported labels", "Exclude deferred windows", "Delayed labels (+6m after exit)",
      "Oracle: no-deferral counterfactual"]
E3n = ["Naive: reported labels", "Exclude deferred windows", "Delayed labels (+6 months after exit)",
       "Oracle: no-deferral counterfactual"]
m, s = ms(e3, "strategy", E3)
rows = []
for i, k in enumerate(E3):
    if i == 3:
        rows.append("\\midrule")
    coef = "n/e" if np.isnan(m.loc[k, "coef: moratorium flag"]) else pm(m.loc[k, "coef: moratorium flag"], s.loc[k, "coef: moratorium flag"], "{:+.2f}").replace("\\pm +", "\\pm ")
    rows.append(f"{E3n[i]} & {100*m.loc[k,'train bad rate']:.1f} & {coef} & {pm(m.loc[k,'OOT AUC'], s.loc[k,'OOT AUC'])} & "
                f"{pm(m.loc[k,'relative calibration, deferred (ideal 1)'], s.loc[k,'relative calibration, deferred (ideal 1)'], '{:.2f}')} & "
                f"{pm(m.loc[k,'OOT Brier'], s.loc[k,'OOT Brier'])} \\\\")
write("tab_e3", "\\begin{tabular}{lccccc}\n\\toprule\n & Training default & Coefficient on & Out-of-time & Relative calibration, & Out-of-time \\\\\n"
      "Label strategy & rate (\\%) & deferral flag & AUC & deferred borrowers & Brier \\\\\n\\midrule\n"
      + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
for key, nm in zip(E3, ["Naive", "Excl", "Delay", "Oracle"]):
    mk(f"EthreeAUC{nm}", f"{m.loc[key,'OOT AUC']:.3f}")
    mk(f"EthreeAUCtwo{nm}", f"{m.loc[key,'OOT AUC']:.2f}")
    mk(f"EthreeRel{nm}", f"{m.loc[key,'relative calibration, deferred (ideal 1)']:.2f}")
    mk(f"EthreeTrain{nm}", f"{100*m.loc[key,'train bad rate']:.1f}")
    if not np.isnan(m.loc[key, "coef: moratorium flag"]):
        mk(f"EthreeCoef{nm}", f"{m.loc[key,'coef: moratorium flag']:+.2f}".replace("-", "$-$"))
mk("EthreeAUCsdNaive", f"{s.loc['Naive: reported labels','OOT AUC']:.3f}")
i3 = pd.read_csv(RES / "info_e3.csv").mean(numeric_only=True)
mk("EthreeFlagBad", f"{100*i3['oot_bad_rate_flagged']:.1f}")
mk("EthreeNoFlagBad", f"{100*i3['oot_bad_rate_not_flagged']:.1f}")
mk("EthreeFlagShare", f"{100*i3['share_flagged_oot']:.0f}")
mk("EthreeDevTrue", f"{100*i3['dev_true_bad_rate']:.1f}")
mk("EthreeDevRep", f"{100*i3['dev_reported_bad_rate']:.1f}")

# ------------------------------------------------------------------ E4
e4 = pd.read_csv(RES / "e4_all.csv")
g = e4.drop(columns=["seed"]).groupby("group").mean()
mk("EfourRelInformal", f"{g.loc['occupation=informal','relative calibration']:.2f}")
mk("EfourRelSalaried", f"{g.loc['occupation=salaried','relative calibration']:.2f}")
mk("EfourRelSelf", f"{g.loc['occupation=self_employed','relative calibration']:.2f}")
mk("EfourRelThin", f"{g.loc['thin_file=True','relative calibration']:.2f}")
mk("EfourRelYoung", f"{g.loc['age_band=<30','relative calibration']:.2f}")
mk("EfourRelOld", f"{g.loc['age_band=60+','relative calibration']:.2f}")
mk("EfourAppInformal", f"{100*g.loc['occupation=informal','approval among goods']:.1f}")
mk("EfourAppSalaried", f"{100*g.loc['occupation=salaried','approval among goods']:.1f}")
mk("EfourAppThin", f"{100*g.loc['thin_file=True','approval among goods']:.1f}")
mk("EfourAppThick", f"{100*g.loc['thin_file=False','approval among goods']:.1f}")
mk("EfourBadInformal", f"{100*g.loc['occupation=informal','bad rate']:.1f}")
mk("EfourBadSalaried", f"{100*g.loc['occupation=salaried','bad rate']:.1f}")
reg = g.loc[[k for k in g.index if k.startswith("region=")], "relative calibration"]
mk("EfourRegMin", f"{reg.min():.2f}"); mk("EfourRegMax", f"{reg.max():.2f}")

# ------------------------------------------------------------------ R1
r1 = pd.read_csv(RES / "r1_all.csv")
R1 = ["Term extension", "Write-down", "Extension + write-down"]
m, s = ms(r1, "offer", R1)
rows = [f"{k} & {m.loc[k,'true mean effect']*100:+.1f} & {m.loc[k,'sd of true effect']*100:.1f} & "
        f"{pm(100*m.loc[k,'T-learner RMSE'], 100*s.loc[k,'T-learner RMSE'], '{:.1f}')} & "
        f"{pm(100*m.loc[k,'DR-learner RMSE'], 100*s.loc[k,'DR-learner RMSE'], '{:.1f}')} \\\\".replace("+-", "$-$").replace("& -", "& $-$")
        for k in R1]
write("tab_r1", "\\begin{tabular}{lcccc}\n\\toprule\n & \\multicolumn{2}{c}{True effect vs.\\ standard (pp)} & \\multicolumn{2}{c}{RMSE of estimated effect (pp)} \\\\\n"
      "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\nOffer & mean & s.d.\\ across cases & T-learner & DR-learner \\\\\n\\midrule\n"
      + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
for k, nm in zip(R1, ["Ext", "Wd", "Both"]):
    mk(f"RoneT{nm}", f"{100*m.loc[k,'T-learner RMSE']:.1f}")
    mk(f"RoneDR{nm}", f"{100*m.loc[k,'DR-learner RMSE']:.1f}")
    mk(f"RoneMean{nm}", f"{100*m.loc[k,'true mean effect']:+.1f}".replace("-", "$-$"))
    mk(f"RoneSd{nm}", f"{100*m.loc[k,'sd of true effect']:.1f}")

# ------------------------------------------------------------------ R2
r2 = pd.read_csv(RES / "r2_all.csv")
R2 = ["Status quo (logged offers)", "All standard", "All term extension", "Predictive targeting",
      "Causal targeting (T-learner)", "Causal targeting (DR-learner)", "Oracle"]
m, s = ms(r2, "policy", R2)
rows = []
for i, k in enumerate(R2):
    if i in (1, 3, 6):
        rows.append("\\midrule")
    rows.append(f"{k} & {pm(m.loc[k,'recovery (% of balance)'], s.loc[k,'recovery (% of balance)'], '{:.2f}')} & "
                f"{m.loc[k,'plan completion (%)']:.1f} & {m.loc[k,'new 90+ elsewhere (%)']:.1f} & "
                f"{m.loc[k,'write-downs (% of balance)']:.1f} \\\\")
write("tab_r2", "\\begin{tabular}{lcccc}\n\\toprule\n & Recovery value & Plan & New 90+ DPD at & Write-downs \\\\\n"
      "Policy & (\\% of balance) & completion (\\%) & other lenders (\\%) & (\\% of balance) \\\\\n\\midrule\n"
      + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
rv = m["recovery (% of balance)"]
for k, nm in zip(R2, ["Sq", "Std", "Ext", "Pred", "CausT", "CausDR", "Oracle"]):
    mk(f"RtwoRec{nm}", f"{rv.loc[k]:.2f}")
    mk(f"RtwoWd{nm}", f"{m.loc[k,'write-downs (% of balance)']:.1f}")
    mk(f"RtwoComp{nm}", f"{m.loc[k,'plan completion (%)']:.1f}")
    mk(f"RtwoSpill{nm}", f"{m.loc[k,'new 90+ elsewhere (%)']:.1f}")
gap = rv["Oracle"] - rv["Status quo (logged offers)"]
mk("RtwoGainDR", f"{rv['Causal targeting (DR-learner)'] - rv['Status quo (logged offers)']:.1f}")
mk("RtwoGainPred", f"{rv['Predictive targeting'] - rv['Status quo (logged offers)']:.1f}")
mk("RtwoGainOracle", f"{gap:.1f}")
mk("RtwoShareDR", f"{100*(rv['Causal targeting (DR-learner)'] - rv['Status quo (logged offers)'])/gap:.0f}")
mk("RtwoSharePred", f"{100*(rv['Predictive targeting'] - rv['Status quo (logged offers)'])/gap:.0f}")
mk("RtwoDRvsPred", f"{rv['Causal targeting (DR-learner)'] - rv['Predictive targeting']:.1f}")
ir = pd.read_csv(RES / "info_r.csv").mean(numeric_only=True)
mk("RlogStd", f"{100*ir['logged_share_Standard']:.0f}")
mk("RlogExt", f"{100*ir['logged_share_Term extension']:.0f}")
mk("RlogWd", f"{100*ir['logged_share_Write-down']:.0f}")
mk("RlogBoth", f"{100*ir['logged_share_Extension + write-down']:.0f}")

# ------------------------------------------------------------------ R3
r3 = pd.read_csv(RES / "r3_all.csv")
f3 = r3.groupby(["policy", "budget (% of balance)"])["recovery (% of balance)"].mean().unstack(0)
mk("RthreeOracleZero", f"{f3.loc[0,'Oracle']:.2f}")
mk("RthreeOracleMax", f"{f3['Oracle'].max():.2f}")
mk("RthreeDRZero", f"{f3.loc[0,'Causal (DR-learner)']:.2f}")
mk("RthreeDRMax", f"{f3['Causal (DR-learner)'].max():.2f}")
mk("RthreeDRArgmax", f"{f3['Causal (DR-learner)'].idxmax():.0f}")
mk("RthreePredMin", f"{f3['Predictive'].min():.2f}")
mk("RthreePredMax", f"{f3['Predictive'].max():.2f}")
mk("RthreeOracleSat", f"{f3.index[np.argmax(f3['Oracle'].values >= f3['Oracle'].max() - 0.01)]:.0f}")

# ------------------------------------------------------------------ R4
r4 = pd.read_csv(RES / "r4_all.csv")
rows = []
for est in ["DM", "IPS", "DR"]:
    err = r4[est] - r4["true"]
    rows.append(f"{ {'DM':'Direct method','IPS':'Inverse propensity (IPS)','DR':'Doubly robust'}[est] } & {np.mean(np.abs(err)):.2f} & {np.mean(err):+.2f} & {np.max(np.abs(err)):.2f} \\\\".replace("+-", "$-$").replace("& -", "& $-$"))
    mk(f"Rfour{est.title()}", f"{np.mean(np.abs(err)):.2f}")
write("tab_r4", "\\begin{tabular}{lccc}\n\\toprule\nEstimator & Mean absolute error (pp) & Mean error (pp) & Largest error (pp) \\\\\n\\midrule\n"
      + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
tl = r4[r4.policy == "Causal targeting (T-learner)"]
mk("RfourDMselfbias", f"{np.mean(tl['DM'] - tl['true']):+.2f}".replace("+", "+"))
mk("RfourDRselfbias", f"{np.mean(tl['DR'] - tl['true']):+.2f}".replace("-", "$-$"))
mk("RfourNpoints", f"{len(r4)}")

# ------------------------------------------------------------------ sanity / calibration table
san = pd.read_csv(RES / "sanity_all.csv").drop(columns=["seed"]).groupby("panel").mean()
ts = pd.read_csv(RES / "fig_e3_timeseries.csv")
pre = ts.loc[ts.t0 == 6, "reported"].iloc[0]
rows = [
    ("Bureau, no moratorium", "12-month default rate, scorable population, $t_0=18$ (window spans the shock)", f"{100*san.loc['clean','12m bad rate, t0=18']:.1f}\\%"),
    ("", "12-month default rate, $t_0=39$ (after the shock)", f"{100*san.loc['clean','12m bad rate, t0=39']:.1f}\\%"),
    ("", "Thin-file share of the scorable population", f"{100*san.loc['clean','thin-file share']:.1f}\\%"),
    ("", "Approval rate, lender 0 / lenient lender", f"{100*san.loc['clean','lender-0 approval']:.0f}\\% / {100*san.loc['clean','lenient-lender approval']:.0f}\\%"),
    ("", "Default rate of approved / rejected applicants at lender 0, $Y(1)$", f"{100*i2['bad_rate_approved']:.1f}\\% / {100*i2['bad_rate_rejected']:.1f}\\%"),
    ("", "Rejected applicants with repayment observable at other lenders", f"{100*i2['share_rejects_observable']:.0f}\\%"),
    ("Bureau, with moratorium", "12-month default rate before the shock, $t_0=6$", f"{100*pre:.1f}\\%"),
    ("", "Borrowers ever deferred", f"{100*san.loc['moratorium','ever deferred (of borrowers)']:.1f}\\%"),
    ("", "Training default rate, reported vs.\\ no-deferral counterfactual", f"{100*i3['dev_reported_bad_rate']:.1f}\\% vs.\\ {100*i3['dev_true_bad_rate']:.1f}\\%"),
    ("Workouts", "Logged offers: standard / extension / write-down / both",
     f"{100*ir['logged_share_Standard']:.0f} / {100*ir['logged_share_Term extension']:.0f} / {100*ir['logged_share_Write-down']:.0f} / {100*ir['logged_share_Extension + write-down']:.0f}\\%"),
    ("", "Write-downs under the logged policy, share of balance", f"{ir['status_quo_writedowns_pct']:.1f}\\%"),
]
body = "\\begin{tabular}{llr}\n\\toprule\nSimulator & Moment & Value \\\\\n\\midrule\n"
prev = None
for sim, mom, val in rows:
    if sim and prev is not None:
        body += "\\midrule\n"
    body += f"{sim} & {mom} & {val} \\\\\n"
    prev = sim or prev
body += "\\bottomrule\n\\end{tabular}\n"
write("tab_calibration", body)

# ------------------------------------------------------------------ scorecard excerpt (appendix)
pts = pd.read_csv(RES / "e5_points.csv")
ex = pd.read_csv(RES / "e5_examples.csv")
meta = json.load(open(RES / "e5_meta.json"))
rng_pts = pts.groupby("feature")["points"].agg(lambda v: v.max() - v.min())
top = list(rng_pts.sort_values(ascending=False).index[:3])
rows = []
for f in top:
    sub = pts[pts.feature == f]
    first = True
    for _, r in sub.iterrows():
        name = f"\\texttt{{{f.replace('_', chr(92) + '_')}}} (IV {r['iv']:.2f})" if first else ""
        first = False
        b = r["bin"].replace("-inf", "@NEG@").replace("inf", "@POS@").replace("@NEG@", "$-\\infty$").replace("@POS@", "$\\infty$")
        rows.append(f"{name} & {b} & {int(r['n']):,} & {100*r['bad_rate']:.1f} & {r['woe']:+.2f} & {r['points']:.0f} \\\\".replace("& -", "& $-$").replace("+-", "$-$"))
    rows.append("\\addlinespace")
write("tab_scorecard", "\\begin{tabular}{llrrrr}\n\\toprule\nAttribute & Bin & Accounts & Default (\\%) & WoE & Points \\\\\n\\midrule\n"
      + "\n".join(rows[:-1]) + "\n\\bottomrule\n\\end{tabular}\n")
rows = []
for _, r in ex.iterrows():
    rc = ", ".join(f"\\texttt{{{x.strip().replace('_', chr(92) + '_')}}}" for x in r["reason codes"].split(","))
    rows.append(f"{r['score percentile']} & {r['score']:.0f} & {rc} \\\\")
write("tab_reasons", "\\begin{tabular}{lcl}\n\\toprule\nScore percentile & Score & Reason codes, largest shortfall first \\\\\n\\midrule\n"
      + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
mk("EfiveCut", f"{ex['cut-off at 70% approval'].iloc[0]:.0f}")
mk("EfiveNfeat", f"{len(meta['scorecard_features'])}")

(PAPER / "results_macros.tex").write_text("% generated by bureau_to_workout.tables (make tables) -- do not edit\n" + "\n".join(MACROS) + "\n")
print(len(MACROS), "macros;", sorted(p.name for p in TAB.iterdir()))
