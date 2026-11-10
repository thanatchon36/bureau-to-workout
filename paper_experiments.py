#!/usr/bin/env python3
"""
paper_experiments.py
Runs every experiment in the paper over five seeds and writes the numbers and figure data to
paper_results/. Tables report mean and standard deviation across seeds; figures use seed 7
(bureau) and seed 107 (workouts) unless they show seed-to-seed spread.

Run: python paper_experiments.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ncb_scoring_lab as A      # noqa: E402
import sam_resolution_lab as B   # noqa: E402

OUT = Path(__file__).resolve().parent / "paper_results"
SEEDS = [7, 8, 9, 10, 11]
N_BORROWERS, N_CASES = 50_000, 50_000


def reliability(y, p, bins=10):
    order = np.argsort(p)
    rows = []
    for chunk in np.array_split(order, bins):
        rows.append({"pred": float(p[chunk].mean()), "actual": float(y[chunk].mean()), "n": len(chunk)})
    return pd.DataFrame(rows)


def stack(frames, seeds, index_name):
    out = []
    for s, f in zip(seeds, frames):
        g = f.copy()
        g.index.name = index_name
        g = g.reset_index()
        g["seed"] = s
        out.append(g)
    return pd.concat(out, ignore_index=True)


def main():
    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    keep = {k: [] for k in ["e1", "e2", "e3", "e4", "r1", "r2", "r3", "r4", "sanity", "info_e1", "info_e2",
                            "info_e3", "info_r"]}
    for seed in SEEDS:
        ts = time.time()
        clean = A.simulate_bureau(A.SimConfig(n_borrowers=N_BORROWERS, moratorium=False, seed=seed))
        cube_c = A.build_feature_cube(clean)
        morat = A.simulate_bureau(A.SimConfig(n_borrowers=N_BORROWERS, moratorium=True, seed=seed + 1000))
        cube_m = A.build_feature_cube(morat)
        for d in (A.describe_panel(clean, "clean"), A.describe_panel(morat, "moratorium")):
            keep["sanity"].append({**d, "seed": seed})

        e1, i1, a1 = A.run_e1(clean, cube_c, seed)
        e4 = A.run_e4(a1["oot"], a1["preds"]["Monotone GBM"][1])
        e2, i2, a2 = A.run_e2(clean, cube_c, seed)
        e3, i3, a3 = A.run_e3(morat, cube_m, seed)
        keep["e1"].append(e1); keep["e4"].append(e4); keep["e2"].append(e2); keep["e3"].append(e3)
        keep["info_e1"].append({k: v for k, v in i1.items() if not isinstance(v, list)} | {"seed": seed})
        keep["info_e2"].append({**i2, "seed": seed})
        keep["info_e3"].append({"seed": seed, "oot_bad_rate_flagged": i3["oot_bad_rate_flagged"],
                                "oot_bad_rate_not_flagged": i3["oot_bad_rate_not_flagged"],
                                "share_flagged_oot": i3["share_flagged_oot"],
                                "dev_true_bad_rate": i3["dev_true_vs_reported_bad_rate"][0],
                                "dev_reported_bad_rate": i3["dev_true_vs_reported_bad_rate"][1]})

        w = B.simulate_workouts(B.WorkoutConfig(n_cases=N_CASES, seed=seed + 100))
        res = B.run_all(w, seed=seed + 100)
        for k in ("r1", "r2", "r4"):
            keep[k].append(res[k])
        keep["r3"].append(res["r3"].assign(seed=seed))
        keep["info_r"].append({"seed": seed, "status_quo_writedowns_pct": res["info"]["status-quo write-downs (% of balance)"],
                               **{f"logged_share_{B.ARMS[k]}": res["info"]["logged offer shares"][k] / N_CASES
                                  for k in range(4)}})

        if seed == SEEDS[0]:
            # ----- figure data from the first seed -----
            yo = a1["y_oot"]
            for name, (pd_dev, pd_oot) in a1["preds"].items():
                fpr, tpr, _ = roc_curve(yo, pd_oot)
                pick = np.unique(np.r_[np.linspace(0, len(fpr) - 1, 400).astype(int), len(fpr) - 1])
                pd.DataFrame({"fpr": fpr[pick], "tpr": tpr[pick]}).to_csv(OUT / f"fig_e1_roc_{name}.csv", index=False)
                reliability(yo, pd_oot).to_csv(OUT / f"fig_e1_rel_{name}.csv", index=False)
            for name, pr in a2["preds"].items():
                reliability(a2["y_te"], pr).to_csv(OUT / f"fig_e2_rel_{name}.csv", index=False)
            A.masking_timeseries(morat, cube_m, list(range(6, 48, 3))).to_csv(OUT / "fig_e3_timeseries.csv", index=False)
            for k, df in res["curves"].items():
                df.to_csv(OUT / f"fig_r_curve_{k}.csv", index=False)
            pts, ex = A.run_e5(a1["sc"], a1["oot"], a1["feats"])
            pts.to_csv(OUT / "e5_points.csv", index=False)
            ex.to_csv(OUT / "e5_examples.csv", index=False)
            json.dump({"scorecard_features": a1["sc"].features, "dropped": a1["sc"].dropped_wrong_sign,
                       "factor": a1["sc"].factor, "offset": a1["sc"].offset},
                      open(OUT / "e5_meta.json", "w"), indent=2)
        print(f"seed {seed} done in {time.time() - ts:.0f}s", flush=True)

    stack(keep["e1"], SEEDS, "model").to_csv(OUT / "e1_all.csv", index=False)
    stack(keep["e2"], SEEDS, "method").to_csv(OUT / "e2_all.csv", index=False)
    stack(keep["e3"], SEEDS, "strategy").to_csv(OUT / "e3_all.csv", index=False)
    stack(keep["e4"], SEEDS, "group").to_csv(OUT / "e4_all.csv", index=False)
    stack(keep["r1"], SEEDS, "offer").to_csv(OUT / "r1_all.csv", index=False)
    stack(keep["r2"], SEEDS, "policy").to_csv(OUT / "r2_all.csv", index=False)
    stack(keep["r4"], SEEDS, "policy").to_csv(OUT / "r4_all.csv", index=False)
    pd.concat(keep["r3"], ignore_index=True).to_csv(OUT / "r3_all.csv", index=False)
    pd.DataFrame(keep["sanity"]).to_csv(OUT / "sanity_all.csv", index=False)
    pd.DataFrame(keep["info_e1"]).to_csv(OUT / "info_e1.csv", index=False)
    pd.DataFrame(keep["info_e2"]).to_csv(OUT / "info_e2.csv", index=False)
    pd.DataFrame(keep["info_e3"]).to_csv(OUT / "info_e3.csv", index=False)
    pd.DataFrame(keep["info_r"]).to_csv(OUT / "info_r.csv", index=False)
    print(f"all done in {time.time() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
