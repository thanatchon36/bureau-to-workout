"""Build (and optionally upload) the Hugging Face dataset and model repositories.

Both come from the paper's first seed: bureau panels with seeds 7 (no moratorium) and 1007 (moratorium),
workouts with seed 107, exactly as `make experiments` draws them. Everything is synthetic.

  build/hf/dataset -> https://huggingface.co/datasets/thanatchon36/bureau-to-workout
  build/hf/model   -> https://huggingface.co/thanatchon36/bureau-to-workout

Models are saved with skops (no pickle) and the WoE scorecard as plain JSON. The model card reports the
metrics of these exact fitted models, which can differ from the paper's seed-7 row in the third decimal
because library versions differ (see the README's Limitations).

Run: make hf            (build only)
     make hf-push       (build and upload; needs `hf auth login`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import shutil
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import skops.io as sio

from bureau_to_workout import bureau as A
from bureau_to_workout import workout as B
from bureau_to_workout.paths import RESULTS, ROOT

REPO_ID = "thanatchon36/bureau-to-workout"
SEED, N = 7, 50_000
PANEL_ARRAYS = ["s_obs", "s_true", "mor", "util", "n_acc", "n_lenders", "limit", "inq", "new_acc"]
E2_FILES = {
    "Approved-only (AO)": "e2_approved_only",
    "AO + IPW on observables": "e2_approved_only_ipw",
    "Fuzzy augmentation": "e2_fuzzy",
    "Bureau-augmented (BA)": "e2_bureau_augmented",
    "BA + IPW for unobserved rejects": "e2_bureau_augmented_ipw",
    "Oracle (all applicants, true Y)": "e2_oracle",
}
E3_FILES = {
    "Naive: reported labels": "e3_naive",
    "Exclude deferred windows": "e3_exclude",
    "Delayed labels (+6m after exit)": "e3_delayed",
    "Oracle: no-deferral counterfactual": "e3_oracle",
}
CARDS = ROOT / "scripts" / "hf_cards"


def panel_long(p: A.BureauPanel) -> pd.DataFrame:
    n, t = p.s_obs.shape
    df = pd.DataFrame({"borrower": np.repeat(np.arange(n, dtype=np.int32), t),
                       "month": np.tile(np.arange(t, dtype=np.int16), n)})
    for name in PANEL_ARRAYS:
        df[name] = getattr(p, name).reshape(-1)
    return df


def write_dataset(out: Path, clean, morat, w, e1, e2, e3, workout_models) -> None:
    (out / "bureau").mkdir(parents=True)
    (out / "scoring").mkdir()
    (out / "workout").mkdir()
    for tag, p in (("clean", clean), ("moratorium", morat)):
        panel_long(p).to_parquet(out / "bureau" / f"panel_{tag}.parquet", index=False)
        p.borrowers.rename_axis("borrower").reset_index().to_parquet(
            out / "bureau" / f"borrowers_{tag}.parquet", index=False)
        p.applications.to_parquet(out / "bureau" / f"applications_{tag}.parquet", index=False)
    e1["dev"].to_parquet(out / "scoring" / "e1_dev.parquet", index=False)
    e1["oot"].to_parquet(out / "scoring" / "e1_oot.parquet", index=False)
    e3["dev"].to_parquet(out / "scoring" / "e3_dev.parquet", index=False)
    for part in ("train", "test"):
        t = e2["table"]
        t[t["split"] == part].drop(columns="split").to_parquet(out / "scoring" / f"e2_{part}.parquet", index=False)
    e3["oot"].to_parquet(out / "scoring" / "e3_oot.parquet", index=False)

    cases = w.X.copy()
    cases.insert(0, "case", np.arange(len(cases), dtype=np.int32))
    cases["balance"], cases["offer"], cases["recovery"], cases["officer"] = w.balance, w.A, w.Y, w.officer
    for k in range(4):
        cases[f"propensity_{k}"] = w.e_true[:, k]
        cases[f"value_{k}"] = w.ev[:, k]
        cases[f"p_complete_{k}"] = w.p_complete[:, k]
        cases[f"spill_{k}"] = w.spill[:, k]
    cases["capacity"], cases["willingness"] = w.latent["capacity"].to_numpy(), w.latent["will"].to_numpy()
    split = np.full(len(cases), "test", dtype=object)
    split[workout_models["train_idx"]] = "train"
    cases["split"] = split
    cases.to_parquet(out / "workout" / "cases.parquet", index=False)

    shutil.copytree(RESULTS, out / "results")


def write_configs(out: Path, clean, morat, wcfg) -> None:
    cfg = {"bureau_clean": vars(clean.cfg), "bureau_moratorium": vars(morat.cfg), "workout": vars(wcfg),
           "arms": B.ARMS, "write_down_haircut": B.HAIRCUT.tolist()}
    (out / "configs.json").write_text(json.dumps(cfg, indent=2))


def write_models(out: Path, e1_res, e1, e2_res, e2, e3_res, e3, r, wx_cols) -> dict:
    (out / "bureau").mkdir(parents=True)
    (out / "workout").mkdir()
    sc = e1["sc"]
    (out / "bureau" / "woe_scorecard.json").write_text(json.dumps(sc.to_dict(), indent=1))
    sc.points_table().to_csv(out / "bureau" / "woe_points_table.csv", index=False)
    sio.dump(e1["gbm"], out / "bureau" / "monotone_gbm.skops")
    sio.dump(e1["gbm_ssl"], out / "bureau" / "gbm_ssl.skops")
    enc = e1["enc"]
    sio.dump({"scaler": enc.scaler, "net": enc.net, "hidden": list(enc.hidden), "mask_frac": enc.mask_frac},
             out / "bureau" / "ssl_encoder.skops")
    sio.dump({"scaler": e1["probe"][0], "model": e1["probe"][1]}, out / "bureau" / "ssl_linear_probe.skops")
    for name, fname in E2_FILES.items():
        sio.dump(e2["models"][name], out / "bureau" / f"{fname}.skops")
    for name, fname in E3_FILES.items():
        s, lr = e3["models"][name]
        sio.dump({"scaler": s, "model": lr}, out / "bureau" / f"{fname}.skops")
    (out / "bureau" / "features.json").write_text(json.dumps({
        "e1_features": e1["feats"], "e1_embedding_columns": e1["ecols"], "e2_features": e2["feats"], "e3_features": e3["feats"],
        "monotone_constraints": A.MONOTONE, "sequence_months": A.SEQ_LEN,
    }, indent=2))

    m = r["models"]
    mu, (psc, prop) = m["nuisances"]
    sio.dump(list(mu), out / "workout" / "outcome_models.skops")
    sio.dump({"scaler": psc, "model": prop}, out / "workout" / "propensity.skops")
    sio.dump({str(k): v for k, v in m["dr_learner"].items()}, out / "workout" / "dr_learner.skops")
    sio.dump(m["predictive"], out / "workout" / "predictive_score.skops")
    (out / "workout" / "features.json").write_text(json.dumps({"features": wx_cols, "arms": B.ARMS}, indent=2))

    metrics = {
        "e1_out_of_time": e1_res.round(4).reset_index().to_dict("records"),
        "e2_reject_inference": e2_res.round(4).reset_index().to_dict("records"),
        "e3_after_moratorium": e3_res.round(4).reset_index().to_dict("records"),
        "r1_cate_rmse": r["r1"].round(4).reset_index().to_dict("records"),
        "r2_policies": r["r2"].round(3).reset_index().to_dict("records"),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    return metrics


def versions() -> dict:
    return {"python": platform.python_version(),
            **{pkg: version(pkg) for pkg in ("numpy", "pandas", "scikit-learn", "skops", "bureau-to-workout")}}


def sha256sums(root: Path) -> None:
    lines = []
    for f in sorted(p for p in root.rglob("*") if p.is_file() and p.name != "SHA256SUMS"):
        lines.append(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.relative_to(root).as_posix()}")
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n")


def fill(template: Path, values: dict) -> str:
    text = template.read_text(encoding="utf-8")
    for k, v in values.items():
        text = text.replace("{{" + k + "}}", str(v))
    if "{{" in text:
        raise RuntimeError(f"unfilled placeholder in {template.name}")
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "build" / "hf"))
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    shutil.rmtree(out, ignore_errors=True)

    clean = A.simulate_bureau(A.SimConfig(n_borrowers=N, moratorium=False, seed=SEED))
    morat = A.simulate_bureau(A.SimConfig(n_borrowers=N, moratorium=True, seed=SEED + 1000))
    cube_c, cube_m = A.build_feature_cube(clean), A.build_feature_cube(morat)
    e1_res, _, e1 = A.run_e1(clean, cube_c, SEED)
    e2_res, _, e2 = A.run_e2(clean, cube_c, SEED)
    e3_res, _, e3 = A.run_e3(morat, cube_m, SEED)
    wcfg = B.WorkoutConfig(n_cases=N, seed=SEED + 100)
    w = B.simulate_workouts(wcfg)
    r = B.run_all(w, seed=SEED + 100)

    write_dataset(out / "dataset", clean, morat, w, e1, e2, e3, r["models"])
    write_configs(out / "dataset", clean, morat, wcfg)
    metrics = write_models(out / "model", e1_res, e1, e2_res, e2, e3_res, e3, r, list(w.X.columns))
    vers = versions()
    for d in ("dataset", "model"):
        (out / d / "versions.json").write_text(json.dumps(vers, indent=2))
    shutil.copyfile(ROOT / "LICENSE", out / "model" / "LICENSE")
    shutil.copyfile(CARDS / "LICENSE-CC-BY-4.0.txt", out / "dataset" / "LICENSE")

    paper = pd.read_csv(RESULTS / "e1_all.csv")
    paper = paper[paper["seed"] == SEED].set_index("model")["AUC"]
    e1_rows = "\n".join(f"| {name} | {row['AUC']:.3f} | {paper[name]:.3f} | {row['PSI dev->OOT']:.2f} |"
                        for name, row in e1_res.iterrows())
    paper2 = pd.read_csv(RESULTS / "e2_all.csv")
    paper2 = paper2[paper2["seed"] == SEED].set_index("method")
    e2_rows = "\n".join(
        f"| {name} | {row['AUC all applicants']:.3f} | {row['PD/actual rejected']:.2f} | "
        f"{paper2.loc[name, 'PD/actual rejected']:.2f} |" for name, row in e2_res.iterrows())
    e3_rows = "\n".join(f"| {name} | {row['OOT AUC']:.3f} | {row['relative calibration, deferred (ideal 1)']:.2f} |"
                        for name, row in e3_res.iterrows())
    r2_rows = "\n".join(f"| {name} | {row['recovery (% of balance)']:.2f} | {row['write-downs (% of balance)']:.1f} |"
                        for name, row in r["r2"].iterrows())
    doi = re.search(r"^doi:\s*(\S+)", (ROOT / "CITATION.cff").read_text(), re.M)
    values = {
        "E1_ROWS": e1_rows, "E2_ROWS": e2_rows, "E3_ROWS": e3_rows, "R2_ROWS": r2_rows,
        "VERSIONS": ", ".join(f"{k} {v}" for k, v in vers.items()),
        "DOI_LINK": f"[{doi.group(1)}](https://doi.org/{doi.group(1)})" if doi else "pending",
        "DOI_BIB": f"\n  doi    = {{{doi.group(1)}}}," if doi else "",
        "N_E1_DEV": f"{len(e1['dev']):,}", "N_E1_OOT": f"{len(e1['oot']):,}",
        "N_E2_TRAIN": f"{(e2['table']['split'] == 'train').sum():,}",
        "N_E2_TEST": f"{(e2['table']['split'] == 'test').sum():,}",
        "N_E3_DEV": f"{len(e3['dev']):,}", "N_E3_OOT": f"{len(e3['oot']):,}",
        "N_APPS_CLEAN": f"{len(clean.applications):,}", "N_APPS_MOR": f"{len(morat.applications):,}",
    }
    (out / "model" / "README.md").write_text(fill(CARDS / "model_card.md", values), encoding="utf-8")
    (out / "dataset" / "README.md").write_text(fill(CARDS / "dataset_card.md", values), encoding="utf-8")
    for d in ("dataset", "model"):
        sha256sums(out / d)
    print(json.dumps({k: len(v) for k, v in metrics.items()}), "->", out)

    if args.push:
        from huggingface_hub import HfApi
        api = HfApi()
        for d, kind in (("dataset", "dataset"), ("model", "model")):
            api.create_repo(REPO_ID, repo_type=kind, exist_ok=True)
            api.upload_folder(folder_path=out / d, repo_id=REPO_ID, repo_type=kind,
                              commit_message="Upload from scripts/export_hf.py", delete_patterns=["*"])
            print("pushed", kind, REPO_ID)


if __name__ == "__main__":
    main()
