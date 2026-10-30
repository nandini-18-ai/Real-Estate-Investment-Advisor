"""Independent verification of the Phase 3 artifacts (targets, split, leakage, baselines, reproducibility, environment).

Run from the project folder, in the SAME Python environment that runs Streamlit:
    python tools/verify_artifacts.py                  # fast checks (seconds)
    python tools/verify_artifacts.py --permutation    # also runs the slow 'is there any signal?' permutation test

It re-derives everything from the saved files and compares it with what model_metadata.json claims.
It never retrains the saved models and never edits any project file (except writing signal_permutation_test.csv
when --permutation is used).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LABEL_DEFINING = ["Price_in_Lakhs", "Price_per_SqFt", "Price_per_SqFt_exact", "Price_per_SqFt_INR", "Size_in_SqFt", "BHK",
                  "Availability_Status", "Parking_Space", "Security", "Public_Transport_Accessibility",
                  "Transport_Level", "Infra_Score", "Size_per_BHK"]
PRICE_DERIVED = ["Price_in_Lakhs", "Price_per_SqFt", "Price_per_SqFt_exact", "Price_per_SqFt_INR"]
FLOOR_COLUMNS = ["Floor_No", "Total_Floors"]


class Report:
    def __init__(self):
        self.rows = []

    def add(self, name, status, detail=""):
        self.rows.append({"check": name, "status": status, "detail": detail})
        print(f"{status:<5} {name}" + (f"  |  {detail}" if detail else ""))

    @property
    def failed(self):
        return [r for r in self.rows if r["status"] == "FAIL"]


def environment_info():
    info = {"python": sys.version.split()[0], "executable": sys.executable}
    for mod in ("sklearn", "xgboost", "pandas", "numpy", "scipy", "streamlit", "mlflow", "joblib"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "installed")
        except Exception:
            info[mod] = None
    return info


def check_environment(rep: Report, meta: dict):
    env = environment_info()
    rep.add("Python interpreter in use", "INFO", f"{env['executable']} (Python {env['python']})")
    rep.add("Installed versions", "INFO", ", ".join(f"{k}={env[k]}" for k in ("sklearn", "xgboost", "pandas", "numpy", "streamlit", "mlflow")))
    trained = (meta.get("library_versions") or {})
    t_sk, i_sk = trained.get("scikit_learn"), env["sklearn"]
    if t_sk and i_sk and t_sk != i_sk:
        rep.add("scikit-learn version matches the training run", "WARN",
                f"models trained with {t_sk}, installed {i_sk}: retrain in this environment (see the run guide) or install {t_sk}")
    else:
        rep.add("scikit-learn version matches the training run", "PASS", f"{i_sk}")
    names = " ".join(str((meta.get(k) or {}).get("model", "")) for k in ("classification", "regression"))
    fallback = "XGBoost not installed" in names
    if "XGBoost" in names and not fallback and env["xgboost"] is None:
        rep.add("XGBoost available for the saved XGBoost models", "FAIL", "the saved models are XGBoost models but xgboost is not importable here")
    elif fallback:
        rep.add("Saved models used XGBoost", "WARN",
                "the training environment had no xgboost, so HistGradientBoosting was used instead; "
                f"xgboost in THIS environment: {env['xgboost']}. Re-run the Phase 3 notebook in an environment where xgboost imports.")
    else:
        rep.add("Saved models used XGBoost", "PASS" if "XGBoost" in names else "INFO", names)
    return env


def load_project(root: Path):
    meta = json.loads((root / "models/model_metadata.json").read_text(encoding="utf-8"))
    rule = json.loads((root / "phase3_outputs/label_rule.json").read_text(encoding="utf-8"))
    data = pd.read_csv(root / "outputs/engineered_features.csv")
    split = pd.read_csv(root / "phase3_outputs/split_ids.csv")
    return meta, rule, data, split


def apply_rule(frame, rule):
    keys = frame["City"] + "|" + frame["Property_Type"]
    med = keys.map(rule["segment_median_ppsf_lakhs"]).fillna(rule["global_median_ppsf_lakhs"])
    q = ((frame["BHK"] >= 3).astype(int) + (frame["Availability_Status"] == "Ready_to_Move").astype(int)
         + (frame["Parking_Space"] == "Yes").astype(int) + (frame["Security"] == "Yes").astype(int)
         + (frame["Public_Transport_Accessibility"] == "High").astype(int))
    return ((frame["Price_per_SqFt_exact"] <= med) & (q >= rule["quality_threshold"])).astype(int)


def verify(root: Path, permutation=False, n_perm=10, perm_rows=40_000):
    rep = Report()
    root = Path(root)
    needed = ["models/classifier_pipeline.joblib", "models/regressor_pipeline.joblib", "models/model_metadata.json",
              "phase3_outputs/label_rule.json", "phase3_outputs/split_ids.csv", "outputs/engineered_features.csv"]
    missing = [f for f in needed if not (root / f).exists()]
    if missing:
        rep.add("Required files present", "FAIL", f"missing: {missing}")
        return rep
    rep.add("Required files present", "PASS")
    meta, rule, data, split = load_project(root)
    check_environment(rep, meta)

    # ---------------- split
    train_ids, test_ids = set(split.loc[split["set"] == "train", "ID"]), set(split.loc[split["set"] == "test", "ID"])
    rep.add("Train and test IDs are disjoint", "PASS" if not (train_ids & test_ids) else "FAIL", f"{len(train_ids):,} train / {len(test_ids):,} test")
    rep.add("Split sizes equal the metadata", "PASS" if (len(train_ids), len(test_ids)) == (meta.get("train_rows"), meta.get("test_rows")) else "FAIL",
            f"file {len(train_ids):,}/{len(test_ids):,} vs metadata {meta.get('train_rows')}/{meta.get('test_rows')}")
    fp = hashlib.sha256(",".join(map(str, sorted(train_ids))).encode()).hexdigest()[:16]
    rep.add("Training-ID fingerprint equals the metadata", "PASS" if fp == meta.get("split_fingerprint") else "FAIL", f"{fp} vs {meta.get('split_fingerprint')}")
    tr = data[data["ID"].isin(train_ids)].reset_index(drop=True)
    te = data[data["ID"].isin(test_ids)].reset_index(drop=True)
    gap = (tr["City"].value_counts(normalize=True) - te["City"].value_counts(normalize=True)).abs().max() * 100
    rep.add("Split is stratified by City (max share gap, percentage points)", "PASS" if gap < 0.5 else "WARN", f"{gap:.3f}")

    # ---------------- targets
    seg_train = tr.groupby(["City", "Property_Type"])["Price_per_SqFt_exact"].median()
    diffs = [abs(seg_train.get(tuple(k.split("|")), np.nan) - v) for k, v in rule["segment_median_ppsf_lakhs"].items()]
    rep.add("Label-rule medians equal medians recomputed from TRAIN rows only", "PASS" if np.nanmax(diffs) < 1e-12 else "FAIL", f"max difference {np.nanmax(diffs):.2e}")
    seg_all = data.groupby(["City", "Property_Type"])["Price_per_SqFt_exact"].median()
    diff_all = np.nanmax([abs(seg_all.get(tuple(k.split("|")), np.nan) - v) for k, v in rule["segment_median_ppsf_lakhs"].items()])
    rep.add("Rule medians differ from full-data medians (so test rows were not used)", "PASS" if diff_all > 1e-12 else "WARN", f"max difference {diff_all:.4f}")
    y_tr, y_te = apply_rule(tr, rule), apply_rule(te, rule)
    rep.add("Class balance (share of 'good')", "INFO", f"train {y_tr.mean():.4f}, test {y_te.mean():.4f}")
    mult = meta["growth_multiplier"]
    rep.add("Growth multiplier = (1 + rate) ** years", "PASS" if abs((1 + meta["growth_rate_assumed"]) ** meta.get("horizon_years", 5) - mult) < 1e-9 else "FAIL", f"{mult:.6f}")
    y_reg_te = te["Price_in_Lakhs"] * mult

    # ---------------- leakage guards
    cls_cols, reg_cols = meta["classification"]["input_columns"], meta["regression"]["input_columns"]
    bad_c = sorted(set(cls_cols) & set(LABEL_DEFINING)); bad_r = sorted(set(reg_cols) & set(PRICE_DERIVED))
    bad_f = sorted((set(cls_cols) | set(reg_cols)) & set(FLOOR_COLUMNS))
    rep.add("Classifier inputs contain no label-defining column", "PASS" if not bad_c else "FAIL", str(bad_c))
    rep.add("Regressor inputs contain no price-derived column", "PASS" if not bad_r else "FAIL", str(bad_r))
    rep.add("Invalid floor columns excluded from both models", "PASS" if not bad_f else "FAIL", str(bad_f))

    # ---------------- pipelines, preprocessing, reproducibility
    import joblib
    from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error, mean_squared_error, precision_score,
                                 r2_score, recall_score, roc_auc_score)
    try:
        clf, reg = joblib.load(root / "models/classifier_pipeline.joblib"), joblib.load(root / "models/regressor_pipeline.joblib")
        rep.add("Saved pipelines load", "PASS")
    except Exception as exc:
        rep.add("Saved pipelines load", "FAIL", f"{type(exc).__name__}: {str(exc)[:200]}")
        return rep
    for name, pipe, cols in (("classifier", clf, cls_cols), ("regressor", reg, reg_cols)):
        seen = list(getattr(pipe, "feature_names_in_", []))
        rep.add(f"{name}: pipeline input columns equal the metadata", "PASS" if set(seen) == set(cols) else "FAIL", "")
        Xt = pipe.named_steps["prep"].transform(te[cols].head(2000))
        rep.add(f"{name}: preprocessing output is finite", "PASS" if np.isfinite(np.asarray(Xt, dtype=float)).all() else "FAIL", f"{Xt.shape[1]} features")

    proba = clf.predict_proba(te[cls_cols])[:, 1]; pred = clf.predict(te[cls_cols])
    got = {"test_roc_auc": roc_auc_score(y_te, proba), "test_f1": f1_score(y_te, pred), "test_accuracy": accuracy_score(y_te, pred),
           "test_precision": precision_score(y_te, pred, zero_division=0), "test_recall": recall_score(y_te, pred, zero_division=0)}
    worst = max(abs(got[k] - meta["classification"]["test"][k]) for k in got)
    rep.add("Classifier: saved model reproduces the metadata test metrics", "PASS" if worst < 1e-6 else "FAIL",
            f"largest difference {worst:.2e}" + ("" if worst < 1e-6 else " (version incompatibility or different split)"))
    pr = reg.predict(te[reg_cols])
    got_r = {"test_rmse": float(np.sqrt(mean_squared_error(y_reg_te, pr))), "test_mae": mean_absolute_error(y_reg_te, pr), "test_r2": r2_score(y_reg_te, pr)}
    worst_r = max(abs(got_r[k] - meta["regression"]["test"][k]) for k in got_r)
    rep.add("Regressor: saved model reproduces the metadata test metrics", "PASS" if worst_r < 1e-6 else "FAIL", f"largest difference {worst_r:.2e}")

    # ---------------- baselines and thresholds
    base_acc = 1 - y_te.mean()
    b = meta["classification"]["baseline_test"]
    rep.add("Majority-class baseline accuracy equals 1 - test base rate", "PASS" if abs(b["test_accuracy"] - base_acc) < 1e-6 else "FAIL", f"{b['test_accuracy']:.5f} vs {base_acc:.5f}")
    rep.add("Majority-class baseline has F1 = 0 (it never predicts 'good')", "PASS" if b["test_f1"] == 0 else "WARN", f"{b['test_f1']}")
    rmse_base = float(np.sqrt(mean_squared_error(y_reg_te, np.full(len(y_reg_te), (tr['Price_in_Lakhs'] * mult).mean()))))
    rep.add("Predict-the-mean baseline RMSE equals the metadata", "PASS" if abs(rmse_base - meta["regression"]["baseline_test"]["test_rmse"]) < 1e-6 else "FAIL", f"{rmse_base:.4f}")
    c, r = meta["classification"], meta["regression"]
    c_ok = c["cv"]["cv_roc_auc_mean"] >= c["skill_threshold_cv_auc"]; r_ok = r["cv"]["cv_r2_mean"] >= r["skill_threshold_cv_r2"]
    rep.add("Classifier skill flag equals (CV AUC >= threshold)", "PASS" if c_ok == c["meets_pre_declared_skill_threshold"] else "FAIL",
            f"CV AUC {c['cv']['cv_roc_auc_mean']:.4f} vs threshold {c['skill_threshold_cv_auc']}: {'passes' if c_ok else 'FAILS'}")
    rep.add("Regressor skill flag equals (CV R2 >= threshold)", "PASS" if r_ok == r["meets_pre_declared_skill_threshold"] else "FAIL",
            f"CV R2 {r['cv']['cv_r2_mean']:.4f} vs threshold {r['skill_threshold_cv_r2']}: {'passes' if r_ok else 'FAILS'}")

    # ---------------- is there any signal at all? (optional, slow)
    if permutation:
        from sklearn.base import clone
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
        from sklearn.model_selection import KFold, StratifiedKFold, permutation_test_score
        from sklearn.pipeline import Pipeline
        rng_idx = np.random.RandomState(42).choice(len(tr), size=min(perm_rows, len(tr)), replace=False)
        sub = tr.iloc[rng_idx]
        y_c, y_r = apply_rule(sub, rule), sub["Price_in_Lakhs"] * mult
        rows = []
        for task, pipe, cols, y, scoring, est, cv in (
                ("classification", clf, cls_cols, y_c, "roc_auc",
                 HistGradientBoostingClassifier(max_iter=100, max_depth=4, min_samples_leaf=100, class_weight="balanced", random_state=42),
                 StratifiedKFold(3, shuffle=True, random_state=42)),
                ("regression", reg, reg_cols, y_r, "r2",
                 HistGradientBoostingRegressor(max_iter=100, max_depth=4, min_samples_leaf=100, random_state=42),
                 KFold(3, shuffle=True, random_state=42))):
            p = Pipeline([("prep", clone(pipe.named_steps["prep"])), ("model", est)])
            score, null, pval = permutation_test_score(p, sub[cols], y, scoring=scoring, cv=cv, n_permutations=n_perm, random_state=42, n_jobs=1)
            rows.append({"task": task, "metric": f"CV {scoring}", "observed_cv_score": score, "null_mean": float(np.mean(null)),
                         "null_std": float(np.std(null)), "null_max": float(np.max(null)), "p_value": float(pval),
                         "n_permutations": n_perm, "sample_rows": len(sub),
                         "reading": "indistinguishable from shuffled targets: no detectable signal" if pval > 0.05 else "differs from shuffled targets"})
            rep.add(f"Signal test ({task}): observed {scoring} vs shuffled-target null", "INFO" if pval > 0.05 else "WARN",
                    f"observed {score:.4f}, null mean {np.mean(null):.4f} (max {np.max(null):.4f}), p = {pval:.3f}")
        out = root / "phase3_outputs" / "signal_permutation_test.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        rep.add("Wrote phase3_outputs/signal_permutation_test.csv", "INFO", "")
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("REIA_PROJECT_ROOT", Path(__file__).resolve().parent.parent))
    ap.add_argument("--permutation", action="store_true")
    ap.add_argument("--n-permutations", type=int, default=10)
    a = ap.parse_args()
    report = verify(Path(a.root), a.permutation, a.n_permutations)
    counts = pd.Series([r["status"] for r in report.rows]).value_counts().to_dict()
    print("\nSUMMARY:", counts)
    sys.exit(1 if report.failed else 0)
