# Phase 3 patch — MLflow skops error + class-imbalance review

Original notebook: `02_ML_Model_Training.ipynb` (unchanged logic). Two cells are **inserted**; nothing else is edited.

## Steps
1. Open your notebook. Insert the **PATCH 1** code cell directly after the Section 9 cell that ends with `print("Helpers ready.")`
   (it must run *before* any `run_classification(...)` call).
2. Insert the **PATCH 2** code cell directly after the Section 11.5 cell that prints `AUTOMATIC READ-OUT (classification)`
   (just before the "## 12. Regression" heading).
3. Save, then **Kernel -> Restart & Run All** (about 12 minutes on a 1-CPU machine).
   Do not re-run single cells on the old kernel: result lists would contain duplicate entries.
   Old half-finished MLflow runs stay in `mlflow/mlflow.db` under the old session id; delete the `mlflow/` folder first for a clean registry.

A ready-made copy with both cells inserted is `02_ML_Model_Training_patched.ipynb`.

## PATCH 1 code
```python
# PATCH 1 - safe MLflow model logging (skops format with a reviewed allowlist of trusted types)
import types, inspect

ALLOWED_SKOPS_PREFIXES = ("sklearn.", "numpy.", "xgboost.")   # extend ONLY after reading the printed type list
FALLBACK_TO_PICKLE = False   # True = accept pickle-style model files if skops fails (only ever load models you created yourself)
SKOPS_REVIEW, _SEEN_TYPES = [], set()

def _supports_skops():
    """True if this MLflow version can save scikit-learn models in the skops format."""
    params = inspect.signature(mlflow.sklearn.log_model).parameters
    return "skops_trusted_types" in params or hasattr(mlflow.sklearn, "SERIALIZATION_FORMAT_SKOPS")

def _review_skops_types(pipe):
    """Ask skops which types this pipeline needs trusted; split them into approved / rejected."""
    import skops.io as sio
    untrusted = sorted(sio.get_untrusted_types(data=sio.dumps(pipe)))
    approved = [t for t in untrusted if t.startswith(ALLOWED_SKOPS_PREFIXES)]
    rejected = [t for t in untrusted if t not in approved]
    return approved, rejected

def _call_log_model(pipe, signature, **extra):
    try:                                                   # MLflow 3.x argument name
        return mlflow.sklearn.log_model(pipe, name="model", signature=signature, **extra)
    except TypeError:                                      # MLflow 2.x argument name
        return mlflow.sklearn.log_model(pipe, artifact_path="model", signature=signature, **extra)

def log_sklearn_model(pipe, signature):
    """Log a fitted pipeline to the ACTIVE run. Never raises; returns an object with .model_uri (None on failure)."""
    failed = types.SimpleNamespace(model_uri=None)
    try:
        if _supports_skops():
            approved, rejected = _review_skops_types(pipe)
            SKOPS_REVIEW.append({"run_id": mlflow.active_run().info.run_id, "approved": approved, "rejected": rejected})
            (PHASE3_DIR / "skops_types_review.json").write_text(json.dumps(SKOPS_REVIEW, indent=2))
            new_types = [t for t in approved if t not in _SEEN_TYPES]
            if new_types:
                print("   skops: newly approved types (allowlist prefixes", ALLOWED_SKOPS_PREFIXES, "):", new_types)
                _SEEN_TYPES.update(new_types)
            if rejected:
                raise RuntimeError(f"types outside the allowlist were NOT trusted: {rejected}")
            return _call_log_model(pipe, signature, serialization_format="skops", skops_trusted_types=approved)
        return _call_log_model(pipe, signature)            # older MLflow without skops support: its own default
    except Exception as err:
        msg = f"{type(err).__name__}: {str(err)[:300]}"
        print("   WARNING: model NOT saved to MLflow for this run ->", msg)
        mlflow.set_tags({"model_logging_failed": "True", "model_logging_error": msg[:500]})
        if FALLBACK_TO_PICKLE:
            try:
                info = _call_log_model(pipe, signature, serialization_format="cloudpickle")
                mlflow.set_tag("model_serialization_fallback", "cloudpickle")
                print("   fallback: saved with cloudpickle (pickle-style; load only models you created yourself)")
                return info
            except Exception as err2:
                print("   WARNING: pickle fallback also failed ->", f"{type(err2).__name__}: {str(err2)[:200]}")
        return failed

print("Patched log_sklearn_model is active. FALLBACK_TO_PICKLE =", FALLBACK_TO_PICKLE)
```

## PATCH 2 code
```python
# PATCH 2 - class-imbalance review for the Good Investment classifier
n_test = len(y_test_cls); n_pos = int(y_test_cls.sum()); n_neg = n_test - n_pos
prev = n_pos / n_test

refs = [
    {"model": "REFERENCE: always 'not good' (majority class)", "accuracy": 1 - prev, "precision": 0.0, "recall": 0.0, "f1": 0.0,
     "roc_auc": 0.5, "pr_auc": prev, "predicted_good_rate": 0.0, "TN": n_neg, "FP": 0, "FN": n_pos, "TP": 0},
    {"model": "REFERENCE: always 'good'", "accuracy": prev, "precision": prev, "recall": 1.0, "f1": 2 * prev / (1 + prev),
     "roc_auc": 0.5, "pr_auc": prev, "predicted_good_rate": 1.0, "TN": 0, "FP": n_neg, "FN": 0, "TP": n_pos},
    {"model": "REFERENCE: random guess at the base rate (expected values)", "accuracy": prev ** 2 + (1 - prev) ** 2,
     "precision": prev, "recall": prev, "f1": prev, "roc_auc": 0.5, "pr_auc": prev, "predicted_good_rate": prev,
     "TN": np.nan, "FP": np.nan, "FN": np.nan, "TP": np.nan},
]
rows = []
for r in CLS_RESULTS:
    tn, fp, fn, tp = [int(v) for v in r["cm"].ravel()]
    rows.append({"model": r["run_name"], "accuracy": r["test_accuracy"], "precision": r["test_precision"], "recall": r["test_recall"],
                 "f1": r["test_f1"], "roc_auc": r["test_roc_auc"], "pr_auc": r["test_pr_auc"],
                 "predicted_good_rate": (tp + fp) / n_test, "TN": tn, "FP": fp, "FN": fn, "TP": tp})
imbalance = pd.DataFrame(refs + rows)
imbalance["pr_auc_minus_base_rate"] = imbalance["pr_auc"] - prev
always_good_f1 = 2 * prev / (1 + prev)
imbalance["f1_beats_always_good"] = [None if m.startswith("REFERENCE") else bool(f > always_good_f1 + 1e-12)
                                     for m, f in zip(imbalance["model"], imbalance["f1"])]
display(imbalance.round(4))
imbalance.to_csv(PHASE3_DIR / "classification_imbalance_review.csv", index=False)

dummy = [r for r in CLS_RESULTS if r["experiment_type"] == "baseline"][0]
d_tn, d_fp, d_fn, d_tp = [int(v) for v in dummy["cm"].ravel()]
print(f"\nBase rate of 'good' in the test set: {prev:.4f} ({n_pos:,} of {n_test:,}).")
print(f"Majority-class baseline: accuracy {dummy['test_accuracy']:.4f}, F1 {dummy['test_f1']:.4f}; confusion matrix "
      f"TN={d_tn:,} FP={d_fp:,} FN={d_fn:,} TP={d_tp:,}.")
print("  -> TP = 0 and FP = 0: it never predicts 'good', so precision/recall/F1 for 'good' are 0 by definition.")
print(f"Reference points: ROC AUC chance = 0.5; PR AUC chance = base rate = {prev:.4f}; "
      f"F1 of 'always good' = {always_good_f1:.4f}; F1 of random guessing = {prev:.4f}.")

honest = [r for r in CLS_RESULTS if r["experiment_type"] == "primary"]
best_h = max(honest, key=lambda r: r["cv_roc_auc_mean"])
threshold = 0.5 + MIN_AUC_ABOVE_CHANCE
print("\nHonest models versus the references:")
for r in honest:
    print(f"- {r['run_name']}: ROC AUC {r['test_roc_auc']:.4f}, PR AUC {r['test_pr_auc']:.4f} (base rate {prev:.4f}), "
          f"F1 {r['test_f1']:.4f} (always-good {always_good_f1:.4f}), accuracy {r['test_accuracy']:.4f} (majority {1 - prev:.4f})")
if best_h["cv_roc_auc_mean"] >= threshold:
    print(f"\nBest honest model ({best_h['run_name']}) meets the pre-declared threshold (CV ROC AUC {best_h['cv_roc_auc_mean']:.4f} >= {threshold:.2f}).")
    print("That is the minimum bar only: it still describes a rule-based label, not real investment outcomes.")
else:
    print(f"\nNOT RELIABLE: the best honest model ({best_h['run_name']}) has CV ROC AUC {best_h['cv_roc_auc_mean']:.4f}, below the "
          f"pre-declared threshold of {threshold:.2f}. It must not be presented as able to identify 'Good Investment' properties.")
    print("Its accuracy and F1 should not be quoted without the reference rows above.")
```


---

# Added in the Phase 4 fixes (notebook now contains PATCH 3, 4 and 5)

### PATCH 3 — environment guard (insert right after the Section 2 cell that prints the library versions)
The notebook, the saved models and the Streamlit app must all run in **one** Python environment. If the notebook's kernel is a different interpreter from the one where `pip install -r requirements.txt` was run (the usual cause of "xgboost is listed but not available" and of a scikit-learn 1.7.2 vs 1.8.0 mismatch), this cell stops early and prints the exact fix instead of silently training with a fallback model.

```python
# PATCH 3 - environment guard
import sys

REQUIRED_SKLEARN = "1.8.0"       # same exact pin as requirements.txt
STRICT_ENVIRONMENT = True        # True = stop with instructions when the environment is wrong; False = print a warning only

print("Python interpreter used by THIS notebook kernel:", sys.executable)
print("scikit-learn:", sklearn.__version__, "| xgboost:", __import__("xgboost").__version__ if HAS_XGB else "NOT importable in this kernel")

problems = []
if sklearn.__version__ != REQUIRED_SKLEARN:
    problems.append(f"scikit-learn is {sklearn.__version__} in this kernel, but requirements.txt pins {REQUIRED_SKLEARN}. "
                    "Models saved here could not be loaded safely by an app running a different version.")
if not HAS_XGB:
    problems.append("xgboost cannot be imported in this kernel, so a HistGradientBoosting fallback would be used instead of XGBoost.")

if problems:
    message = ("ENVIRONMENT PROBLEM:\n- " + "\n- ".join(problems) +
               "\n\nFix: in a terminal with the project's virtual environment ACTIVE run\n"
               "    python -m pip install -r requirements-dev.txt\n"
               "    python -m ipykernel install --user --name reia --display-name \"Python (reia)\"\n"
               "then in Jupyter choose Kernel -> Change kernel -> 'Python (reia)', restart, and run the notebook again.\n"
               "(Set STRICT_ENVIRONMENT = False only if you knowingly accept the fallback.)")
    if STRICT_ENVIRONMENT:
        raise RuntimeError(message)
    print("WARNING:", message)
else:
    print("PASS: scikit-learn matches the pin and xgboost is available in this kernel.")
```

### PATCH 4 — is there any signal to find? (insert right after the cell that shows the "LEAKY model" importance, before Section 14)
Re-trains a fast boosted model on a sample of the training rows with the **targets randomly shuffled** several times. If the real cross-validated score is no better than the shuffled scores, the honest inputs contain no detectable information about the target — which would explain why no model can clear its threshold. The result is saved for the app.

```python
# PATCH 4 - permutation (shuffled-target) signal test
from sklearn.model_selection import permutation_test_score
from sklearn.ensemble import HistGradientBoostingClassifier as _HGBC, HistGradientBoostingRegressor as _HGBR

PERM_ROWS = 10_000 if QUICK_RUN else 40_000
PERM_N = 5 if QUICK_RUN else 10
sub_idx = np.random.RandomState(SEED).choice(len(df_train), size=min(PERM_ROWS, len(df_train)), replace=False)

signal_rows = []
for task, feats, y, scoring, est, cv in (
        ("classification", CLS_FEATURES, y_train_cls, "roc_auc",
         _HGBC(max_iter=100, max_depth=4, min_samples_leaf=100, class_weight="balanced", random_state=SEED), CLS_CV),
        ("regression", REG_FEATURES, y_train_reg, "r2",
         _HGBR(max_iter=100, max_depth=4, min_samples_leaf=100, random_state=SEED), REG_CV)):
    pipe = Pipeline([("prep", build_preprocessor(feats, task)), ("model", est)])
    score, null_scores, pval = permutation_test_score(pipe, df_train[feats].iloc[sub_idx], y.iloc[sub_idx], scoring=scoring,
                                                      cv=cv, n_permutations=PERM_N, random_state=SEED, n_jobs=1)
    signal_rows.append({"task": task, "metric": f"CV {scoring}", "observed_cv_score": score, "null_mean": float(np.mean(null_scores)),
                        "null_std": float(np.std(null_scores)), "null_max": float(np.max(null_scores)), "p_value": float(pval),
                        "n_permutations": PERM_N, "sample_rows": len(sub_idx),
                        "reading": "indistinguishable from shuffled targets: no detectable signal" if pval > 0.05 else "differs from shuffled targets"})
signal_table = pd.DataFrame(signal_rows)
display(signal_table.round(4))
signal_table.to_csv(PHASE3_DIR / "signal_permutation_test.csv", index=False)
for r in signal_rows:
    print(f"{r['task'].capitalize():<15s} observed {r['observed_cv_score']:.4f} | shuffled-target scores: mean {r['null_mean']:.4f}, max {r['null_max']:.4f} | p = {r['p_value']:.3f} -> {r['reading']}")
print("\nA p-value above 0.05 means the real score cannot be told apart from scores obtained with meaningless targets.")
```

### PATCH 5 — record the training environment (insert right after the Section 14 cell that saves the pipelines)
Adds the Python/library versions and the model family (XGBoost or the fallback) to `models/model_metadata.json`, so the app can show exactly what the models were trained with.

```python
# PATCH 5 - record the training environment inside model_metadata.json
meta_path = MODELS_DIR / "model_metadata.json"
meta_saved = json.loads(meta_path.read_text(encoding="utf-8"))
family_text = f"{champ_cls['run_name']} {champ_reg['run_name']}"
meta_saved["environment"] = {
    "python_executable": sys.executable, "python": sys.version.split()[0], "scikit_learn": sklearn.__version__,
    "xgboost": __import__("xgboost").__version__ if HAS_XGB else None, "pandas": pd.__version__, "numpy": np.__version__,
    "required_sklearn_pin": REQUIRED_SKLEARN, "strict_environment": STRICT_ENVIRONMENT,
}
meta_saved["model_family"] = {"uses_xgboost": ("XGBoost" in family_text) and ("not installed" not in family_text),
                              "fallback_used": "XGBoost not installed" in family_text}
meta_path.write_text(json.dumps(meta_saved, indent=2), encoding="utf-8")
print("Recorded environment:", meta_saved["environment"]["scikit_learn"], "| XGBoost model family:", meta_saved["model_family"])
```