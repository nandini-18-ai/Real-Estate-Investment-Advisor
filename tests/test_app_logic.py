"""Logic tests for the Real Estate Investment Advisor app (no browser, no Streamlit needed).

Run from the project folder:   python tests/test_app_logic.py        (or:  python -m pytest tests -q)
Tests that need a saved project file print SKIP when that file is missing.
"""
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("REIA_PROJECT_ROOT", Path(__file__).resolve().parent.parent)).resolve()
sys.path.insert(0, str(ROOT / "app"))
import app_logic as L  # noqa: E402


class Skip(Exception):
    pass


def need(rel):
    p = ROOT / rel
    if not p.exists():
        raise Skip(f"{rel} not found")
    return p


def load_ready():
    art = L.load_artifacts(ROOT)
    if not art.ready:
        raise Skip("saved models/metadata/label rule not loadable: " + " | ".join(art.errors))
    return art


def sample_rows(n=2000):
    path = need("outputs/engineered_features.csv")
    return pd.read_csv(path).sample(n, random_state=7).reset_index(drop=True)


# ---------------------------------------------------------------- rule: app == notebook logic
def test_rule_matches_phase3_notebook_logic():
    art = load_ready()
    rows = sample_rows(3000)
    rule = art.label_rule
    keys = rows["City"] + "|" + rows["Property_Type"]
    medians = keys.map(rule["segment_median_ppsf_lakhs"]).fillna(rule["global_median_ppsf_lakhs"])
    quality = ((rows["BHK"] >= 3).astype(int) + (rows["Availability_Status"] == "Ready_to_Move").astype(int)
               + (rows["Parking_Space"] == "Yes").astype(int) + (rows["Security"] == "Yes").astype(int)
               + (rows["Public_Transport_Accessibility"] == "High").astype(int))
    expected = ((rows["Price_per_SqFt_exact"] <= medians) & (quality >= rule["quality_threshold"])).astype(int)
    got = np.array([int(L.apply_label_rule(r.to_dict(), rule)["meets_screen"]) for _, r in rows.iterrows()])
    assert (got == expected.values).all(), f"{(got != expected.values).sum()} rows differ between app rule and notebook rule"


def test_rule_unknown_segment_uses_global_median():
    art = load_ready()
    vals = {"City": "Atlantis", "Property_Type": "Villa", "Price_in_Lakhs": 100.0, "Size_in_SqFt": 2000.0, "BHK": 3,
            "Availability_Status": "Ready_to_Move", "Parking_Space": "Yes", "Security": "Yes", "Public_Transport_Accessibility": "High"}
    r = L.apply_label_rule(vals, art.label_rule)
    assert r["used_global_median"] is True
    assert abs(r["median_ppsf_lakhs"] - art.label_rule["global_median_ppsf_lakhs"]) < 1e-12


# ---------------------------------------------------------------- form -> model columns
def test_form_derivation_matches_training_columns_and_pipelines():
    art = load_ready()
    rows = sample_rows(40)
    meta = art.metadata
    amen = L.amenity_names(meta)
    for _, r in rows.iterrows():
        form = {k: r[k] for k in ["City", "Property_Type", "BHK", "Size_in_SqFt", "Price_in_Lakhs", "Year_Built", "Furnished_Status",
                                   "Facing", "Owner_Type", "Availability_Status", "Parking_Space", "Security",
                                   "Public_Transport_Accessibility", "Nearby_Schools", "Nearby_Hospitals"]}
        form["Amenities"] = [a for a in amen if r[f"Has_{a}"] == 1]
        out = L.run_assessment(art, form)
        direct_c = art.classifier.predict_proba(pd.DataFrame([r])[meta["classification"]["input_columns"]])[0, 1]
        direct_r = art.regressor.predict(pd.DataFrame([r])[meta["regression"]["input_columns"]])[0]
        assert abs(out["classifier"]["probability"] - direct_c) < 1e-9
        assert abs(out["regressor"]["scenario_price_lakhs"] - direct_r) < 1e-6
        assert abs(out["regressor"]["implied_current_price_lakhs"] * meta["growth_multiplier"] - direct_r) < 1e-6


def test_category_options_come_from_fitted_pipeline():
    art = load_ready()
    opts = L.get_options(art)
    assert len(opts["City"]) >= 2
    assert "Apartment" in opts["Property_Type"]
    assert opts["Public_Transport_Accessibility"] == ["Low", "Medium", "High"]


# ---------------------------------------------------------------- scenario arithmetic
def test_scenario_table_math():
    t = L.scenario_table(200.0, rates=(0.05, 0.08), years=5, assumed_rate=0.08)
    assert abs(t.iloc[1, 2] - round(200 * 1.08 ** 5, 2)) < 1e-9
    assert "assumed" in t.iloc[1, 0]
    assert abs(1.08 ** 5 - 1.4693280768) < 1e-9


def test_five_year_scenarios_use_asking_price_and_decimal_rates():
    table = L.scenario_table(254.0, rates=(0.0, 0.05, 0.08, 0.10), years=5, assumed_rate=0.08)
    assert table["Scenario price after 5 years (lakhs)"].tolist() == [254.0, 324.18, 373.21, 409.07]
    assert L.scenario_price_lakhs(254.0, 0.0, 5) == 254.0
    assert abs(L.scenario_price_lakhs(254.0, 0.05, 5) - 324.175516875) < 1e-9


def test_scenario_calculation_rejects_invalid_price_growth_and_horizon():
    for price in (0, -1, None, float("nan"), float("inf"), True):
        try:
            L.scenario_price_lakhs(price, 0.05)
        except ValueError as exc:
            assert "Asking price" in str(exc)
        else:
            raise AssertionError(f"invalid asking price {price!r} should be rejected")
    for rate in (-0.01, 0.151, 5, None, float("nan"), float("inf"), True):
        try:
            L.scenario_table(254.0, rates=(rate,))
        except ValueError as exc:
            assert "growth rate" in str(exc).lower()
        else:
            raise AssertionError(f"invalid decimal growth rate {rate!r} should be rejected")
    try:
        L.scenario_table(254.0, years=0)
    except ValueError as exc:
        assert "horizon" in str(exc).lower()
    else:
        raise AssertionError("a zero-year horizon should be rejected")


# ---------------------------------------------------------------- validation
def _form():
    return {"City": "Pune", "Property_Type": "Villa", "BHK": 3, "Size_in_SqFt": 2000.0, "Price_in_Lakhs": 200.0, "Year_Built": 2005,
            "Furnished_Status": "Furnished", "Facing": "East", "Owner_Type": "Owner", "Availability_Status": "Ready_to_Move",
            "Parking_Space": "Yes", "Security": "Yes", "Public_Transport_Accessibility": "High", "Nearby_Schools": 5,
            "Nearby_Hospitals": 5, "Amenities": []}


def _opts():
    o = {k: list(v) for k, v in L.FALLBACK_OPTIONS.items()}
    o["City"] = ["Pune", "Mumbai"]
    return o


def test_validation_accepts_good_input():
    e, w = L.validate_inputs(_form(), _opts(), L.DEFAULT_RANGES)
    assert e == [] and w == []


def test_validation_rejects_bad_input():
    cases = [("Size_in_SqFt", 0), ("Size_in_SqFt", -5), ("Size_in_SqFt", float("nan")), ("Price_in_Lakhs", 0), ("Price_in_Lakhs", None),
             ("Size_in_SqFt", float("inf")), ("Size_in_SqFt", L.MAX_SIZE_SQFT + 1),
             ("Price_in_Lakhs", -1), ("Price_in_Lakhs", float("inf")), ("Price_in_Lakhs", L.MAX_PRICE_LAKHS + 1),
             ("BHK", 0), ("BHK", 2.5), ("BHK", L.MAX_BHK + 1), ("Year_Built", 2030), ("Year_Built", 1700),
             ("Nearby_Schools", -1), ("Nearby_Hospitals", L.MAX_NEARBY_FACILITIES + 1),
             ("City", "Atlantis"), ("Facing", "Up"), ("Parking_Space", "Maybe")]
    for key, bad in cases:
        form = _form(); form[key] = bad
        e, _ = L.validate_inputs(form, _opts(), L.DEFAULT_RANGES)
        assert e, f"{key}={bad!r} should be rejected"


def test_validation_handles_missing_and_unexpected_values():
    errors, _ = L.validate_inputs(None, _opts(), L.DEFAULT_RANGES)
    assert errors and "provided as a form" in errors[0]
    form = _form()
    form["Unexpected_Feature"] = float("nan")
    errors, _ = L.validate_inputs(form, _opts(), L.DEFAULT_RANGES)
    assert any("Unexpected_Feature" in error and "finite number" in error for error in errors)
    form = _form()
    form["Size_in_SqFt"] = "not a size"
    errors, _ = L.validate_inputs(form, _opts(), L.DEFAULT_RANGES)
    assert any("Property size" in error for error in errors)


def test_run_assessment_rejects_invalid_values_before_model_prediction():
    art = load_ready()
    form = _form()
    form["Price_in_Lakhs"] = 0
    try:
        L.run_assessment(art, form)
    except ValueError as exc:
        assert "Asking price" in str(exc)
    else:
        raise AssertionError("run_assessment should reject a zero asking price")


def test_validation_warns_outside_training_range():
    form = _form(); form["Size_in_SqFt"] = 9000.0; form["Price_in_Lakhs"] = 900.0
    e, w = L.validate_inputs(form, _opts(), L.DEFAULT_RANGES)
    assert e == [] and any("outside the range" in x for x in w)
    form = _form(); form["BHK"] = 12
    e, w = L.validate_inputs(form, _opts(), L.DEFAULT_RANGES)
    assert e == [] and any("unusually small" in x for x in w)


# ---------------------------------------------------------------- loading problems are reported, never raised
def _copy_project(dst):
    import shutil
    for rel in ("models", "phase3_outputs"):
        if (ROOT / rel).exists():
            shutil.copytree(ROOT / rel, Path(dst) / rel)


def test_missing_files_reported():
    with tempfile.TemporaryDirectory() as d:
        art = L.load_artifacts(d)
        assert not art.ready and len([e for e in art.errors if e.startswith("Missing required file")]) == 4


def test_corrupt_model_file_reported():
    load_ready()
    with tempfile.TemporaryDirectory() as d:
        _copy_project(d)
        (Path(d) / "models" / "regressor_pipeline.joblib").write_bytes(b"not a model")
        art = L.load_artifacts(d)
        assert not art.ready and any("Could not load models/regressor_pipeline.joblib" in e for e in art.errors)


def test_bad_metadata_reported():
    load_ready()
    with tempfile.TemporaryDirectory() as d:
        _copy_project(d)
        meta_path = Path(d) / "models" / "model_metadata.json"
        meta_path.write_text("{ broken json")
        assert any("Could not read" in e for e in L.load_artifacts(d).errors)
        meta = json.loads((ROOT / "models" / "model_metadata.json").read_text())
        del meta["regression"]["input_columns"]
        meta_path.write_text(json.dumps(meta))
        assert any("input_columns" in e for e in L.load_artifacts(d).errors)


def test_bad_label_rule_reported():
    load_ready()
    with tempfile.TemporaryDirectory() as d:
        _copy_project(d)
        p = Path(d) / "phase3_outputs" / "label_rule.json"
        rule = json.loads(p.read_text()); rule["segment_median_ppsf_lakhs"] = {}
        p.write_text(json.dumps(rule))
        assert any("segment_median_ppsf_lakhs" in e for e in L.load_artifacts(d).errors)


# ---------------------------------------------------------------- reliability messages are fail-safe
def test_reliability_failsafe():
    assert L.reliability({})["classification"]["passed"] is False
    meta = {"classification": {"meets_pre_declared_skill_threshold": "true"}, "regression": {"meets_pre_declared_skill_threshold": True}}
    r = L.reliability(meta)
    assert r["classification"]["passed"] is False and r["regression"]["passed"] is True
    assert "FAILED" in r["classification"]["message"]


def test_real_metadata_flags_are_shown_consistently():
    art = load_ready()
    rel = L.reliability(art.metadata)
    for task in ("classification", "regression"):
        flag = art.metadata[task]["meets_pre_declared_skill_threshold"]
        assert rel[task]["passed"] is (flag is True)
        assert ("PASSED" in rel[task]["message"]) == (flag is True)


# ---------------------------------------------------------------- browsing helpers
def test_filter_properties():
    rows = sample_rows(1500)
    sub = L.filter_properties(rows, cities=[rows["City"].iloc[0]], bhk=(2, 4), price=(100, 300))
    assert len(sub) > 0 and sub["BHK"].between(2, 4).all() and sub["Price_in_Lakhs"].between(100, 300).all()
    assert set(sub["City"]) == {rows["City"].iloc[0]}
    assert len(L.filter_properties(rows)) == len(rows)


def test_base_rate_reader():
    df = pd.DataFrame({"model": ["REFERENCE: always 'good'", "x"], "precision": [0.2459, 0.3]})
    assert abs(L.base_rate_from_imbalance(df) - 0.2459) < 1e-12
    assert L.base_rate_from_imbalance(pd.DataFrame({"a": [1]})) is None



# ---------------------------------------------------------------- Phase 4 fixes: formatting, environment, self-check
def test_indian_formatting():
    assert L.indian_number(1234567) == "12,34,567"
    assert L.indian_number(100000) == "1,00,000"
    assert L.indian_number(999) == "999"
    assert L.indian_number(-12345.5, 1) == "-12,345.5"
    assert L.format_lakhs(373.7) == "₹373.7 lakh (₹3.74 crore)"
    assert L.format_lakhs(48.2) == "₹48.2 lakh"
    assert L.format_lakhs(250, with_crore=False) == "₹250.0 lakh"
    assert L.format_inr(9245.4) == "₹9,245"
    assert L.format_ppsf(0.0924) == "₹9,240/sq ft"


def test_startup_self_check_passes_on_real_models():
    art = load_ready()
    assert L.self_check(art) is None


def test_self_check_detects_a_model_that_cannot_predict():
    import copy

    class Broken:
        classes_ = [0, 1]
        named_steps = {}
        feature_names_in_ = []
        def predict_proba(self, X):
            raise RuntimeError("incompatible pickle")
        def predict(self, X):
            raise RuntimeError("incompatible pickle")

    art = copy.copy(load_ready())
    art.classifier = Broken()
    problem = L.self_check(art)
    assert problem and "failed a start-up test prediction" in problem and "incompatible pickle" in problem


def test_version_mismatch_is_a_warning_not_an_error():
    load_ready()
    with tempfile.TemporaryDirectory() as d:
        _copy_project(d)
        p = Path(d) / "models" / "model_metadata.json"
        meta = json.loads(p.read_text(encoding="utf-8"))
        meta.setdefault("library_versions", {})["scikit_learn"] = "0.0.1"
        p.write_text(json.dumps(meta))
        art = L.load_artifacts(d)
        assert art.ready, art.errors
        assert any("trained with scikit-learn 0.0.1" in w for w in art.warnings)


def test_model_family_note_logic():
    fallback = {"classification": {"model": "CLS primary: HistGradientBoosting (XGBoost not installed)"},
                "regression": {"model": "REG primary: HistGradientBoosting (XGBoost not installed)"}}
    assert L.model_family_note(fallback, {"xgboost": "3.1.3"})["level"] == "warn"
    assert "Re-run the Phase 3 notebook" in L.model_family_note(fallback, {"xgboost": None})["text"]
    xgb = {"classification": {"model": "CLS primary: XGBoost"}, "regression": {"model": "REG primary: XGBoost"}}
    assert L.model_family_note(xgb, {"xgboost": None})["level"] == "fail"
    assert L.model_family_note(xgb, {"xgboost": "3.1.3"})["level"] == "ok"


def test_environment_report_keys():
    env = L.environment_report()
    assert {"python", "executable", "sklearn", "xgboost", "pandas", "numpy", "streamlit"} <= set(env)
    assert env["sklearn"] is not None


def test_threshold_table_and_reliability_agree_with_metadata():
    art = load_ready()
    t = L.threshold_table(art.metadata)
    rel = L.reliability(art.metadata)
    assert list(t["Task"]) == ["Classification", "Regression"]
    assert (t.loc[0, "Meets threshold"] == "Yes") == rel["classification"]["passed"]
    assert (t.loc[1, "Meets threshold"] == "Yes") == rel["regression"]["passed"]


def test_confusion_from_review():
    df = pd.DataFrame({"model": ["m1"], "TN": [10], "FP": [2], "FN": [3], "TP": [5]})
    assert L.confusion_from_review(df, "m1").tolist() == [[10, 2], [3, 5]]
    assert L.confusion_from_review(df, "missing") is None


def test_chart_runs_excludes_leakage_demonstrations():
    df = pd.DataFrame({"run_name": ["CLS baseline: Dummy", "CLS primary: A", "CLS rule-recovery (LEAKY): A"],
                       "experiment_type": ["baseline", "primary", "rule_recovery"],
                       "cv_roc_auc_mean": [0.5, 0.51, 0.99], "cv_roc_auc_std": [0, 0.001, 0.001]})
    out = L.chart_runs(df, "classification")
    assert list(out["label"]) == ["Dummy", "A"]


def test_city_spread():
    df = pd.DataFrame({"City": ["A", "A", "B", "B"], "Price_in_Lakhs": [100, 120, 200, 220]})
    s = L.city_spread(df)
    assert s["lowest_city"] == "A" and s["highest_city"] == "B" and s["n_cities"] == 2 and s["spread_pct"] > 0


def test_results_csv_matches_metadata_for_the_selected_models():
    art = load_ready()
    meta = art.metadata
    cls_csv = L.read_optional_csv(ROOT, "phase3_outputs/classification_results.csv")
    reg_csv = L.read_optional_csv(ROOT, "phase3_outputs/regression_results.csv")
    if cls_csv is None or reg_csv is None:
        raise Skip("results CSV files not found")
    c = cls_csv[cls_csv["run_name"] == meta["classification"]["model"]].iloc[0]
    r = reg_csv[reg_csv["run_name"] == meta["regression"]["model"]].iloc[0]
    assert abs(c["test_roc_auc"] - meta["classification"]["test"]["test_roc_auc"]) < 1e-9
    assert abs(c["cv_roc_auc_mean"] - meta["classification"]["cv"]["cv_roc_auc_mean"]) < 1e-9
    assert abs(r["test_r2"] - meta["regression"]["test"]["test_r2"]) < 1e-9
    assert abs(r["cv_r2_mean"] - meta["regression"]["cv"]["cv_r2_mean"]) < 1e-9


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    passed = failed = skipped = 0
    for name, fn in tests:
        try:
            fn(); passed += 1; print(f"PASS  {name}")
        except Skip as why:
            skipped += 1; print(f"SKIP  {name}: {why}")
        except Exception:
            failed += 1; print(f"FAIL  {name}"); traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed, {skipped} skipped")
    sys.exit(1 if failed else 0)
