"""Pure-Python logic for the Real Estate Investment Advisor app.

No Streamlit imports here, so everything in this file can be unit-tested without a browser.

What the app is allowed to claim (Phase 3 decisions):
  * a SCENARIO estimate of the price after 5 years under an ASSUMED annual growth rate (8%), not a market forecast;
  * a RULE-BASED screen ("Good Investment" label) - not proof of profitability and not a predicted return.
"""
from __future__ import annotations

import json
import math
import re
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------------------------
# Constants (documented in the Phase 2 / Phase 3 notebooks)
# ----------------------------------------------------------------------------------------------
REFERENCE_YEAR = 2025            # Age_of_Property = 2025 - Year_Built in the dataset
RUPEES_PER_LAKH = 100_000        # ASSUMPTION used since Phase 2: the currency is rupees
SUPPORTED_RULE_VERSIONS = {"v1"}
SCENARIO_RATES = (0.05, 0.08, 0.10)
MAX_SIZE_SQFT = 100_000
MAX_PRICE_LAKHS = 1_000_000
MAX_BHK = 20
MAX_NEARBY_FACILITIES = 100

REQUIRED_FILES = {
    "classifier": "models/classifier_pipeline.joblib",
    "regressor": "models/regressor_pipeline.joblib",
    "metadata": "models/model_metadata.json",
    "label_rule": "phase3_outputs/label_rule.json",
}
OPTIONAL_FILES = {
    "engineered_features": "outputs/engineered_features.csv",
    "classification_results": "phase3_outputs/classification_results.csv",
    "regression_results": "phase3_outputs/regression_results.csv",
    "imbalance_review": "phase3_outputs/classification_imbalance_review.csv",
    "uncertainty": "phase3_outputs/test_uncertainty.csv",
    "floor_slices": "phase3_outputs/floor_flag_slices.csv",
    "importance_cls": "phase3_outputs/importance_cls_champion.csv",
    "importance_reg": "phase3_outputs/importance_reg_champion.csv",
}

# Ranges seen in the training data (Phase 2 profile). Replaced by values computed from the data when it is available.
DEFAULT_RANGES = {
    "Size_in_SqFt": (500.0, 5000.0), "Price_in_Lakhs": (10.0, 500.0), "BHK": (1, 5),
    "Age_of_Property": (2, 35), "Nearby_Schools": (1, 10), "Nearby_Hospitals": (1, 10),
    "Year_Built": (1990, 2023), "ppsf_lakhs": (0.002, 0.992),
}
# Category lists seen in the training data; used only if they cannot be read from the fitted pipelines.
FALLBACK_OPTIONS = {
    "Property_Type": ["Apartment", "Independent House", "Villa"],
    "Furnished_Status": ["Furnished", "Semi-furnished", "Unfurnished"],
    "Owner_Type": ["Broker", "Builder", "Owner"],
    "Facing": ["East", "North", "South", "West"],
    "Parking_Space": ["No", "Yes"], "Security": ["No", "Yes"],
    "Availability_Status": ["Ready_to_Move", "Under_Construction"],
    "Public_Transport_Accessibility": ["Low", "Medium", "High"],
}
DEFAULT_FORM = {
    "Size_in_SqFt": 2750.0, "Price_in_Lakhs": 250.0, "BHK": 3, "Year_Built": 2005,
    "Nearby_Schools": 5, "Nearby_Hospitals": 5,
}
CATEGORICAL_FORM_FIELDS = ["City", "Property_Type", "Furnished_Status", "Owner_Type", "Facing", "Parking_Space",
                           "Security", "Availability_Status", "Public_Transport_Accessibility"]
BROWSE_COLUMNS = ["ID", "State", "City", "Property_Type", "BHK", "Size_in_SqFt", "Price_in_Lakhs", "Year_Built",
                  "Age_of_Property", "Furnished_Status", "Availability_Status", "Parking_Space", "Security",
                  "Public_Transport_Accessibility", "Facing", "Owner_Type", "Price_per_SqFt_INR"]


def slug(text: str) -> str:
    """Same file-name slug the Phase 3 notebook used for figure names."""
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()


def fmt_num(x, digits: int = 4) -> str:
    try:
        if x is None or (isinstance(x, float) and not math.isfinite(x)):
            return "n/a"
        return f"{float(x):,.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


# ----------------------------------------------------------------------------------------------
# Loading and validating the saved project files
# ----------------------------------------------------------------------------------------------
@dataclass
class Artifacts:
    root: Path
    classifier: object = None
    regressor: object = None
    metadata: dict = field(default_factory=dict)
    label_rule: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)       # problems that block predictions
    warnings: list = field(default_factory=list)     # problems worth showing but not blocking
    status: list = field(default_factory=list)       # one row per file, for the sidebar

    @property
    def ready(self) -> bool:
        return not self.errors and self.classifier is not None and self.regressor is not None


def validate_metadata(meta) -> list:
    problems = []
    if not isinstance(meta, dict):
        return ["model_metadata.json is not a JSON object."]
    for task in ("classification", "regression"):
        section = meta.get(task)
        if not isinstance(section, dict):
            problems.append(f"model_metadata.json has no '{task}' section.")
            continue
        cols = section.get("input_columns")
        if not isinstance(cols, list) or not cols or not all(isinstance(c, str) for c in cols):
            problems.append(f"model_metadata.json: '{task}.input_columns' is missing or invalid.")
    for key in ("growth_rate_assumed", "growth_multiplier"):
        value = meta.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            problems.append(f"model_metadata.json: '{key}' is missing or invalid.")
    return problems


def validate_label_rule(rule) -> list:
    problems = []
    if not isinstance(rule, dict):
        return ["label_rule.json is not a JSON object."]
    seg = rule.get("segment_median_ppsf_lakhs")
    if not isinstance(seg, dict) or not seg:
        problems.append("label_rule.json: 'segment_median_ppsf_lakhs' is missing or empty.")
    g = rule.get("global_median_ppsf_lakhs")
    if isinstance(g, bool) or not isinstance(g, (int, float)) or not math.isfinite(g) or g <= 0:
        problems.append("label_rule.json: 'global_median_ppsf_lakhs' is missing or invalid.")
    q = rule.get("quality_threshold")
    if isinstance(q, bool) or not isinstance(q, int) or not 0 <= q <= 5:
        problems.append("label_rule.json: 'quality_threshold' is missing or invalid.")
    if rule.get("rule_version") not in SUPPORTED_RULE_VERSIONS:
        problems.append(f"label_rule.json: rule_version {rule.get('rule_version')!r} is not supported by this app "
                        f"(supported: {sorted(SUPPORTED_RULE_VERSIONS)}).")
    return problems


def _read_json(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_pipeline(path: Path, kind: str):
    """Load one pipeline. Returns (pipeline, [version messages]). Only load files you created yourself."""
    import joblib
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        obj = joblib.load(path)
    needed = "predict_proba" if kind == "classifier" else "predict"
    if not hasattr(obj, "named_steps") or not hasattr(obj, needed):
        raise TypeError(f"the file does not contain a fitted scikit-learn Pipeline with '{needed}'")
    versions = set()
    for w in caught:
        m = re.search(r"from version (\S+) when using version (\S+)", str(w.message))
        if m:
            versions.add((m.group(1), m.group(2)))
    msgs = [f"scikit-learn reports that the {kind} file was saved with version {a} but is being loaded with {b}."
            for a, b in sorted(versions)]
    return obj, msgs


def load_artifacts(root) -> Artifacts:
    """Load models, metadata and label rule. Never raises: problems are collected in .errors / .warnings."""
    root = Path(root)
    art = Artifacts(root=root)
    hint = " Run the patched Phase 3 notebook (02_ML_Model_Training_patched.ipynb) from the project folder first."

    paths = {key: root / rel for key, rel in REQUIRED_FILES.items()}
    for key, rel in REQUIRED_FILES.items():
        art.status.append({"file": rel, "required": True, "found": paths[key].exists(), "ok": False, "note": ""})
    row = {r["file"]: r for r in art.status}

    def fail(key, message):
        art.errors.append(message)
        row[REQUIRED_FILES[key]]["note"] = message

    for key, rel in REQUIRED_FILES.items():
        if not paths[key].exists():
            fail(key, f"Missing required file: {rel}.{hint}")

    if paths["metadata"].exists():
        try:
            art.metadata = _read_json(paths["metadata"])
            problems = validate_metadata(art.metadata)
            if problems:
                art.metadata = {}
                fail("metadata", " ".join(problems))
            else:
                row[REQUIRED_FILES["metadata"]]["ok"] = True
        except Exception as exc:                                     # corrupt JSON, permissions, ...
            art.metadata = {}
            fail("metadata", f"Could not read {REQUIRED_FILES['metadata']}: {type(exc).__name__}: {exc}")

    if paths["label_rule"].exists():
        try:
            art.label_rule = _read_json(paths["label_rule"])
            problems = validate_label_rule(art.label_rule)
            if problems:
                art.label_rule = {}
                fail("label_rule", " ".join(problems))
            else:
                row[REQUIRED_FILES["label_rule"]]["ok"] = True
        except Exception as exc:
            art.label_rule = {}
            fail("label_rule", f"Could not read {REQUIRED_FILES['label_rule']}: {type(exc).__name__}: {exc}")

    for key, attr in (("classifier", "classifier"), ("regressor", "regressor")):
        if not paths[key].exists():
            continue
        try:
            pipe, version_msgs = _load_pipeline(paths[key], key)
            setattr(art, attr, pipe)
            row[REQUIRED_FILES[key]]["ok"] = True
            for msg in version_msgs:
                if msg not in art.warnings:
                    art.warnings.append(msg)
        except Exception as exc:
            extra = " If the message mentions 'xgboost', run: pip install xgboost." if "xgboost" in str(exc).lower() else ""
            fail(key, f"Could not load {REQUIRED_FILES[key]}: {type(exc).__name__}: {str(exc)[:300]}.{extra} "
                      "(Pipelines must be loaded with the same scikit-learn version that saved them.)")

    for key, rel in OPTIONAL_FILES.items():
        art.status.append({"file": rel, "required": False, "found": (root / rel).exists(), "ok": (root / rel).exists(), "note": ""})

    # ---- consistency warnings (non-blocking) ----
    if art.metadata:
        trained = (art.metadata.get("library_versions") or {}).get("scikit_learn")
        try:
            import sklearn
            if trained and trained != sklearn.__version__:
                art.warnings.append(f"The models were trained with scikit-learn {trained}, but {sklearn.__version__} is installed. "
                                    "The app runs a test prediction at start-up (it passed if you can use the Assessment page), and "
                                    "tools/verify_artifacts.py checks that the saved models reproduce their reported metrics. "
                                    "To remove this notice, re-run the Phase 3 notebook in this same environment.")
        except ImportError:
            art.warnings.append("scikit-learn is not installed; models cannot be used.")
        if art.metadata.get("quick_run") is True:
            art.warnings.append("These models come from a QUICK_RUN trial (small data sample). Re-run the Phase 3 notebook with "
                                "QUICK_RUN = False before using any numbers.")
        rate = art.metadata.get("growth_rate_assumed")
        years = art.metadata.get("horizon_years", 5)
        mult = art.metadata.get("growth_multiplier")
        if isinstance(rate, (int, float)) and isinstance(mult, (int, float)) and isinstance(years, int):
            if abs((1 + rate) ** years - mult) > 1e-6:
                art.warnings.append("growth_multiplier in model_metadata.json does not equal (1 + growth_rate) ** horizon_years.")
        for key, attr in (("classification", "classifier"), ("regression", "regressor")):
            pipe = getattr(art, attr)
            cols = (art.metadata.get(key) or {}).get("input_columns")
            seen = getattr(pipe, "feature_names_in_", None)
            if pipe is not None and cols and seen is not None and set(cols) != set(seen):
                art.errors.append(f"The {key} pipeline expects columns {sorted(seen)} but model_metadata.json lists {sorted(cols)}. "
                                  "Re-run the Phase 3 notebook so both files come from the same run.")
    if art.ready:                                   # prove the loaded models can actually predict
        problem = self_check(art)
        if problem:
            art.errors.append(problem)
    return art


def read_optional_csv(root, rel: str):
    path = Path(root) / rel
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def load_browse_data(root):
    path = Path(root) / OPTIONAL_FILES["engineered_features"]
    if not path.exists():
        return None
    try:
        header = pd.read_csv(path, nrows=0).columns
        cols = [c for c in BROWSE_COLUMNS if c in header]
        if not {"City", "Price_in_Lakhs", "Size_in_SqFt"} <= set(cols):
            return None
        return pd.read_csv(path, usecols=cols)
    except Exception:
        return None


def figure_path(root, name: str):
    p = Path(root) / "phase3_outputs" / "figures" / name
    return p if p.exists() else None


# ----------------------------------------------------------------------------------------------
# Inputs: options, ranges, defaults, validation
# ----------------------------------------------------------------------------------------------
def category_options(pipe) -> dict:
    """Read the categories each encoder learned during training, so the app offers exactly those values."""
    out = {}
    try:
        prep = pipe.named_steps["prep"]
        for _, transformer, cols in prep.transformers_:
            if isinstance(transformer, str) or not hasattr(transformer, "categories_"):
                continue
            for col, cats in zip(cols, transformer.categories_):
                out[col] = [str(c) for c in cats]
    except Exception:
        return {}
    return out


def get_options(art: Artifacts, browse_df=None) -> dict:
    opts = {}
    for pipe in (art.classifier, art.regressor):
        if pipe is not None:
            for k, v in category_options(pipe).items():
                opts.setdefault(k, v)
    for k, v in FALLBACK_OPTIONS.items():
        opts.setdefault(k, list(v))
    if "City" not in opts and browse_df is not None and "City" in browse_df:
        opts["City"] = sorted(browse_df["City"].dropna().unique().tolist())
    return opts


def numeric_ranges(browse_df=None) -> dict:
    ranges = dict(DEFAULT_RANGES)
    if browse_df is None or len(browse_df) == 0:
        return ranges
    for col in ("Size_in_SqFt", "Price_in_Lakhs", "BHK", "Age_of_Property", "Nearby_Schools", "Nearby_Hospitals", "Year_Built"):
        if col in browse_df:
            ranges[col] = (browse_df[col].min().item(), browse_df[col].max().item())
    if {"Price_in_Lakhs", "Size_in_SqFt"} <= set(browse_df.columns):
        ppsf = browse_df["Price_in_Lakhs"] / browse_df["Size_in_SqFt"]
        ranges["ppsf_lakhs"] = (float(ppsf.min()), float(ppsf.max()))
    return ranges


def default_values(browse_df=None) -> dict:
    d = dict(DEFAULT_FORM)
    if browse_df is not None and len(browse_df):
        d["Size_in_SqFt"] = float(round(browse_df["Size_in_SqFt"].median()))
        d["Price_in_Lakhs"] = float(round(browse_df["Price_in_Lakhs"].median()))
        if "BHK" in browse_df:
            d["BHK"] = int(browse_df["BHK"].mode().iloc[0])
        if "Year_Built" in browse_df:
            d["Year_Built"] = int(browse_df["Year_Built"].median())
        for col in ("Nearby_Schools", "Nearby_Hospitals"):
            if col in browse_df:
                d[col] = int(browse_df[col].median())
    return d


def _num(value):
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        x = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return x if math.isfinite(x) else None


def validate_inputs(form: dict, options: dict, ranges: dict):
    """Return (errors, warnings). Errors block the assessment; warnings only inform."""
    errors, warnings = [], []
    if not isinstance(form, dict):
        return ["Property inputs must be provided as a form."], warnings
    if not isinstance(options, dict):
        return ["Property options are unavailable; the inputs cannot be validated."], warnings
    if not isinstance(ranges, dict):
        return ["Training-data ranges are unavailable; the inputs cannot be validated."], warnings

    size, price = _num(form.get("Size_in_SqFt")), _num(form.get("Price_in_Lakhs"))
    if size is None or size <= 0:
        errors.append("Property size must be a number greater than 0 sq ft.")
    elif size > MAX_SIZE_SQFT:
        errors.append(f"Property size cannot exceed {MAX_SIZE_SQFT:,} sq ft.")
    if price is None or price <= 0:
        errors.append("Asking price must be a number greater than 0 lakhs.")
    elif price > MAX_PRICE_LAKHS:
        errors.append(f"Asking price cannot exceed {MAX_PRICE_LAKHS:,} lakhs.")

    bhk = _num(form.get("BHK"))
    if bhk is None or bhk < 1 or bhk != int(bhk):
        errors.append("BHK must be a whole number of at least 1.")
    elif bhk > MAX_BHK:
        errors.append(f"BHK cannot exceed {MAX_BHK}.")
    year = _num(form.get("Year_Built"))
    if year is None or year != int(year) or not 1800 <= year <= REFERENCE_YEAR:
        errors.append(f"Year built must be a whole year between 1800 and {REFERENCE_YEAR}.")
    for key, label in (("Nearby_Schools", "Nearby schools"), ("Nearby_Hospitals", "Nearby hospitals")):
        v = _num(form.get(key))
        if v is None or v < 0 or v != int(v):
            errors.append(f"{label} must be a whole number of 0 or more.")
        elif v > MAX_NEARBY_FACILITIES:
            errors.append(f"{label} cannot exceed {MAX_NEARBY_FACILITIES}.")

    for key in CATEGORICAL_FORM_FIELDS:
        allowed = options.get(key)
        if not allowed:
            errors.append(f"No list of valid values is available for '{key}' (cannot validate it).")
        elif not isinstance(form.get(key), str) or form.get(key) not in allowed:
            errors.append(f"'{form.get(key)}' is not a valid value for {key}.")

    known_fields = set(CATEGORICAL_FORM_FIELDS) | {
        "BHK", "Size_in_SqFt", "Price_in_Lakhs", "Year_Built",
        "Nearby_Schools", "Nearby_Hospitals", "Amenities",
    }
    for key, value in form.items():
        if key in known_fields:
            continue
        allowed = options.get(key)
        if allowed:
            if not isinstance(value, str) or value not in allowed:
                errors.append(f"'{value}' is not a valid value for {key}.")
        elif _num(value) is None:
            errors.append(f"{key} must be a finite number.")

    if size is not None and size > 0 and price is not None and price > 0 and not math.isfinite(price / size):
        errors.append("Asking price per sq ft is too large to calculate; check the price and size.")

    if errors:
        return errors, warnings

    def outside(name, value, label, unit=""):
        bounds = ranges.get(name)
        if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
            warnings.append(f"The training-data range for {label.lower()} is unavailable; extrapolation could not be checked.")
            return
        lo, hi = _num(bounds[0]), _num(bounds[1])
        if lo is None or hi is None or lo > hi:
            warnings.append(f"The training-data range for {label.lower()} is invalid; extrapolation could not be checked.")
            return
        if value < lo or value > hi:
            warnings.append(f"{label} ({value:,.0f}{unit}) is outside the range seen in the training data "
                            f"({lo:,.0f}-{hi:,.0f}{unit}); any model output is an extrapolation.")
    outside("Size_in_SqFt", size, "Size", " sq ft")
    outside("Price_in_Lakhs", price, "Asking price", " lakhs")
    outside("BHK", bhk, "BHK")
    outside("Year_Built", year, "Year built")
    outside("Nearby_Schools", _num(form["Nearby_Schools"]), "Nearby schools")
    outside("Nearby_Hospitals", _num(form["Nearby_Hospitals"]), "Nearby hospitals")
    ppsf_bounds = ranges.get("ppsf_lakhs")
    ppsf_lo, ppsf_hi = (None, None)
    if isinstance(ppsf_bounds, (tuple, list)) and len(ppsf_bounds) == 2:
        ppsf_lo, ppsf_hi = _num(ppsf_bounds[0]), _num(ppsf_bounds[1])
    if ppsf_lo is None or ppsf_hi is None or ppsf_lo > ppsf_hi:
        warnings.append("The training-data range for price per sq ft is unavailable or invalid; extrapolation could not be checked.")
    elif not ppsf_lo <= price / size <= ppsf_hi:
        warnings.append(f"The implied price per sq ft (Rs {price / size * RUPEES_PER_LAKH:,.0f}) is outside the range seen in the "
                        f"training data (Rs {ppsf_lo * RUPEES_PER_LAKH:,.0f}-{ppsf_hi * RUPEES_PER_LAKH:,.0f}).")
    if size / bhk < 200:
        warnings.append(f"Only {size / bhk:,.0f} sq ft per BHK: unusually small. Please double-check size and BHK.")
    return errors, warnings


# ----------------------------------------------------------------------------------------------
# Rule-based screening, scenario arithmetic and model predictions
# ----------------------------------------------------------------------------------------------
def amenity_names(metadata: dict) -> list:
    cols = set((metadata.get("classification") or {}).get("input_columns", [])) | \
           set((metadata.get("regression") or {}).get("input_columns", []))
    return sorted(c[4:] for c in cols if c.startswith("Has_"))


def derive_model_values(form: dict, amenities: list) -> dict:
    """Turn what the user typed into the exact columns the pipelines were trained on."""
    v = dict(form)
    v["Size_in_SqFt"] = float(form["Size_in_SqFt"])
    v["Price_in_Lakhs"] = float(form["Price_in_Lakhs"])
    for key in ("BHK", "Nearby_Schools", "Nearby_Hospitals"):
        v[key] = int(form[key])
    v["Age_of_Property"] = REFERENCE_YEAR - int(form["Year_Built"])
    chosen = set(form.get("Amenities") or [])
    for name in amenities:
        v[f"Has_{name}"] = int(name in chosen)
    return v


def build_frame(values: dict, columns: list) -> pd.DataFrame:
    missing = [c for c in columns if c not in values]
    if missing:
        raise ValueError(f"The form does not provide these model inputs: {missing}")
    return pd.DataFrame([{c: values[c] for c in columns}], columns=columns)


def apply_label_rule(values: dict, rule: dict) -> dict:
    """Apply the saved rule (fitted on training data only) to ONE property. Mirrors the Phase 3 notebook."""
    ppsf = values["Price_in_Lakhs"] / values["Size_in_SqFt"]
    key = f"{values['City']}|{values['Property_Type']}"
    segment = rule["segment_median_ppsf_lakhs"].get(key)
    used_global = segment is None
    median = rule["global_median_ppsf_lakhs"] if used_global else segment
    value_ok = bool(ppsf <= median)
    criteria = [
        ("BHK is 3 or more", values["BHK"] >= 3),
        ("Ready to move", values["Availability_Status"] == "Ready_to_Move"),
        ("Parking space available", values["Parking_Space"] == "Yes"),
        ("Security available", values["Security"] == "Yes"),
        ("Public transport access is High", values["Public_Transport_Accessibility"] == "High"),
    ]
    score = int(sum(met for _, met in criteria))
    threshold = int(rule["quality_threshold"])
    return {
        "meets_screen": bool(value_ok and score >= threshold),
        "value_ok": value_ok, "segment": key, "used_global_median": used_global,
        "ppsf_lakhs": float(ppsf), "median_ppsf_lakhs": float(median),
        "ppsf_inr": float(ppsf * RUPEES_PER_LAKH), "median_ppsf_inr": float(median * RUPEES_PER_LAKH),
        "quality_score": score, "quality_threshold": threshold,
        "criteria": [{"criterion": name, "met": bool(met)} for name, met in criteria],
    }


def scenario_multiplier(annual_growth_rate: float, years: int = 5) -> float:
    """Return a multiplier; rates are decimals (0.05 means 5%), not percentages."""
    if isinstance(annual_growth_rate, bool):
        raise ValueError("Annual growth rate must be a decimal from 0 to 0.15 (for example, 0.05 for 5%).")
    rate = _num(annual_growth_rate)
    if rate is None or not 0 <= rate <= 0.15:
        raise ValueError("Annual growth rate must be a decimal from 0 to 0.15 (for example, 0.05 for 5%).")
    if isinstance(years, bool) or not isinstance(years, int) or years < 1:
        raise ValueError("Scenario horizon must be a positive whole number of years.")
    try:
        multiplier = (1 + rate) ** years
    except OverflowError as exc:
        raise ValueError("Scenario horizon is too large to calculate safely.") from exc
    if not math.isfinite(multiplier):
        raise ValueError("Scenario multiplier is not finite.")
    return multiplier


def scenario_price_lakhs(asking_price_lakhs: float, annual_growth_rate: float, years: int = 5) -> float:
    """Calculate an arithmetic scenario from an asking price in lakhs and a decimal growth rate."""
    if isinstance(asking_price_lakhs, bool):
        raise ValueError("Asking price must be a finite number greater than 0 lakhs.")
    price = _num(asking_price_lakhs)
    if price is None or price <= 0:
        raise ValueError("Asking price must be a finite number greater than 0 lakhs.")
    result = price * scenario_multiplier(annual_growth_rate, years)
    if not math.isfinite(result):
        raise ValueError("Scenario price is not finite.")
    return result


def scenario_table(price_lakhs: float, rates=SCENARIO_RATES, years: int = 5, assumed_rate=None) -> pd.DataFrame:
    if isinstance(price_lakhs, bool) or _num(price_lakhs) is None or _num(price_lakhs) <= 0:
        raise ValueError("Asking price must be a finite number greater than 0 lakhs.")
    if not isinstance(rates, (tuple, list, set)):
        raise ValueError("Scenario growth rates must be a sequence of decimal rates.")
    if not rates:
        raise ValueError("At least one scenario growth rate is required.")
    normalized_rates = []
    for rate in rates:
        scenario_multiplier(rate, years)
        normalized_rates.append(_num(rate))
    if assumed_rate is not None:
        scenario_multiplier(assumed_rate, years)
        assumed_rate = _num(assumed_rate)
    price_lakhs = float(price_lakhs)
    rows = []
    for r in normalized_rates:
        mult = scenario_multiplier(r, years)
        label = f"{r:.1%}".replace(".0%", "%")
        if assumed_rate is not None and abs(r - assumed_rate) < 1e-9:
            label += " (the assumed rate)"
        future_value = scenario_price_lakhs(price_lakhs, r, years)
        rows.append({"Assumed annual growth": label, f"{years}-year multiplier": round(mult, 4),
                     f"Scenario price after {years} years (lakhs)": round(future_value, 2),
                     "Scenario change vs asking price (lakhs)": round(future_value - price_lakhs, 2)})
    return pd.DataFrame(rows)


def predict_classifier(pipe, frame: pd.DataFrame) -> dict:
    proba = pipe.predict_proba(frame)[0]
    classes = list(getattr(pipe, "classes_", [0, 1]))
    p = float(proba[classes.index(1)])
    if not math.isfinite(p):
        raise ValueError("the classifier returned a non-finite probability")
    return {"probability": p, "predicted_label": int(pipe.predict(frame)[0])}


def predict_regressor(pipe, frame: pd.DataFrame, multiplier: float) -> dict:
    value = float(pipe.predict(frame)[0])
    if not math.isfinite(value):
        raise ValueError("the regressor returned a non-finite value")
    return {"scenario_price_lakhs": value, "implied_current_price_lakhs": value / multiplier}


def run_assessment(art: Artifacts, form: dict) -> dict:
    errors, _ = validate_inputs(form, get_options(art), DEFAULT_RANGES)
    if errors:
        raise ValueError("Invalid property inputs: " + " ".join(errors))
    meta = art.metadata
    values = derive_model_values(form, amenity_names(meta))
    rule = apply_label_rule(values, art.label_rule)
    cls = predict_classifier(art.classifier, build_frame(values, meta["classification"]["input_columns"]))
    reg = predict_regressor(art.regressor, build_frame(values, meta["regression"]["input_columns"]), meta["growth_multiplier"])
    return {"values": values, "rule": rule, "classifier": cls, "regressor": reg}


# ----------------------------------------------------------------------------------------------
# Reliability (skill-threshold) information - fail-safe: anything unclear counts as NOT passed
# ----------------------------------------------------------------------------------------------
def reliability(meta: dict) -> dict:
    cls, reg = meta.get("classification") or {}, meta.get("regression") or {}
    cv_auc = (cls.get("cv") or {}).get("cv_roc_auc_mean")
    cv_r2 = (reg.get("cv") or {}).get("cv_r2_mean")
    thr_auc, thr_r2 = cls.get("skill_threshold_cv_auc"), reg.get("skill_threshold_cv_r2")
    cls_ok = cls.get("meets_pre_declared_skill_threshold") is True
    reg_ok = reg.get("meets_pre_declared_skill_threshold") is True
    return {
        "classification": {"passed": cls_ok, "cv": cv_auc, "threshold": thr_auc, "model": cls.get("model", "unknown"),
            "message": (f"The classification model PASSED the minimum skill threshold (cross-validated ROC AUC {fmt_num(cv_auc)} "
                        f">= {fmt_num(thr_auc, 2)}). That is only a minimum bar: it still describes a constructed, rule-based label.")
                       if cls_ok else
                       (f"The classification model FAILED its pre-declared skill threshold (cross-validated ROC AUC {fmt_num(cv_auc)} "
                        f"vs the required {fmt_num(thr_auc, 2)}; 0.5 is a coin flip). Its output is no better than guessing and "
                        "must not be relied on.")},
        "regression": {"passed": reg_ok, "cv": cv_r2, "threshold": thr_r2, "model": reg.get("model", "unknown"),
            "message": (f"The regression model PASSED the minimum skill threshold (cross-validated R2 {fmt_num(cv_r2)} "
                        f">= {fmt_num(thr_r2, 2)}). That is only a minimum bar: the target is an assumed-growth scenario.")
                       if reg_ok else
                       (f"The regression model FAILED its pre-declared skill threshold (cross-validated R2 {fmt_num(cv_r2)} vs the "
                        f"required {fmt_num(thr_r2, 2)}; 0 means no better than predicting the average price). Its output is no "
                        "better than a constant guess and must not be relied on.")},
    }


def base_rate_from_imbalance(df):
    """Share of 'good' properties in the test set, read from the Phase 3 imbalance review (if present)."""
    try:
        row = df[df["model"].str.contains("always 'good'", regex=False)].iloc[0]
        return float(row["precision"])
    except Exception:
        return None


# ----------------------------------------------------------------------------------------------
# Browsing / insight helpers
# ----------------------------------------------------------------------------------------------
def filter_properties(df, cities=None, property_types=None, bhk=None, price=None, size=None):
    mask = pd.Series(True, index=df.index)
    if cities:
        mask &= df["City"].isin(cities)
    if property_types and "Property_Type" in df:
        mask &= df["Property_Type"].isin(property_types)
    if bhk and "BHK" in df:
        mask &= df["BHK"].between(bhk[0], bhk[1])
    if price:
        mask &= df["Price_in_Lakhs"].between(price[0], price[1])
    if size:
        mask &= df["Size_in_SqFt"].between(size[0], size[1])
    return df[mask]


def year_band_medians(df):
    if "Year_Built" not in df or len(df) == 0:
        return None
    bins = [1989, 1994, 1999, 2004, 2009, 2014, 2019, 2023]
    labels = ["1990-94", "1995-99", "2000-04", "2005-09", "2010-14", "2015-19", "2020-23"]
    band = pd.cut(df["Year_Built"], bins=bins, labels=labels)
    out = df.groupby(band, observed=True)["Price_in_Lakhs"].agg(["median", "count"])
    return out if len(out) else None


# ----------------------------------------------------------------------------------------------
# Number formatting (Indian digit grouping, rupees and lakhs)
# ----------------------------------------------------------------------------------------------
def indian_number(x, decimals: int = 0) -> str:
    """1234567.8 -> '12,34,568' (Indian grouping)."""
    x = float(x)
    neg = x < 0
    ip, _, fp = f"{abs(x):.{decimals}f}".partition(".")
    if len(ip) > 3:
        head, tail = ip[:-3], ip[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        ip = ",".join(groups + [tail])
    return ("-" if neg else "") + ip + ("." + fp if decimals else "")


def format_inr(rupees, decimals: int = 0) -> str:
    return "₹" + indian_number(rupees, decimals)


def format_lakhs(lakhs, decimals: int = 1, with_crore: bool = True) -> str:
    text = f"₹{indian_number(lakhs, decimals)} lakh"
    if with_crore and abs(lakhs) >= 100:
        text += f" (₹{indian_number(lakhs / 100, 2)} crore)"
    return text


def format_ppsf(lakhs_per_sqft) -> str:
    return f"{format_inr(lakhs_per_sqft * RUPEES_PER_LAKH)}/sq ft"


# ----------------------------------------------------------------------------------------------
# Environment, model family, start-up self-check
# ----------------------------------------------------------------------------------------------
def environment_report() -> dict:
    info = {"python": sys.version.split()[0], "executable": sys.executable}
    for mod in ("sklearn", "xgboost", "pandas", "numpy", "streamlit", "joblib"):
        try:
            module = __import__(mod)
            info[mod] = getattr(module, "__version__", "installed")
        except Exception:
            info[mod] = None
    return info


def model_family_note(meta: dict, env: dict | None = None):
    """Explain which boosting library the saved models use, and whether that matches the environment."""
    env = env or environment_report()
    names = " ".join(str((meta.get(k) or {}).get("model", "")) for k in ("classification", "regression"))
    fallback = "XGBoost not installed" in names
    uses_xgb = "XGBoost" in names and not fallback
    if fallback:
        return {"level": "warn", "uses_xgboost": False,
                "text": "The saved models use scikit-learn's HistGradientBoosting because XGBoost was not importable in the environment "
                        "that trained them (listing xgboost in requirements.txt does not install it into the notebook's Python). "
                        + (f"XGBoost {env['xgboost']} is available in the environment running this app. " if env.get("xgboost") else
                           "XGBoost is also not importable in the environment running this app. ")
                        + "Re-run the Phase 3 notebook in a single environment where xgboost imports (see PHASE4_RUN_GUIDE.md)."}
    if uses_xgb and not env.get("xgboost"):
        return {"level": "fail", "uses_xgboost": True,
                "text": "The saved models are XGBoost models but xgboost is not importable here. Install it into this environment."}
    return {"level": "ok", "uses_xgboost": uses_xgb,
            "text": "The saved models use XGBoost." if uses_xgb else "No XGBoost dependency in the saved models."}


def smoke_form(art: Artifacts):
    opts = get_options(art)
    form = {"Size_in_SqFt": 2000.0, "Price_in_Lakhs": 200.0, "BHK": 3, "Year_Built": 2005,
            "Nearby_Schools": 5, "Nearby_Hospitals": 5, "Amenities": []}
    for key in CATEGORICAL_FORM_FIELDS:
        choices = opts.get(key)
        if not choices:
            return None
        form[key] = choices[0]
    needed = set((art.metadata.get("classification") or {}).get("input_columns", [])) | \
             set((art.metadata.get("regression") or {}).get("input_columns", []))
    for col in needed:
        if col not in form and col not in ("Age_of_Property",) and not col.startswith("Has_"):
            form[col] = opts[col][0] if opts.get(col) else 0.0
    return form


def self_check(art: Artifacts):
    """Return None if both loaded pipelines can predict a test property, else a plain-language problem."""
    form = smoke_form(art)
    if form is None:
        return "Start-up test prediction could not be built because some category lists are unavailable."
    try:
        run_assessment(art, form)
        return None
    except Exception as exc:
        return (f"The saved models loaded but failed a start-up test prediction ({type(exc).__name__}: {str(exc)[:200]}). "
                "This usually means the models and the installed library versions do not match. Re-run the Phase 3 notebook in this "
                "environment, or install the library versions shown in model_metadata.json.")


# ----------------------------------------------------------------------------------------------
# Presentation helpers for the Model Performance and Market Insights pages
# ----------------------------------------------------------------------------------------------
def threshold_table(meta: dict) -> pd.DataFrame:
    c, r = meta.get("classification") or {}, meta.get("regression") or {}
    rel = reliability(meta)
    def row(task, metric, cv, thr, chance, passed, digits):
        gap = (cv - thr) if isinstance(cv, (int, float)) and isinstance(thr, (int, float)) else None
        return {"Task": task, "Measure (cross-validated, training data)": metric, "Measured": fmt_num(cv, digits),
                "Required minimum": fmt_num(thr, 2), "No-skill reference": chance,
                "Gap to required": ("n/a" if gap is None else f"{gap:+.{digits}f}"), "Meets threshold": "Yes" if passed else "No"}
    return pd.DataFrame([
        row("Classification", "ROC AUC", (c.get("cv") or {}).get("cv_roc_auc_mean"), c.get("skill_threshold_cv_auc"),
            "0.50 (coin flip)", rel["classification"]["passed"], 4),
        row("Regression", "R²", (r.get("cv") or {}).get("cv_r2_mean"), r.get("skill_threshold_cv_r2"),
            "0.00 (predict the average)", rel["regression"]["passed"], 4)])


def confusion_from_review(df, model_name: str):
    """[[TN, FP], [FN, TP]] for a model, read from the Phase 3 imbalance review (None if unavailable)."""
    try:
        r = df[df["model"] == model_name].iloc[0]
        return np.array([[int(r["TN"]), int(r["FP"])], [int(r["FN"]), int(r["TP"])]])
    except Exception:
        return None


def chart_runs(df, kind: str):
    """Baseline and honest candidate models from a Phase 3 results CSV (leakage demonstrations excluded)."""
    if df is None or "experiment_type" not in df:
        return None
    sub = df[df["experiment_type"].isin(["baseline", "primary"])].copy()
    metric = "cv_roc_auc" if kind == "classification" else "cv_r2"
    if f"{metric}_mean" not in sub or sub.empty:
        return None
    sub["label"] = sub["run_name"].str.replace(r"^(CLS|REG) (baseline|primary): ", "", regex=True)
    sub["mean"], sub["std"] = sub[f"{metric}_mean"], sub.get(f"{metric}_std", 0.0)
    return sub[["label", "experiment_type", "mean", "std"]]


def city_spread(df):
    if df is None or "City" not in df or len(df) == 0:
        return None
    med = df.groupby("City")["Price_in_Lakhs"].median()
    overall = float(df["Price_in_Lakhs"].median())
    return {"overall_median": overall, "lowest_city": med.idxmin(), "lowest": float(med.min()),
            "highest_city": med.idxmax(), "highest": float(med.max()), "spread_pct": float((med.max() - med.min()) / overall * 100),
            "n_cities": int(len(med))}
