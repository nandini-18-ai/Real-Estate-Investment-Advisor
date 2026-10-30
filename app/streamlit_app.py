"""Real Estate Investment Advisor - Streamlit app.

Run from the project folder:   python -m streamlit run app/streamlit_app.py

The app only LOADS what Phase 3 saved. It never retrains a model and never invents a metric.
It presents (1) a rule-based screen and (2) a scenario estimate under an ASSUMED growth rate,
never a prediction of returns, a verified market forecast, or proof of profitability.
"""
from __future__ import annotations

import html
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))
PROJECT_ROOT = Path(os.environ.get("REIA_PROJECT_ROOT", APP_DIR.parent)).resolve()

import app_logic as L  # noqa: E402  (needs the sys.path line above)

st.set_page_config(page_title="Real Estate Investment Advisor", page_icon="🏠", layout="wide",
                   initial_sidebar_state="collapsed")

NAVY, BLUE, LIGHT, GREY, RED, GREEN = "#1F3A5F", "#3B6EA8", "#A9C1E0", "#8A97A8", "#B42318", "#067647"
PAGES = ["Home", "Property Assessment", "Model Performance", "Market Insights", "Methodology & About"]

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.edgecolor": "#C9D3E0", "axes.labelcolor": "#1B2430", "text.color": "#1B2430",
    "xtick.color": "#46546A", "ytick.color": "#46546A", "font.size": 10, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.titlecolor": NAVY, "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#E6EBF2",
    "grid.linewidth": 0.8, "legend.frameon": False,
})

CSS = """
<style>
.block-container { padding-top: 1.6rem; max-width: 1180px; }
html, body, [class*="css"] { font-family: "Inter", "Segoe UI", system-ui, -apple-system, Roboto, Arial, sans-serif; }
h1, h2, h3, h4 { color: #1F3A5F; letter-spacing: -0.01em; }
.reia-brand { font-size: 0.95rem; font-weight: 700; color: #1F3A5F; letter-spacing: 0.04em; text-transform: uppercase; }
.reia-title { font-size: 2.1rem; font-weight: 750; color: #1F3A5F; line-height: 1.15; margin: 0.1rem 0 0.3rem 0; }
.reia-sub { font-size: 1.05rem; color: #46546A; margin-bottom: 1rem; max-width: 62rem; }
.reia-kpi { background: #F6F8FB; border: 1px solid #D8E0EA; border-radius: 10px; padding: 14px 16px; height: 100%; }
.reia-kpi .k-label { font-size: 0.78rem; color: #5B6B7F; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600; }
.reia-kpi .k-value { font-size: 1.55rem; font-weight: 700; color: #1F3A5F; margin: 4px 0 2px 0; line-height: 1.2; }
.reia-kpi .k-sub { font-size: 0.82rem; color: #5B6B7F; }
.reia-kpi.fail { border-left: 4px solid #B42318; } .reia-kpi.fail .k-value { color: #B42318; }
.reia-kpi.ok { border-left: 4px solid #067647; } .reia-kpi.ok .k-value { color: #067647; }
.reia-kpi.warn { border-left: 4px solid #B54708; }
.reia-callout { border: 1px solid #D8E0EA; border-left: 4px solid #3B6EA8; background: #F5F9FF; border-radius: 6px;
                padding: 11px 15px; margin: 8px 0; color: #1B2430; font-size: 0.95rem; }
.reia-callout.fail { border-left-color: #B42318; background: #FEF3F2; }
.reia-callout.ok { border-left-color: #067647; background: #ECFDF3; }
.reia-callout.warn { border-left-color: #B54708; background: #FFFAEB; }
.reia-callout strong { color: #1F3A5F; }
.reia-callout.fail strong { color: #B42318; }
.reia-section { font-size: 1.05rem; font-weight: 700; color: #1F3A5F; margin: 0 0 0.4rem 0; }
.reia-foot { color: #5B6B7F; font-size: 0.82rem; border-top: 1px solid #D8E0EA; margin-top: 2rem; padding-top: 0.6rem; }
div[data-testid="stMetric"] { background: #F6F8FB; border: 1px solid #D8E0EA; border-radius: 10px; padding: 12px 14px; }
div[data-testid="stRadio"] > div[role="radiogroup"] { gap: 0.35rem; flex-wrap: wrap; }
div[data-testid="stRadio"] label { background: #F3F6FA; border: 1px solid #D8E0EA; border-radius: 999px; padding: 4px 14px; }
</style>
"""


# ------------------------------------------------------------------ cached loaders
@st.cache_resource(show_spinner="Loading saved models ...")
def get_artifacts(root: str):
    return L.load_artifacts(Path(root))


@st.cache_data(show_spinner="Loading property data ...")
def get_browse_data(root: str):
    return L.load_browse_data(Path(root))


@st.cache_data(show_spinner=False)
def get_csv(root: str, rel: str):
    return L.read_optional_csv(Path(root), rel)


# ------------------------------------------------------------------ small UI helpers
def esc(x) -> str:
    return html.escape(str(x))


def callout(kind: str, title: str, body=""):
    parts = [body] if isinstance(body, str) else list(body)
    inner = "".join(f"<div>{esc(p)}</div>" for p in parts if p)
    st.markdown(f'<div class="reia-callout {kind}"><strong>{esc(title)}</strong>{inner}</div>', unsafe_allow_html=True)


def kpi(label: str, value: str, sub: str = "", tone: str = "neutral"):
    st.markdown(f'<div class="reia-kpi {tone}"><div class="k-label">{esc(label)}</div>'
                f'<div class="k-value">{esc(value)}</div><div class="k-sub">{esc(sub)}</div></div>', unsafe_allow_html=True)


def page_header(title: str, subtitle: str = ""):
    st.markdown(f'<div class="reia-title">{esc(title)}</div>' + (f'<div class="reia-sub">{esc(subtitle)}</div>' if subtitle else ""),
                unsafe_allow_html=True)


def section(title: str):
    st.markdown(f'<div class="reia-section">{esc(title)}</div>', unsafe_allow_html=True)


def go_to(page: str):
    st.session_state["nav_page"] = page


def reload_all():
    st.cache_resource.clear()
    st.cache_data.clear()
    st.session_state.pop("result", None)


def show_figure(title: str, filename: str):
    path = L.figure_path(PROJECT_ROOT, filename)
    if path is None:
        st.caption(f"({title}: `{filename}` was not found in phase3_outputs/figures)")
    else:
        st.image(str(path), caption=title)


def reliability_callout(info: dict, title: str):
    if info["passed"]:
        callout("ok", f"{title}: passed the minimum reliability threshold", info["message"])
    else:
        callout("fail", f"{title}: below its reliability threshold", info["message"])


def lakh_axis(ax, axis="x"):
    fmt = FuncFormatter(lambda v, _: L.indian_number(v, 0))
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(fmt)


def done(fig):
    st.pyplot(fig)
    plt.close(fig)


# ------------------------------------------------------------------ HOME
def render_home(art, df):
    meta = art.metadata
    page_header("Real Estate Investment Advisor",
                "A student project that screens properties with a documented rule and shows five-year price scenarios, "
                "alongside an honest report on how well the machine-learning models perform.")
    growth = meta.get("growth_rate_assumed", 0.08) if meta else 0.08
    rel = L.reliability(meta) if meta else None
    total = (meta.get("train_rows", 0) + meta.get("test_rows", 0)) if meta else (len(df) if df is not None else None)
    n_pass = sum(1 for k in ("classification", "regression") if rel and rel[k]["passed"]) if rel else 0

    c = st.columns(4)
    with c[0]:
        kpi("Properties analysed", L.indian_number(total) if total else "n/a", "used for training and testing the models")
    with c[1]:
        kpi("Cities covered", str(df["City"].nunique()) if df is not None else "n/a", "in the project dataset")
    with c[2]:
        kpi("Assumed growth", f"{growth:.0%} a year", "an assumption, not a market forecast")
    with c[3]:
        kpi("Models above threshold", f"{n_pass} of 2" if rel else "n/a", "pre-declared minimum skill",
            "ok" if rel and n_pass == 2 else ("fail" if rel and n_pass == 0 else "warn"))

    st.write("")
    if rel and n_pass < 2:
        callout("fail", "Machine-learning reliability notice",
                ["At least one model is below the minimum skill level that was set before training. "
                 "Its output is no better than a simple guess and is shown for transparency only.",
                 "The rule-based screen and the growth scenarios are plain calculations and do not depend on the models."])
    if meta and meta.get("quick_run") is True:
        callout("warn", "Trial models", "These models were trained on a small trial sample. Re-run the Phase 3 notebook with QUICK_RUN = False.")
    for e in art.errors:
        callout("fail", "Setup problem", e)

    st.write("")
    a, b = st.columns([2, 1])
    with a:
        section("What you can do here")
        st.markdown("- **Screen a property** against a documented rule (price per sq ft versus similar properties, plus a quality checklist).\n"
                    f"- **See five-year price scenarios** based on an assumed annual growth rate (default {growth:.0%}).\n"
                    "- **Check how reliable the models are** against baselines and pre-declared thresholds.\n"
                    "- **Explore the dataset** with filters and descriptive charts.")
    with b:
        st.write("")
        st.button("Assess a property  →", on_click=go_to, args=("Property Assessment",), type="primary", key="cta_assess")
        st.button("View model performance", on_click=go_to, args=("Model Performance",), key="cta_perf")

    st.write("")
    callout("info", "How to read the results",
            ["The five-year figure is a scenario: today's price grown at an assumed rate. It is not a verified market forecast.",
             "\"Meets the screen\" means a property satisfies a rule we wrote. It is not proof that the property is, or will be, profitable."])


# ------------------------------------------------------------------ PROPERTY ASSESSMENT
def render_assessment(art, options, ranges, defaults):
    meta = art.metadata
    amenities = L.amenity_names(meta)
    page_header("Property Assessment", "Describe a property to see the rule-based screen, the model estimates and five-year scenarios.")
    callout("info", "Scenario estimates and rule-based screening only.",
            "Nothing on this page is a guarantee, a forecast of market prices or investment advice.")
    if art.warnings:
        st.caption(f"{len(art.warnings)} environment notice(s) are listed under Methodology & About → Technical details.")

    needed = set(meta["classification"]["input_columns"]) | set(meta["regression"]["input_columns"])
    known = set(L.CATEGORICAL_FORM_FIELDS) | {"BHK", "Size_in_SqFt", "Age_of_Property", "Nearby_Schools", "Nearby_Hospitals"}
    extra_cols = sorted(c for c in needed if c not in known and not c.startswith("Has_"))

    def pick(label, key, default=None, help_text=None):
        choices = options.get(key, [])
        idx = choices.index(default) if default in choices else 0
        return st.selectbox(label, choices, index=idx, key=f"in_{key}", help=help_text)

    with st.form("property_form"):
        with st.container(border=True):
            section("1 · Location and type")
            c1, c2, c3 = st.columns(3)
            with c1:
                city = pick("City", "City")
                ptype = pick("Property type", "Property_Type")
            with c2:
                furnished = pick("Furnishing", "Furnished_Status")
                facing = pick("Facing", "Facing")
            with c3:
                owner = pick("Owner type", "Owner_Type")
                availability = pick("Availability", "Availability_Status", "Ready_to_Move")
        with st.container(border=True):
            section("2 · Size, age and price")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                bhk = st.number_input("BHK (bedrooms)", min_value=0, max_value=L.MAX_BHK,
                                      value=int(defaults["BHK"]), step=1, key="in_bhk")
            with c2:
                size = st.number_input("Size (sq ft)", min_value=0.0, max_value=float(L.MAX_SIZE_SQFT), value=float(defaults["Size_in_SqFt"]),
                                       step=50.0, key="in_size")
            with c3:
                year = st.number_input(f"Year built (age = {L.REFERENCE_YEAR} − year)", min_value=1800, max_value=L.REFERENCE_YEAR,
                                       value=int(min(defaults["Year_Built"], L.REFERENCE_YEAR)), step=1, key="in_year")
            with c4:
                price = st.number_input("Asking price (₹ lakh)", min_value=0.0, max_value=float(L.MAX_PRICE_LAKHS),
                                        value=float(defaults["Price_in_Lakhs"]), step=5.0, key="in_price",
                                        help="1 lakh = ₹1,00,000. Used for the rule-based screen and the scenario table.")
        with st.container(border=True):
            section("3 · Surroundings and facilities")
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                parking = pick("Parking space", "Parking_Space")
            with c2:
                security = pick("Security", "Security")
            with c3:
                transport = pick("Public transport access", "Public_Transport_Accessibility", "Medium")
            with c4:
                schools = st.number_input("Nearby schools", min_value=0, max_value=L.MAX_NEARBY_FACILITIES,
                                          value=int(defaults["Nearby_Schools"]), step=1, key="in_schools")
            with c5:
                hospitals = st.number_input("Nearby hospitals", min_value=0, max_value=L.MAX_NEARBY_FACILITIES,
                                            value=int(defaults["Nearby_Hospitals"]), step=1, key="in_hospitals")
        with st.container(border=True):
            section("4 · Amenities")
            chosen = []
            if amenities:
                cols = st.columns(len(amenities))
                for col, name in zip(cols, amenities):
                    with col:
                        if st.checkbox(name, value=False, key=f"in_amen_{name}"):
                            chosen.append(name)
            extras = {}
            for col_name in extra_cols:
                extras[col_name] = pick(col_name, col_name) if options.get(col_name) else st.number_input(col_name, value=0.0, key=f"in_extra_{col_name}")
        submitted = st.form_submit_button("Assess this property", type="primary")

    if submitted:
        form = {"City": city, "Property_Type": ptype, "BHK": bhk, "Size_in_SqFt": size, "Price_in_Lakhs": price,
                "Year_Built": year, "Furnished_Status": furnished, "Facing": facing, "Owner_Type": owner,
                "Availability_Status": availability, "Parking_Space": parking, "Security": security,
                "Public_Transport_Accessibility": transport, "Nearby_Schools": schools, "Nearby_Hospitals": hospitals,
                "Amenities": chosen, **extras}
        errors, warnings = L.validate_inputs(form, options, ranges)
        if errors:
            st.session_state.pop("result", None)
            callout("fail", "Please correct the following before assessing the property", errors)
        else:
            try:
                st.session_state["result"] = {"form": form, "warnings": warnings, "assessment": L.run_assessment(art, form)}
            except Exception as exc:  # model/input mismatch, version problems, ...
                st.session_state.pop("result", None)
                callout("fail", "The assessment could not be completed",
                        f"{type(exc).__name__}: {str(exc)[:300]}. Check the environment notices under Methodology & About → Technical details.")
    result = st.session_state.get("result")
    if result:
        render_results(art, result)


def render_results(art, result):
    meta = art.metadata
    a = result["assessment"]
    rel = L.reliability(meta)
    growth, years = meta["growth_rate_assumed"], meta.get("horizon_years", 5)
    scenario_mult, model_mult = L.scenario_multiplier(growth, years), meta["growth_multiplier"]
    price = a["values"]["Price_in_Lakhs"]
    r, c, g = a["rule"], a["classifier"], a["regressor"]
    both_models_below_threshold = not rel["classification"]["passed"] and not rel["regression"]["passed"]

    st.write("")
    st.markdown("### Results")
    for w in result["warnings"]:
        callout("warn", "Input notice", w)

    k = st.columns(4)
    with k[0]:
        kpi("Rule-based screen", "Meets screen" if r["meets_screen"] else "Does not meet",
            f"{L.format_ppsf(r['ppsf_lakhs'])} vs {L.format_ppsf(r['median_ppsf_lakhs'])} for similar properties · quality {r['quality_score']}/5",
            "ok" if r["meets_screen"] else "neutral")
    with k[1]:
        kpi("ML estimate: rule-label probability", f"{c['probability']:.1%}",
            "probability of the constructed rule label, not profitability")
    with k[2]:
        kpi(f"ML estimate: scenario target in {years} years", L.format_lakhs(g["scenario_price_lakhs"], 2),
            "estimate of the constructed assumed-growth target")
    with k[3]:
        kpi(f"Arithmetic scenario at {growth:.0%} for {years} years",
            L.format_lakhs(L.scenario_price_lakhs(price, growth, years), 2),
            f"your asking price {L.format_lakhs(price)} × {scenario_mult:.4f} · no model involved")

    st.write("")
    section("1 · Rule-based screen (deterministic)")
    if r["meets_screen"]:
        callout("ok", "This property meets the documented screen",
                f"Its price per sq ft is at or below the typical level of similar properties and it meets {r['quality_score']} of 5 "
                f"quality criteria (needs {r['quality_threshold']}). This is a rule, not proof that the property is a good investment.")
    else:
        why = []
        if not r["value_ok"]:
            why.append("its price per sq ft is above the typical level of similar properties")
        if r["quality_score"] < r["quality_threshold"]:
            why.append(f"only {r['quality_score']} of 5 quality criteria are met (needs {r['quality_threshold']})")
        callout("info", "This property does not meet the documented screen",
                "Because " + " and ".join(why) + ". That does not mean it is a poor investment; the rule is only a screening convention.")
    with st.expander("How the screen was calculated"):
        seg = r["segment"].replace("|", " / ")
        note = " (no matching group in the training data, so the overall median was used)" if r["used_global_median"] else ""
        st.write(f"**Value test.** Your price per sq ft is {L.format_ppsf(r['ppsf_lakhs'])}. The training-set median for *{seg}*{note} is "
                 f"{L.format_ppsf(r['median_ppsf_lakhs'])}. The test is passed when yours is at or below the median: "
                 f"**{'passed' if r['value_ok'] else 'not passed'}**.")
        st.write(f"**Quality checklist** (at least {r['quality_threshold']} of 5 needed):")
        st.dataframe(pd.DataFrame([{"Criterion": x["criterion"], "Met": "Yes" if x["met"] else "No"} for x in r["criteria"]]), hide_index=True)
        st.caption(f"Medians were fitted on training rows only ({L.indian_number(art.label_rule.get('fitted_on_rows', 0))} rows). "
                   f"Rupee figures assume 1 lakh = ₹{L.indian_number(L.RUPEES_PER_LAKH)}.")

    st.write("")
    section("2 · Machine-learning model estimates")
    if both_models_below_threshold:
        callout("fail", "Neither model meets its reliability threshold",
                "Both outputs are shown for transparency only and must not be used to judge this property. "
                "The rule-based screen and arithmetic scenarios below are separate from these model estimates.")
    else:
        reliability_callout(rel["classification"], "Classification model")
        reliability_callout(rel["regression"], "Regression model")
    left, right = st.columns(2)
    with left:
        base = L.base_rate_from_imbalance(get_csv(str(PROJECT_ROOT), L.OPTIONAL_FILES["imbalance_review"]))
        st.write(f"The model's probability that the property carries the rule-based label is **{c['probability']:.1%}**; "
                 f"at its own 0.5 cut-off it says **{'yes' if c['predicted_label'] == 1 else 'no'}**.")
        st.caption((f"For reference, {base:.1%} of test-set properties carry the label. " if base is not None else "")
                   + "This probability is not calibrated and is not the chance that the property will be profitable.")
    with right:
        st.write(f"The model's scenario price after {years} years is **{L.format_lakhs(g['scenario_price_lakhs'])}**, which corresponds to "
                 f"a present-day value of {L.format_lakhs(g['implied_current_price_lakhs'])} (your asking price: {L.format_lakhs(price)}).")
        st.caption(f"The model predicts the constructed assumed-growth target from other property attributes; it does not use your asking price. "
                   f"The target uses today's price × {model_mult:.4f} and is not a market forecast.")

    st.write("")
    section(f"3 · {years}-year scenarios from your asking price (arithmetic only)")
    what_if_percent = st.slider("What-if annual growth rate (%)", min_value=0.0, max_value=15.0,
                                value=float(round(growth * 100, 1)), step=0.5, key="what_if_rate",
                                help="Changes only this table and chart. The models above always use the assumed rate.")
    what_if_rate = round(what_if_percent / 100, 6)
    rates = sorted(set(L.SCENARIO_RATES) | {what_if_rate})
    table = L.scenario_table(price, rates=rates, years=years, assumed_rate=growth)
    show = table.copy()
    for col in show.columns:
        if "lakhs" in col:
            show[col] = show[col].map(lambda v: L.format_lakhs(v, 2, with_crore=False))
    st.dataframe(show.rename(columns=lambda x: x.replace("(lakhs)", "(₹ lakh)")), hide_index=True)
    fig, ax = plt.subplots(figsize=(7.5, 3.3))
    scenario_values = [L.scenario_price_lakhs(price, rate, years) for rate in rates]
    ax.bar([f"{x:.1%}" for x in rates], scenario_values, color=BLUE, width=0.6)
    ax.axhline(price, color=RED, linestyle="--", linewidth=1.2, label=f"asking price today ({L.format_lakhs(price, 1, False)})")
    ax.set_xlabel("Assumed annual growth"); ax.set_ylabel("Scenario price (₹ lakh)"); lakh_axis(ax, "y"); ax.grid(axis="x", visible=False)
    ax.set_title(f"Price after {years} years under different assumed growth rates"); ax.legend(loc="upper left")
    done(fig)
    st.caption(f"Scenario price = asking price × (1 + r)^{years}. Nothing in the data supports a particular r; real prices can rise, stay flat or fall.")


# ------------------------------------------------------------------ MODEL PERFORMANCE
def bar_candidates(runs, title, xlabel, threshold, chance, chance_label):
    runs = runs.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 0.55 * len(runs) + 1.6))
    colors = [GREY if t == "baseline" else BLUE for t in runs["experiment_type"]]
    ax.barh(runs["label"], runs["mean"], xerr=runs["std"], color=colors, ecolor="#46546A", height=0.55)
    ax.axvline(chance, color=GREY, linestyle=":", label=chance_label)
    ax.axvline(threshold, color=RED, linestyle="--", label=f"required minimum ({threshold:g})")
    lo, hi = min(runs["mean"].min(), chance) - 0.02, max(runs["mean"].max(), threshold) + 0.02
    ax.set_xlim(lo, hi); ax.set_xlabel(xlabel); ax.set_title(title); ax.grid(axis="y", visible=False); ax.legend(loc="lower right", fontsize=8)
    return fig


def render_performance(art):
    meta = art.metadata
    rel = L.reliability(meta)
    root = str(PROJECT_ROOT)
    page_header("Model Performance", "Measured results from the Phase 3 evaluation, compared with simple baselines and the thresholds set before training.")

    cls, reg = meta["classification"], meta["regression"]
    both_fail = not rel["classification"]["passed"] and not rel["regression"]["passed"]
    if both_fail:
        callout("fail", "Neither model meets its reliability threshold",
                ["Both machine-learning models perform no better than simple baselines (see the table below).",
                 "Their outputs must not be used to judge a property. The thresholds were fixed before training and have not been relaxed."])
    else:
        reliability_callout(rel["classification"], "Classification model")
        reliability_callout(rel["regression"], "Regression model")
    st.dataframe(L.threshold_table(meta), hide_index=True)

    review = get_csv(root, L.OPTIONAL_FILES["imbalance_review"])
    base_rate = L.base_rate_from_imbalance(review)
    t_c, b_c, cv_c = cls.get("test") or {}, cls.get("baseline_test") or {}, cls.get("cv") or {}
    t_r, b_r, cv_r = reg.get("test") or {}, reg.get("baseline_test") or {}, reg.get("cv") or {}

    st.markdown(f"### Classification — {cls.get('model', 'unknown')}")
    k = st.columns(4)
    with k[0]:
        kpi("ROC AUC (test)", L.fmt_num(t_c.get("test_roc_auc"), 4),
            f"coin flip = 0.500 · cross-validated {L.fmt_num(cv_c.get('cv_roc_auc_mean'), 4)}")
    with k[1]:
        kpi("PR AUC (test)", L.fmt_num(t_c.get("test_pr_auc"), 4),
            f"no-skill reference = share of 'good' ({L.fmt_num(base_rate, 4)})" if base_rate else "no-skill = share of 'good'")
    with k[2]:
        kpi("F1 for 'good' (test)", L.fmt_num(t_c.get("test_f1"), 4),
            f"labelling every property 'good' scores {L.fmt_num(2 * base_rate / (1 + base_rate), 4)}" if base_rate else "see class-imbalance table")
    with k[3]:
        kpi("Accuracy (test)", L.fmt_num(t_c.get("test_accuracy"), 4),
            f"always answering 'not good' scores {L.fmt_num(b_c.get('test_accuracy'), 4)}")
    k = st.columns(4)
    with k[0]:
        kpi("Precision (test)", L.fmt_num(t_c.get("test_precision"), 4), f"no-skill ≈ {L.fmt_num(base_rate, 4)}" if base_rate else "")
    with k[1]:
        kpi("Recall (test)", L.fmt_num(t_c.get("test_recall"), 4), "share of 'good' properties found")
    with k[2]:
        kpi("Balanced accuracy (test)", L.fmt_num(t_c.get("test_balanced_accuracy"), 4), "0.500 = no skill")
    with k[3]:
        kpi("Cross-validated F1", L.fmt_num(cv_c.get("cv_f1_mean"), 4), "training data, 3 folds")
    b_acc, t_acc = b_c.get("test_accuracy"), t_c.get("test_accuracy")
    if isinstance(b_acc, (int, float)) and isinstance(t_acc, (int, float)) and b_acc > t_acc:
        callout("warn", "Why accuracy is misleading here",
                f"Only about {L.fmt_num((base_rate or 0) * 100, 0)}% of properties carry the label, so answering 'not good' every time scores "
                f"{b_acc:.1%} accuracy with an F1 of {L.fmt_num(b_c.get('test_f1'), 2)} for the label. The model's lower accuracy comes from "
                "guessing 'good' for some properties; judge it by ROC AUC and PR AUC against the no-skill references instead.")

    left, right = st.columns(2)
    with left:
        cm = L.confusion_from_review(review, cls.get("model", "")) if review is not None else None
        if cm is not None:
            fig, ax = plt.subplots(figsize=(4.6, 3.8))
            ax.imshow(cm, cmap="Blues"); ax.grid(False)
            for (i, j), v in np.ndenumerate(cm):
                ax.text(j, i, L.indian_number(v), ha="center", va="center", color="white" if v > cm.max() / 2 else "#1B2430", fontsize=12)
            ax.set_xticks([0, 1], ["Not good", "Good"]); ax.set_yticks([0, 1], ["Not good", "Good"])
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion matrix (test set)")
            done(fig)
        else:
            show_figure("Confusion matrix of the selected classifier", f"cm_{L.slug(cls.get('model', ''))}.png")
    with right:
        runs = L.chart_runs(get_csv(root, L.OPTIONAL_FILES["classification_results"]), "classification")
        if runs is not None:
            done(bar_candidates(runs, "Classification: cross-validated ROC AUC", "ROC AUC (training data, 3 folds)",
                                cls.get("skill_threshold_cv_auc", 0.52), 0.5, "coin flip (0.5)"))
        else:
            st.caption("(Model comparison chart unavailable: classification_results.csv not found.)")
    if review is not None:
        with st.expander("Class-imbalance review: every model next to three no-skill reference predictors"):
            st.dataframe(review.round(4), hide_index=True)

    st.markdown(f"### Regression — {reg.get('model', 'unknown')}")
    k = st.columns(4)
    with k[0]:
        kpi("R² (test)", L.fmt_num(t_r.get("test_r2"), 4), f"predicting the average scores {L.fmt_num(b_r.get('test_r2'), 4)}")
    with k[1]:
        kpi("RMSE (test)", L.format_lakhs(t_r["test_rmse"], 1, False) if "test_rmse" in t_r else "n/a",
            f"predicting the average: {L.format_lakhs(b_r['test_rmse'], 1, False)}" if "test_rmse" in b_r else "")
    with k[2]:
        kpi("MAE (test)", L.format_lakhs(t_r["test_mae"], 1, False) if "test_mae" in t_r else "n/a",
            f"predicting the average: {L.format_lakhs(b_r['test_mae'], 1, False)}" if "test_mae" in b_r else "")
    with k[3]:
        kpi("Cross-validated R²", L.fmt_num(cv_r.get("cv_r2_mean"), 4), f"required ≥ {L.fmt_num(reg.get('skill_threshold_cv_r2'), 2)}",
            "ok" if rel["regression"]["passed"] else "fail")
    t_rmse, b_rmse = t_r.get("test_rmse"), b_r.get("test_rmse")
    if not both_fail and isinstance(t_rmse, (int, float)) and isinstance(b_rmse, (int, float)) and t_rmse >= b_rmse * 0.999:
        callout("warn", "No improvement over the average", "The model's error is essentially the same as always predicting the average scenario price.")
    runs = L.chart_runs(get_csv(root, L.OPTIONAL_FILES["regression_results"]), "regression")
    if runs is not None:
        done(bar_candidates(runs, "Regression: cross-validated R²", "R² (training data, 3 folds)", reg.get("skill_threshold_cv_r2", 0.01), 0.0, "predict the average (0)"))

    signal = get_csv(root, "phase3_outputs/signal_permutation_test.csv")
    if signal is not None:
        st.markdown("### Is there any signal to find?")
        st.write("Each model was re-trained with the targets randomly shuffled. If the real score is no better than the shuffled scores, "
                 "the inputs contain no detectable information about the target.")
        show = signal.copy()
        for col in ("observed_cv_score", "null_mean", "null_std", "null_max", "p_value"):
            if col in show:
                show[col] = show[col].map(lambda v: f"{v:.4f}")
        st.dataframe(show, hide_index=True)

    st.markdown("### Details")
    for rel_path, title in (("phase3_outputs/classification_results.csv", "All classification runs (including leakage demonstrations)"),
                            ("phase3_outputs/regression_results.csv", "All regression runs (including leakage demonstrations)"),
                            ("phase3_outputs/test_uncertainty.csv", "Test-set uncertainty (95% bootstrap intervals)"),
                            ("phase3_outputs/floor_flag_slices.csv", "Performance on rows with consistent vs impossible floor data")):
        d = get_csv(root, rel_path)
        if d is not None:
            with st.expander(title):
                st.dataframe(d.round(4), hide_index=True)
                if "results" in rel_path:
                    st.caption("Rows labelled LEAKY, rule-recovery or formula-recovery are demonstrations: their inputs contain the answer, "
                               "so their high scores say nothing about real predictive skill.")
    shown = False
    for key, title in (("importance_cls", "Classifier"), ("importance_reg", "Regressor")):
        d = get_csv(root, L.OPTIONAL_FILES[key])
        if d is not None and {"feature", "score_drop_mean"} <= set(d.columns):
            shown = True
            with st.expander(f"Feature importance — {title}"):
                st.bar_chart(d.sort_values("score_drop_mean", ascending=False).set_index("feature")["score_drop_mean"])
    if not shown:
        callout("info", "Feature importance is not shown",
                "The Phase 3 notebook computes importance only for a model that clears its skill threshold, because ranking the inputs of a model "
                "that is no better than guessing would only describe noise.")
    note = L.model_family_note(meta)
    if note["level"] != "ok":
        callout("warn" if note["level"] == "warn" else "fail", "Model family", note["text"])


# ------------------------------------------------------------------ MARKET INSIGHTS
def render_insights(df):
    page_header("Market Insights", "Descriptive statistics of the project dataset. Nothing here is a market forecast.")
    if df is None:
        callout("info", "Dataset not available", "outputs/engineered_features.csv (written by the Phase 2 notebook) was not found or could not be read. "
                "Predictions are not affected.")
        return
    with st.expander("Filters", expanded=True):
        f1, f2 = st.columns(2)
        with f1:
            cities = st.multiselect("City (empty = all)", sorted(df["City"].unique()), default=[], key="f_city")
            ptypes = st.multiselect("Property type (empty = all)", sorted(df["Property_Type"].unique()), default=[], key="f_ptype") if "Property_Type" in df else []
            lo_b, hi_b = int(df["BHK"].min()), int(df["BHK"].max())
            bhk = st.slider("BHK", lo_b, hi_b, (lo_b, hi_b), key="f_bhk") if "BHK" in df and lo_b < hi_b else None
        with f2:
            p_lo, p_hi = float(df["Price_in_Lakhs"].min()), float(df["Price_in_Lakhs"].max())
            price = st.slider("Asking price (₹ lakh)", p_lo, p_hi, (p_lo, p_hi), key="f_price")
            s_lo, s_hi = float(df["Size_in_SqFt"].min()), float(df["Size_in_SqFt"].max())
            size = st.slider("Size (sq ft)", s_lo, s_hi, (s_lo, s_hi), key="f_size")
    sub = L.filter_properties(df, cities, ptypes, bhk, price, size)
    if len(sub) == 0:
        callout("info", "No properties match these filters", "Widen the filters to see results.")
        return

    k = st.columns(4)
    with k[0]:
        kpi("Properties in selection", L.indian_number(len(sub)), f"of {L.indian_number(len(df))} in the dataset")
    with k[1]:
        kpi("Median asking price", L.format_lakhs(sub["Price_in_Lakhs"].median()), "middle value of the selection")
    with k[2]:
        kpi("Median price per sq ft", L.format_inr(sub["Price_in_Lakhs"].div(sub["Size_in_SqFt"]).median() * L.RUPEES_PER_LAKH),
            "asking price ÷ size")
    with k[3]:
        kpi("Median size", f"{L.indian_number(sub['Size_in_SqFt'].median())} sq ft", "middle value of the selection")

    spread = L.city_spread(sub)
    if spread and spread["n_cities"] > 1:
        callout("info", "How much do cities differ?",
                f"Among the {spread['n_cities']} cities in this selection, the median asking price runs from {L.format_lakhs(spread['lowest'], 1, False)} "
                f"({spread['lowest_city']}) to {L.format_lakhs(spread['highest'], 1, False)} ({spread['highest_city']}): a gap of "
                f"{spread['spread_pct']:.1f}% of the overall median. In the Phase 2 analysis, gaps of this size were about what random variation "
                "produces, so rankings here are descriptive and not evidence that an area is a better investment.")

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6.2, 3.6))
        ax.hist(sub["Price_in_Lakhs"], bins=40, color=BLUE, edgecolor="white")
        ax.axvline(sub["Price_in_Lakhs"].median(), color=RED, linestyle="--", label=f"median {L.format_lakhs(sub['Price_in_Lakhs'].median(), 1, False)}")
        ax.set_xlabel("Asking price (₹ lakh)"); ax.set_ylabel("Number of properties"); ax.set_title("Distribution of asking prices"); lakh_axis(ax, "x")
        ax.legend(); done(fig)
    with c2:
        if "Property_Type" in sub and "Price_per_SqFt_INR" in sub:
            med = sub.groupby("Property_Type")["Price_per_SqFt_INR"].median().sort_values()
            fig, ax = plt.subplots(figsize=(6.2, 3.6))
            bars = ax.barh(med.index, med.values, color=NAVY, height=0.5)
            for b, v in zip(bars, med.values):
                ax.text(v, b.get_y() + b.get_height() / 2, " " + L.format_inr(v), va="center", fontsize=9)
            ax.set_xlim(0, med.max() * 1.25); ax.set_xlabel("Median price per sq ft (₹)"); ax.set_title("Price per sq ft by property type")
            ax.grid(axis="y", visible=False); lakh_axis(ax, "x"); done(fig)
    c3, c4 = st.columns(2)
    with c3:
        med = sub.groupby("City")["Price_in_Lakhs"].median().sort_values()
        fig, ax = plt.subplots(figsize=(6.2, max(3.2, 0.26 * len(med))))
        ax.barh(med.index, med.values, color=BLUE, height=0.65)
        ax.axvline(sub["Price_in_Lakhs"].median(), color=RED, linestyle="--", label="median of selection")
        ax.set_xlabel("Median asking price (₹ lakh)"); ax.set_title("Median asking price by city"); ax.grid(axis="y", visible=False)
        lakh_axis(ax, "x"); ax.legend(fontsize=8); done(fig)
    with c4:
        if "BHK" in sub:
            counts = sub["BHK"].value_counts().sort_index()
            fig, ax = plt.subplots(figsize=(6.2, 3.6))
            ax.bar(counts.index.astype(str), counts.values, color=BLUE, width=0.6)
            ax.set_xlabel("BHK"); ax.set_ylabel("Number of properties"); ax.set_title("Properties by number of bedrooms"); ax.grid(axis="x", visible=False)
            lakh_axis(ax, "y"); done(fig)
    if {"Property_Type", "Price_per_SqFt_INR"} <= set(sub.columns) and sub["City"].nunique() > 1:
        piv = sub.pivot_table(index="City", columns="Property_Type", values="Price_per_SqFt_INR", aggfunc="median")
        try:
            import seaborn as sns
            fig, ax = plt.subplots(figsize=(7.5, max(3.5, 0.26 * len(piv))))
            sns.heatmap(piv, annot=len(piv) <= 15, fmt=".0f", cmap="Blues", ax=ax, cbar_kws={"label": "Median ₹ per sq ft"}, linewidths=0.4, linecolor="white")
            ax.set_title("Median price per sq ft: city × property type"); ax.grid(False); ax.set_xlabel(""); ax.set_ylabel("")
            done(fig)
        except ImportError:
            st.dataframe(piv.round(0))
    bands = L.year_band_medians(sub)
    if bands is not None:
        fig, ax = plt.subplots(figsize=(7.5, 3.3))
        ax.plot(bands.index.astype(str), bands["median"], marker="o", color=NAVY)
        ax.set_xlabel("Year-built band"); ax.set_ylabel("Median asking price (₹ lakh)"); ax.set_title("Median asking price by construction-year band")
        lakh_axis(ax, "y"); done(fig)
        st.caption("This is not a price trend over time: the dataset has no sale dates. It compares different buildings by construction year, "
                   "and the pattern is noise.")
    section("Matching properties")
    cols = [c for c in ("ID", "City", "Property_Type", "BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt_INR", "Furnished_Status", "Availability_Status") if c in sub]
    table = sub[cols].head(200).rename(columns={"Price_in_Lakhs": "Price (₹ lakh)", "Price_per_SqFt_INR": "₹ per sq ft", "Size_in_SqFt": "Size (sq ft)"})
    st.dataframe(table, hide_index=True)
    st.caption("Showing the first 200 matching rows.")


# ------------------------------------------------------------------ METHODOLOGY & ABOUT
def render_about(art):
    meta, rule = art.metadata, art.label_rule
    page_header("Methodology & About", "How the numbers are produced, what they mean, and where the data and models fall short.")
    if meta:
        growth, years = meta.get("growth_rate_assumed"), meta.get("horizon_years", 5)
        mult = L.scenario_multiplier(growth, years)
        section("Purpose")
        st.write("This project helps a user explore a property dataset, screen a property against a transparent rule and view five-year price "
                 "scenarios. It is an academic demonstration of a data-science workflow, not a financial product.")
        section("The rule-based screen (the \"Good Investment\" label)")
        st.write("A property meets the screen when **both** conditions hold:")
        st.markdown("1. **Value test:** its price per sq ft is at or below the median of similar properties (same city and property type), "
                    "measured on the training data only.\n"
                    "2. **Quality test:** it meets at least " + str(rule.get("quality_threshold", 3)) + " of 5 criteria: BHK of 3 or more, ready to move, "
                    "parking, security, and high public-transport access.")
        callout("info", "What the label is not", "It is a convention we wrote, not an observed outcome. It does not show that a property is, or will be, profitable.")
        section("The five-year growth assumption")
        st.write(f"The five-year price is today's price multiplied by (1 + {growth:.0%})^{years} = {mult:.4f}. The {growth:.0%} annual rate is an "
                 "**assumption we chose**: the dataset contains no sale dates or price history, so no growth rate can be measured from it. "
                 "The result is a scenario, not a verified market forecast.")
        section("Data and its limitations")
        st.markdown("- The dataset behaves like randomly generated data: features are almost unrelated to price, and several records are internally contradictory "
                    "(for example floor numbers above the building's height, in about 46.5% of records, and under-construction buildings with past construction years).\n"
                    "- Neither a real future price nor a real investment label exists, so both targets are constructed.\n"
                    "- Because of this, high model scores would say little about real markets, and low scores are expected.")
        section("Model approach")
        st.markdown("- The data was split into training and test sets **before** the label rule or any statistics were computed; the rule uses training data only.\n"
                    "- The classifier does not see the label's own ingredients (price, size, BHK, status, parking, security, transport); the price model does not "
                    "see price. Otherwise they would simply recompute the answer.\n"
                    "- Floor columns are excluded because they are unreliable.\n"
                    "- Models are compared with simple baselines using cross-validation. A model is treated as reliable only if it clears a minimum skill threshold "
                    "that was set before training: cross-validated ROC AUC ≥ "
                    f"{L.fmt_num((meta.get('classification') or {}).get('skill_threshold_cv_auc'), 2)} and R² ≥ "
                    f"{L.fmt_num((meta.get('regression') or {}).get('skill_threshold_cv_r2'), 2)}.")
        rel = L.reliability(meta)
        both_below_threshold = not rel["classification"]["passed"] and not rel["regression"]["passed"]
        if both_below_threshold:
            callout("fail", "Neither model meets its reliability threshold",
                    "Both models failed their pre-declared skill checks. Their outputs must not be treated as evidence of profitability or future prices.")
        else:
            reliability_callout(rel["classification"], "Classification model")
            reliability_callout(rel["regression"], "Regression model")
        section("Statements this app may and may not make")
        a, b = st.columns(2)
        with a:
            st.markdown("**May say**")
            for s in meta.get("allowed_claims", []):
                st.write("• " + s)
        with b:
            st.markdown("**May not say**")
            for s in meta.get("NOT_allowed_claims", []):
                st.write("• " + s)
        if meta.get("disclaimer"):
            st.caption(meta["disclaimer"])
    if rule:
        with st.expander("Saved rule details"):
            st.write(rule.get("description", ""))
            st.write(f"Fitted on {L.indian_number(rule.get('fitted_on_rows', 0))} training rows; "
                     f"{len(rule.get('segment_median_ppsf_lakhs', {}))} city × property-type groups.")
            st.dataframe(pd.DataFrame({"Quality criterion": rule.get("quality_criteria", [])}), hide_index=True)

    with st.expander("Technical details"):
        env = L.environment_report()
        env["streamlit"] = getattr(st, "__version__", env.get("streamlit"))
        trained = (meta.get("library_versions") or {}) if meta else {}
        st.dataframe(pd.DataFrame([
            {"Component": "Python", "Running now": env["python"], "Used when models were trained": trained.get("python", "n/a")},
            {"Component": "scikit-learn", "Running now": env["sklearn"] or "not installed", "Used when models were trained": trained.get("scikit_learn", "n/a")},
            {"Component": "XGBoost", "Running now": env["xgboost"] or "not installed", "Used when models were trained": trained.get("xgboost") or "not installed"},
            {"Component": "pandas", "Running now": env["pandas"] or "not installed", "Used when models were trained": trained.get("pandas", "n/a")},
            {"Component": "numpy", "Running now": env["numpy"] or "not installed", "Used when models were trained": trained.get("numpy", "n/a")},
            {"Component": "Streamlit", "Running now": env["streamlit"] or "n/a", "Used when models were trained": "n/a"}]), hide_index=True)
        st.caption(f"Python interpreter running this app: {env['executable']}")
        if meta:
            note = L.model_family_note(meta, env)
            callout({"ok": "ok", "warn": "warn", "fail": "fail"}[note["level"]], "Model library", note["text"])
        for w in art.warnings:
            callout("warn", "Notice", w)
        for e in art.errors:
            callout("fail", "Problem", e)
        st.button("Reload models and data", on_click=reload_all, key="reload_btn")


# ------------------------------------------------------------------ main
def main():
    st.markdown(CSS, unsafe_allow_html=True)
    art = get_artifacts(str(PROJECT_ROOT))
    df = get_browse_data(str(PROJECT_ROOT))

    st.markdown('<div class="reia-brand">🏠 Real Estate Investment Advisor</div>', unsafe_allow_html=True)
    if "nav_page" not in st.session_state:
        st.session_state["nav_page"] = PAGES[0]
    page = st.radio("Navigate", PAGES, horizontal=True, key="nav_page", label_visibility="collapsed")
    st.divider()

    if page == "Home":
        render_home(art, df)
    elif page == "Property Assessment":
        if not art.ready:
            callout("fail", "Predictions are unavailable", art.errors or ["The saved models could not be loaded."])
        else:
            options = L.get_options(art, df)
            if not options.get("City"):
                callout("fail", "No list of cities is available", "Neither the fitted pipelines nor the data file contain one.")
            else:
                render_assessment(art, options, L.numeric_ranges(df), L.default_values(df))
    elif page == "Model Performance":
        if art.metadata:
            render_performance(art)
        else:
            callout("fail", "Model metadata is unavailable", art.errors or ["models/model_metadata.json could not be read."])
    elif page == "Market Insights":
        render_insights(df)
    else:
        render_about(art)

    st.markdown('<div class="reia-foot">Scenario estimates and rule-based screening for an academic project. Not investment advice; '
                'no returns or future prices are guaranteed or verified.</div>', unsafe_allow_html=True)


main()
