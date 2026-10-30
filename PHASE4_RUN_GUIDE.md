# Phase 4 fixes — run guide, change log and verification checklist

## 1. What the investigation found

### Model reliability (classification AUC ≈ 0.50, regression R² ≈ −0.0002)
`tools/verify_artifacts.py` re-derives everything from the saved files and compares it with `model_metadata.json`. Result: **no defect**.

| Checked | Result |
|---|---|
| Train/test split | disjoint IDs, sizes and training-ID fingerprint equal the metadata, stratified by city (max share gap 0.001 percentage points) |
| Label rule | medians equal medians recomputed from **training rows only**; they differ from full-data medians, so no test rows were used |
| Class balance | 'good' = 23.86% of train, 23.54% of test; majority-class baseline accuracy 0.7646 with F1 = 0 (it never predicts 'good') |
| Target definitions | regression target = price × 1.469328 (8% for 5 years, an assumption); label = value test AND quality score ≥ 3 |
| Leakage | the classifier sees no label-defining column; the regressor sees no price-derived column; both floor columns are excluded |
| Preprocessing | pipeline inputs equal the metadata; encoded output is finite |
| Reproducibility | the saved models reproduce every metadata test metric (difference < 1e-6) |
| Skill flags | the metadata flags equal (CV score ≥ threshold): AUC 0.5000 < 0.52 and R² −0.0002 < 0.01 |
| **Is there any signal?** | Shuffled-target (permutation) test: classification observed CV AUC 0.5041 vs shuffled-target scores with mean 0.4989 and maximum 0.5045 (p = 0.18); regression observed R² −0.0010 vs shuffled mean −0.0008 (p = 0.64). The real scores cannot be told apart from scores obtained with meaningless targets. |

**Conclusion.** The models fail their thresholds because the honest inputs carry no detectable information about the constructed targets
(consistent with Phase 2: every feature's relationship with price was at chance level). Nothing was tuned, relaxed or hidden: the
thresholds (0.52 and 0.01), the warnings and the 8% assumption are unchanged. A model clearing the threshold here would be a red flag for leakage, not progress.
(The numbers above come from my re-run of the same pipeline; run the tool on your own files to confirm yours.)

### Environment (scikit-learn 1.7.2 vs 1.8.0; XGBoost "unavailable")
* The notebook kernel and the Streamlit app are **two different Python environments**: the models were trained under scikit-learn 1.7.2 and the app runs 1.8.0.
  The same explanation fits XGBoost: `pip install -r requirements.txt` was evidently run in the terminal environment, not the one the notebook's kernel uses, so the
  notebook silently trained with the HistGradientBoosting fallback ("XGBoost not installed" is in your saved model name).
* I cannot inspect your machine, so confirm it yourself (commands in section 3).
* The old requirement `xgboost>=2.1.4` was also too loose: the first XGBoost release with scikit-learn 1.8 compatibility fixes is 3.1.3 (per its release notes), so it is now `xgboost>=3.1.3,<4`.
* scikit-learn is pinned exactly to **1.8.0** (the version your app already runs). Retraining under it is the only reproducible fix: a pickled model should be loaded with the version that saved it.

## 2. Files in this delivery

| File | Status | Purpose |
|---|---|---|
| `app/streamlit_app.py` | **replaced** | Light professional UI; five pages; no sidebar |
| `app/app_logic.py` | **modified** | Indian rupee/lakh formatting, environment report, start-up test prediction, version-warning capture, performance helpers |
| `.streamlit/config.toml` | **new** | Forces the light theme (white background, navy accent) locally and on Streamlit Cloud |
| `requirements.txt` | **modified** | Runtime dependencies with the exact scikit-learn pin and corrected XGBoost lower bound |
| `requirements-dev.txt` | **new** | Adds notebook, MLflow, scipy and pytest |
| `02_ML_Model_Training_patched.ipynb` | **modified** | Adds PATCH 3 (environment guard), PATCH 4 (shuffled-target signal test), PATCH 5 (records the environment); patches 1–2 retained |
| `PHASE3_PATCH.md` | **modified** | Code of the new notebook cells appended |
| `tools/verify_artifacts.py` | **new** | Independent verification described above |
| `tests/test_app_logic.py` | **modified** | 27 logic tests |
| `tests/test_model_artifacts.py` | **new** | Runs the independent verification as a test |
| `PHASE4_RUN_GUIDE.md` | **replaced** | This file |

Copy each file to the same relative path inside your project folder (the one that contains `models/`, `phase3_outputs/`, `outputs/`).

## 3. Commands to run (in this order)

**a) One environment for everything** (project folder; macOS/Linux shown, Windows activation in the comment)
```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m ipykernel install --user --name reia --display-name "Python (reia)"
python -c "import sys, sklearn, xgboost; print(sys.executable); print(sklearn.__version__, xgboost.__version__)"
```
Expected: the path points inside `.venv`, scikit-learn prints `1.8.0`, xgboost prints 3.1.3 or higher. If `import xgboost` fails, fix that before continuing.

**b) Retrain in that environment** (needed because the saved models and metadata must come from the same environment)
1. `jupyter notebook`, open `02_ML_Model_Training_patched.ipynb`, choose **Kernel → Change kernel → Python (reia)**.
2. **Kernel → Restart & Run All** (all cells: pickles, metadata, CSVs and figures must come from one run; about 15–25 minutes).
   The new guard cell (right after the version printout) stops immediately with instructions if scikit-learn is not 1.8.0 or xgboost cannot be imported.
3. Optional clean start: delete the old `mlflow/` folder first.

**c) Verify and test** (same terminal, same environment)
```bash
python tools/verify_artifacts.py --permutation   # expect no FAIL lines; the signal test takes a few minutes
python -m pytest tests -q                         # or: python tests/test_app_logic.py && python tests/test_model_artifacts.py
```

**d) Launch the app**
```bash
python -m streamlit run app/streamlit_app.py     # http://localhost:8501
```
`python -m streamlit` guarantees the app uses the same interpreter you just checked.

**What to expect after retraining:** the models will probably still be below their thresholds. The app shows whatever is measured; do not change the thresholds.

## 4. Manual verification checklist (about 10 minutes)
1. Page is white with dark text and navy accents; **no sidebar** and no file list anywhere.
2. Five navigation pills: Home, Property Assessment, Model Performance, Market Insights, Methodology & About.
3. Home: title, four key-figure cards, red reliability notice if a model is below threshold, prominent **Assess a property →** button (it switches to the assessment page).
4. Assessment: four organised input sections. Size 0, price 0 or BHK 0 → red message, no results. Size 9000 with price 900 → results plus an "outside the range" notice.
5. Results: four cards (rule screen, model probability, model scenario price, arithmetic scenario). Rupees use Indian grouping (₹12,34,567) and prices read "₹250.0 lakh" (with crore above 100 lakh).
6. Failing models carry a red "below its reliability threshold" label on their cards; the 8% is described as an assumption and the label as a rule, not proof.
7. Rule check: price 10, size 5000, BHK 4, ready to move, parking Yes, security Yes, transport High → "Meets screen"; price 500, size 500 → "Does not meet".
8. What-if slider changes only the scenario table and chart.
9. Model Performance: red summary when thresholds are missed; threshold table; metric cards with baselines beside them; confusion matrix; model-comparison charts with the threshold line; class-imbalance and signal-test sections (if their CSVs exist).
10. Metrics shown equal `models/model_metadata.json` (compare three).
11. Market Insights: filters change the cards and charts; impossible filters → "No properties match"; the year-built chart is captioned "not a price trend".
12. Methodology & About: rule, growth assumption, data limits, model approach, may/may-not statements; **Technical details** expander shows Python, scikit-learn and XGBoost versions now vs at training time.
13. Resilience: rename `models/classifier_pipeline.joblib` and press **Reload models and data** (Technical details) → red setup message, assessment disabled, other pages still work; restore and reload.

## 5. Change log
* **Reliability:** audited targets, split, leakage, preprocessing, baselines and reproducibility; added a shuffled-target signal test; thresholds and warnings untouched.
* **Environment:** exact `scikit-learn==1.8.0`, `xgboost>=3.1.3,<4`; guard cell stops the notebook in a mismatched environment; training environment recorded in the metadata; app shows trained-vs-running versions; unpickling version warnings are captured; start-up test prediction proves the models work before predictions are offered.
* **UI:** light theme via `.streamlit/config.toml` plus CSS; navy/blue palette; five-page navigation with a Home call to action; no sidebar; metric cards; native confusion matrix and comparison charts; Indian rupee/lakh formatting; permanent disclaimers.
* **Safety/UX:** invalid-input messages, out-of-range notices, graceful handling of missing, corrupt or incompatible files; fail-safe treatment of a missing skill flag.
* **Tests:** 27 logic tests (formatting, rule parity with the notebook, validation, loading failures, version warnings, reliability messages, results-vs-metadata consistency, start-up check) plus the artifact verification test.

## 6. Known limits of this delivery
* I could not run the real Streamlit, XGBoost or MLflow here (not installed, no network). The UI was exercised with a stand-in `streamlit` module (47 checks) and the logic with real saved pipelines (27 tests); widget look and feel needs your visual check.
* I cannot change your machine: your existing `models/` folder was trained under scikit-learn 1.7.2 and stays that way until you retrain (step 3b).
* The notebook update was test-run end to end on a 15% sample with stand-ins for MLflow and XGBoost; the full-size run happens on your machine.
