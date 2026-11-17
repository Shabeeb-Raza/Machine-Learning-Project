# ModelLab

A free-to-deploy Streamlit machine-learning workspace for a university assignment. Load a CSV or a built-in sample, work through five guided steps, train a model, and read the results in plain English.

The app runs as a single Python process and is designed for Streamlit Community Cloud's free tier and a 1 GB memory budget.

## The five-step wizard

1. **Data** — upload a CSV or load a sample, choose the delimiter and encoding, confirm the target and task, select features, and inspect the dataset. Data quality flags call out IDs, constants, missing values, and high-cardinality columns.
2. **Preprocess** — review a per-column recipe for type overrides, missing values, outlier handling, categorical encoding, rare-category grouping, scaling, Yeo-Johnson, and variance/correlation filters. Apply the recipe, inspect a before/after preview, and download it as JSON.
3. **Split** — set the test size, random seed, and optional stratification. The app stores train/test indices and shows class or target distributions. A small test set is allowed but clearly flagged.
4. **Model** — choose an estimator from the shared registry, read a short model card, edit hyperparameters or reset them, optionally run a capped grid search, and train on the training split only.
5. **Results** — read held-out metrics, baseline comparisons, a plain-English summary, confusion/ROC/PR plots, per-class metrics, regression errors, feature importance, cross-validation stability, and downloadable artifacts.

The stepper is gated. Only completed steps and the next eligible step can be opened, and only the active step's code runs. There is no sidebar.

Every step has a short explanation, a bordered workspace, a step-summary card, and Back/Continue controls. Continue is disabled with a one-line reason when the step is incomplete. Changing an upstream choice clears downstream artifacts and reports what was reset. Start over asks for confirmation.

## Data handling

- CSV only, maximum 20 MB.
- Delimiter can be auto-detected or set to comma, semicolon, tab, or pipe.
- Encoding can be auto-detected, UTF-8, UTF-8 with BOM, or Latin-1.
- Datasets over 50,000 rows are randomly sampled to 50,000 rows, with a visible notice.
- Classification and regression are auto-detected from the target and can be overridden.
- Target labels are encoded for fitting and decoded again for metrics, curves, and tables.
- The app keeps one working dataframe, split indices, one fitted pipeline, and small result tables in `st.session_state`. State keys use the `wz_` prefix.

## Preprocessing and leakage

The recipe table covers every selected column. Global settings control the one-hot category cap, rare-category frequency, variance filter, and correlation filter.

Row-level cleaning happens before the split:

- rows with a missing target are dropped;
- rows with missing values in a column set to **drop** are removed;
- rows outside the IQR fence in a numeric column set to **remove** are removed.

Column-level transforms are fitted on training rows only:

- missing-value statistics;
- outlier clip bounds and outlier flags;
- rare-category grouping and category order;
- ordinal, one-hot, and frequency encoders;
- standard, min-max, robust, and Yeo-Johnson scaling;
- variance and correlation filters.

The Step 2 before/after table is an illustrative preview of up to 5,000 rows. It is not the fitted model preprocessor. During training, the exact same recipe is fitted on the training split inside an sklearn `Pipeline`.

All custom transformers live at module level in `preprocess.py`, so a fitted pipeline can be saved with joblib and loaded in another Python process that has this project on its import path.

## Results and cross-validation

- Compact metric cards and a baseline comparison.
- A "What this means" panel with three to five plain-English sentences.
- Confusion matrix with counts or row percentages.
- ROC and precision-recall curves for binary classification.
- Predicted-probability histogram when supported.
- Per-class precision, recall, F1, and support.
- Regression predicted-vs-actual, residuals, and largest-error tables.
- Feature importance from built-in coefficients/importances or permutation importance.
- A warning when a classifier predicts only one class.
- Cross-validation on the training split with 3–10 requested folds, mean, standard deviation, and a per-fold chart.

Cross-validation reduces the fold count when class counts are too small, when there are too few rows, or when the training rows × folds cap would be exceeded.

## Free-tier limits enforced in code

| Limit | Value |
| --- | --- |
| Maximum upload size | 20 MB |
| Maximum training rows | 50,000 (larger datasets are randomly sampled) |
| Maximum grid-search combinations | 50 |
| Cross-validation folds | 3–10 requested |
| Cross-validation rows × folds cap | 250,000 |
| Parallel jobs | Forced off where supported (`n_jobs=1`) |
| Preview rows | 5,000 |
| Visualisation rows | 5,000 |
| Feature-importance budget | 20 seconds, at most 500 test rows and 3 repeats |
| Target deployment memory | 1 GB design budget |

Streamlit Community Cloud apps sleep after 12 hours of inactivity. CSV parsing, sample loading, and EDA statistics use `st.cache_data`. Every Matplotlib figure is closed after rendering.

## Requirements

- Python 3.12
- The packages pinned in `requirements.txt`

No additional dependencies are required.

## Run locally

```bash
cd ml-dashboard
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

On Windows, activate the virtual environment with:

```powershell
.venv\Scripts\activate
```

The app opens at `http://localhost:8501`.

## Quick check

1. Choose **Sample dataset** and keep **Classification**.
2. Continue through **Data** with `churn` as the target.
3. In **Preprocess**, review the recipe and click **Apply recipe**.
4. In **Split**, keep the defaults and click **Create split**.
5. In **Model**, keep **Logistic Regression** and click **Train model**.
6. Review the baseline uplift, per-class table, confusion matrix, ROC, precision-recall, probability histogram, feature importance, and cross-validation panel.
7. Switch the sample to **Regression**, repeat the steps with **Ridge Regression**, and review the regression plots.
8. Use **Download trained model (.joblib)** and reload it with `joblib.load` to confirm the preprocessing recipe travels with the model.

## Project structure

```text
ml-dashboard/
├── app.py
├── wizard.py
├── preprocess.py
├── model_info.py
├── eda.py
├── plots.py
├── models.py
├── pipeline.py
├── requirements.txt
├── README.md
├── DEPLOY_CHECKLIST.md
├── .gitignore
├── .streamlit/
│   └── config.toml
└── data/
    ├── sample_classification.csv
    └── sample_regression.csv
```

- `app.py` owns the Streamlit layout, active-step dispatch, step renders, result views, and downloads.
- `wizard.py` owns `wz_`-prefixed state, completion tracking, gated navigation, the stepper, headers, summaries, and footers.
- `preprocess.py` owns recipe defaults and validation, row-level cleaning, target encoding, the module-level recipe transformer, previews, and result label decoding. It has no Streamlit calls.
- `model_info.py` contains short model cards: summary, strengths, weaknesses, and typical use.
- `eda.py` contains pandas/numpy-only statistics and plain-English insight generation. It has no Streamlit or Matplotlib calls.
- `plots.py` contains Matplotlib figure builders with one shared colour-blind-safe palette. It has no Streamlit calls.
- `models.py` is the single model registry. The UI loops over it and has no per-model widget code.
- `pipeline.py` contains splitting, training, evaluation, baselines, feature importance, grid search, and cross-validation. It has no Streamlit calls.

## Add a new model in three steps

All model definitions live in `models.py`.

1. Import the estimator class at the top of `models.py`:

```python
from sklearn.ensemble import ExtraTreesClassifier
```

2. Add an entry to `MODEL_REGISTRY`:

```python
"Extra Trees": {
    "task": CLASSIFICATION,
    "class": ExtraTreesClassifier,
    "fixed_params": {},
    "hyperparameters": {
        "n_estimators": {
            "type": "int",
            "min": 10,
            "max": 300,
            "step": 10,
            "default": 100,
            "help": "Number of trees in the forest.",
        },
        "max_depth": {
            "type": "int",
            "min": 1,
            "max": 30,
            "step": 1,
            "default": 10,
            "help": "Maximum depth of each tree.",
        },
    },
},
```

3. Restart or reload the app. The model appears in the correct task dropdown and its widgets are generated automatically.

Supported hyperparameter spec types are:

- `select`: `options`, `default`, `help`
- `int`: `min`, `max`, `step`, `default`, `help`
- `float`: `min`, `max`, `step`, `default`, `help`

No changes to `app.py` or `pipeline.py` are needed for a model that uses these types. The pipeline automatically injects `random_state` when the estimator accepts it and disables parallelism where meaningful.

## Implementation notes

- Regression targets are standardised internally before fitting and transformed back for predictions. Metrics and plots are therefore in the original target units.
- Classification precision, recall, and F1 use weighted averaging so the metrics also work for multiclass targets.
- ROC and precision-recall curves are shown only for binary classification.
- Per-class metrics are always shown for classification.
- Feature importance uses built-in tree importances or linear coefficients when available. Otherwise it uses permutation importance on at most 500 test rows with three repeats and a 20-second budget.
- Scale-sensitive models are SVM, SVR, KNN, KNN Regressor, and Logistic Regression. The app warns when the recipe has no numeric scaling.
- Training, search, and cross-validation errors are caught and shown as friendly `st.error` messages. Raw tracebacks are not displayed.
- Custom transformers must stay module-level in `preprocess.py` for joblib compatibility.
- The split is stored as indices, not a duplicated dataframe. This leaves room for a future clustering step and a future new-record step without changing the earlier contract.

## Deploy to Streamlit Community Cloud

See `DEPLOY_CHECKLIST.md` for the exact click-by-click steps. In short:

1. Push this folder to a GitHub repository.
2. Sign in at <https://share.streamlit.io> with GitHub.
3. Create a new app from the repository, branch `main`, main file `app.py`.
4. Deploy. No credit card or paid service is required.
5. Confirm the app's memory limit. The app is designed for 1 GB.
