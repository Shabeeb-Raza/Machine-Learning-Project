# ML Dashboard

A free-to-deploy Streamlit dashboard for a university machine learning assignment. Upload a CSV or load a sample dataset, choose a model, tune hyperparameters in the UI, and evaluate the model on a held-out test set.

The app is a single Python process. It is designed to fit inside Streamlit Community Cloud's free tier and to stay within a 1 GB memory budget.

## What it does

- Uploads a CSV (maximum 20 MB) or loads a built-in sample dataset.
- Shows the dataset shape, column dtypes, missing-value counts, and the first 10 rows.
- Selects a target column and auto-detects classification vs regression. The task type can be overridden.
- Handles missing values by dropping rows, mean, median, or most frequent.
- Encodes categorical columns with one-hot or ordinal encoding.
- Scales numeric columns with standard, min-max, or no scaling.
- Configures the train/test split with a test-size slider, random seed, and optional stratified sampling.
- Renders model-specific hyperparameter controls from a single registry.
- Trains a model and reports the training time.
- Shows classification metrics, a confusion matrix, and a binary ROC curve.
- Shows regression metrics and a predicted-vs-actual plot.
- Downloads the fitted pipeline as a `.joblib` file and the metrics as a CSV.
- Supports an optional small grid search, capped at 50 combinations.

## Free-tier limits enforced in code

| Limit | Value |
| --- | --- |
| Maximum upload size | 20 MB |
| Maximum training rows | 50,000 (larger datasets are randomly sampled) |
| Maximum grid-search combinations | 50 |
| Parallel jobs | `n_jobs=1` everywhere |
| Target deployment memory | 1 GB design budget |

Streamlit Community Cloud apps sleep after 12 hours of inactivity. The UI and data-loading functions use `st.cache_data`; only small pieces of UI state are kept in `st.session_state`.

## Requirements

- Python 3.12 (the pinned packages were installed with Python 3.12)
- The packages in `requirements.txt`

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

1. Click **Use sample dataset** with **Classification** selected.
2. Keep the target as `churn`.
3. Choose **SVM**, keep `kernel=rbf`, `C=1.0`, `gamma=scale`, and `degree=3`.
4. Click **Train model**. Accuracy, precision, recall, F1, the confusion matrix, and the ROC curve appear.
5. Switch the sample dataset to **Regression**, reload it, choose **SVR**, and click **Train model**. R-squared, MAE, RMSE, and the predicted-vs-actual plot appear.

## Project structure

```text
ml-dashboard/
├── app.py
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

- `app.py` contains the Streamlit UI and the free-tier guardrails that need UI feedback.
- `models.py` is the single model registry. The UI loops over it and has no per-model widget code.
- `pipeline.py` contains preprocessing, splitting, training, evaluation, and search validation. It contains no Streamlit calls.

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

No changes to `app.py` or `pipeline.py` are needed for a model that uses these types. The pipeline automatically injects `random_state` when the estimator accepts it and always forces `n_jobs=1` where supported.

## Implementation notes

- Regression targets are standardised internally before fitting and transformed back for predictions. Metrics and plots are therefore in the original target units. This makes SVR behave well with a reasonable default `C`.
- Classification precision, recall, and F1 use weighted averaging so the metrics also work for multiclass targets.
- The ROC curve is shown only for binary classification.
- Recommended preprocessing for distance- and kernel-based models: one-hot encoding plus standard scaling.
- Training and search errors are caught and shown as friendly `st.error` messages. Raw tracebacks are not displayed.

## Deploy to Streamlit Community Cloud

See `DEPLOY_CHECKLIST.md` for the exact click-by-click steps. In short:

1. Push this folder to a GitHub repository.
2. Sign in at <https://share.streamlit.io> with GitHub.
3. Create a new app from the repository, branch `main`, main file `app.py`.
4. Deploy. No credit card or paid service is required.
5. Record the memory limit shown for the app. The app is designed for 1 GB.
