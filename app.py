import io
import re
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from models import (
    hyperparameters_for,
    models_for_task,
    search_candidates,
)
from pipeline import (
    MAX_SEARCH_COMBINATIONS,
    MAX_TRAIN_ROWS,
    MAX_UPLOAD_MB,
    PipelineError,
    count_combinations,
    grid_search_and_evaluate,
    infer_task_type,
    limit_rows,
    train_and_evaluate,
    validate_upload,
)

st.set_page_config(
    page_title="ML Dashboard",
    page_icon=None,
    layout="wide",
)


@st.cache_data(show_spinner=False, max_entries=2)
def parse_uploaded_csv(payload):
    return pd.read_csv(io.BytesIO(payload))


@st.cache_data(show_spinner=False, max_entries=2)
def load_sample_csv(path):
    return pd.read_csv(path)


def render_single_hyperparameters(model_name, specs):
    values = {}
    for name, spec in specs.items():
        key = f"hp_{model_name}_{name}"
        if spec["type"] == "select":
            options = list(spec["options"])
            values[name] = st.selectbox(
                name,
                options,
                index=options.index(spec["default"]),
                help=spec["help"],
                key=key,
                format_func=str,
            )
        elif spec["type"] == "int":
            values[name] = st.slider(
                name,
                min_value=int(spec["min"]),
                max_value=int(spec["max"]),
                value=int(spec["default"]),
                step=int(spec["step"]),
                help=spec["help"],
                key=key,
            )
        elif spec["type"] == "float":
            values[name] = st.slider(
                name,
                min_value=float(spec["min"]),
                max_value=float(spec["max"]),
                value=float(spec["default"]),
                step=float(spec["step"]),
                help=spec["help"],
                key=key,
            )
        else:
            st.warning(f"Unsupported hyperparameter type for {name}.")
            values[name] = spec["default"]
    return values


def render_search_hyperparameters(model_name, specs):
    grid = {}
    for name, spec in specs.items():
        grid[name] = st.multiselect(
            f"{name} values",
            options=search_candidates(spec),
            default=[spec["default"]],
            help=spec["help"],
            key=f"search_hp_{model_name}_{name}",
            format_func=str,
        )
    return grid


def render_classification_results(result):
    metrics = result["metrics"]
    columns = st.columns(4)
    columns[0].metric("Accuracy", f"{metrics['accuracy']:.4f}")
    columns[1].metric("Precision", f"{metrics['precision']:.4f}")
    columns[2].metric("Recall", f"{metrics['recall']:.4f}")
    columns[3].metric("F1 score", f"{metrics['f1']:.4f}")

    if "roc_auc" in metrics:
        st.caption(f"ROC AUC: {metrics['roc_auc']:.4f}")

    st.subheader("Confusion matrix")
    matrix = result["confusion_matrix"]
    labels = [str(label) for label in result["classes"]]
    figure, axis = plt.subplots(figsize=(6, 4.5))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    axis.set_title("Confusion matrix")
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("Actual label")
    axis.set_xticks(np.arange(len(labels)), labels=labels, rotation=45, ha="right")
    axis.set_yticks(np.arange(len(labels)), labels=labels)
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                f"{matrix[row, column]:,}",
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    st.pyplot(figure, width="stretch")
    plt.close(figure)

    if result["roc"] is not None:
        st.subheader("ROC curve")
        roc = result["roc"]
        figure, axis = plt.subplots(figsize=(6, 4.5))
        axis.plot(
            roc["fpr"],
            roc["tpr"],
            label=f"AUC = {roc['auc']:.3f}",
        )
        axis.plot([0, 1], [0, 1], linestyle="--", color="gray")
        axis.set_xlabel("False positive rate")
        axis.set_ylabel("True positive rate")
        axis.set_title("ROC curve")
        axis.legend(loc="lower right")
        figure.tight_layout()
        st.pyplot(figure, width="stretch")
        plt.close(figure)


def render_regression_results(result):
    metrics = result["metrics"]
    columns = st.columns(3)
    columns[0].metric("R²", f"{metrics['r2']:.4f}")
    columns[1].metric("MAE", f"{metrics['mae']:.2f}")
    columns[2].metric("RMSE", f"{metrics['rmse']:.2f}")

    st.subheader("Predicted vs actual")
    actual = np.asarray(result["y_test"], dtype=float)
    predicted = np.asarray(result["y_pred"], dtype=float)
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.scatter(actual, predicted, alpha=0.55, s=18)
    minimum = float(min(actual.min(), predicted.min()))
    maximum = float(max(actual.max(), predicted.max()))
    axis.plot([minimum, maximum], [minimum, maximum], linestyle="--", color="gray")
    axis.set_xlabel("Actual value")
    axis.set_ylabel("Predicted value")
    axis.set_title("Predicted vs actual")
    figure.tight_layout()
    st.pyplot(figure, width="stretch")
    plt.close(figure)


def metrics_frame(result):
    return pd.DataFrame(
        [
            {
                "model": result["model_name"],
                "task": result["task"],
                "metric": name,
                "value": value,
            }
            for name, value in result["metrics"].items()
        ]
    )


def slugify(value):
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


st.title("Machine Learning Dashboard")
st.caption(
    "Upload a CSV, choose a model, tune its hyperparameters, "
    "and evaluate it on a held-out test set."
)

st.header("1. Upload CSV")
uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"],
    help=f"Maximum file size: {MAX_UPLOAD_MB} MB.",
    max_upload_size=MAX_UPLOAD_MB,
)

if uploaded_file is not None:
    st.session_state["use_sample"] = False

st.write("or")
sample_column, button_column = st.columns([2, 1])
with sample_column:
    sample_kind = st.selectbox(
        "Sample dataset",
        ["Classification", "Regression"],
        key="sample_kind",
    )
with button_column:
    st.write("")
    if st.button("Use sample dataset", key="use_sample_button"):
        st.session_state["use_sample"] = True

df = None
source_name = None
original_rows = 0

if uploaded_file is not None:
    try:
        validate_upload(uploaded_file.name, uploaded_file.size)
        df = parse_uploaded_csv(uploaded_file.getvalue())
        source_name = uploaded_file.name
        original_rows = len(df)
    except PipelineError as exc:
        st.error(str(exc))
    except Exception:
        st.error(
            "Could not parse this file as CSV. "
            "Check that it is comma-separated and has a header row."
        )
elif st.session_state.get("use_sample", False):
    sample_path = (
        Path(__file__).parent / "data" / f"sample_{sample_kind.lower()}.csv"
    )
    try:
        df = load_sample_csv(str(sample_path))
        source_name = f"Sample {sample_kind.lower()} dataset"
        original_rows = len(df)
    except Exception:
        st.error("The sample dataset could not be loaded.")

if df is None:
    st.info("Upload a CSV or load a sample dataset to begin.")
    st.stop()

if df.empty:
    st.error("The dataset has no rows. Please upload a CSV with data.")
    st.stop()

if original_rows > MAX_TRAIN_ROWS:
    st.warning(
        f"The dataset has {original_rows:,} rows. "
        f"A random sample of {MAX_TRAIN_ROWS:,} rows is used to stay "
        "within the free-tier limit."
    )
    df, _ = limit_rows(df)

st.caption(f"Source: {source_name} | Rows: {len(df):,} | Columns: {df.shape[1]:,}")

st.header("2. Data preview")
summary = pd.DataFrame(
    {
        "dtype": df.dtypes.astype(str),
        "missing": df.isna().sum(),
        "missing_percent": (df.isna().mean() * 100).round(1),
    }
)
summary.index.name = "column"
st.write(f"Shape: {df.shape[0]:,} rows x {df.shape[1]:,} columns")
st.dataframe(summary, width="stretch")
st.write("First 10 rows")
st.dataframe(df.head(10), width="stretch")

st.header("3. Target column")
target_column = st.selectbox(
    "Select the target column",
    options=list(df.columns),
    index=len(df.columns) - 1,
)

if df[target_column].notna().sum() == 0:
    st.error("The selected target column has no values.")
    st.stop()

try:
    detected_task = infer_task_type(df[target_column])
except PipelineError as exc:
    st.error(str(exc))
    st.stop()

task_label = st.radio(
    "Task type",
    ["Classification", "Regression"],
    index=0 if detected_task == "classification" else 1,
    horizontal=True,
    help="Auto-detected from the target column. You can override it.",
    key=f"task_type_{target_column}_{detected_task}",
)
task = task_label.lower()
st.caption(
    f"Auto-detected task type: **{detected_task}**. "
    "Use the control above to override it."
)

st.header("4. Preprocessing options")
preprocessing_columns = st.columns(3)
with preprocessing_columns[0]:
    missing_strategy = st.selectbox(
        "Missing values",
        ["drop", "mean", "median", "most_frequent"],
        format_func={
            "drop": "Drop rows with missing values",
            "mean": "Mean",
            "median": "Median",
            "most_frequent": "Most frequent",
        }.get,
        help=(
            "Mean and median apply to numeric columns; categorical columns "
            "always use the most frequent value."
        ),
    )
with preprocessing_columns[1]:
    encoding = st.selectbox(
        "Categorical encoding",
        ["one-hot", "ordinal"],
        format_func={
            "one-hot": "One-hot encoding",
            "ordinal": "Ordinal encoding",
        }.get,
        help="How text and category columns are converted to numbers.",
    )
with preprocessing_columns[2]:
    scaling = st.selectbox(
        "Numeric scaling",
        ["none", "standard", "minmax"],
        format_func={
            "none": "None",
            "standard": "Standard scaler",
            "minmax": "Min-max scaler",
        }.get,
        help="Scaling is applied after missing values are handled.",
    )

st.header("5. Train/test split")
split_columns = st.columns(3)
with split_columns[0]:
    test_size = st.slider(
        "Test set size",
        min_value=0.1,
        max_value=0.5,
        value=0.2,
        step=0.05,
    )
with split_columns[1]:
    random_state = st.number_input(
        "Random seed",
        min_value=0,
        max_value=2_147_483_647,
        value=42,
        step=1,
    )
with split_columns[2]:
    stratify = st.checkbox(
        "Stratify split (classification)",
        value=task == "classification",
        disabled=task != "classification",
        help="Keep the class proportions similar in the train and test sets.",
        key=f"stratify_split_{task}",
    )

if task == "regression":
    stratify = False

st.header("6. Model")
model_options = models_for_task(task)
if not model_options:
    st.error("No models are registered for this task type.")
    st.stop()

model_name = st.selectbox(
    "Model",
    options=model_options,
    key=f"model_{task}",
)
hyperparameters = hyperparameters_for(model_name)

run_search = st.checkbox(
    "Run a small grid search (optional)",
    key=f"search_mode_{task}_{model_name}",
    help=(
        "Search several hyperparameter values with cross-validation. "
        f"The maximum is {MAX_SEARCH_COMBINATIONS} combinations."
    ),
)

search_grid = {}
search_combinations = None
if run_search:
    st.caption(
        "Choose one or more values for each hyperparameter. "
        f"Total combinations must be {MAX_SEARCH_COMBINATIONS} or fewer."
    )
    search_grid = render_search_hyperparameters(model_name, hyperparameters)
    search_combinations = count_combinations(search_grid)
    if search_combinations < 1:
        st.error("Select at least one value for every hyperparameter.")
    elif search_combinations > MAX_SEARCH_COMBINATIONS:
        st.error(
            f"This search space has {search_combinations} combinations. "
            f"The free-tier limit is {MAX_SEARCH_COMBINATIONS}. "
            "Reduce the values selected for each hyperparameter."
        )
    else:
        st.caption(
            f"Search space: {search_combinations} combinations "
            f"(limit {MAX_SEARCH_COMBINATIONS})."
        )
else:
    selected_params = render_single_hyperparameters(
        model_name,
        hyperparameters,
    )

st.header("7. Train")
search_is_blocked = run_search and (
    search_combinations is None
    or search_combinations < 1
    or search_combinations > MAX_SEARCH_COMBINATIONS
)
train_clicked = st.button(
    "Train model",
    type="primary",
    disabled=search_is_blocked,
)

if train_clicked:
    started = time.perf_counter()
    try:
        with st.spinner(
            "Running grid search and training..."
            if run_search
            else "Training model..."
        ):
            if run_search:
                result = grid_search_and_evaluate(
                    df,
                    target=target_column,
                    task=task,
                    model_name=model_name,
                    param_grid=search_grid,
                    missing_strategy=missing_strategy,
                    encoding=encoding,
                    scaling=scaling,
                    test_size=test_size,
                    random_state=int(random_state),
                    stratify=stratify,
                )
            else:
                result = train_and_evaluate(
                    df,
                    target=target_column,
                    task=task,
                    model_name=model_name,
                    params=selected_params,
                    missing_strategy=missing_strategy,
                    encoding=encoding,
                    scaling=scaling,
                    test_size=test_size,
                    random_state=int(random_state),
                    stratify=stratify,
                )
        elapsed = time.perf_counter() - started
    except PipelineError as exc:
        st.error(str(exc))
    except Exception:
        st.error(
            "Training failed unexpectedly. "
            "Check the dataset, preprocessing settings, and hyperparameters."
        )
    else:
        st.success(f"Training completed in {elapsed:.2f} seconds.")
        st.caption(
            f"Model: {model_name} | Task: {task} | "
            f"Train rows: {result['train_rows']:,} | "
            f"Test rows: {result['test_rows']:,}"
        )
        if result["rows_dropped"]:
            st.info(
                f"{result['rows_dropped']:,} rows were dropped because the "
                "target or selected features were missing."
            )
        if result["best_params"] is not None:
            st.info(
                "Best cross-validation score: "
                f"{result['best_cv_score']:.4f} "
                f"(search space: {result['search_combinations']} combinations)"
            )
            st.write("Best hyperparameters")
            st.json(result["best_params"])

        st.header("8. Results")
        if task == "classification":
            render_classification_results(result)
        else:
            render_regression_results(result)

        st.header("9. Downloads")
        model_slug = slugify(model_name)
        model_buffer = io.BytesIO()
        joblib.dump(result["model"], model_buffer)

        download_columns = st.columns(2)
        with download_columns[0]:
            st.download_button(
                "Download trained model (.joblib)",
                data=model_buffer.getvalue(),
                file_name=f"{model_slug}.joblib",
                mime="application/octet-stream",
                width="stretch",
            )
        with download_columns[1]:
            st.download_button(
                "Download metrics summary (.csv)",
                data=metrics_frame(result).to_csv(index=False).encode("utf-8"),
                file_name=f"{model_slug}_metrics.csv",
                mime="text/csv",
                width="stretch",
            )
