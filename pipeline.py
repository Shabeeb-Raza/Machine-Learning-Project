import inspect

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

from models import CLASSIFICATION, MODEL_REGISTRY, REGRESSION

MAX_UPLOAD_MB = 20
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_TRAIN_ROWS = 50_000
MAX_SEARCH_COMBINATIONS = 50


class PipelineError(Exception):
    pass


def validate_upload(file_name, size_bytes):
    if not file_name or not file_name.lower().endswith(".csv"):
        raise PipelineError("Please upload a CSV file with a .csv extension.")
    if size_bytes > MAX_UPLOAD_BYTES:
        raise PipelineError(
            f"This file is larger than {MAX_UPLOAD_MB} MB. "
            "Please upload a smaller CSV."
        )


def limit_rows(df):
    if len(df) <= MAX_TRAIN_ROWS:
        return df, False
    sampled = df.sample(n=MAX_TRAIN_ROWS, random_state=42).reset_index(drop=True)
    return sampled, True


def infer_task_type(target):
    values = target.dropna()
    if values.empty:
        raise PipelineError("The selected target column has no values.")
    if not pd.api.types.is_numeric_dtype(values):
        return CLASSIFICATION
    if values.nunique() <= 20:
        return CLASSIFICATION
    return REGRESSION


def prepare_data(df, target, task, missing_strategy):
    if target not in df.columns:
        raise PipelineError("Select a target column that exists in the dataset.")
    if task not in {CLASSIFICATION, REGRESSION}:
        raise PipelineError("Task type must be classification or regression.")
    if df.empty:
        raise PipelineError("The dataset has no rows.")

    work = df.copy()
    y = work[target]

    if task == REGRESSION:
        try:
            y = pd.to_numeric(y, errors="raise")
        except (TypeError, ValueError) as exc:
            raise PipelineError(
                "A regression target must contain numeric values. "
                "Choose classification or select a different target column."
            ) from exc

    target_mask = y.notna()
    work = work.loc[target_mask]
    y = y.loc[target_mask]

    X = work.drop(columns=[target])

    if missing_strategy == "drop":
        feature_mask = X.notna().all(axis=1)
        X = X.loc[feature_mask]
        y = y.loc[feature_mask]

    dropped_rows = len(df) - len(y)

    if X.shape[1] == 0:
        raise PipelineError("The dataset needs at least one feature column.")
    if len(X) < 2:
        raise PipelineError(
            "Fewer than two usable rows remain. "
            "Try a different missing-value strategy or target column."
        )
    if task == CLASSIFICATION and y.nunique() < 2:
        raise PipelineError(
            "Classification needs at least two classes. "
            "Choose a different target column or task type."
        )
    if task == REGRESSION and y.nunique() < 2:
        raise PipelineError(
            "The regression target has only one unique value. "
            "Choose a different target column or task type."
        )

    return X, y, dropped_rows


def build_preprocessor(X, missing_strategy, encoding, scaling):
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [
        column for column in X.columns if column not in numeric_features
    ]

    numeric_imputer = (
        "most_frequent"
        if missing_strategy in {"drop", "most_frequent"}
        else missing_strategy
    )

    transformers = []

    if numeric_features:
        numeric_steps = [
            ("imputer", SimpleImputer(strategy=numeric_imputer)),
        ]
        if scaling == "standard":
            numeric_steps.append(("scaler", StandardScaler()))
        elif scaling == "minmax":
            numeric_steps.append(("scaler", MinMaxScaler()))
        transformers.append(
            ("numeric", Pipeline(steps=numeric_steps), numeric_features)
        )

    if categorical_features:
        if encoding == "ordinal":
            encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )
        else:
            encoder = OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            )
        categorical_steps = [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", encoder),
        ]
        transformers.append(
            ("categorical", Pipeline(steps=categorical_steps), categorical_features)
        )

    if not transformers:
        raise PipelineError("No usable feature columns were found.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_estimator(model_name, params, random_state):
    if model_name not in MODEL_REGISTRY:
        raise PipelineError("The selected model is not in the registry.")

    entry = MODEL_REGISTRY[model_name]
    kwargs = dict(entry.get("fixed_params", {}))
    kwargs.update(params)

    signature = inspect.signature(entry["class"])

    if "n_jobs" in signature.parameters:
        kwargs["n_jobs"] = 1
    if (
        "random_state" in signature.parameters
        and "random_state" not in kwargs
    ):
        kwargs["random_state"] = random_state

    try:
        estimator = entry["class"](**kwargs)
    except TypeError as exc:
        raise PipelineError(
            "The selected hyperparameters are not valid for this model."
        ) from exc

    if entry["task"] == REGRESSION:
        estimator = TransformedTargetRegressor(
            regressor=estimator,
            transformer=StandardScaler(),
        )

    return estimator


def make_pipeline(
    model_name,
    params,
    X_train,
    missing_strategy,
    encoding,
    scaling,
    random_state,
):
    preprocessor = build_preprocessor(
        X_train,
        missing_strategy=missing_strategy,
        encoding=encoding,
        scaling=scaling,
    )
    estimator = build_estimator(
        model_name,
        params=params,
        random_state=random_state,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )


def split_data(X, y, task, test_size, random_state, stratify):
    stratify_values = None
    if stratify and task == CLASSIFICATION:
        class_counts = y.value_counts()
        if class_counts.min() < 2:
            raise PipelineError(
                "A stratified split needs at least two rows in every class."
            )
        stratify_values = y

    try:
        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_values,
        )
    except ValueError as exc:
        raise PipelineError(
            "Could not create the train/test split with these settings. "
            "Try a different test size, random seed, or disable stratification."
        ) from exc


def _positive_scores(estimator, model, X_test, positive_label):
    try:
        if hasattr(estimator, "predict_proba"):
            probabilities = np.asarray(estimator.predict_proba(X_test))
            classes = list(model.classes_)
            if probabilities.ndim == 2 and len(classes) == 2:
                return probabilities[:, classes.index(positive_label)]
    except Exception:
        pass

    try:
        if hasattr(estimator, "decision_function"):
            scores = np.asarray(estimator.decision_function(X_test))
            if scores.ndim == 1:
                return scores.ravel()
            if scores.ndim == 2 and scores.shape[1] == 2:
                classes = list(model.classes_)
                return scores[:, classes.index(positive_label)]
    except Exception:
        pass

    return None


def _evaluate(estimator, task, X_test, y_test):
    y_pred = estimator.predict(X_test)
    model = estimator.named_steps["model"]

    if task == CLASSIFICATION:
        labels = np.asarray(model.classes_)
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(
                precision_score(
                    y_test,
                    y_pred,
                    average="weighted",
                    zero_division=0,
                )
            ),
            "recall": float(
                recall_score(
                    y_test,
                    y_pred,
                    average="weighted",
                    zero_division=0,
                )
            ),
            "f1": float(
                f1_score(
                    y_test,
                    y_pred,
                    average="weighted",
                    zero_division=0,
                )
            ),
        }
        matrix = confusion_matrix(y_test, y_pred, labels=labels)
        roc = None

        if len(labels) == 2:
            positive_label = labels[1]
            scores = _positive_scores(
                estimator,
                model,
                X_test,
                positive_label,
            )
            if scores is not None:
                try:
                    fpr, tpr, _ = roc_curve(
                        np.asarray(y_test),
                        scores,
                        pos_label=positive_label.item()
                        if hasattr(positive_label, "item")
                        else positive_label,
                    )
                    roc_auc = float(auc(fpr, tpr))
                    metrics["roc_auc"] = roc_auc
                    roc = {
                        "fpr": fpr,
                        "tpr": tpr,
                        "auc": roc_auc,
                    }
                except (TypeError, ValueError):
                    roc = None

        return metrics, {
            "confusion_matrix": matrix,
            "classes": labels,
            "roc": roc,
            "y_test": np.asarray(y_test),
            "y_pred": np.asarray(y_pred),
        }

    metrics = {
        "r2": float(r2_score(y_test, y_pred)),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
    }
    return metrics, {
        "y_test": np.asarray(y_test),
        "y_pred": np.asarray(y_pred),
    }


def train_and_evaluate(
    df,
    target,
    task,
    model_name,
    params,
    missing_strategy,
    encoding,
    scaling,
    test_size,
    random_state,
    stratify,
):
    X, y, dropped_rows = prepare_data(
        df,
        target=target,
        task=task,
        missing_strategy=missing_strategy,
    )
    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
        task=task,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    estimator = make_pipeline(
        model_name,
        params=params,
        X_train=X_train,
        missing_strategy=missing_strategy,
        encoding=encoding,
        scaling=scaling,
        random_state=random_state,
    )

    try:
        estimator.fit(X_train, y_train)
        metrics, extra = _evaluate(estimator, task, X_test, y_test)
    except PipelineError:
        raise
    except Exception as exc:
        raise PipelineError(
            "Training failed for this model and data. "
            "Try simpler hyperparameters, a different model, or different preprocessing."
        ) from exc

    result = {
        "model": estimator,
        "model_name": model_name,
        "task": task,
        "params": params,
        "metrics": metrics,
        "rows_dropped": dropped_rows,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "best_params": None,
        "best_cv_score": None,
        "search_combinations": None,
    }
    result.update(extra)
    return result


def count_combinations(param_grid):
    total = 1
    for values in param_grid.values():
        total *= len(values)
    return total


def validate_search_space(param_grid):
    total = count_combinations(param_grid)
    if total < 1:
        raise PipelineError(
            "Select at least one value for every hyperparameter in the search."
        )
    if total > MAX_SEARCH_COMBINATIONS:
        raise PipelineError(
            f"This search space has {total} combinations. "
            f"The free-tier limit is {MAX_SEARCH_COMBINATIONS}. "
            "Reduce the values selected for each hyperparameter."
        )
    return total


def grid_search_and_evaluate(
    df,
    target,
    task,
    model_name,
    param_grid,
    missing_strategy,
    encoding,
    scaling,
    test_size,
    random_state,
    stratify,
):
    combinations = validate_search_space(param_grid)

    X, y, dropped_rows = prepare_data(
        df,
        target=target,
        task=task,
        missing_strategy=missing_strategy,
    )
    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
        task=task,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    if task == CLASSIFICATION:
        folds = min(3, int(y_train.value_counts().min()))
        if folds < 2:
            raise PipelineError(
                "Grid search needs at least two rows per class in the training set."
            )
        cross_validation = StratifiedKFold(
            n_splits=folds,
            shuffle=True,
            random_state=random_state,
        )
        scoring = "accuracy"
    else:
        folds = min(3, len(y_train))
        if folds < 2:
            raise PipelineError(
                "Grid search needs at least two training rows."
            )
        cross_validation = KFold(
            n_splits=folds,
            shuffle=True,
            random_state=random_state,
        )
        scoring = "r2"

    base_pipeline = make_pipeline(
        model_name,
        params={},
        X_train=X_train,
        missing_strategy=missing_strategy,
        encoding=encoding,
        scaling=scaling,
        random_state=random_state,
    )
    param_prefix = "model__regressor__" if task == REGRESSION else "model__"
    search_grid = {
        f"{param_prefix}{name}": list(values)
        for name, values in param_grid.items()
    }

    search = GridSearchCV(
        estimator=base_pipeline,
        param_grid=search_grid,
        scoring=scoring,
        cv=cross_validation,
        n_jobs=1,
        refit=True,
        error_score="raise",
    )

    try:
        search.fit(X_train, y_train)
        metrics, extra = _evaluate(
            search.best_estimator_,
            task,
            X_test,
            y_test,
        )
    except PipelineError:
        raise
    except Exception as exc:
        raise PipelineError(
            "Grid search or training failed. "
            "Reduce the search space, choose simpler hyperparameters, or try another model."
        ) from exc

    best_params = {
        name.replace(param_prefix, "", 1): value
        for name, value in search.best_params_.items()
    }
    result = {
        "model": search.best_estimator_,
        "model_name": model_name,
        "task": task,
        "params": best_params,
        "metrics": metrics,
        "rows_dropped": dropped_rows,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "best_params": best_params,
        "best_cv_score": float(search.best_score_),
        "search_combinations": combinations,
    }
    result.update(extra)
    return result
