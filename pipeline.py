import inspect
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_recall_fscore_support,
    precision_score,
    r2_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

import preprocess
from models import CLASSIFICATION, MODEL_REGISTRY, REGRESSION

MAX_UPLOAD_MB = 20
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_TRAIN_ROWS = 50_000
MAX_SEARCH_COMBINATIONS = 50
MIN_CV_FOLDS = 3
MAX_CV_FOLDS = 10
MAX_CV_ROWS_X_FOLDS = 250_000
SCALE_SENSITIVE_MODELS = {
    "SVM",
    "SVR",
    "KNN Classifier",
    "KNN Regressor",
    "Logistic Regression",
}


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

    if (
        "n_jobs" in signature.parameters
        and entry["class"].__name__ != "LogisticRegression"
    ):
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


def _python_scalar(value):
    return value.item() if hasattr(value, "item") else value


def _predict_probabilities(estimator, X_test):
    try:
        if hasattr(estimator, "predict_proba"):
            return np.asarray(estimator.predict_proba(X_test))
    except Exception:
        return None
    return None


def _baseline_metrics(task, y_train, y_test, metrics):
    if task == CLASSIFICATION:
        majority = pd.Series(y_train).value_counts().idxmax()
        baseline_pred = np.full(len(y_test), majority)
        baseline_accuracy = float(accuracy_score(y_test, baseline_pred))
        baseline_f1 = float(
            f1_score(
                y_test,
                baseline_pred,
                average="weighted",
                zero_division=0,
            )
        )
        return {
            "name": "Majority class",
            "label": str(majority),
            "accuracy": baseline_accuracy,
            "f1": baseline_f1,
            "accuracy_uplift": float(metrics["accuracy"] - baseline_accuracy),
            "f1_uplift": float(metrics["f1"] - baseline_f1),
        }

    train_values = np.asarray(y_train, dtype=float)
    baseline_value = float(np.mean(train_values))
    baseline_pred = np.full(len(y_test), baseline_value)
    baseline_r2 = float(r2_score(y_test, baseline_pred))
    baseline_mae = float(mean_absolute_error(y_test, baseline_pred))
    mae_reduction = (
        float((baseline_mae - metrics["mae"]) / baseline_mae * 100)
        if baseline_mae
        else 0.0
    )
    return {
        "name": "Mean predictor",
        "value": baseline_value,
        "r2": baseline_r2,
        "mae": baseline_mae,
        "r2_uplift": float(metrics["r2"] - baseline_r2),
        "mae_reduction_pct": mae_reduction,
    }


def _unwrapped_estimator(estimator):
    model = estimator.named_steps["model"]
    if isinstance(model, TransformedTargetRegressor):
        return model.regressor_ if hasattr(model, "regressor_") else model.regressor
    return model


def _raw_feature_for(output_name, raw_features):
    if output_name in raw_features:
        return output_name
    direct = [
        feature
        for feature in raw_features
        if output_name.startswith(feature + "__")
        or output_name.startswith(feature + "_")
    ]
    if direct:
        return max(direct, key=len)
    stripped = output_name.split("__", 1)[-1]
    if stripped in raw_features:
        return stripped
    candidates = [
        feature
        for feature in raw_features
        if stripped == feature or stripped.startswith(feature + "_")
    ]
    if candidates:
        return max(candidates, key=len)
    return stripped


def _transformed_feature_names(estimator, raw_features):
    try:
        names = estimator.named_steps["preprocessor"].get_feature_names_out()
    except Exception:
        return None
    return [_raw_feature_for(str(name), raw_features) for name in names]


def _builtin_importance(estimator, raw_features):
    model = _unwrapped_estimator(estimator)
    names = _transformed_feature_names(estimator, raw_features)
    if names is None:
        return None

    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float).ravel()
    elif hasattr(model, "coef_"):
        coefficients = np.asarray(model.coef_, dtype=float)
        if coefficients.ndim == 1:
            values = np.abs(coefficients)
        else:
            values = np.abs(coefficients).mean(axis=0)
    else:
        return None

    if values.size != len(names):
        return None

    aggregated = defaultdict(float)
    for name, value in zip(names, values):
        aggregated[name] += float(value)
    frame = pd.DataFrame(
        [
            {"feature": feature, "importance": importance}
            for feature, importance in aggregated.items()
        ]
    ).sort_values("importance", ascending=False)
    frame = frame.reset_index(drop=True)
    return {
        "data": frame,
        "method": "Built-in",
        "complete": True,
        "seconds": 0.0,
    }


def _permutation_importance(
    estimator,
    X_test,
    y_test,
    n_repeats=3,
    budget_seconds=20.0,
):
    started = time.perf_counter()
    if len(X_test) == 0 or X_test.shape[1] == 0:
        return {
            "data": pd.DataFrame(columns=["feature", "importance"]),
            "method": None,
            "complete": False,
            "seconds": 0.0,
        }

    sample = X_test.sample(n=min(500, len(X_test)), random_state=42)
    if hasattr(y_test, "loc"):
        y_sample = y_test.loc[sample.index]
    else:
        positions = [X_test.index.get_loc(index) for index in sample.index]
        y_sample = np.asarray(y_test)[positions]

    try:
        baseline_score = float(estimator.score(sample, y_sample))
    except Exception:
        return {
            "data": pd.DataFrame(columns=["feature", "importance"]),
            "method": None,
            "complete": False,
            "seconds": time.perf_counter() - started,
        }

    rows = []
    complete = True
    rng = np.random.default_rng(42)
    columns = list(sample.columns)
    for position, feature in enumerate(columns):
        if time.perf_counter() - started >= budget_seconds:
            complete = False
            break
        drops = []
        original = sample[feature].to_numpy(copy=True)
        for _ in range(n_repeats):
            shuffled = sample.copy()
            shuffled[feature] = rng.permutation(original)
            try:
                score = float(estimator.score(shuffled, y_sample))
            except Exception:
                score = np.nan
            if np.isfinite(score):
                drops.append(baseline_score - score)
        if drops:
            rows.append(
                {
                    "feature": feature,
                    "importance": max(float(np.mean(drops)), 0.0),
                }
            )
        if (
            time.perf_counter() - started >= budget_seconds
            and position < len(columns) - 1
        ):
            complete = False
            break

    frame = pd.DataFrame(rows, columns=["feature", "importance"])
    if not frame.empty:
        frame = frame.sort_values("importance", ascending=False).reset_index(drop=True)
    return {
        "data": frame,
        "method": "Permutation",
        "complete": complete,
        "seconds": time.perf_counter() - started,
    }


def _feature_importance(estimator, X_test, y_test):
    raw_features = list(X_test.columns)
    try:
        builtin = _builtin_importance(estimator, raw_features)
        if builtin is not None:
            return builtin
    except Exception:
        pass
    try:
        return _permutation_importance(estimator, X_test, y_test)
    except Exception:
        return {
            "data": pd.DataFrame(columns=["feature", "importance"]),
            "method": None,
            "complete": False,
            "seconds": 0.0,
        }


def _evaluate(estimator, task, X_test, y_test, y_train=None):
    if y_train is None:
        y_train = y_test
    y_pred = estimator.predict(X_test)
    model = estimator.named_steps["model"]
    importance = _feature_importance(estimator, X_test, y_test)
    importance_extra = {
        "feature_importance": importance["data"],
        "feature_importance_method": importance["method"],
        "feature_importance_complete": importance["complete"],
        "feature_importance_seconds": float(importance["seconds"]),
    }

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
        per_precision, per_recall, per_f1, per_support = (
            precision_recall_fscore_support(
                y_test,
                y_pred,
                labels=labels,
                average=None,
                zero_division=0,
            )
        )
        per_class = pd.DataFrame(
            {
                "class": [str(label) for label in labels],
                "precision": per_precision.astype(float),
                "recall": per_recall.astype(float),
                "f1": per_f1.astype(float),
                "support": per_support.astype(int),
            }
        )
        baseline = _baseline_metrics(task, y_train, y_test, metrics)
        roc = None
        precision_recall = None
        probabilities = _predict_probabilities(estimator, X_test)
        positive_label = labels[1] if len(labels) == 2 else None

        if len(labels) == 2:
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
                        pos_label=_python_scalar(positive_label),
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
                try:
                    pr_precision, pr_recall, _ = precision_recall_curve(
                        np.asarray(y_test),
                        scores,
                        pos_label=_python_scalar(positive_label),
                    )
                    average_precision = float(
                        average_precision_score(
                            np.asarray(y_test),
                            scores,
                            pos_label=_python_scalar(positive_label),
                        )
                    )
                    precision_recall = {
                        "precision": pr_precision,
                        "recall": pr_recall,
                        "average_precision": average_precision,
                        "baseline": float(
                            np.mean(np.asarray(y_test) == positive_label)
                        ),
                    }
                except (TypeError, ValueError):
                    precision_recall = None

        return metrics, {
            "confusion_matrix": matrix,
            "classes": labels,
            "positive_class": positive_label,
            "per_class": per_class,
            "baseline": baseline,
            "roc": roc,
            "precision_recall": precision_recall,
            "y_proba": probabilities,
            "y_test": np.asarray(y_test),
            "y_pred": np.asarray(y_pred),
            **importance_extra,
        }

    metrics = {
        "r2": float(r2_score(y_test, y_pred)),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
    }
    baseline = _baseline_metrics(task, y_train, y_test, metrics)
    actual = np.asarray(y_test, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    residuals = actual - predicted
    absolute_error = np.abs(residuals)
    indices = list(y_test.index) if hasattr(y_test, "index") else list(range(len(actual)))
    largest_errors = (
        pd.DataFrame(
            {
                "row": indices,
                "actual": actual,
                "predicted": predicted,
                "absolute_error": absolute_error,
            }
        )
        .sort_values("absolute_error", ascending=False)
        .head(5)
        .reset_index(drop=True)
    )
    return metrics, {
        "baseline": baseline,
        "y_test": actual,
        "y_pred": predicted,
        "residuals": residuals,
        "largest_errors": largest_errors,
        **importance_extra,
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
        metrics, extra = _evaluate(
            estimator,
            task,
            X_test,
            y_test,
            y_train=y_train,
        )
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


def make_recipe_pipeline(model_name, params, recipe, random_state):
    estimator = build_estimator(
        model_name,
        params=params,
        random_state=random_state,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocess.build_preprocessor(recipe)),
            ("model", estimator),
        ]
    )


def _wizard_result(
    estimator,
    task,
    model_name,
    params,
    X_train,
    y_train,
    X_test,
    y_test,
    dropped_rows=0,
    split_meta=None,
):
    try:
        estimator.fit(X_train, y_train)
        metrics, extra = _evaluate(
            estimator,
            task,
            X_test,
            y_test,
            y_train=y_train,
        )
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
        "rows_dropped": int(dropped_rows),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "best_params": None,
        "best_cv_score": None,
        "search_combinations": None,
        "split": split_meta or {},
    }
    result.update(extra)
    return result


def train_with_recipe(
    X_train,
    y_train,
    X_test,
    y_test,
    task,
    model_name,
    params,
    recipe,
    random_state,
    dropped_rows=0,
    split_meta=None,
):
    estimator = make_recipe_pipeline(
        model_name,
        params=params,
        recipe=recipe,
        random_state=random_state,
    )
    return _wizard_result(
        estimator,
        task,
        model_name,
        params,
        X_train,
        y_train,
        X_test,
        y_test,
        dropped_rows=dropped_rows,
        split_meta=split_meta,
    )


def grid_search_with_recipe(
    X_train,
    y_train,
    X_test,
    y_test,
    task,
    model_name,
    param_grid,
    recipe,
    random_state,
    dropped_rows=0,
    split_meta=None,
):
    combinations = validate_search_space(param_grid)
    if task == CLASSIFICATION:
        folds = min(3, int(pd.Series(y_train).value_counts().min()))
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
            raise PipelineError("Grid search needs at least two training rows.")
        cross_validation = KFold(
            n_splits=folds,
            shuffle=True,
            random_state=random_state,
        )
        scoring = "r2"

    base_pipeline = make_recipe_pipeline(
        model_name,
        params={},
        recipe=recipe,
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
    result = _wizard_result(
        search.best_estimator_,
        task,
        model_name,
        best_params,
        X_train,
        y_train,
        X_test,
        y_test,
        dropped_rows=dropped_rows,
        split_meta=split_meta,
    )
    result["best_params"] = best_params
    result["best_cv_score"] = float(search.best_score_)
    result["search_combinations"] = combinations
    return result


def max_cv_folds(n_rows, y, task, requested):
    folds = max(int(requested), MIN_CV_FOLDS)
    capped = min(folds, MAX_CV_FOLDS, max(int(n_rows), 2))
    if task == CLASSIFICATION:
        counts = pd.Series(y).value_counts()
        if not counts.empty:
            capped = min(capped, int(counts.min()))
    if capped < 2:
        raise PipelineError(
            "Cross-validation needs at least two rows per fold, "
            "and at least two rows per class for classification."
        )
    row_cap = max(2, MAX_CV_ROWS_X_FOLDS // max(len(y), 1))
    return max(2, min(capped, row_cap))


def cross_validate_recipe(
    X_train,
    y_train,
    task,
    model_name,
    params,
    recipe,
    random_state,
    folds,
):
    used_folds = max_cv_folds(len(y_train), y_train, task, folds)
    if task == CLASSIFICATION:
        cross_validation = StratifiedKFold(
            n_splits=used_folds,
            shuffle=True,
            random_state=random_state,
        )
        scoring = "accuracy"
        metric = "Accuracy"
    else:
        cross_validation = KFold(
            n_splits=used_folds,
            shuffle=True,
            random_state=random_state,
        )
        scoring = "r2"
        metric = "R2"
    estimator = make_recipe_pipeline(
        model_name,
        params=params,
        recipe=recipe,
        random_state=random_state,
    )
    try:
        scores = cross_validate(
            estimator,
            X_train,
            y_train,
            scoring=scoring,
            cv=cross_validation,
            n_jobs=1,
            error_score="raise",
        )["test_score"]
    except PipelineError:
        raise
    except Exception as exc:
        raise PipelineError(
            "Cross-validation failed. Try fewer folds or a different model."
        ) from exc
    values = np.asarray(scores, dtype=float)
    return {
        "scores": values,
        "folds": list(range(1, len(values) + 1)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "metric": metric,
        "requested_folds": int(folds),
        "used_folds": int(used_folds),
        "capped": int(folds) != int(used_folds),
        "rows": int(len(y_train)),
    }


def has_numeric_scaling(recipe):
    for item in recipe.get("columns", []):
        if item.get("kind") == preprocess.NUMERIC and item.get("scaling") not in {
            None,
            "none",
        }:
            return True
    return False


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
            y_train=y_train,
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


def _format_narrative_value(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if abs(value) >= 1_000:
        return f"{value:,.0f}"
    if abs(value) >= 10:
        return f"{value:,.1f}"
    return f"{value:,.2f}"


def result_narrative(result):
    metrics = result["metrics"]
    baseline = result.get("baseline", {})
    model_name = result.get("model_name", "model")
    test_rows = result.get("test_rows", 0)
    importance = result.get("feature_importance")
    top_features = []
    if isinstance(importance, pd.DataFrame) and not importance.empty:
        top_features = importance.head(2)["feature"].tolist()
    driver_note = ""
    if not top_features:
        driver_note = (
            "Feature importance could not be estimated within the 20-second budget, "
            "so driver interpretation is limited."
        )
    elif not result.get("feature_importance_complete", False):
        driver_note = (
            "The driver ranking is partial because the 20-second importance budget "
            "was reached."
        )

    if result["task"] == CLASSIFICATION:
        accuracy = metrics["accuracy"]
        f1 = metrics["f1"]
        sentences = [
            f"The {model_name} model reached {accuracy:.1%} accuracy and "
            f"{f1:.1%} weighted F1 on {test_rows:,} held-out rows."
        ]
        if baseline.get("name") == "Majority class":
            uplift = float(baseline.get("accuracy_uplift", 0.0)) * 100
            if uplift >= 0:
                sentences.append(
                    f"That is {uplift:.1f} percentage points above the "
                    f"{baseline['accuracy']:.1%} majority-class baseline."
                )
            else:
                sentences.append(
                    f"That is {abs(uplift):.1f} percentage points below the "
                    f"{baseline['accuracy']:.1%} majority-class baseline, so treat "
                    "the result with caution."
                )
        if "roc_auc" in metrics:
            roc_auc = metrics["roc_auc"]
            if roc_auc >= 0.85:
                strength = "strong"
            elif roc_auc >= 0.75:
                strength = "useful"
            elif roc_auc >= 0.65:
                strength = "modest"
            else:
                strength = "weak"
            sentences.append(
                f"ROC AUC was {roc_auc:.3f}, indicating {strength} separation "
                "between the classes."
            )
        positive = result.get("positive_class")
        per_class = result.get("per_class")
        if positive is not None and isinstance(per_class, pd.DataFrame):
            row = per_class.loc[per_class["class"] == str(positive)]
            if not row.empty:
                row = row.iloc[0]
                sentences.append(
                    f"For class {positive}, the model catches "
                    f"{row['recall']:.1%} of actual cases; when it predicts that "
                    f"class, it is right {row['precision']:.1%} of the time."
                )
        if top_features:
            drivers = " and ".join(f"`{feature}`" for feature in top_features)
            sentences.append(f"The strongest drivers were {drivers}.")
            if driver_note:
                sentences.append(driver_note)
        elif driver_note:
            sentences.append(driver_note)
        return sentences[:5]

    r2 = metrics["r2"]
    sentences = [
        f"The {model_name} model reached R² = {r2:.3f} on {test_rows:,} test rows, "
        f"with MAE {_format_narrative_value(metrics['mae'])} and RMSE "
        f"{_format_narrative_value(metrics['rmse'])}."
    ]
    if baseline.get("name") == "Mean predictor":
        mae_reduction = float(baseline.get("mae_reduction_pct", 0.0))
        direction = "lowers" if mae_reduction >= 0 else "raises"
        sentences.append(
            f"Against the mean-predictor baseline (R² {baseline['r2']:.3f}, MAE "
            f"{_format_narrative_value(baseline['mae'])}), this model improves R² "
            f"by {baseline['r2_uplift']:+.3f} and {direction} MAE by "
            f"{abs(mae_reduction):.1f}%."
        )
    largest = result.get("largest_errors")
    if isinstance(largest, pd.DataFrame) and not largest.empty:
        row = largest.iloc[0]
        row_label = row["row"]
        if isinstance(row_label, (float, np.floating)) and float(row_label).is_integer():
            row_label = int(row_label)
        sentences.append(
            f"The largest miss was row {row_label}: actual "
            f"{_format_narrative_value(row['actual'])}, predicted "
            f"{_format_narrative_value(row['predicted'])}."
        )
    residuals = result.get("residuals")
    if residuals is not None and len(residuals):
        mean_residual = float(np.mean(residuals))
        if abs(mean_residual) < 1e-9:
            sentences.append("Residuals are centred on zero.")
        else:
            lean = "low" if mean_residual > 0 else "high"
            sentences.append(
                f"Residuals average {_format_narrative_value(mean_residual)}, "
                f"so predictions lean slightly {lean} overall."
            )
    if top_features:
        drivers = " and ".join(f"`{feature}`" for feature in top_features)
        sentences.append(f"The strongest drivers were {drivers}.")
        if driver_note:
            sentences.append(driver_note)
    elif driver_note:
        sentences.append(driver_note)
    return sentences[:5]
