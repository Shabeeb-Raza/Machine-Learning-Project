import copy
import json
from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (
    MinMaxScaler,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)

NUMERIC = "numeric"
CATEGORICAL = "categorical"
CLASSIFICATION = "classification"
REGRESSION = "regression"

NUMERIC_MISSING = ["median", "mean", "most_frequent", "constant", "drop", "none"]
CATEGORICAL_MISSING = ["most_frequent", "constant", "drop", "none"]
OUTLIER_ACTIONS = ["none", "clip", "remove", "flag"]
ENCODINGS = ["one-hot", "ordinal", "frequency", "none"]
SCALINGS = ["none", "standard", "minmax", "robust", "yeo-johnson"]
RARE_ACTIONS = ["none", "group"]
MISSING_SENTINEL = "__missing__"
OTHER_SENTINEL = "__other__"
DEFAULT_ONEHOT_CATEGORIES = 20
DEFAULT_RARE_FREQUENCY = 0.01
MAX_RECIPE_COLUMNS = 60
MAX_PREVIEW_ROWS = 5_000


class PreprocessError(Exception):
    pass


def infer_column_kind(series):
    if pd.api.types.is_bool_dtype(series):
        return CATEGORICAL
    if pd.api.types.is_numeric_dtype(series):
        return NUMERIC
    return CATEGORICAL


def _value_key(value):
    if value is None:
        return MISSING_SENTINEL
    try:
        if pd.isna(value):
            return MISSING_SENTINEL
    except (TypeError, ValueError):
        pass
    return str(value)


def _json_value(value):
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        value = value.item()
    return value


def default_column_recipe(name, series):
    kind = infer_column_kind(series)
    return {
        "name": name,
        "kind": kind,
        "missing": "median" if kind == NUMERIC else "most_frequent",
        "constant": 0.0 if kind == NUMERIC else MISSING_SENTINEL,
        "outlier": "none",
        "encoding": "none" if kind == NUMERIC else "one-hot",
        "scaling": "standard" if kind == NUMERIC else "none",
        "rare": "none",
        "category_order": "",
        "included": True,
    }


def default_recipe(df, features):
    return {
        "version": 1,
        "onehot_max_categories": DEFAULT_ONEHOT_CATEGORIES,
        "rare_min_frequency": DEFAULT_RARE_FREQUENCY,
        "variance_threshold": 0.0,
        "correlation_threshold": 1.0,
        "columns": [default_column_recipe(name, df[name]) for name in features],
    }


def sync_recipe(recipe, df, features):
    existing = {item["name"]: item for item in recipe.get("columns", [])}
    synced = copy.deepcopy(recipe)
    columns = []
    for name in features:
        if name not in df.columns:
            continue
        if name in existing:
            item = copy.deepcopy(existing[name])
            item["kind"] = infer_column_kind(df[name])
        else:
            item = default_column_recipe(name, df[name])
        item["included"] = True
        columns.append(item)
    synced["columns"] = columns
    synced.setdefault("version", 1)
    synced.setdefault("onehot_max_categories", DEFAULT_ONEHOT_CATEGORIES)
    synced.setdefault("rare_min_frequency", DEFAULT_RARE_FREQUENCY)
    synced.setdefault("variance_threshold", 0.0)
    synced.setdefault("correlation_threshold", 1.0)
    return synced


def recipe_to_json(recipe):
    return json.dumps(recipe, indent=2, sort_keys=False)


def validate_recipe(df, features, recipe):
    errors = []
    warnings = []
    known = set(df.columns)
    names = []
    for item in recipe.get("columns", []):
        name = item.get("name")
        if name not in known:
            errors.append(f"`{name}` is not in the dataset.")
            continue
        if name not in features:
            continue
        names.append(name)
        kind = item.get("kind", infer_column_kind(df[name]))
        if item.get("missing") not in NUMERIC_MISSING + CATEGORICAL_MISSING:
            errors.append(f"`{name}` has an unsupported missing-value strategy.")
        if item.get("missing") == "constant":
            value = item.get("constant")
            if value is None or (
                isinstance(value, str) and not value.strip()
            ):
                errors.append(f"`{name}` needs a constant fill value.")
        if kind == NUMERIC and item.get("outlier") not in OUTLIER_ACTIONS:
            errors.append(f"`{name}` has an unsupported outlier action.")
        if kind == CATEGORICAL and item.get("outlier") not in {None, "none"}:
            warnings.append(f"`{name}` is categorical, so its outlier setting is ignored.")
        if item.get("encoding") not in ENCODINGS:
            errors.append(f"`{name}` has an unsupported encoding.")
        if item.get("scaling") not in SCALINGS:
            errors.append(f"`{name}` has an unsupported scaling option.")
        if kind == CATEGORICAL and item.get("scaling") not in {None, "none"}:
            warnings.append(f"`{name}` is categorical, so its scaling setting is ignored.")
        if item.get("missing") == "none" and df[name].isna().any():
            warnings.append(
                f"`{name}` keeps missing values under the current setting; "
                "most models cannot train on NaNs."
            )
        if kind == CATEGORICAL and item.get("encoding") == "none":
            warnings.append(
                f"`{name}` is text; choose one-hot, ordinal, or frequency encoding "
                "for reliable training."
            )
        if kind == NUMERIC:
            values = pd.to_numeric(df[name], errors="coerce")
            if values.notna().sum() == 0:
                errors.append(f"`{name}` is marked numeric but has no numeric values.")
            elif values.nunique(dropna=True) < 2 and item.get("scaling") == "yeo-johnson":
                warnings.append(
                    f"`{name}` is constant, so the Yeo-Johnson transform is unnecessary."
                )

    missing = [name for name in features if name not in names]
    if missing:
        errors.append(
            "The recipe is missing settings for: " + ", ".join(f"`{name}`" for name in missing)
        )
    if not features:
        errors.append("Select at least one feature.")
    if len(features) > MAX_RECIPE_COLUMNS:
        errors.append(
            f"Select at most {MAX_RECIPE_COLUMNS} features so the recipe table stays usable."
        )
    if not 1 <= int(recipe.get("onehot_max_categories", DEFAULT_ONEHOT_CATEGORIES)) <= 100:
        errors.append("One-hot maximum categories must be between 1 and 100.")
    if not 0.0 <= float(recipe.get("rare_min_frequency", DEFAULT_RARE_FREQUENCY)) <= 0.5:
        errors.append("Rare-category frequency must be between 0% and 50%.")
    if not 0.0 <= float(recipe.get("variance_threshold", 0.0)) <= 1.0:
        errors.append("Variance threshold must be between 0 and 1.")
    if not 0.5 <= float(recipe.get("correlation_threshold", 1.0)) <= 1.0:
        errors.append("Correlation threshold must be between 0.5 and 1.0.")
    return errors, warnings


def _mode(series):
    values = series.dropna()
    if values.empty:
        return MISSING_SENTINEL
    modes = values.mode(dropna=True)
    if modes.empty:
        return values.iloc[0]
    return modes.iloc[0]


def _outlier_bounds(values):
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) < 4:
        return None, None
    q1 = float(clean.quantile(0.25))
    q3 = float(clean.quantile(0.75))
    spread = q3 - q1
    if spread <= 0:
        return q1, q3
    return q1 - 1.5 * spread, q3 + 1.5 * spread


def outlier_summary(df, column):
    values = pd.to_numeric(df[column], errors="coerce")
    lower, upper = _outlier_bounds(values)
    if lower is None:
        return {
            "column": column,
            "lower": np.nan,
            "upper": np.nan,
            "outliers": 0,
            "outlier_pct": 0.0,
        }
    mask = (values < lower) | (values > upper)
    count = int(mask.sum())
    return {
        "column": column,
        "lower": float(lower),
        "upper": float(upper),
        "outliers": count,
        "outlier_pct": float(count / max(len(values.dropna()), 1) * 100),
    }


def clean_rows(df, features, target, task, recipe):
    if target not in df.columns:
        raise PreprocessError("Select a target column that exists in the dataset.")
    if not features:
        raise PreprocessError("Select at least one feature.")
    if target in features:
        raise PreprocessError("The target cannot also be a feature.")
    if df.empty:
        raise PreprocessError("The dataset has no rows.")
    columns = list(OrderedDict.fromkeys(features + [target]))
    work = df.loc[:, columns].copy()
    input_rows = len(work)
    dropped_target = int(work[target].isna().sum())
    work = work.loc[work[target].notna()]
    if task == REGRESSION:
        numeric_target = pd.to_numeric(work[target], errors="coerce")
        invalid_target = int(numeric_target.isna().sum())
        work = work.loc[~numeric_target.isna()].copy()
        work[target] = numeric_target.loc[~numeric_target.isna()].astype(float)
        dropped_target += invalid_target
    work = work.reset_index(drop=True)

    missing_drop = 0
    for item in recipe.get("columns", []):
        if item.get("name") in features and item.get("missing") == "drop":
            before = len(work)
            work = work.loc[work[item["name"]].notna()]
            missing_drop += before - len(work)

    outlier_drop = 0
    for item in recipe.get("columns", []):
        name = item.get("name")
        if name not in features or item.get("kind") != NUMERIC:
            continue
        if item.get("outlier") != "remove":
            continue
        lower, upper = _outlier_bounds(work[name])
        if lower is None:
            continue
        before = len(work)
        values = pd.to_numeric(work[name], errors="coerce")
        keep = values.isna() | ((values >= lower) & (values <= upper))
        work = work.loc[keep]
        outlier_drop += before - len(work)

    work = work.reset_index(drop=True)
    if work.empty:
        raise PreprocessError(
            "No rows remain after target and row-level cleaning. "
            "Review missing-value drops and outlier removal."
        )
    if work.shape[1] < 2:
        raise PreprocessError("The dataset needs at least one feature column.")
    summary = {
        "input_rows": int(input_rows),
        "rows": int(len(work)),
        "dropped_target": int(dropped_target),
        "dropped_missing": int(missing_drop),
        "dropped_outlier": int(outlier_drop),
        "dropped_total": int(input_rows - len(work)),
    }
    return work, summary


def encode_target(y, task):
    if task == REGRESSION:
        values = pd.to_numeric(pd.Series(y), errors="raise")
        if values.nunique(dropna=True) < 2:
            raise PreprocessError(
                "The regression target has only one unique value. "
                "Choose a different target or task type."
            )
        return values.to_numpy(dtype=float), {
            "task": task,
            "classes": [],
            "mapping": {},
            "lookup": {},
        }

    values = pd.Series(y)
    if values.nunique(dropna=True) < 2:
        raise PreprocessError(
            "Classification needs at least two classes. "
            "Choose a different target or task type."
        )
    try:
        codes, uniques = pd.factorize(values, sort=True)
    except TypeError:
        codes, uniques = pd.factorize(values.astype(str), sort=True)
    classes = [_json_value(value) for value in uniques.tolist()]
    mapping = {_value_key(value): index for index, value in enumerate(uniques)}
    lookup = {index: value for index, value in enumerate(uniques)}
    return np.asarray(codes, dtype=int), {
        "task": task,
        "classes": classes,
        "mapping": mapping,
        "lookup": lookup,
    }


def prepare_xy(df, features, target, task, recipe):
    cleaned, summary = clean_rows(df, features, target, task, recipe)
    X = cleaned.loc[:, features].copy()
    y = cleaned[target]
    encoded, target_meta = encode_target(y, task)
    return X, y, summary, target_meta, encoded


def classify_column(df, column):
    series = df[column]
    non_null = series.dropna()
    unique = int(non_null.nunique(dropna=True))
    kind = infer_column_kind(series)
    flags = []
    if unique <= 1:
        flags.append("Constant")
    if len(non_null) and unique == len(non_null):
        flags.append("Unique per row")
    if kind == CATEGORICAL and unique > 50:
        flags.append("High cardinality")
    return {
        "column": column,
        "dtype": str(series.dtype),
        "kind": kind,
        "missing": int(series.isna().sum()),
        "missing_pct": float(series.isna().mean() * 100),
        "unique": unique,
        "flags": ", ".join(flags) if flags else "",
    }


def profile_columns(df):
    return pd.DataFrame([classify_column(df, column) for column in df.columns])


def _as_text(series, fill):
    values = series.astype("object")
    return values.map(lambda value: _value_key(value) if not pd.isna(value) else fill)


class ColumnRecipeTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        recipe=None,
        variance_threshold=0.0,
        correlation_threshold=1.0,
    ):
        self.recipe = recipe
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        recipe = copy.deepcopy(self.recipe or {})
        specs = [
            copy.deepcopy(item)
            for item in recipe.get("columns", [])
            if item.get("name") in X.columns
        ]
        if not specs:
            raise PreprocessError("No feature columns were available for preprocessing.")
        onehot_cap = int(recipe.get("onehot_max_categories", DEFAULT_ONEHOT_CATEGORIES))
        rare_frequency = float(
            recipe.get("rare_min_frequency", DEFAULT_RARE_FREQUENCY)
        )
        self.feature_names_in_ = np.asarray(list(X.columns), dtype=object)
        self.output_feature_names_ = []
        self.column_states_ = []
        for spec in specs:
            state = self._fit_column(
                X,
                spec,
                onehot_cap=onehot_cap,
                rare_frequency=rare_frequency,
            )
            self.column_states_.append(state)
        matrix, names = self._build_matrix(X, self.column_states_)
        keep = self._keep_mask(matrix, names)
        self.keep_mask_ = keep
        self.n_features_in_ = X.shape[1]
        self.output_feature_names_ = [
            name for name, keep_value in zip(names, keep) if keep_value
        ]
        if not self.output_feature_names_:
            raise PreprocessError(
                "All features were removed by filtering. Lower the variance or "
                "correlation thresholds and try again."
            )
        return self

    def _fit_column(self, X, spec, onehot_cap, rare_frequency):
        name = spec["name"]
        kind = spec.get("kind", infer_column_kind(X[name]))
        missing = spec.get("missing")
        state = {
            "name": name,
            "kind": kind,
            "missing": missing,
            "constant": spec.get("constant"),
            "outlier": spec.get("outlier", "none"),
            "encoding": spec.get("encoding", "none"),
            "scaling": spec.get("scaling", "none"),
            "rare": spec.get("rare", "none"),
            "missing_fill": None,
            "clip_lower": None,
            "clip_upper": None,
            "flag_name": None,
            "categories": [],
            "category_index": {},
            "frequency_map": {},
            "scaler": None,
            "scale_name": "none",
            "output_names": [],
        }
        series = X[name]
        if missing == "drop":
            missing = "most_frequent" if kind == CATEGORICAL else "median"
        if kind == NUMERIC:
            values = pd.to_numeric(series, errors="coerce")
            clean = values.dropna()
            if missing == "mean":
                fill = float(clean.mean()) if len(clean) else 0.0
            elif missing == "most_frequent":
                fill = _mode(values)
                try:
                    fill = float(fill)
                except (TypeError, ValueError):
                    fill = float(clean.iloc[0]) if len(clean) else 0.0
            elif missing == "constant":
                try:
                    fill = float(spec.get("constant", 0.0))
                except (TypeError, ValueError):
                    fill = 0.0
            elif missing == "none":
                fill = None
            else:
                fill = float(clean.median()) if len(clean) else 0.0
            state["missing_fill"] = fill
            if state["outlier"] in {"clip", "flag"}:
                lower, upper = _outlier_bounds(values)
                state["clip_lower"] = lower
                state["clip_upper"] = upper
            if state["outlier"] == "flag":
                state["flag_name"] = f"{name}__outlier"
            if state["scaling"] in {"standard", "minmax", "robust"}:
                filled = self._numeric_values(series, state).reshape(-1, 1)
                finite = np.isfinite(filled).ravel()
                if not finite.any():
                    state["scaler"] = None
                    state["scale_name"] = "none"
                else:
                    scaler = {
                        "standard": StandardScaler,
                        "minmax": MinMaxScaler,
                        "robust": RobustScaler,
                    }[state["scaling"]]()
                    scaler.fit(filled[finite])
                    state["scaler"] = scaler
                    state["scale_name"] = state["scaling"]
            elif state["scaling"] == "yeo-johnson":
                filled = self._numeric_values(series, state)
                finite = np.isfinite(filled)
                if finite.sum() >= 2 and len(np.unique(filled[finite])) > 1:
                    transformer = PowerTransformer(
                        method="yeo-johnson",
                        standardize=True,
                    )
                    transformer.fit(filled[finite].reshape(-1, 1))
                    state["scaler"] = transformer
                    state["scale_name"] = "yeo-johnson"
            state["output_names"] = [name]
            if state["flag_name"]:
                state["output_names"].append(state["flag_name"])
            return state

        fill = state["constant"]
        if missing == "most_frequent":
            fill = _mode(series)
        elif missing == "constant":
            fill = spec.get("constant", MISSING_SENTINEL)
        state["missing_fill"] = _value_key(fill)
        text = _as_text(series, state["missing_fill"])
        counts = text.value_counts(dropna=False)
        rare = state.get("rare") == "group"
        if rare:
            minimum = max(rare_frequency, 1.0 / max(len(text), 1))
            keep_categories = [
                value
                for value, count in counts.items()
                if count / max(len(text), 1) >= minimum
            ]
            keep_set = set(keep_categories)
            text = text.map(
                lambda value: value if value in keep_set else OTHER_SENTINEL
            )
            counts = text.value_counts(dropna=False)
        if state.get("encoding") == "one-hot":
            categories = counts.head(onehot_cap).index.tolist()
        else:
            categories = counts.index.tolist()
        custom_order = [
            value.strip()
            for value in str(spec.get("category_order", "")).split(",")
            if value.strip()
        ]
        if custom_order:
            categories = [
                value for value in custom_order if value in categories
            ] + [
                value for value in categories if value not in custom_order
            ]
            if state.get("encoding") == "one-hot":
                categories = categories[:onehot_cap]
        state["categories"] = list(categories)
        state["category_index"] = {
            category: index for index, category in enumerate(categories)
        }
        state["frequency_map"] = {
            str(category): float(count / max(len(text), 1))
            for category, count in counts.items()
        }
        if state["encoding"] == "one-hot":
            state["output_names"] = [f"{name}__{category}" for category in categories]
        else:
            state["output_names"] = [name]
        return state

    def _numeric_values(self, series, state):
        values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        fill = state.get("missing_fill")
        if fill is not None:
            values = np.where(np.isnan(values), float(fill), values)
        lower = state.get("clip_lower")
        upper = state.get("clip_upper")
        if state.get("outlier") in {"clip", "flag"} and lower is not None:
            values = np.clip(values, lower, upper)
        return values

    def _text_values(self, series, state):
        fill = state.get("missing_fill") or MISSING_SENTINEL
        return _as_text(series, fill).to_numpy(dtype=object)

    def _transform_column(self, series, state):
        name = state["name"]
        if state["kind"] == NUMERIC:
            values = self._numeric_values(series, state)
            columns = []
            if state.get("scaler") is not None:
                transformed = state["scaler"].transform(values.reshape(-1, 1))
                values = transformed.ravel()
            columns.append((name, values))
            if state.get("flag_name"):
                original = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
                lower = state.get("clip_lower")
                upper = state.get("clip_upper")
                if lower is None:
                    flag = np.zeros(len(original), dtype=float)
                else:
                    flag = ((original < lower) | (original > upper)).astype(float)
                columns.append((state["flag_name"], flag))
            return columns

        text = self._text_values(series, state)
        encoding = state.get("encoding")
        if encoding == "one-hot":
            columns = []
            for category, output_name in zip(
                state["categories"],
                state["output_names"],
            ):
                columns.append(
                    (
                        output_name,
                        (text == category).astype(float),
                    )
                )
            return columns
        if encoding == "frequency":
            mapping = state["frequency_map"]
            values = np.asarray(
                [mapping.get(str(value), 0.0) for value in text],
                dtype=float,
            )
            return [(name, values)]
        index = state.get("category_index", {})
        values = np.asarray(
            [float(index.get(str(value), -1)) for value in text],
            dtype=float,
        )
        return [(name, values)]

    def _build_matrix(self, X, states):
        arrays = []
        names = []
        for state in states:
            for name, values in self._transform_column(X[state["name"]], state):
                arrays.append(np.asarray(values, dtype=float))
                names.append(name)
        if not arrays:
            return np.empty((len(X), 0)), []
        return np.column_stack(arrays), names

    def _keep_mask(self, matrix, names):
        if matrix.shape[1] == 0:
            return np.zeros(0, dtype=bool)
        keep = np.ones(matrix.shape[1], dtype=bool)
        variance = np.var(matrix, axis=0)
        variance_threshold = float(self.variance_threshold or 0.0)
        keep &= variance > variance_threshold
        correlation_threshold = float(self.correlation_threshold or 1.0)
        if correlation_threshold < 1.0 and keep.sum() > 1:
            frame = pd.DataFrame(matrix[:, keep], columns=np.asarray(names)[keep])
            correlation = frame.corr().abs().fillna(0.0)
            columns = list(frame.columns)
            remove = set()
            for index, left in enumerate(columns):
                if left in remove:
                    continue
                for right in columns[index + 1 :]:
                    if right in remove:
                        continue
                    if correlation.loc[left, right] >= correlation_threshold:
                        remove.add(right)
            kept_names = np.asarray(names)[keep]
            for position, name in enumerate(kept_names):
                if name in remove:
                    source = np.where(np.asarray(names) == name)[0][0]
                    keep[source] = False
        return keep

    def transform(self, X):
        if not hasattr(self, "column_states_"):
            raise PreprocessError("The preprocessing recipe has not been fitted.")
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=list(self.feature_names_in_))
        matrix, names = self._build_matrix(X, self.column_states_)
        if hasattr(self, "keep_mask_"):
            matrix = matrix[:, self.keep_mask_]
            names = [
                name for name, keep_value in zip(names, self.keep_mask_) if keep_value
            ]
        return matrix

    def get_feature_names_out(self, input_features=None):
        if not hasattr(self, "output_feature_names_"):
            raise PreprocessError("The preprocessing recipe has not been fitted.")
        return np.asarray(self.output_feature_names_, dtype=object)


def build_preprocessor(recipe):
    return ColumnRecipeTransformer(
        recipe=copy.deepcopy(recipe),
        variance_threshold=float(recipe.get("variance_threshold", 0.0)),
        correlation_threshold=float(recipe.get("correlation_threshold", 1.0)),
    )


def preview_transform(df, features, target, task, recipe):
    cleaned, summary = clean_rows(df, features, target, task, recipe)
    sample = cleaned
    if len(cleaned) > MAX_PREVIEW_ROWS:
        sample = cleaned.sample(n=MAX_PREVIEW_ROWS, random_state=42)
    X = sample.loc[:, features]
    transformer = build_preprocessor(recipe)
    matrix = transformer.fit_transform(X)
    names = transformer.get_feature_names_out()
    preview = pd.DataFrame(matrix, columns=names, index=X.index)
    return cleaned, summary, preview, transformer


def recipe_from_editor(frame, base_recipe):
    recipe = copy.deepcopy(base_recipe)
    edited = {str(row["column"]): row for _, row in frame.iterrows()}
    columns = []
    for item in recipe.get("columns", []):
        row = edited.get(item["name"])
        if row is None:
            continue
        updated = copy.deepcopy(item)
        updated["kind"] = row["type"]
        updated["missing"] = row["missing"]
        updated["outlier"] = row["outlier"]
        updated["encoding"] = row["encoding"]
        updated["scaling"] = row["scaling"]
        updated["rare"] = row["rare"]
        category_order = row.get("category_order", updated.get("category_order", ""))
        if category_order is None or (
            isinstance(category_order, float) and np.isnan(category_order)
        ):
            category_order = ""
        updated["category_order"] = str(category_order).strip()
        constant = row.get("constant", updated.get("constant"))
        if updated["kind"] == NUMERIC:
            try:
                constant = float(constant)
            except (TypeError, ValueError):
                pass
        elif constant is None or (
            isinstance(constant, float) and np.isnan(constant)
        ):
            constant = MISSING_SENTINEL
        updated["constant"] = constant
        columns.append(updated)
    recipe["columns"] = columns
    return recipe


def decode_classification_result(result, target_meta):
    if result.get("task") != CLASSIFICATION:
        return result
    lookup = target_meta.get("lookup", {})
    classes = target_meta.get("classes", [])

    def decode(value):
        try:
            key = int(value)
        except (TypeError, ValueError):
            return value
        return lookup.get(key, value)

    result["classes"] = np.asarray([decode(value) for value in result.get("classes", [])])
    if result.get("y_test") is not None:
        result["y_test"] = np.asarray([decode(value) for value in result["y_test"]])
    if result.get("positive_class") is not None:
        result["positive_class"] = decode(result["positive_class"])
    per_class = result.get("per_class")
    if isinstance(per_class, pd.DataFrame) and not per_class.empty:
        per_class = per_class.copy()
        per_class["class"] = per_class["class"].map(
            lambda value: str(decode(value))
        )
        result["per_class"] = per_class
    baseline = result.get("baseline")
    if isinstance(baseline, dict) and baseline.get("label") is not None:
        baseline["label"] = decode(baseline["label"])
    result["target_meta"] = target_meta
    return result
