import math

import numpy as np
import pandas as pd

MAX_PLOT_ROWS = 5_000
MAX_HISTOGRAMS = 12
MAX_GROUPED_NUMERIC = 6
MAX_CATEGORICAL_CHARTS = 6
MAX_CORRELATION_COLUMNS = 15
MAX_TOP_CATEGORIES = 10
MAX_ASSOCIATION_CATEGORIES = 20

CLASSIFICATION = "classification"
REGRESSION = "regression"


def _sample_frame(frame, n=MAX_PLOT_ROWS):
    if len(frame) <= n:
        return frame
    return frame.sample(n=n, random_state=42)


def _numeric_columns(frame):
    return frame.select_dtypes(include=["number"]).columns.tolist()


def _categorical_columns(frame):
    numeric = set(_numeric_columns(frame))
    return [column for column in frame.columns if column not in numeric]


def _factorize_target(series):
    values = series.dropna()
    codes, uniques = pd.factorize(values, sort=True)
    encoded = pd.Series(np.nan, index=series.index, dtype=float)
    encoded.loc[values.index] = codes.astype(float)
    return encoded, [value for value in uniques]


def _target_numeric(series, task):
    if task == REGRESSION:
        values = pd.to_numeric(series, errors="coerce")
        return values, []
    return _factorize_target(series)


def _clean_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def _format_number(value, decimals=1):
    value = _clean_float(value)
    if value is None:
        return "n/a"
    if abs(value) >= 10_000:
        return f"{value:,.0f}"
    if abs(value) >= 1_000:
        return f"{value:,.0f}"
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.{decimals}f}"
    return f"{value:.{max(decimals, 2)}f}"


def _format_pct(value, decimals=1):
    value = _clean_float(value)
    if value is None:
        return "n/a"
    return f"{value:.{decimals}f}%"


def _format_label(value):
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _format_edge(value):
    value = _clean_float(value)
    if value is None:
        return "n/a"
    value = round(value, 2)
    if abs(value) < 1e-9:
        return "0"
    if float(value).is_integer():
        return str(int(value))
    return _format_number(value)


def _format_interval(interval):
    try:
        return f"{_format_edge(interval.left)} to {_format_edge(interval.right)}"
    except AttributeError:
        return str(interval)


def _strength_word(value):
    value = _clean_float(value) or 0.0
    if value >= 0.5:
        return "strong"
    if value >= 0.3:
        return "moderate"
    if value >= 0.15:
        return "modest"
    return "weak"


def _safe_corr(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 3:
        return None
    left = left[valid]
    right = right[valid]
    if np.std(left) == 0 or np.std(right) == 0:
        return None
    value = _clean_float(np.corrcoef(left, right)[0, 1])
    return value


def _eta_squared(values, codes):
    frame = pd.DataFrame({"value": values, "code": codes}).dropna()
    if len(frame) < 3 or frame["code"].nunique() < 2:
        return 0.0
    overall = frame["value"].mean()
    total = ((frame["value"] - overall) ** 2).sum()
    if total <= 0:
        return 0.0
    between = 0.0
    for _, group in frame.groupby("code")["value"]:
        between += len(group) * (group.mean() - overall) ** 2
    return float(np.clip(between / total, 0.0, 1.0))


def _numeric_relationship(series, codes, task, binary):
    frame = pd.DataFrame({"value": series, "code": codes}).dropna()
    if len(frame) < 3:
        return {"strength": 0.0, "direction": None, "metric": "n/a"}
    if task == REGRESSION or binary:
        correlation = _safe_corr(frame["value"], frame["code"])
        if correlation is None:
            return {"strength": 0.0, "direction": None, "metric": "|r|"}
        return {
            "strength": abs(correlation),
            "direction": correlation,
            "metric": "|r|",
        }
    strength = _eta_squared(frame["value"], frame["code"])
    return {"strength": strength, "direction": None, "metric": "eta²"}


def _cramers_v(series, codes):
    frame = pd.DataFrame({"value": series, "code": codes}).dropna()
    if len(frame) < 3:
        return 0.0
    counts = frame["value"].value_counts()
    if len(counts) > MAX_ASSOCIATION_CATEGORIES:
        keep = set(counts.head(MAX_ASSOCIATION_CATEGORIES).index)
        frame["value"] = frame["value"].where(frame["value"].isin(keep), "__other__")
    observed = pd.crosstab(frame["value"], frame["code"]).to_numpy(dtype=float)
    if observed.shape[0] < 2 or observed.shape[1] < 2:
        return 0.0
    total = observed.sum()
    if total <= 0:
        return 0.0
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / total
    mask = expected > 0
    chi_square = float((((observed - expected) ** 2)[mask] / expected[mask]).sum())
    denominator = total * (min(observed.shape) - 1)
    if denominator <= 0:
        return 0.0
    return float(np.clip(math.sqrt(chi_square / denominator), 0.0, 1.0))


def _correlation_ratio(series, values):
    frame = pd.DataFrame({"value": series, "target": values}).dropna()
    if len(frame) < 3:
        return 0.0
    counts = frame["value"].value_counts()
    if len(counts) > MAX_ASSOCIATION_CATEGORIES:
        keep = set(counts.head(MAX_ASSOCIATION_CATEGORIES).index)
        frame["value"] = frame["value"].where(frame["value"].isin(keep), "__other__")
    overall = frame["target"].mean()
    total = ((frame["target"] - overall) ** 2).sum()
    if total <= 0:
        return 0.0
    between = 0.0
    for _, group in frame.groupby("value")["target"]:
        between += len(group) * (group.mean() - overall) ** 2
    return float(np.clip(between / total, 0.0, 1.0))


def _categorical_relationship(series, codes, task):
    if task == REGRESSION:
        return {
            "strength": _correlation_ratio(series, codes),
            "direction": None,
            "metric": "correlation ratio",
        }
    return {
        "strength": _cramers_v(series, codes),
        "direction": None,
        "metric": "Cramér's V",
    }


def _category_table(series, target_values, task, positive_code):
    frame = pd.DataFrame({"category": series, "target": target_values}).dropna()
    if frame.empty:
        return pd.DataFrame(
            columns=["category", "count", "share_pct", "target_value"]
        )
    counts = frame["category"].value_counts().head(MAX_TOP_CATEGORIES)
    rows = []
    total = len(frame)
    for category, count in counts.items():
        group = frame.loc[frame["category"] == category, "target"]
        if task == CLASSIFICATION:
            target_value = float((group == positive_code).mean() * 100)
        else:
            target_value = float(group.mean())
        rows.append(
            {
                "category": str(category),
                "count": int(count),
                "share_pct": float(count / total * 100),
                "target_value": target_value,
            }
        )
    return pd.DataFrame(rows)


def _numeric_groups(series, codes, task, labels):
    if task == REGRESSION:
        frame = pd.DataFrame({"x": series, "y": codes}).dropna()
        if frame.empty:
            return {"x": np.array([]), "y": np.array([])}
        return {
            "x": frame["x"].to_numpy(dtype=float),
            "y": frame["y"].to_numpy(dtype=float),
        }

    frame = pd.DataFrame({"value": series, "code": codes}).dropna()
    if frame.empty:
        return {"groups": []}
    frequencies = frame["code"].value_counts().head(8)
    groups = []
    for code in frequencies.index:
        values = frame.loc[frame["code"] == code, "value"].to_numpy(dtype=float)
        label = labels[int(code)] if int(code) < len(labels) else str(code)
        groups.append((_format_label(label), values))
    return {"groups": groups}


def overview(df):
    numeric = _numeric_columns(df)
    categorical = _categorical_columns(df)
    missing_cells = int(df.isna().sum().sum())
    total_cells = int(df.shape[0] * df.shape[1])
    missing_pct = missing_cells / total_cells * 100 if total_cells else 0.0
    duplicate_rows = int(df.duplicated().sum())

    rows = []
    for column in df.columns:
        series = df[column]
        non_null = series.dropna()
        if non_null.empty:
            example = ""
        else:
            example = str(non_null.iloc[0])
        if len(example) > 48:
            example = example[:45] + "..."
        rows.append(
            {
                "column": column,
                "dtype": str(series.dtype),
                "missing": int(series.isna().sum()),
                "missing_pct": float(series.isna().mean() * 100),
                "unique": int(series.nunique(dropna=True)),
                "example": example,
            }
        )

    column_summary = pd.DataFrame(rows)
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "numeric_count": len(numeric),
        "categorical_count": len(categorical),
        "missing_cells": missing_cells,
        "missing_pct": float(missing_pct),
        "duplicate_rows": duplicate_rows,
        "memory_mb": float(df.memory_usage(deep=True).sum() / (1024**2)),
        "column_summary": column_summary,
    }


def _target_summary(df, target, task):
    series = df[target]
    if task == CLASSIFICATION:
        encoded, labels = _factorize_target(series)
        counts = series.value_counts(dropna=True).reindex(labels, fill_value=0)
        percentages = counts / counts.sum() * 100 if counts.sum() else counts * 0
        distribution = pd.DataFrame(
            {
                "class": [_format_label(label) for label in labels],
                "label": labels,
                "count": counts.to_numpy(dtype=int),
                "pct": percentages.to_numpy(dtype=float),
            }
        )
        minority_pct = float(percentages.min()) if len(percentages) else 0.0
        majority_pct = float(percentages.max()) if len(percentages) else 0.0
        positive_label = labels[-1] if labels else None
        positive_pct = (
            float(percentages.iloc[-1]) if len(percentages) > 0 else 0.0
        )
        return {
            "task": task,
            "name": target,
            "n": int(series.notna().sum()),
            "classes": labels,
            "positive_class": positive_label,
            "positive_code": len(labels) - 1 if labels else None,
            "positive_pct": positive_pct,
            "distribution": distribution,
            "minority_pct": minority_pct,
            "majority_pct": majority_pct,
            "imbalanced": minority_pct < 20.0,
        }

    values = pd.to_numeric(series, errors="coerce").dropna()
    sample = _sample_frame(values)
    return {
        "task": task,
        "name": target,
        "n": int(len(values)),
        "mean": float(values.mean()) if len(values) else None,
        "median": float(values.median()) if len(values) else None,
        "std": float(values.std()) if len(values) > 1 else None,
        "min": float(values.min()) if len(values) else None,
        "max": float(values.max()) if len(values) else None,
        "skew": float(values.skew()) if len(values) > 2 else None,
        "values": sample.to_numpy(dtype=float),
    }


def _missing_table(df):
    missing = df.isna().mean() * 100
    missing = missing[missing > 0].sort_values(ascending=False)
    return pd.DataFrame(
        {
            "column": missing.index,
            "missing_pct": missing.to_numpy(dtype=float),
        }
    )


def _outlier_table(df, numeric):
    rows = []
    for column in numeric:
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        if len(values) < 4:
            continue
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        spread = q3 - q1
        if spread <= 0:
            lower, upper = q1, q3
        else:
            lower = q1 - 1.5 * spread
            upper = q3 + 1.5 * spread
        count = int(((values < lower) | (values > upper)).sum())
        if count:
            rows.append(
                {
                    "column": column,
                    "lower": float(lower),
                    "upper": float(upper),
                    "outliers": count,
                    "outlier_pct": float(count / len(values) * 100),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=["column", "lower", "upper", "outliers", "outlier_pct"]
        )
    return pd.DataFrame(rows).sort_values("outlier_pct", ascending=False)


def _correlation_data(working, numeric, codes, target):
    if not numeric:
        return None
    correlations = {}
    for column in numeric:
        frame = pd.DataFrame({"value": working[column], "target": codes}).dropna()
        correlation = _safe_corr(frame["value"], frame["target"])
        correlations[column] = abs(correlation) if correlation is not None else 0.0
    ranked = sorted(correlations, key=correlations.get, reverse=True)
    selected = ranked[: MAX_CORRELATION_COLUMNS - 1]
    if not selected:
        return None
    heat = working[selected].apply(pd.to_numeric, errors="coerce").copy()
    heat[target] = codes
    matrix = heat.corr().fillna(0.0).round(4)
    return {
        "matrix": matrix,
        "target": target,
        "columns": list(matrix.columns),
    }


def _relationship_sentence(association, lookup):
    feature = association["feature"]
    kind = association["kind"]
    strength = association["strength"]
    direction = association["direction"]
    target = lookup["target_name"]
    task = lookup["task"]

    if kind == "numeric":
        if task == CLASSIFICATION and lookup["binary"]:
            if direction is None:
                return (
                    f"`{feature}` has no measurable linear relationship with "
                    f"{target} in the usable rows."
                )
            sign = "positive" if direction > 0 else "negative"
            verb = "more" if sign == "positive" else "less"
            return (
                f"`{feature}` has a {_strength_word(strength)} {sign} relationship "
                f"with {target} (r = {direction:+.2f}): higher values are associated "
                f"with {verb} {target}."
            )
        if task == REGRESSION:
            if direction is None:
                return (
                    f"`{feature}` has no measurable linear relationship with "
                    f"{target} in the usable rows."
                )
            sign = "positive" if direction > 0 else "negative"
            movement = "higher" if direction > 0 else "lower"
            return (
                f"`{feature}` has a {_strength_word(strength)} {sign} relationship "
                f"with {target} (r = {direction:+.2f}): higher values tend to mean "
                f"{movement} {target}."
            )
        return (
            f"`{feature}` separates the {target} classes "
            f"(eta² = {strength:.2f})."
        )

    table = lookup["category_tables"].get(feature)
    if table is None or table.empty:
        return f"`{feature}` is associated with {target} ({association['metric']} = {strength:.2f})."
    usable = table.loc[table["count"] >= 3]
    if len(usable) < 2:
        return (
            f"`{feature}` is {_strength_word(strength)}ly associated with {target} "
            f"({association['metric']} = {strength:.2f}); category-level rates are "
            "too sparse to compare reliably."
        )
    highest = usable.loc[usable["target_value"].idxmax()]
    lowest = usable.loc[usable["target_value"].idxmin()]
    if highest["category"] == lowest["category"]:
        return (
            f"`{feature}` is {_strength_word(strength)}ly associated with {target} "
            f"({association['metric']} = {strength:.2f}); target rates are similar "
            "across its categories."
        )
    if task == CLASSIFICATION:
        return (
            f"`{feature}` is {_strength_word(strength)}ly associated with {target} "
            f"({association['metric']} = {strength:.2f}): "
            f"{highest['category']} has the highest rate at {_format_pct(highest['target_value'])}, "
            f"versus {_format_pct(lowest['target_value'])} for {lowest['category']}."
        )
    return (
        f"`{feature}` is {_strength_word(strength)}ly associated with {target} "
        f"({association['metric']} = {strength:.2f}): average {target} is "
        f"{_format_number(highest['target_value'])} for {highest['category']}, "
        f"versus {_format_number(lowest['target_value'])} for {lowest['category']}."
    )


def _correlation_pairs(working, numeric):
    if not numeric:
        return (
            "No numeric features are available, so feature-to-feature "
            "correlation cannot be checked."
        )
    if len(numeric) == 1:
        return (
            "Only one numeric feature is available, so feature-to-feature "
            "correlation cannot be checked."
        )
    pairs = []
    for index, left in enumerate(numeric):
        for right in numeric[index + 1 :]:
            value = _safe_corr(
                pd.to_numeric(working[left], errors="coerce"),
                pd.to_numeric(working[right], errors="coerce"),
            )
            if value is not None and abs(value) >= 0.8:
                pairs.append((left, right, value))
    pairs.sort(key=lambda item: abs(item[2]), reverse=True)
    if not pairs:
        return "No numeric feature pair exceeds |r| = 0.80, so multicollinearity is limited."
    left, right, value = pairs[0]
    sentence = (
        f"`{left}` and `{right}` are highly correlated (r = {value:+.2f}); "
        "they carry overlapping information."
    )
    if len(pairs) > 1:
        sentence += f" {len(pairs) - 1} other pair(s) also exceed the 0.80 threshold."
    return sentence


def _range_sentence(
    working,
    codes,
    target,
    task,
    numeric_associations,
    positive_code,
):
    if not numeric_associations:
        return None
    best = numeric_associations[0]["feature"]
    values = pd.to_numeric(working[best], errors="coerce")
    frame = pd.DataFrame({"value": values, "target": codes}).dropna()
    if (
        len(frame) < 8
        or frame["value"].nunique() < 2
        or frame["target"].nunique() < 2
    ):
        return None
    try:
        frame["bin"] = pd.qcut(frame["value"], q=4, duplicates="drop")
    except (TypeError, ValueError):
        return None
    if frame["bin"].nunique() < 2:
        return None
    if task == CLASSIFICATION:
        frame["positive"] = (frame["target"] == positive_code).astype(float)
        grouped = frame.groupby("bin", observed=True)["positive"].mean() * 100
    else:
        grouped = frame.groupby("bin", observed=True)["target"].mean()
    highest = grouped.idxmax()
    lowest = grouped.idxmin()
    if highest == lowest or grouped.max() == grouped.min():
        return None
    if task == CLASSIFICATION:
        return (
            f"The highest {target} rate is {_format_pct(grouped.max())} when "
            f"`{best}` is {_format_interval(highest)}, and the lowest is "
            f"{_format_pct(grouped.min())} when it is {_format_interval(lowest)}."
        )
    return (
        f"Average {target} peaks at {_format_number(grouped.max())} when `{best}` "
        f"is {_format_interval(highest)}, and bottoms out at "
        f"{_format_number(grouped.min())} when it is {_format_interval(lowest)}."
    )


def _build_insights(
    df,
    working,
    target,
    task,
    numeric,
    categorical,
    codes,
    target_summary,
    numeric_associations,
    categorical_associations,
    associations,
    missing,
    outliers,
    category_tables,
):
    insights = []
    if task == CLASSIFICATION:
        if len(target_summary.get("classes", [])) < 2:
            insights.append(
                f"`{target}` has only one usable class. Classification needs "
                "at least two classes."
            )
        else:
            positive = _format_label(target_summary["positive_class"])
            insights.append(
                f"`{target}` = {positive} makes up "
                f"{_format_pct(target_summary['positive_pct'])} of usable rows, "
                f"so the largest class is "
                f"{_format_pct(target_summary['majority_pct'])}. "
                + (
                    "This is an imbalanced target, so accuracy alone can be misleading."
                    if target_summary["imbalanced"]
                    else "The target is reasonably balanced."
                )
            )
    elif target_summary["std"] is None or target_summary["std"] <= 0:
        insights.append(
            f"`{target}` is constant at {_format_number(target_summary['mean'])}. "
            "There is no variation for a regression model to learn."
        )
    else:
        insights.append(
            f"`{target}` ranges from {_format_number(target_summary['min'])} to "
            f"{_format_number(target_summary['max'])} with a mean of "
            f"{_format_number(target_summary['mean'])} and median of "
            f"{_format_number(target_summary['median'])}."
        )

    lookup = {
        "target_name": target,
        "task": task,
        "binary": task == CLASSIFICATION and len(target_summary.get("classes", [])) == 2,
        "positive_class": target_summary.get("positive_class"),
        "category_tables": category_tables,
    }
    has_target_variation = (
        task == CLASSIFICATION and len(target_summary.get("classes", [])) >= 2
    ) or (
        task == REGRESSION
        and target_summary.get("std") is not None
        and target_summary["std"] > 0
    )
    if has_target_variation:
        for association in associations[:3]:
            insights.append(_relationship_sentence(association, lookup))

        range_sentence = _range_sentence(
            working,
            codes,
            target,
            task,
            numeric_associations,
            target_summary.get("positive_code"),
        )
        if range_sentence:
            insights.append(range_sentence)

    insights.append(_correlation_pairs(working, numeric))

    high_missing = missing.loc[missing["missing_pct"] > 20]
    if not high_missing.empty:
        names = ", ".join(
            f"`{row.column}` ({_format_pct(row.missing_pct)})"
            for row in high_missing.head(4).itertuples()
        )
        insights.append(f"High missingness: {names} exceed the 20% threshold.")
    elif not missing.empty:
        top = missing.iloc[0]
        insights.append(
            f"No column exceeds 20% missing; `{top['column']}` is highest at "
            f"{_format_pct(top['missing_pct'])}."
        )
    else:
        insights.append("No missing values were found in the selected dataset.")

    skews = []
    for column in numeric:
        series = df[column].dropna()
        if len(series) < 3 or series.nunique() < 2:
            continue
        value = _clean_float(series.skew())
        if value is not None and abs(value) > 1:
            skews.append((column, value))
    if skews:
        column, value = max(skews, key=lambda item: abs(item[1]))
        direction = "right" if value > 0 else "left"
        insights.append(
            f"`{column}` is strongly {direction}-skewed (skew {value:+.2f}); "
            "consider a transform or a model that handles skew."
        )
    elif numeric:
        insights.append(
            "No numeric feature has |skew| above 1, so the distributions are broadly symmetric."
        )
    else:
        insights.append("No numeric features were detected.")

    near_constant = []
    for column in df.columns:
        series = df[column].dropna()
        if series.empty:
            continue
        top_share = series.value_counts(normalize=True).iloc[0] * 100
        if series.nunique() <= 1 or top_share >= 95:
            near_constant.append((column, top_share, series.nunique()))
    if near_constant:
        column, share, unique = near_constant[0]
        insights.append(
            f"`{column}` is nearly constant ({unique} unique value(s); "
            f"{_format_pct(share)} in the top value), so it adds little signal."
        )
    else:
        insights.append("No constant or near-constant columns were found.")

    high_cardinality = []
    for column in categorical:
        series = df[column].dropna()
        if series.empty:
            continue
        unique = series.nunique()
        ratio = unique / len(df) if len(df) else 0
        if unique > 20 and ratio > 0.20:
            high_cardinality.append((column, unique, ratio))
    if high_cardinality:
        column, unique, ratio = max(high_cardinality, key=lambda item: item[1])
        insights.append(
            f"`{column}` is high-cardinality ({unique} unique values); "
            "one-hot encoding could create many sparse columns."
        )
    elif categorical:
        insights.append(
            "No categorical column is high-cardinality, so encoding should stay compact."
        )
    else:
        insights.append("No categorical features were detected.")

    if not outliers.empty:
        top = outliers.iloc[0]
        insights.append(
            f"IQR outliers are most common in `{top['column']}` "
            f"({_format_pct(top['outlier_pct'])} of rows); tree models are usually less sensitive."
        )
    elif numeric:
        insights.append("No IQR outliers were detected in the numeric columns.")
    else:
        insights.append("No numeric features were available for outlier checks.")

    return insights[:12]


def explore(df, target, task):
    if target not in df.columns:
        raise ValueError("The target column is not in the dataframe.")
    if task not in {CLASSIFICATION, REGRESSION}:
        raise ValueError("Task must be classification or regression.")

    target_summary = _target_summary(df, target, task)
    codes, labels = _target_numeric(df[target], task)
    target_mask = codes.notna()
    working = df.loc[target_mask].copy()
    codes = codes.loc[target_mask]

    feature_frame = df.drop(columns=[target])
    numeric = _numeric_columns(feature_frame)
    categorical = _categorical_columns(feature_frame)
    binary = task == CLASSIFICATION and len(labels) == 2
    positive_code = len(labels) - 1 if labels else None

    numeric_associations = []
    for column in numeric:
        relation = _numeric_relationship(working[column], codes, task, binary)
        numeric_associations.append(
            {
                "feature": column,
                "kind": "numeric",
                "strength": relation["strength"],
                "direction": relation["direction"],
                "metric": relation["metric"],
            }
        )
    numeric_associations.sort(key=lambda item: item["strength"], reverse=True)

    categorical_associations = []
    category_tables = {}
    for column in categorical:
        relation = _categorical_relationship(working[column], codes, task)
        categorical_associations.append(
            {
                "feature": column,
                "kind": "categorical",
                "strength": relation["strength"],
                "direction": None,
                "metric": relation["metric"],
            }
        )
        category_tables[column] = _category_table(
            working[column], codes, task, positive_code
        )
    categorical_associations.sort(key=lambda item: item["strength"], reverse=True)

    associations = sorted(
        numeric_associations + categorical_associations,
        key=lambda item: item["strength"],
        reverse=True,
    )

    sample = _sample_frame(working)
    sample_codes = codes.loc[sample.index]

    histograms = []
    for column in numeric:
        series = working[column]
        if series.nunique(dropna=True) < 2:
            continue
        axis_values = series.dropna()
        axis_sample = _sample_frame(axis_values)
        histograms.append(
            {
                "name": column,
                "values": axis_sample.to_numpy(dtype=float),
                "skew": _clean_float(series.skew()),
            }
        )
        if len(histograms) >= MAX_HISTOGRAMS:
            break

    grouped_numeric = []
    for association in numeric_associations[:MAX_GROUPED_NUMERIC]:
        column = association["feature"]
        groups = _numeric_groups(
            sample[column],
            sample_codes,
            task,
            [ _format_label(label) for label in labels],
        )
        grouped_numeric.append(
            {
                "name": column,
                "strength": association["strength"],
                "direction": association["direction"],
                "metric": association["metric"],
                **groups,
            }
        )

    categorical_charts = []
    for association in categorical_associations:
        if len(categorical_charts) >= MAX_CATEGORICAL_CHARTS:
            break
        column = association["feature"]
        table = category_tables[column]
        if table.empty:
            continue
        categorical_charts.append(
            {
                "name": column,
                "strength": association["strength"],
                "metric": association["metric"],
                "table": table,
            }
        )

    correlation = _correlation_data(working, numeric, codes, target)
    missing = _missing_table(df)
    outliers = _outlier_table(df, numeric)

    insights = _build_insights(
        df,
        working,
        target,
        task,
        numeric,
        categorical,
        codes,
        target_summary,
        numeric_associations,
        categorical_associations,
        associations,
        missing,
        outliers,
        category_tables,
    )

    return {
        "target": target_summary,
        "numeric_count": len(numeric),
        "categorical_count": len(categorical),
        "numeric_associations": numeric_associations,
        "categorical_associations": categorical_associations,
        "associations": associations,
        "histograms": histograms,
        "grouped_numeric": grouped_numeric,
        "categorical_charts": categorical_charts,
        "category_tables": category_tables,
        "correlation": correlation,
        "missing": missing,
        "outliers": outliers,
        "insights": insights,
    }
