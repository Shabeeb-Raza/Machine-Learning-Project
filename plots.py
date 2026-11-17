import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PALETTE = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#CC79A7",
    "#56B4E9",
    "#D55E00",
    "#7F7F7F",
    "#F0E442",
]
ACCENT = PALETTE[0]
SEQUENTIAL = "Blues"
DIVERGING = "RdBu_r"
GRID = "#E5E7EB"
SPINE = "#D1D5DB"
TEXT = "#1F2937"
MUTED = "#6B7280"


def _short(value, length=22):
    text = str(value)
    return text if len(text) <= length else text[: length - 1] + "…"


def _style_axis(axis):
    axis.set_facecolor("white")
    axis.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(SPINE)
    axis.tick_params(colors=MUTED, labelsize=9)


def _title(axis, text, size=11):
    axis.set_title(
        text,
        loc="left",
        fontsize=size,
        fontweight="bold",
        color=TEXT,
        pad=9,
    )


def _finish(figure, top=None):
    if top is None:
        figure.tight_layout()
    else:
        figure.tight_layout(rect=(0, 0, 1, top))


def target_distribution(target):
    name = target["name"]
    figure, axis = plt.subplots(figsize=(7.2, 3.7))
    _style_axis(axis)

    if target["task"] == "classification":
        distribution = target["distribution"]
        labels = [_short(str(label), 18) for label in distribution["class"]]
        counts = distribution["count"].to_numpy(dtype=float)
        percentages = distribution["pct"].to_numpy(dtype=float)
        colors = PALETTE[: len(labels)]
        bars = axis.bar(labels, counts, color=colors, width=0.62)
        maximum = counts.max() if len(counts) else 1
        for bar, count, pct in zip(bars, counts, percentages):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + maximum * 0.02,
                f"{int(count):,} ({pct:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=9,
                color=TEXT,
            )
        axis.set_ylabel("Rows")
        axis.set_ylim(0, maximum * 1.18 if maximum else 1)
        _title(
            axis,
            f"{name} split: {percentages[-1]:.1f}% in the positive class",
        )
    else:
        values = np.asarray(target.get("values", []), dtype=float)
        if len(values):
            axis.hist(values, bins=28, color=ACCENT, alpha=0.82, edgecolor="white")
        axis.axvline(
            target["mean"],
            color=PALETTE[5],
            linewidth=2,
            label=f"Mean {target['mean']:,.0f}",
        )
        axis.axvline(
            target["median"],
            color=PALETTE[2],
            linewidth=2,
            linestyle="--",
            label=f"Median {target['median']:,.0f}",
        )
        axis.set_xlabel(name)
        axis.set_ylabel("Rows")
        axis.legend(frameon=False, fontsize=9)
        _title(
            axis,
            f"{name}: mean {target['mean']:,.0f}, median {target['median']:,.0f}",
        )

    _finish(figure)
    return figure


def histogram_grid(histograms):
    count = len(histograms)
    columns = min(3, max(1, count))
    rows = max(1, math.ceil(count / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(3.35 * columns, 2.55 * rows),
        squeeze=False,
    )
    flat = axes.ravel()
    for axis, histogram in zip(flat, histograms):
        _style_axis(axis)
        axis.hist(
            histogram["values"],
            bins=24,
            color=ACCENT,
            alpha=0.82,
            edgecolor="white",
            linewidth=0.4,
        )
        skew = histogram["skew"]
        skew_label = f"skew {skew:+.2f}" if skew is not None else "skew n/a"
        _title(axis, f"{_short(histogram['name'], 20)} · {skew_label}", size=10)
        axis.set_ylabel("Rows", fontsize=8)
    for axis in flat[count:]:
        axis.set_visible(False)
    figure.suptitle(
        "Numeric distributions",
        x=0.01,
        ha="left",
        fontsize=13,
        fontweight="bold",
        color=TEXT,
    )
    _finish(figure, top=0.90)
    return figure


def numeric_by_target(items, target_name, task):
    count = len(items)
    columns = min(3, max(1, count))
    rows = max(1, math.ceil(count / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(3.6 * columns, 2.8 * rows),
        squeeze=False,
    )
    flat = axes.ravel()

    for axis, item in zip(flat, items):
        _style_axis(axis)
        if task == "classification":
            groups = item.get("groups", [])
            values = [values for _, values in groups if len(values)]
            labels = [label for label, values in groups if len(values)]
            if values:
                box = axis.boxplot(
                    values,
                    patch_artist=True,
                    widths=0.58,
                    showfliers=False,
                    medianprops={"color": TEXT, "linewidth": 1.3},
                    whiskerprops={"color": MUTED},
                    capprops={"color": MUTED},
                )
                for patch, color in zip(box["boxes"], PALETTE):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.75)
                    patch.set_edgecolor("white")
                axis.set_xticks(
                    range(1, len(labels) + 1),
                    [_short(label, 10) for label in labels],
                )
            strength = item.get("strength", 0)
            _title(
                axis,
                f"{_short(item['name'], 18)} by {_short(target_name, 12)} "
                f"({item['metric']} {strength:.2f})",
                size=10,
            )
            axis.set_ylabel("Value", fontsize=8)
        else:
            x = np.asarray(item.get("x", []), dtype=float)
            y = np.asarray(item.get("y", []), dtype=float)
            axis.scatter(x, y, s=10, alpha=0.45, color=ACCENT, edgecolors="none")
            if len(x) >= 2 and np.ptp(x) > 0:
                slope, intercept = np.polyfit(x, y, 1)
                line_x = np.linspace(float(x.min()), float(x.max()), 50)
                axis.plot(
                    line_x,
                    slope * line_x + intercept,
                    color=PALETTE[5],
                    linewidth=2,
                )
            direction = item.get("direction")
            correlation = (
                f"r = {direction:+.2f}" if direction is not None else item["metric"]
            )
            _title(
                axis,
                f"{_short(item['name'], 18)} vs {_short(target_name, 12)} "
                f"({correlation})",
                size=10,
            )
            axis.set_ylabel(_short(target_name, 14), fontsize=8)

    for axis in flat[count:]:
        axis.set_visible(False)
    figure.suptitle(
        "Feature patterns by target",
        x=0.01,
        ha="left",
        fontsize=13,
        fontweight="bold",
        color=TEXT,
    )
    _finish(figure, top=0.90)
    return figure


def categorical_target(items, target_name, task, positive_class=None):
    count = len(items)
    columns = min(2, max(1, count))
    rows = max(1, math.ceil(count / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(4.9 * columns, 3.0 * rows),
        squeeze=False,
    )
    flat = axes.ravel()

    for axis, item in zip(flat, items):
        _style_axis(axis)
        table = item["table"].sort_values("target_value")
        if table.empty:
            axis.set_xticks([])
            axis.set_yticks([])
            _title(axis, f"{_short(item['name'], 18)}: no usable categories", size=10)
            continue
        labels = [_short(label, 20) for label in table["category"]]
        values = table["target_value"].to_numpy(dtype=float)
        bars = axis.barh(labels, values, color=ACCENT, alpha=0.85, height=0.62)
        maximum = values.max() if len(values) else 1
        for bar, value in zip(bars, values):
            label = f"{value:.1f}%" if task == "classification" else f"{value:,.0f}"
            axis.text(
                value + maximum * 0.02,
                bar.get_y() + bar.get_height() / 2,
                label,
                va="center",
                fontsize=8,
                color=TEXT,
            )
        axis.set_xlim(0, maximum * 1.22 if maximum else 1)
        if task == "classification":
            axis.set_xlabel(f"% with {_short(positive_class, 14)}", fontsize=8)
            best = table.loc[table["target_value"].idxmax()]
            _title(
                axis,
                f"{_short(item['name'], 16)}: {_short(best['category'], 16)} "
                "has the highest rate",
                size=10,
            )
        else:
            axis.set_xlabel(f"Mean {_short(target_name, 14)}", fontsize=8)
            best = table.loc[table["target_value"].idxmax()]
            _title(
                axis,
                f"{_short(item['name'], 16)}: {_short(best['category'], 16)} "
                "has the highest average",
                size=10,
            )

    for axis in flat[count:]:
        axis.set_visible(False)
    figure.suptitle(
        "Categorical patterns by target",
        x=0.01,
        ha="left",
        fontsize=13,
        fontweight="bold",
        color=TEXT,
    )
    _finish(figure, top=0.90)
    return figure


def correlation_heatmap(correlation, target_name):
    matrix = correlation["matrix"]
    labels = [_short(label, 17) for label in matrix.columns]
    size = max(5.6, 0.46 * len(labels) + 2.6)
    figure, axis = plt.subplots(figsize=(size, size * 0.84))
    values = matrix.to_numpy(dtype=float)
    image = axis.imshow(values, cmap=DIVERGING, vmin=-1, vmax=1)
    axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=7)
    axis.set_yticks(range(len(labels)), labels, fontsize=7)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            axis.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=6.5,
                color="white" if abs(value) > 0.55 else TEXT,
            )
    axis.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=1.5)
    axis.tick_params(which="minor", length=0)
    axis.set_title(
        f"Correlation heatmap: strongest numeric links to {target_name}",
        loc="left",
        fontsize=11,
        fontweight="bold",
        color=TEXT,
        pad=10,
    )
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    return figure


def association_ranking(associations, target_name):
    frame = pd.DataFrame(associations)
    if frame.empty:
        figure, axis = plt.subplots(figsize=(7.4, 3.8))
        _style_axis(axis)
        axis.set_xticks([])
        axis.set_yticks([])
        _title(axis, f"No features available to rank for {_short(target_name, 18)}")
        _finish(figure)
        return figure
    frame = frame.sort_values("strength", ascending=True)
    height = max(3.8, 0.31 * len(frame) + 1.4)
    figure, axis = plt.subplots(figsize=(7.4, height))
    _style_axis(axis)
    labels = [_short(label, 24) for label in frame["feature"]]
    colors = [
        ACCENT if kind == "numeric" else PALETTE[2]
        for kind in frame["kind"]
    ]
    bars = axis.barh(labels, frame["strength"], color=colors, alpha=0.86, height=0.62)
    maximum = frame["strength"].max() if len(frame) else 1
    for bar, value in zip(bars, frame["strength"]):
        axis.text(
            value + maximum * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            va="center",
            fontsize=8,
            color=TEXT,
        )
    axis.set_xlim(0, maximum * 1.16 if maximum else 1)
    axis.set_xlabel("Strength (absolute correlation or association)", fontsize=9)
    if len(frame):
        top = frame.iloc[-1]
        _title(
            axis,
            f"{_short(top['feature'], 24)} is the strongest signal for "
            f"{_short(target_name, 18)}",
        )
    _finish(figure)
    return figure


def missing_values(missing):
    frame = missing.sort_values("missing_pct")
    height = max(3.2, 0.34 * len(frame) + 1.2)
    figure, axis = plt.subplots(figsize=(7.2, height))
    _style_axis(axis)
    bars = axis.barh(
        [_short(column, 24) for column in frame["column"]],
        frame["missing_pct"],
        color=ACCENT,
        alpha=0.85,
        height=0.62,
    )
    maximum = frame["missing_pct"].max() if len(frame) else 1
    for bar, value in zip(bars, frame["missing_pct"]):
        axis.text(
            value + maximum * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}%",
            va="center",
            fontsize=8,
            color=TEXT,
        )
    axis.set_xlim(0, maximum * 1.18 if maximum else 1)
    axis.set_xlabel("% missing", fontsize=9)
    if len(frame):
        top = frame.iloc[-1]
        _title(axis, f"{_short(top['column'], 22)} has the most missing data")
    _finish(figure)
    return figure


def confusion_matrix_chart(matrix, labels, normalize=False):
    matrix = np.asarray(matrix, dtype=float)
    if normalize:
        totals = matrix.sum(axis=1, keepdims=True)
        display = np.divide(
            matrix,
            totals,
            out=np.zeros_like(matrix),
            where=totals != 0,
        ) * 100
        title = "Confusion matrix: row-normalised (%)"
    else:
        display = matrix
        title = "Confusion matrix: counts"

    figure, axis = plt.subplots(figsize=(4.5, 3.8))
    image = axis.imshow(display, cmap=SEQUENTIAL, interpolation="nearest")
    label_text = [_short(label, 12) for label in labels]
    axis.set_xticks(range(len(label_text)), label_text, rotation=35, ha="right")
    axis.set_yticks(range(len(label_text)), label_text)
    axis.set_xlabel("Predicted label", fontsize=9)
    axis.set_ylabel("Actual label", fontsize=9)
    axis.set_title(title, loc="left", fontsize=10, fontweight="bold", color=TEXT)
    threshold = display.max() / 2 if display.size else 0
    for row in range(display.shape[0]):
        for column in range(display.shape[1]):
            value = display[row, column]
            text = f"{value:.0f}%" if normalize else f"{int(round(value)):,}"
            axis.text(
                column,
                row,
                text,
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value > threshold else TEXT,
            )
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    return figure


def roc_curve(roc):
    figure, axis = plt.subplots(figsize=(4.5, 3.8))
    _style_axis(axis)
    axis.plot(
        roc["fpr"],
        roc["tpr"],
        color=ACCENT,
        linewidth=2.2,
        label=f"AUC = {roc['auc']:.3f}",
    )
    axis.fill_between(
        roc["fpr"],
        roc["tpr"],
        color=ACCENT,
        alpha=0.10,
    )
    axis.plot([0, 1], [0, 1], linestyle="--", color=MUTED, linewidth=1)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.02)
    axis.set_xlabel("False positive rate", fontsize=9)
    axis.set_ylabel("True positive rate", fontsize=9)
    axis.legend(loc="lower right", frameon=False, fontsize=9)
    _title(axis, f"ROC curve: AUC {roc['auc']:.3f}", size=10)
    _finish(figure)
    return figure


def precision_recall_curve_chart(precision_recall, baseline):
    figure, axis = plt.subplots(figsize=(5.2, 3.8))
    _style_axis(axis)
    axis.plot(
        precision_recall["recall"],
        precision_recall["precision"],
        color=ACCENT,
        linewidth=2.1,
        label=f"AP = {precision_recall['average_precision']:.3f}",
    )
    axis.axhline(
        baseline,
        color=MUTED,
        linestyle="--",
        linewidth=1,
        label=f"Baseline {baseline:.1%}",
    )
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.04)
    axis.set_xlabel("Recall", fontsize=9)
    axis.set_ylabel("Precision", fontsize=9)
    axis.legend(loc="upper right", frameon=False, fontsize=8)
    _title(
        axis,
        f"Precision-recall: AP {precision_recall['average_precision']:.3f}",
        size=10,
    )
    _finish(figure)
    return figure


def probability_histogram(y_test, y_proba, classes, target_name, positive_class):
    probabilities = np.asarray(y_proba, dtype=float)
    actual = np.asarray(y_test)
    classes = list(classes)
    figure, axis = plt.subplots(figsize=(5.8, 3.8))
    _style_axis(axis)

    if probabilities.ndim == 2 and len(classes) == 2:
        positive_index = classes.index(positive_class)
        scores = probabilities[:, positive_index]
        for index, label in enumerate(classes):
            mask = actual == label
            if mask.any():
                axis.hist(
                    scores[mask],
                    bins=18,
                    range=(0, 1),
                    alpha=0.55,
                    color=PALETTE[index],
                    label=f"Actual {_short(label, 12)}",
                    edgecolor="white",
                    linewidth=0.4,
                )
        axis.axvline(0.5, color=MUTED, linestyle="--", linewidth=1)
        axis.set_xlabel(f"Predicted probability of {_short(positive_class, 14)}")
        axis.set_ylabel("Rows")
        axis.legend(frameon=False, fontsize=8)
        _title(
            axis,
            f"Predicted {target_name} probability separates the classes",
            size=10,
        )
    else:
        class_index = {label: index for index, label in enumerate(classes)}
        valid = np.array([label in class_index for label in actual])
        if valid.any():
            actual = actual[valid]
            probabilities = probabilities[valid]
            actual_indices = np.array(
                [class_index[label] for label in actual], dtype=int
            )
            true_probabilities = probabilities[
                np.arange(len(actual_indices)), actual_indices
            ]
            counts = pd.Series(actual).value_counts().head(6)
            for index, label in enumerate(counts.index):
                mask = actual == label
                if mask.any():
                    axis.hist(
                        true_probabilities[mask],
                        bins=18,
                        range=(0, 1),
                        alpha=0.55,
                        color=PALETTE[index % len(PALETTE)],
                        label=f"Actual {_short(label, 12)}",
                        edgecolor="white",
                        linewidth=0.4,
                    )
        axis.set_xlabel("Predicted probability of the true class")
        axis.set_ylabel("Rows")
        axis.legend(frameon=False, fontsize=8)
        _title(axis, "True-class probability by actual class", size=10)

    _finish(figure)
    return figure


def predicted_vs_actual(y_test, y_pred, r2):
    actual = np.asarray(y_test, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    figure, axis = plt.subplots(figsize=(4.9, 4.2))
    _style_axis(axis)
    axis.scatter(actual, predicted, s=14, alpha=0.5, color=ACCENT, edgecolors="none")
    minimum = float(min(actual.min(), predicted.min())) if len(actual) else 0.0
    maximum = float(max(actual.max(), predicted.max())) if len(actual) else 1.0
    axis.plot(
        [minimum, maximum],
        [minimum, maximum],
        color=PALETTE[5],
        linestyle="--",
        linewidth=1.4,
        label="Perfect prediction",
    )
    axis.set_xlabel("Actual value", fontsize=9)
    axis.set_ylabel("Predicted value", fontsize=9)
    axis.legend(frameon=False, fontsize=8)
    _title(axis, f"Predicted vs actual (R² = {r2:.3f})", size=10)
    _finish(figure)
    return figure


def residuals_plot(y_test, y_pred):
    actual = np.asarray(y_test, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    residuals = actual - predicted
    figure, (scatter_axis, histogram_axis) = plt.subplots(
        1,
        2,
        figsize=(10.0, 3.8),
    )
    _style_axis(scatter_axis)
    _style_axis(histogram_axis)
    scatter_axis.scatter(
        predicted,
        residuals,
        s=14,
        alpha=0.48,
        color=ACCENT,
        edgecolors="none",
    )
    scatter_axis.axhline(0, color=PALETTE[5], linestyle="--", linewidth=1.3)
    scatter_axis.set_xlabel("Predicted value", fontsize=9)
    scatter_axis.set_ylabel("Residual", fontsize=9)
    _title(scatter_axis, "Residuals vs predicted", size=10)

    histogram_axis.hist(
        residuals,
        bins=24,
        color=ACCENT,
        alpha=0.82,
        edgecolor="white",
        linewidth=0.4,
    )
    histogram_axis.axvline(0, color=PALETTE[5], linestyle="--", linewidth=1.3)
    histogram_axis.set_xlabel("Residual", fontsize=9)
    histogram_axis.set_ylabel("Rows", fontsize=9)
    _title(histogram_axis, "Residual distribution", size=10)

    figure.suptitle(
        "Residuals should sit around zero without an obvious pattern",
        x=0.01,
        ha="left",
        fontsize=13,
        fontweight="bold",
        color=TEXT,
    )
    _finish(figure, top=0.90)
    return figure


def feature_importance_chart(importance, target_name):
    frame = importance.sort_values("importance", ascending=True).tail(10)
    height = max(3.4, 0.38 * len(frame) + 1.2)
    figure, axis = plt.subplots(figsize=(7.2, height))
    _style_axis(axis)
    bars = axis.barh(
        [_short(label, 24) for label in frame["feature"]],
        frame["importance"],
        color=ACCENT,
        alpha=0.86,
        height=0.62,
    )
    maximum = frame["importance"].max() if len(frame) else 1
    for bar, value in zip(bars, frame["importance"]):
        axis.text(
            value + maximum * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            fontsize=8,
            color=TEXT,
        )
    axis.set_xlim(0, maximum * 1.18 if maximum else 1)
    axis.set_xlabel("Importance", fontsize=9)
    if len(frame):
        top = frame.iloc[-1]
        _title(
            axis,
            f"Top driver for {_short(target_name, 18)}: {_short(top['feature'], 22)}",
        )
    _finish(figure)
    return figure


def split_distribution(train_values, test_values, target_name, task):
    figure, axis = plt.subplots(figsize=(8.0, 3.8))
    _style_axis(axis)
    if task == "classification":
        labels = list(
            dict.fromkeys(
                [str(value) for value in pd.Series(train_values).dropna().tolist()]
                + [str(value) for value in pd.Series(test_values).dropna().tolist()]
            )
        )
        train_counts = pd.Series(train_values).astype(str).value_counts()
        test_counts = pd.Series(test_values).astype(str).value_counts()
        positions = np.arange(len(labels))
        width = 0.38
        axis.bar(
            positions - width / 2,
            [train_counts.get(label, 0) for label in labels],
            width,
            color=ACCENT,
            label="Train",
        )
        axis.bar(
            positions + width / 2,
            [test_counts.get(label, 0) for label in labels],
            width,
            color=PALETTE[1],
            label="Test",
        )
        axis.set_xticks(positions, [_short(label, 16) for label in labels], rotation=25, ha="right")
        axis.set_ylabel("Rows")
        axis.legend(frameon=False, fontsize=9)
        _title(axis, f"Class balance in train and test for {_short(target_name, 18)}")
    else:
        train = pd.to_numeric(pd.Series(train_values), errors="coerce").dropna()
        test = pd.to_numeric(pd.Series(test_values), errors="coerce").dropna()
        bins = np.histogram_bin_edges(pd.concat([train, test]), bins=24)
        axis.hist(train, bins=bins, color=ACCENT, alpha=0.65, label="Train")
        axis.hist(test, bins=bins, color=PALETTE[1], alpha=0.65, label="Test")
        axis.set_xlabel(target_name)
        axis.set_ylabel("Rows")
        axis.legend(frameon=False, fontsize=9)
        _title(axis, f"{_short(target_name, 18)} distribution in train and test")
    _finish(figure)
    return figure


def cv_score_chart(cv_result):
    scores = np.asarray(cv_result["scores"], dtype=float)
    folds = np.asarray(cv_result["folds"])
    figure, axis = plt.subplots(figsize=(8.0, 3.6))
    _style_axis(axis)
    bars = axis.bar(
        [f"Fold {fold}" for fold in folds],
        scores,
        color=ACCENT,
        alpha=0.86,
        width=0.62,
    )
    mean = float(cv_result["mean"])
    axis.axhline(
        mean,
        color=PALETTE[5],
        linestyle="--",
        linewidth=1.4,
        label=f"Mean {mean:.3f}",
    )
    for bar, value in zip(bars, scores):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=TEXT,
        )
    axis.set_ylabel(cv_result["metric"], fontsize=9)
    axis.legend(frameon=False, fontsize=8)
    _title(axis, f"Training-only {cv_result['metric']} across {len(scores)} folds", size=10)
    _finish(figure)
    return figure
