import io
import json
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
from sklearn.model_selection import train_test_split

import eda
import pipeline
import plots
import preprocess
import wizard
from model_info import info_for
from models import hyperparameters_for, models_for_task, search_candidates
from pipeline import (
    MAX_SEARCH_COMBINATIONS,
    MAX_TRAIN_ROWS,
    MAX_UPLOAD_MB,
    PipelineError,
    count_combinations,
    grid_search_with_recipe,
    infer_task_type,
    limit_rows,
    train_with_recipe,
    validate_upload,
)

st.set_page_config(
    page_title="ModelLab",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

DATA_VIEWS = ["Overview", "Explore"]
MISSING_OPTIONS = list(
    dict.fromkeys(preprocess.NUMERIC_MISSING + preprocess.CATEGORICAL_MISSING)
)
DELIMITERS = {
    "Auto": None,
    "Comma": ",",
    "Semicolon": ";",
    "Tab": "\t",
    "Pipe": "|",
}
ENCODINGS = ["Auto", "utf-8", "utf-8-sig", "latin-1"]


@st.cache_data(show_spinner=False, max_entries=4)
def parse_uploaded_csv(payload, delimiter, encoding):
    encoding_options = (
        [encoding] if encoding != "Auto" else ["utf-8-sig", "utf-8", "latin-1"]
    )
    separator = DELIMITERS.get(delimiter)
    last_error = None
    for candidate in encoding_options:
        try:
            return pd.read_csv(
                io.BytesIO(payload),
                sep=separator,
                engine="python" if separator is None else "c",
                encoding=candidate,
            )
        except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
            last_error = exc
    raise ValueError("The CSV could not be parsed with the selected settings.") from last_error


@st.cache_data(show_spinner=False, max_entries=2)
def load_sample_csv(path):
    return pd.read_csv(path)


@st.cache_data(show_spinner=False, max_entries=4)
def cached_overview(df):
    return eda.overview(df)


@st.cache_data(show_spinner=False, max_entries=4)
def cached_explore(df, target, task):
    return eda.explore(df, target, task)


def render_figure(figure):
    try:
        st.pyplot(figure, width="stretch")
    finally:
        plt.close(figure)


def slugify(value):
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def score_text(value):
    value = float(value)
    if abs(value) < 0.005:
        value = 0.0
    return f"{value:.2f}"


def safe_json(value):
    return json.dumps(value, sort_keys=True, default=str)


def data_signature(source_key, target, task, features):
    return safe_json(
        {
            "source": source_key,
            "target": target,
            "task": task,
            "features": list(features),
        }
    )


def model_signature(source_key, target, task, features, recipe, split, model_name, params, search):
    return safe_json(
        {
            "data": data_signature(source_key, target, task, features),
            "recipe": recipe,
            "split": {
                "test_size": split.get("test_size") if split else None,
                "random_state": split.get("random_state") if split else None,
                "stratify": split.get("stratify") if split else None,
            },
            "model": model_name,
            "params": params,
            "search": bool(search),
        }
    )


def bump_recipe_version():
    wizard.put("recipe_version", int(wizard.get("recipe_version", 0)) + 1)


def remember_data_mode():
    wizard.put("data_mode_value", st.session_state.get("wz_data_mode", "Upload CSV"))


def remember_sample_kind():
    wizard.put("sample_kind_value", st.session_state.get("wz_sample_kind", "Classification"))


def remember_delimiter():
    wizard.put("delimiter_value", st.session_state.get("wz_delimiter", "Auto"))


def remember_encoding():
    wizard.put("encoding_value", st.session_state.get("wz_encoding", "Auto"))


def recipe_changed():
    wizard.put("recipe_applied", False)
    wizard.invalidate_from(3, "Preprocessing changed. Apply the recipe before continuing.")


def split_settings_changed():
    wizard.invalidate_from(4, "Split settings changed. Recreate the split to refresh results.")


def model_settings_changed():
    wizard.clear_results()
    wizard.mark_incomplete_from(4)
    wizard.set_flash("Model settings changed. Train again to refresh the results.")


def clear_dataset():
    wizard.put("df", None)
    wizard.put("source_name", None)
    wizard.put("source_key", None)
    wizard.put("original_rows", 0)
    wizard.put("row_notice", None)
    wizard.put("target", None)
    wizard.put("task", None)
    wizard.put("features", [])
    wizard.put("data_applied", None)
    wizard.put("completed", [])
    wizard.put("step", 1)
    wizard.put("recipe", None)
    wizard.put("recipe_applied", False)
    wizard.put("clean_summary", None)
    wizard.put("target_meta", None)
    wizard.put("split", None)
    wizard.put("result", None)
    wizard.put("result_signature", None)
    wizard.put("cv_result", None)
    wizard.put("trained_at", None)
    bump_recipe_version()


def adopt_dataset(df, source_name, source_key, original_rows, row_notice):
    if source_key == wizard.get("source_key"):
        wizard.put("row_notice", row_notice)
        return
    wizard.put("df", df)
    wizard.put("source_name", source_name)
    wizard.put("source_key", source_key)
    wizard.put("original_rows", int(original_rows))
    wizard.put("row_notice", row_notice)
    target = df.columns[-1]
    wizard.put("target", target)
    try:
        detected = infer_task_type(df[target])
    except PipelineError:
        detected = "classification"
    wizard.put("task", detected)
    wizard.put("features", [column for column in df.columns if column != target])
    wizard.put("data_applied", None)
    wizard.put("completed", [])
    wizard.put("step", 1)
    wizard.put("recipe", None)
    wizard.put("recipe_applied", False)
    wizard.put("clean_summary", None)
    wizard.put("target_meta", None)
    wizard.put("split", None)
    wizard.put("result", None)
    wizard.put("result_signature", None)
    wizard.put("cv_result", None)
    wizard.put("trained_at", None)
    wizard.put("confirm_restart", False)
    bump_recipe_version()
    wizard.set_flash(
        "New dataset loaded. Previous preprocessing, split, and results were cleared."
    )


def render_status_strip():
    df = wizard.get("df")
    if df is None:
        return
    target = wizard.get("target") or "not set"
    task = wizard.get("task") or "not set"
    model_name = wizard.get("model_name") or "not chosen"
    st.caption(
        f"**{wizard.get('source_name', 'Dataset')}** · "
        f"{len(df):,} rows × {df.shape[1]:,} columns · "
        f"target `{target}` · {task} · model **{model_name}**"
    )


def recipe_editor_frame(recipe):
    rows = []
    for item in recipe.get("columns", []):
        rows.append(
            {
                "column": item["name"],
                "type": item["kind"],
                "missing": item["missing"],
                "constant": "" if item.get("constant") is None else str(item.get("constant")),
                "outlier": item.get("outlier", "none"),
                "encoding": item.get("encoding", "none"),
                "scaling": item.get("scaling", "none"),
                "rare": item.get("rare", "none"),
                "category_order": item.get("category_order", ""),
            }
        )
    return pd.DataFrame(rows)


def begin():
    wizard.init_state()
    wizard.apply_page_style()
    st.title("ModelLab")
    st.caption(
        "Five guided steps from a raw CSV to a model you can explain, "
        "download, and defend."
    )
    wizard.render_flash()
    wizard.render_stepper()
    render_status_strip()
    step = int(wizard.get("step", 1))
    if step > 1 and wizard.get("df") is None:
        wizard.go(1)
        wizard.set_flash("Load a dataset before opening later steps.")
        st.rerun()
    if step == 1:
        render_data_step()
    elif step == 2:
        render_preprocess_step()
    elif step == 3:
        render_split_step()
    elif step == 4:
        render_model_step()
    else:
        render_results_step()
    if wizard.get("confirm_restart"):
        with st.container(border=True):
            st.warning("Start over and clear the loaded dataset, recipe, split, and model?")
            columns = st.columns(2)
            if columns[0].button("Yes, start over", type="primary", key="wz_confirm_yes"):
                wizard.put("confirm_restart", False)
                for widget_key in (
                    "wz_data_mode",
                    "wz_sample_kind",
                    "wz_upload",
                    "wz_delimiter",
                    "wz_encoding",
                    "wz_data_view",
                ):
                    st.session_state.pop(widget_key, None)
                wizard.put("data_mode_value", "Upload CSV")
                wizard.put("sample_kind_value", "Classification")
                wizard.put("delimiter_value", "Auto")
                wizard.put("encoding_value", "Auto")
                wizard.reset_all("Workspace cleared. Load a dataset to begin.")
                bump_recipe_version()
                st.rerun()
            if columns[1].button("Cancel", key="wz_confirm_no"):
                wizard.put("confirm_restart", False)
                st.rerun()


def render_data_step():
    wizard.render_header(
        1,
        "Data",
        "Load a CSV or sample, choose the target, and confirm which columns the model may use.",
    )
    with st.container(border=True):
        st.subheader("Data source")
        mode_options = ["Upload CSV", "Sample dataset"]
        stored_mode = wizard.get("data_mode_value", "Upload CSV")
        if stored_mode not in mode_options:
            stored_mode = "Upload CSV"
        data_mode = st.radio(
            "Where should the dataset come from?",
            mode_options,
            index=mode_options.index(stored_mode),
            horizontal=True,
            key="wz_data_mode",
            on_change=remember_data_mode,
        )
        expected_prefix = "upload::" if data_mode == "Upload CSV" else "sample::"
        current_source = wizard.get("source_key")
        if current_source and not str(current_source).startswith(expected_prefix):
            clear_dataset()
        loaded = False
        if data_mode == "Upload CSV":
            uploaded = st.file_uploader(
                "Upload a CSV file",
                type=["csv"],
                help=f"Maximum file size: {MAX_UPLOAD_MB} MB.",
                max_upload_size=MAX_UPLOAD_MB,
                key="wz_upload",
            )
            delimiter_options = list(DELIMITERS)
            stored_delimiter = wizard.get("delimiter_value", "Auto")
            if stored_delimiter not in delimiter_options:
                stored_delimiter = "Auto"
            delimiter = st.selectbox(
                "Delimiter",
                delimiter_options,
                index=delimiter_options.index(stored_delimiter),
                key="wz_delimiter",
                help="Auto detects common separators.",
                on_change=remember_delimiter,
            )
            stored_encoding = wizard.get("encoding_value", "Auto")
            if stored_encoding not in ENCODINGS:
                stored_encoding = "Auto"
            encoding = st.selectbox(
                "Text encoding",
                ENCODINGS,
                index=ENCODINGS.index(stored_encoding),
                key="wz_encoding",
                help="Auto tries UTF-8 first and falls back to Latin-1.",
                on_change=remember_encoding,
            )
            if uploaded is None:
                st.info(
                    "Choose a .csv file to begin, or replace the currently loaded file."
                )
                loaded = bool(
                    current_source and str(current_source).startswith("upload::")
                )
            else:
                try:
                    validate_upload(uploaded.name, uploaded.size)
                    parsed = parse_uploaded_csv(
                        uploaded.getvalue(),
                        delimiter,
                        encoding,
                    )
                except PipelineError as exc:
                    if wizard.get("source_key"):
                        clear_dataset()
                    st.error(str(exc))
                except Exception:
                    if wizard.get("source_key"):
                        clear_dataset()
                    st.error(
                        "Could not parse this file as CSV. Check the delimiter, "
                        "encoding, and header row."
                    )
                else:
                    if parsed.empty:
                        st.error("The dataset has no rows.")
                    else:
                        original_rows = len(parsed)
                        notice = None
                        if original_rows > MAX_TRAIN_ROWS:
                            parsed, _ = limit_rows(parsed)
                            notice = (
                                f"The dataset has {original_rows:,} rows. A random "
                                f"sample of {MAX_TRAIN_ROWS:,} rows is used to stay "
                                "within the free-tier limit."
                            )
                        source_key = (
                            f"upload::{uploaded.name}::{uploaded.size}::"
                            f"{delimiter}::{encoding}"
                        )
                        adopt_dataset(
                            parsed,
                            uploaded.name,
                            source_key,
                            original_rows,
                            notice,
                        )
                        loaded = True
        else:
            sample_options = ["Classification", "Regression"]
            stored_kind = wizard.get("sample_kind_value", "Classification")
            if stored_kind not in sample_options:
                stored_kind = "Classification"
            sample_kind = st.selectbox(
                "Sample dataset",
                sample_options,
                index=sample_options.index(stored_kind),
                key="wz_sample_kind",
                on_change=remember_sample_kind,
            )
            sample_path = (
                Path(__file__).parent / "data" / f"sample_{sample_kind.lower()}.csv"
            )
            try:
                parsed = load_sample_csv(str(sample_path))
            except Exception:
                st.error("The sample dataset could not be loaded.")
            else:
                adopt_dataset(
                    parsed,
                    f"Sample {sample_kind.lower()} dataset",
                    f"sample::{sample_kind}",
                    len(parsed),
                    None,
                )
                loaded = True
            st.caption(
                "Classification: customer churn with contract, tenure, support "
                "calls, and spend. Regression: home prices from size, age, "
                "location, and garage."
            )

    df = wizard.get("df") if loaded else None
    if df is None or df.empty:
        wizard.render_footer(1, False, "Load a CSV or the sample dataset to continue.")
        return

    source_name = wizard.get("source_name")
    source_key = wizard.get("source_key")
    if wizard.get("row_notice"):
        st.warning(wizard.get("row_notice"))

    st.subheader("Target and features")
    with st.container(border=True):
        previous_target = wizard.get("target")
        target_options = list(df.columns)
        default_target = (
            previous_target if previous_target in target_options else target_options[-1]
        )
        target = st.selectbox(
            "Target column",
            target_options,
            index=target_options.index(default_target),
            key=f"wz_target_select_{source_key}",
        )
        target_changed = target != previous_target
        if target_changed and previous_target is not None:
            wizard.put("target", target)
            wizard.put("features", [column for column in target_options if column != target])
            wizard.invalidate_from(
                2,
                f"Target changed to `{target}`. Preprocessing, split, and results were reset.",
            )
        else:
            wizard.put("target", target)

        target_error = None
        detected = None
        if df[target].notna().sum() == 0:
            target_error = "The selected target column has no values."
        else:
            try:
                detected = infer_task_type(df[target])
            except PipelineError as exc:
                target_error = str(exc)

        task = None
        if target_error is None:
            stored_task = wizard.get("task")
            if target_changed or stored_task is None:
                stored_task = detected
            task_label = st.radio(
                "Task type",
                ["Classification", "Regression"],
                index=0 if stored_task == "classification" else 1,
                horizontal=True,
                key=f"wz_task_{source_key}_{target}",
                help="Auto-detected from the target. You can override it.",
            )
            task = task_label.lower()
            if task != wizard.get("task") and wizard.get("task") is not None:
                wizard.put("task", task)
                wizard.invalidate_from(
                    2,
                    "Task type changed. Preprocessing, split, and results were reset.",
                )
            else:
                wizard.put("task", task)
            st.caption(f"Auto-detected as **{detected}**. Override above if needed.")

    if target_error is not None:
        st.error(target_error)
        wizard.render_footer(1, False, "Choose a valid target column.")
        return

    available_features = [column for column in df.columns if column != target]
    stored_features = [
        column for column in wizard.get("features", []) if column in available_features
    ]
    if target_changed or not stored_features:
        stored_features = available_features
    features = st.multiselect(
        "Feature columns",
        available_features,
        default=stored_features,
        key=f"wz_features_{source_key}_{target}",
        help="Leave out IDs, free-text fields, and columns unavailable at prediction time.",
    )
    if set(features) != set(stored_features):
        wizard.invalidate_from(
            2,
            "Feature selection changed. Preprocessing, split, and results were reset.",
        )
    wizard.put("features", features)

    warnings = []
    profile = preprocess.profile_columns(df)
    selected_profile = profile.loc[profile["column"].isin(features)]
    flagged = selected_profile.loc[selected_profile["flags"] != ""]
    for _, row in flagged.iterrows():
        warnings.append(f"`{row['column']}`: {row['flags'].lower()}.")
    if not features:
        warnings.append("Select at least one feature.")
    if len(features) > preprocess.MAX_RECIPE_COLUMNS:
        warnings.append(
            f"Select at most {preprocess.MAX_RECIPE_COLUMNS} features so the "
            "recipe table stays usable."
        )
    for message in warnings:
        st.warning(message)

    st.subheader("Dataset at a glance")
    with st.container(border=True):
        overview_columns = st.columns(4)
        overview_columns[0].metric("Rows", f"{len(df):,}")
        overview_columns[1].metric("Columns", f"{df.shape[1]:,}")
        overview_columns[2].metric(
            "Missing cells",
            f"{df.isna().sum().sum() / max(df.shape[0] * df.shape[1], 1) * 100:.1f}%",
        )
        overview_columns[3].metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")
        st.caption(
            f"**{source_name}** · target `{target}` treated as {task} · "
            f"{len(features)} feature columns selected."
        )
        display_profile = profile.copy()
        display_profile["missing_pct"] = display_profile["missing_pct"].map(
            lambda value: f"{value:.1f}%"
        )
        st.dataframe(
            display_profile.rename(
                columns={
                    "column": "Column",
                    "dtype": "Dtype",
                    "kind": "Type",
                    "missing": "Missing",
                    "missing_pct": "Missing %",
                    "unique": "Unique",
                    "flags": "Flags",
                }
            ),
            hide_index=True,
            width="stretch",
        )

    view = st.segmented_control(
        "Data view",
        DATA_VIEWS,
        default="Overview",
        selection_mode="single",
        key="wz_data_view",
        label_visibility="collapsed",
    )
    view = view or "Overview"
    if view == "Overview":
        with st.container(border=True):
            st.subheader("First rows")
            st.dataframe(df.head(10), hide_index=True, width="stretch")
        st.info(
            "Open **Explore** for distributions, target patterns, data quality, "
            "and feature associations."
        )
    else:
        render_explore(df, target, task)

    wizard.summary_card(
        [
            ("Source", source_name or "n/a"),
            ("Shape", f"{len(df):,} × {df.shape[1]:,}"),
            ("Target", f"`{target}`"),
            ("Task", str(task).title()),
            ("Features", f"{len(features)} selected"),
        ]
    )
    wizard.render_footer(
        1,
        bool(features)
        and target_error is None
        and len(features) <= preprocess.MAX_RECIPE_COLUMNS,
        "Select at least one feature column, up to the recipe limit.",
    )


def render_preprocess_step():
    wizard.render_header(
        2,
        "Preprocess",
        "Choose how each column is cleaned, filled, encoded, and scaled. "
        "The final transforms are fitted on training rows only.",
    )
    df = wizard.get("df")
    features = wizard.get("features", [])
    target = wizard.get("target")
    task = wizard.get("task")
    if df is None or not features or target is None or task is None:
        st.warning("Complete the Data step first.")
        wizard.render_footer(2, False, "Return to Data and choose a target and features.")
        return

    recipe = wizard.get("recipe")
    if not recipe or [item.get("name") for item in recipe.get("columns", [])] != list(features):
        recipe = preprocess.sync_recipe(
            recipe or preprocess.default_recipe(df, features),
            df,
            features,
        )
        wizard.put("recipe", recipe)
        wizard.put("recipe_applied", False)

    st.subheader("Quality scan")
    with st.container(border=True):
        quality = preprocess.profile_columns(df[features + [target]])
        quality["missing_pct"] = quality["missing_pct"].map(lambda value: f"{value:.1f}%")
        st.dataframe(
            quality.rename(
                columns={
                    "column": "Column",
                    "dtype": "Dtype",
                    "kind": "Suggested type",
                    "missing": "Missing",
                    "missing_pct": "Missing %",
                    "unique": "Unique",
                    "flags": "Flags",
                }
            ),
            hide_index=True,
            width="stretch",
        )
        numeric = [
            item["name"]
            for item in recipe.get("columns", [])
            if item.get("kind") == preprocess.NUMERIC
        ]
        if numeric:
            outliers = pd.DataFrame(
                [preprocess.outlier_summary(df, column) for column in numeric]
            )
            outliers = outliers.loc[outliers["outliers"] > 0]
            if not outliers.empty:
                st.markdown("**IQR outlier counts**")
                display = outliers.copy()
                display["lower"] = display["lower"].map(lambda value: f"{value:,.2f}")
                display["upper"] = display["upper"].map(lambda value: f"{value:,.2f}")
                display["outlier_pct"] = display["outlier_pct"].map(
                    lambda value: f"{value:.1f}%"
                )
                st.dataframe(
                    display.rename(
                        columns={
                            "column": "Column",
                            "lower": "Lower fence",
                            "upper": "Upper fence",
                            "outliers": "Rows outside",
                            "outlier_pct": "% of rows",
                        }
                    ),
                    hide_index=True,
                    width="stretch",
                )

    if wizard.get("reset_recipe_globals"):
        for reset_key in (
            "wz_onehot_cap",
            "wz_rare_freq",
            "wz_variance",
            "wz_correlation",
        ):
            st.session_state.pop(reset_key, None)
        wizard.put("reset_recipe_globals", False)

    st.subheader("Column recipe")
    with st.container(border=True):
        st.caption(
            "Row-level cleaning (target drops, missing-row drops, outlier removal) "
            "happens before the split. Imputation, clipping, encoding, scaling, "
            "and filters learn from training rows only."
        )
        global_columns = st.columns(4)
        onehot_cap = global_columns[0].number_input(
            "One-hot max categories",
            min_value=5,
            max_value=60,
            value=int(recipe.get("onehot_max_categories", 20)),
            step=5,
            key="wz_onehot_cap",
            on_change=recipe_changed,
        )
        rare_frequency = global_columns[1].number_input(
            "Rare category threshold (%)",
            min_value=0.0,
            max_value=50.0,
            value=float(recipe.get("rare_min_frequency", 0.01)) * 100,
            step=0.5,
            key="wz_rare_freq",
            on_change=recipe_changed,
        )
        variance_threshold = global_columns[2].number_input(
            "Variance filter",
            min_value=0.0,
            max_value=1.0,
            value=float(recipe.get("variance_threshold", 0.0)),
            step=0.01,
            format="%.2f",
            key="wz_variance",
            on_change=recipe_changed,
            help="Drop transformed columns at or below this variance. 0 keeps all non-constant columns.",
        )
        correlation_threshold = global_columns[3].slider(
            "Correlation filter",
            min_value=0.5,
            max_value=1.0,
            value=float(recipe.get("correlation_threshold", 1.0)),
            step=0.01,
            key="wz_correlation",
            on_change=recipe_changed,
            help="Drop later columns with absolute correlation at or above this value. 1 disables the filter.",
        )

        editor_version = int(wizard.get("recipe_version", 0))
        frame = recipe_editor_frame(recipe)
        edited = st.data_editor(
            frame,
            hide_index=True,
            width="stretch",
            num_rows="fixed",
            disabled=["column"],
            key=f"wz_recipe_editor_{editor_version}",
            on_change=recipe_changed,
            column_config={
                "column": st.column_config.TextColumn("Column", width="medium"),
                "type": st.column_config.SelectboxColumn(
                    "Type",
                    options=[preprocess.NUMERIC, preprocess.CATEGORICAL],
                ),
                "missing": st.column_config.SelectboxColumn(
                    "Missing",
                    options=MISSING_OPTIONS,
                ),
                "constant": st.column_config.TextColumn("Constant fill"),
                "outlier": st.column_config.SelectboxColumn(
                    "Outliers",
                    options=preprocess.OUTLIER_ACTIONS,
                ),
                "encoding": st.column_config.SelectboxColumn(
                    "Encoding",
                    options=preprocess.ENCODINGS,
                ),
                "scaling": st.column_config.SelectboxColumn(
                    "Scaling",
                    options=preprocess.SCALINGS,
                ),
                "rare": st.column_config.SelectboxColumn(
                    "Rare values",
                    options=preprocess.RARE_ACTIONS,
                ),
                "category_order": st.column_config.TextColumn(
                    "Category order",
                    help="Optional comma-separated order for ordinal encoding.",
                ),
            },
        )
        candidate = preprocess.recipe_from_editor(edited, recipe)
        candidate["onehot_max_categories"] = int(onehot_cap)
        candidate["rare_min_frequency"] = float(rare_frequency) / 100
        candidate["variance_threshold"] = float(variance_threshold)
        candidate["correlation_threshold"] = float(correlation_threshold)

        errors, warnings = preprocess.validate_recipe(df, features, candidate)
        for message in errors:
            st.error(message)
        for message in warnings:
            st.warning(message)

        action_columns = st.columns([1, 1, 2])
        apply_clicked = action_columns[0].button(
            "Apply recipe",
            type="primary",
            width="stretch",
            disabled=bool(errors),
            key="wz_apply_recipe",
        )
        reset_clicked = action_columns[1].button(
            "Reset recipe",
            width="stretch",
            key="wz_reset_recipe",
        )
        action_columns[2].download_button(
            "Download recipe JSON",
            data=preprocess.recipe_to_json(candidate).encode("utf-8"),
            file_name="preprocessing_recipe.json",
            mime="application/json",
            width="stretch",
        )
        if reset_clicked:
            wizard.put("recipe", preprocess.default_recipe(df, features))
            wizard.put("recipe_applied", False)
            wizard.put("reset_recipe_globals", True)
            wizard.invalidate_from(3, "Recipe reset. Apply or continue with the defaults.")
            bump_recipe_version()
            st.rerun()
        if apply_clicked and not errors:
            try:
                cleaned, summary, preview, transformer = preprocess.preview_transform(
                    df,
                    features,
                    target,
                    task,
                    candidate,
                )
            except Exception as exc:
                st.error(f"Could not apply this recipe: {exc}")
            else:
                wizard.put("recipe", candidate)
                wizard.put("clean_summary", summary)
                wizard.put("recipe_applied", True)
                wizard.invalidate_from(3)
                wizard.complete(2)
                wizard.set_flash(
                    f"Recipe applied. {summary['rows']:,} rows remain; "
                    f"{preview.shape[1]} transformed columns are produced.",
                    "success",
                )
                st.rerun()

    stored_recipe = wizard.get("recipe") or candidate
    unapplied = safe_json(candidate) != safe_json(stored_recipe)
    if unapplied:
        st.info("The recipe table has unapplied changes. Apply them before continuing.")

    if wizard.get("recipe_applied"):
        try:
            cleaned, summary, preview, transformer = preprocess.preview_transform(
                df,
                features,
                target,
                task,
                stored_recipe,
            )
        except Exception as exc:
            st.error(f"The applied recipe could not be previewed: {exc}")
            summary = None
            preview = None
        else:
            st.subheader("Before and after")
            st.caption(
                "The after-table is an illustrative preview of up to 5,000 rows. When "
                "you train, the same recipe is fitted on the training split only."
            )
            before_columns, after_columns = st.columns(2)
            with before_columns:
                st.markdown("**Before · first 5 rows**")
                st.dataframe(cleaned[features].head(5), hide_index=True, width="stretch")
            with after_columns:
                st.markdown("**After · first 5 transformed rows**")
                st.dataframe(preview.head(5), hide_index=True, width="stretch")
            st.caption(
                f"Dropped {summary['dropped_total']:,} rows "
                f"(target {summary['dropped_target']:,}, missing "
                f"{summary['dropped_missing']:,}, outlier {summary['dropped_outlier']:,}). "
                f"Output: {preview.shape[1]:,} numeric columns."
            )

    clean_summary = wizard.get("clean_summary") or {}
    wizard.summary_card(
        [
            ("Rows kept", f"{clean_summary.get('rows', len(df)):,}"),
            ("Rows dropped", f"{clean_summary.get('dropped_total', 0):,}"),
            ("Output columns", f"{len(stored_recipe.get('columns', []))} input → "
             f"{preview.shape[1] if wizard.get('recipe_applied') and preview is not None else 'n/a'}"),
            ("Recipe status", "Applied" if wizard.get("recipe_applied") else "Not applied"),
        ]
    )
    wizard.render_footer(
        2,
        bool(wizard.get("recipe_applied")) and not unapplied and not errors,
        "Apply a valid recipe to continue.",
    )


def render_split_step():
    wizard.render_header(
        3,
        "Split",
        "Hold out a test set, check the balance, and keep the split-ready indices "
        "for training and future cluster analysis.",
    )
    df = wizard.get("df")
    recipe = wizard.get("recipe")
    features = wizard.get("features", [])
    target = wizard.get("target")
    task = wizard.get("task")
    if df is None or recipe is None or not features:
        st.warning("Complete the Data and Preprocess steps first.")
        wizard.render_footer(3, False, "Preprocessing must be applied first.")
        return
    try:
        X, y, summary, target_meta, encoded = preprocess.prepare_xy(
            df,
            features,
            target,
            task,
            recipe,
        )
    except Exception as exc:
        st.error(f"Could not prepare the split: {exc}")
        wizard.render_footer(3, False, "Fix the preprocessing recipe first.")
        return
    wizard.put("clean_summary", summary)
    wizard.put("target_meta", target_meta)

    with st.container(border=True):
        st.subheader("Split settings")
        setting_columns = st.columns(3)
        test_size = setting_columns[0].slider(
            "Test set size",
            min_value=0.1,
            max_value=0.5,
            value=0.2,
            step=0.05,
            key="wz_test_size",
            on_change=split_settings_changed,
        )
        random_state = setting_columns[1].number_input(
            "Random seed",
            min_value=0,
            max_value=2_147_483_647,
            value=42,
            step=1,
            key="wz_random_state",
            on_change=split_settings_changed,
        )
        stratify = setting_columns[2].checkbox(
            "Stratify split",
            value=task == "classification",
            disabled=task != "classification",
            help="Keep class proportions similar in train and test.",
            key="wz_stratify",
            on_change=split_settings_changed,
        )
        if task == "regression":
            stratify = False
        st.caption(
            f"{summary['rows']:,} rows are ready after row-level cleaning. "
            f"{summary['dropped_total']:,} rows were dropped."
        )
        if st.button("Create split", type="primary", key="wz_create_split"):
            indices = np.arange(len(X))
            stratify_values = encoded if (stratify and task == "classification") else None
            if stratify_values is not None:
                counts = pd.Series(encoded).value_counts()
                if not counts.empty and int(counts.min()) < 2:
                    st.error("A stratified split needs at least two rows in every class.")
                else:
                    try:
                        train_indices, test_indices = train_test_split(
                            indices,
                            test_size=float(test_size),
                            random_state=int(random_state),
                            stratify=stratify_values,
                        )
                    except ValueError:
                        st.error(
                            "Could not create the split. Try a different test size, "
                            "seed, or disable stratification."
                        )
                    else:
                        wizard.put(
                            "split",
                            {
                                "train": train_indices,
                                "test": test_indices,
                                "rows": int(len(X)),
                                "test_size": float(test_size),
                                "random_state": int(random_state),
                                "stratify": bool(stratify),
                            },
                        )
                        wizard.invalidate_from(4)
                        wizard.complete(3)
                        wizard.set_flash(
                            f"Split created: {len(train_indices):,} train and "
                            f"{len(test_indices):,} test rows.",
                            "success",
                        )
                        st.rerun()
            else:
                try:
                    train_indices, test_indices = train_test_split(
                        indices,
                        test_size=float(test_size),
                        random_state=int(random_state),
                    )
                except ValueError:
                    st.error("Could not create the split with these settings.")
                else:
                    wizard.put(
                        "split",
                        {
                            "train": train_indices,
                            "test": test_indices,
                            "rows": int(len(X)),
                            "test_size": float(test_size),
                            "random_state": int(random_state),
                            "stratify": bool(stratify),
                        },
                    )
                    wizard.invalidate_from(4)
                    wizard.complete(3)
                    wizard.set_flash(
                        f"Split created: {len(train_indices):,} train and "
                        f"{len(test_indices):,} test rows.",
                        "success",
                    )
                    st.rerun()

    split = wizard.get("split")
    split_matches = (
        split is not None
        and int(split.get("rows", -1)) == len(X)
        and abs(float(split.get("test_size", -1)) - float(test_size)) < 1e-9
        and int(split.get("random_state", -1)) == int(random_state)
        and bool(split.get("stratify", False)) == bool(stratify)
    )
    if split is not None and not split_matches:
        st.warning("The split settings changed. Create the split again before continuing.")
    if split_matches:
        train_indices = np.asarray(split["train"])
        test_indices = np.asarray(split["test"])
        train_y = y.iloc[train_indices]
        test_y = y.iloc[test_indices]
        metric_columns = st.columns(3)
        metric_columns[0].metric("Train rows", f"{len(train_indices):,}")
        metric_columns[1].metric("Test rows", f"{len(test_indices):,}")
        metric_columns[2].metric(
            "Test share",
            f"{len(test_indices) / max(len(X), 1) * 100:.0f}%",
        )
        if len(test_indices) < 50:
            st.warning(
                f"The test set has only {len(test_indices):,} rows. Treat test "
                "metrics as noisy and rely on the cross-validation panel in Results."
            )
        with st.container(border=True):
            render_figure(
                plots.split_distribution(
                    train_y,
                    test_y,
                    target,
                    task,
                )
            )
            if task == "classification":
                train_counts = train_y.astype(str).value_counts()
                test_counts = test_y.astype(str).value_counts()
                table = pd.DataFrame(
                    {
                        "Class": train_counts.index,
                        "Train": train_counts.to_numpy(dtype=int),
                        "Test": [int(test_counts.get(label, 0)) for label in train_counts.index],
                    }
                )
                table["Train %"] = (table["Train"] / max(len(train_y), 1) * 100).round(1)
                table["Test %"] = (table["Test"] / max(len(test_y), 1) * 100).round(1)
                st.dataframe(table, hide_index=True, width="stretch")
            else:
                st.caption(
                    f"Train mean {train_y.mean():,.2f} · test mean "
                    f"{test_y.mean():,.2f} · train median {train_y.median():,.2f} · "
                    f"test median {test_y.median():,.2f}."
                )
    wizard.summary_card(
        [
            ("Ready rows", f"{summary['rows']:,}"),
            ("Dropped", f"{summary['dropped_total']:,}"),
            ("Test size", f"{float(test_size):.0%}"),
            ("Seed", f"{int(random_state)}"),
            ("Split status", "Created" if split_matches else "Not created"),
        ]
    )
    wizard.render_footer(3, bool(split_matches), "Create a split that matches the current settings.")


def render_single_hyperparameters(model_name, specs):
    values = {}
    for name, spec in specs.items():
        key = f"wz_hp_{model_name}_{name}"
        if spec["type"] == "select":
            options = list(spec["options"])
            values[name] = st.selectbox(
                name,
                options,
                index=options.index(spec["default"]),
                help=spec["help"],
                key=key,
                format_func=str,
                on_change=model_settings_changed,
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
                on_change=model_settings_changed,
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
                on_change=model_settings_changed,
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
            key=f"wz_search_hp_{model_name}_{name}",
            format_func=str,
            on_change=model_settings_changed,
        )
    return grid


def render_model_step():
    wizard.render_header(
        4,
        "Model",
        "Pick an estimator, tune it, and train on the training split only.",
    )
    df = wizard.get("df")
    recipe = wizard.get("recipe")
    split = wizard.get("split")
    features = wizard.get("features", [])
    target = wizard.get("target")
    task = wizard.get("task")
    if df is None or recipe is None or split is None or not features:
        st.warning("Complete the earlier steps first.")
        wizard.render_footer(4, False, "A dataset, recipe, and split are required.")
        return
    try:
        X, y, summary, target_meta, encoded = preprocess.prepare_xy(
            df,
            features,
            target,
            task,
            recipe,
        )
    except Exception as exc:
        st.error(f"Could not prepare the model data: {exc}")
        wizard.render_footer(4, False, "Fix the preprocessing recipe first.")
        return
    if int(split.get("rows", -1)) != len(X):
        st.warning("The saved split no longer matches the cleaned data. Recreate it in Split.")
        wizard.render_footer(4, False, "Return to Split and recreate the split.")
        return

    model_options = models_for_task(task)
    if not model_options:
        st.error("No models are registered for this task type.")
        wizard.render_footer(4, False, "No model is available.")
        return
    default_model = (
        "Logistic Regression"
        if task == "classification" and "Logistic Regression" in model_options
        else "Ridge Regression"
        if task == "regression" and "Ridge Regression" in model_options
        else model_options[0]
    )
    stored_model = wizard.get("model_name")
    if stored_model not in model_options:
        stored_model = default_model
    model_name = st.selectbox(
        "Model",
        model_options,
        index=model_options.index(stored_model),
        key=f"wz_model_select_{task}",
        on_change=model_settings_changed,
    )
    wizard.put("model_name", model_name)
    if wizard.get("reset_model_params"):
        for state_key in list(st.session_state.keys()):
            if str(state_key).startswith(("wz_hp_", "wz_search_hp_", "wz_search_mode_")):
                st.session_state.pop(state_key, None)
        wizard.put("reset_model_params", False)
    info = info_for(model_name)
    with st.container(border=True):
        st.subheader(model_name)
        st.write(info["summary"])
        info_columns = st.columns(2)
        with info_columns[0]:
            st.markdown("**Strengths**")
            st.markdown("\n".join(f"- {item}" for item in info["strengths"]))
        with info_columns[1]:
            st.markdown("**Watch out for**")
            st.markdown("\n".join(f"- {item}" for item in info["weaknesses"]))
        st.caption(f"Typical use: {info['use_cases']}")

    hyperparameters = hyperparameters_for(model_name)
    with st.container(border=True):
        header_columns = st.columns([3, 1])
        header_columns[0].subheader("Hyperparameters")
        if header_columns[1].button(
            "Reset defaults",
            key="wz_reset_model_params_button",
            width="stretch",
        ):
            wizard.put("reset_model_params", True)
            model_settings_changed()
            st.rerun()
        run_search = st.checkbox(
            "Tune with a small grid search",
            key=f"wz_search_mode_{task}_{model_name}",
            help=(
                "Search several hyperparameter values with cross-validation. "
                f"Maximum {MAX_SEARCH_COMBINATIONS} combinations."
            ),
            on_change=model_settings_changed,
        )
        search_grid = {}
        selected_params = {}
        search_combinations = None
        search_blocked = False
        if run_search:
            st.caption(
                "Choose one or more values per parameter. "
                f"Total combinations must be {MAX_SEARCH_COMBINATIONS} or fewer."
            )
            search_grid = render_search_hyperparameters(model_name, hyperparameters)
            search_combinations = count_combinations(search_grid)
            if search_combinations < 1:
                search_blocked = True
                st.error("Select at least one value for every hyperparameter.")
            elif search_combinations > MAX_SEARCH_COMBINATIONS:
                search_blocked = True
                st.error(
                    f"This search has {search_combinations} combinations; "
                    f"the limit is {MAX_SEARCH_COMBINATIONS}."
                )
            else:
                st.caption(
                    f"Search space: {search_combinations} combinations "
                    f"(limit {MAX_SEARCH_COMBINATIONS})."
                )
        else:
            selected_params = render_single_hyperparameters(model_name, hyperparameters)

    if (
        model_name in pipeline.SCALE_SENSITIVE_MODELS
        and not pipeline.has_numeric_scaling(recipe)
    ):
        st.warning(
            f"{model_name} is sensitive to feature scale, but this recipe has no "
            "numeric scaling. Add standard or min-max scaling in Preprocess."
        )

    parameters = search_grid if run_search else selected_params
    train_clicked = st.button(
        "Train model",
        type="primary",
        width="stretch",
        disabled=search_blocked,
        key="wz_train_model",
    )
    if train_clicked and not search_blocked:
        train_indices = np.asarray(split["train"])
        test_indices = np.asarray(split["test"])
        X_train = X.iloc[train_indices].reset_index(drop=True)
        X_test = X.iloc[test_indices].reset_index(drop=True)
        y_train = pd.Series(encoded).iloc[train_indices].reset_index(drop=True)
        y_test = pd.Series(encoded).iloc[test_indices].reset_index(drop=True)
        started = time.perf_counter()
        try:
            with st.spinner(
                "Running grid search and training..." if run_search else "Training model..."
            ):
                if run_search:
                    result = grid_search_with_recipe(
                        X_train,
                        y_train,
                        X_test,
                        y_test,
                        task=task,
                        model_name=model_name,
                        param_grid=search_grid,
                        recipe=recipe,
                        random_state=int(split["random_state"]),
                        dropped_rows=summary["dropped_total"],
                        split_meta={
                            "test_size": split["test_size"],
                            "random_state": split["random_state"],
                            "stratify": split["stratify"],
                        },
                    )
                else:
                    result = train_with_recipe(
                        X_train,
                        y_train,
                        X_test,
                        y_test,
                        task=task,
                        model_name=model_name,
                        params=selected_params,
                        recipe=recipe,
                        random_state=int(split["random_state"]),
                        dropped_rows=summary["dropped_total"],
                        split_meta={
                            "test_size": split["test_size"],
                            "random_state": split["random_state"],
                            "stratify": split["stratify"],
                        },
                    )
            elapsed = time.perf_counter() - started
        except PipelineError as exc:
            st.error(str(exc))
        except Exception:
            st.error(
                "Training failed unexpectedly. Check the recipe, split, and "
                "hyperparameters, then try again."
            )
        else:
            result = preprocess.decode_classification_result(result, target_meta)
            result["elapsed_seconds"] = elapsed
            result["signature"] = model_signature(
                wizard.get("source_key"),
                target,
                task,
                features,
                recipe,
                split,
                model_name,
                parameters,
                run_search,
            )
            result["recipe"] = recipe
            wizard.put("result", result)
            wizard.put("result_signature", result["signature"])
            wizard.put("cv_result", None)
            wizard.put("trained_at", time.strftime("%Y-%m-%d %H:%M:%S"))
            wizard.complete(4)
            wizard.go(5)
            wizard.set_flash(f"Training completed in {elapsed:.2f} seconds.", "success")
            st.rerun()

    wizard.summary_card(
        [
            ("Model", model_name),
            ("Task", str(task).title()),
            ("Train rows", f"{len(split['train']):,}"),
            ("Test rows", f"{len(split['test']):,}"),
            ("Search", f"{search_combinations} combos" if run_search else "Off"),
        ]
    )
    wizard.render_footer(
        4,
        wizard.get("result") is not None,
        "Train a model to unlock Results.",
        next_label="View results",
    )


def metrics_frame(result):
    rows = []
    for name, value in result["metrics"].items():
        rows.append(
            {
                "model": result["model_name"],
                "task": result["task"],
                "scope": "model",
                "metric": name,
                "value": float(value),
            }
        )
    baseline = result.get("baseline", {})
    if baseline.get("name") == "Majority class":
        for name in ("accuracy", "f1"):
            if name in baseline:
                rows.append(
                    {
                        "model": result["model_name"],
                        "task": result["task"],
                        "scope": "baseline",
                        "metric": f"majority_{name}",
                        "value": float(baseline[name]),
                    }
                )
    elif baseline.get("name") == "Mean predictor":
        for name in ("r2", "mae"):
            if name in baseline:
                rows.append(
                    {
                        "model": result["model_name"],
                        "task": result["task"],
                        "scope": "baseline",
                        "metric": f"mean_{name}",
                        "value": float(baseline[name]),
                    }
                )
    return pd.DataFrame(rows)


def render_key_insights(insights):
    with st.container(border=True):
        st.subheader("Key insights")
        st.markdown("\n".join(f"- {insight}" for insight in insights))


def render_explore(df, target, task):
    try:
        analysis = cached_explore(df, target, task)
    except Exception:
        st.error(
            "Could not compute the exploratory analysis for this dataset. "
            "Try a different target or check unusual column values."
        )
        return
    st.subheader("What the data is telling us")
    st.write(
        "Statistics use the full loaded dataset. Charts use a random sample "
        "of at most 5,000 rows so the page stays responsive."
    )
    render_key_insights(analysis["insights"])
    st.markdown("### Target profile")
    render_figure(plots.target_distribution(analysis["target"]))
    st.markdown("### Data quality")
    quality_columns = st.columns(2)
    with quality_columns[0]:
        if analysis["missing"].empty:
            st.success("No missing values were found.")
        else:
            render_figure(plots.missing_values(analysis["missing"]))
    with quality_columns[1]:
        st.markdown("**IQR outlier summary**")
        if analysis["outliers"].empty:
            st.success("No IQR outliers were detected.")
        else:
            outliers = analysis["outliers"].head(6).copy()
            outliers["outliers"] = outliers["outliers"].map(lambda value: f"{value:,}")
            outliers["outlier_pct"] = outliers["outlier_pct"].map(
                lambda value: f"{value:.1f}%"
            )
            outliers = outliers.rename(
                columns={
                    "column": "Column",
                    "outliers": "Rows outside fence",
                    "outlier_pct": "% of rows",
                }
            )
            st.dataframe(
                outliers[["Column", "Rows outside fence", "% of rows"]],
                hide_index=True,
                width="stretch",
            )
            st.caption(
                "Fence: Q1 − 1.5 × IQR to Q3 + 1.5 × IQR. "
                "Outliers are signals to inspect, not automatic errors."
            )
    if analysis["histograms"]:
        st.markdown("### Numeric distributions")
        render_figure(plots.histogram_grid(analysis["histograms"]))
        st.caption(
            "Skew is calculated on the full column; the histogram uses the "
            "5,000-row plot sample."
        )
    if analysis["grouped_numeric"]:
        st.markdown("### How numeric features move with the target")
        render_figure(
            plots.numeric_by_target(
                analysis["grouped_numeric"],
                target,
                task,
            )
        )
    if analysis["categorical_charts"]:
        st.markdown("### Categorical patterns")
        render_figure(
            plots.categorical_target(
                analysis["categorical_charts"],
                target,
                task,
                analysis["target"].get("positive_class"),
            )
        )
    st.markdown("### Which features carry signal?")
    signal_columns = st.columns([1.08, 1])
    with signal_columns[0]:
        if analysis["correlation"] is not None:
            render_figure(plots.correlation_heatmap(analysis["correlation"], target))
            st.caption(
                "At most 15 numeric columns, chosen by absolute correlation "
                "with the target."
            )
        else:
            st.info("The dataset has no numeric features for a correlation heatmap.")
    with signal_columns[1]:
        if analysis["associations"]:
            render_figure(plots.association_ranking(analysis["associations"], target))
            st.caption(
                "Blue bars are numeric correlations or eta²; green bars are "
                "categorical association measures."
            )
        else:
            st.info("No features were available for association ranking.")


def render_feature_importance(result, target):
    st.markdown("### What drives the prediction")
    importance = result.get("feature_importance")
    if not isinstance(importance, pd.DataFrame) or importance.empty:
        st.info(
            "Feature importance was not available for this model and run. "
            "Try a linear model, tree model, or a larger dataset."
        )
        return
    if not result.get("feature_importance_complete", False):
        st.warning(
            "Feature importance is partial: the 20-second permutation budget "
            f"was reached after {result.get('feature_importance_seconds', 0):.1f} seconds."
        )
    render_figure(plots.feature_importance_chart(importance, target))
    st.caption(
        f"Method: {result.get('feature_importance_method', 'unknown')}. "
        "Importance describes model behaviour, not causation."
    )


def render_classification_results(result, target):
    metrics = result["metrics"]
    baseline = result["baseline"]
    per_class = result["per_class"]
    positive = result.get("positive_class")
    positive_recall = None
    if positive is not None and isinstance(per_class, pd.DataFrame):
        row = per_class.loc[per_class["class"] == str(positive)]
        if not row.empty:
            positive_recall = float(row.iloc[0]["recall"])
    if np.unique(np.asarray(result.get("y_pred", []))).size == 1:
        st.warning(
            "The model predicted only one class. Check feature scaling, the "
            "target, and class balance before trusting these results."
        )
    metric_columns = st.columns(4)
    metric_columns[0].metric("Accuracy", score_text(metrics["accuracy"]))
    metric_columns[1].metric("Weighted F1", score_text(metrics["f1"]))
    metric_columns[2].metric(
        "ROC AUC", score_text(metrics["roc_auc"]) if "roc_auc" in metrics else "n/a"
    )
    metric_columns[3].metric(
        f"Recall · class {positive}" if positive is not None else "Classes",
        score_text(positive_recall)
        if positive_recall is not None
        else f"{len(result['classes'])}",
    )
    with st.container(border=True):
        st.subheader("Baseline check")
        baseline_columns = st.columns(3)
        baseline_columns[0].metric(
            "Majority-class accuracy", score_text(baseline["accuracy"])
        )
        baseline_columns[1].metric(
            "Model accuracy",
            score_text(metrics["accuracy"]),
            delta=f"{baseline['accuracy_uplift'] * 100:+.1f} pts",
        )
        baseline_columns[2].metric("Weighted F1 uplift", f"{baseline['f1_uplift']:+.2f}")
        st.caption(
            f"Accuracy is {baseline['accuracy_uplift'] * 100:+.1f} points versus "
            "always predicting the most common training class."
        )
    with st.container(border=True):
        st.subheader("What this means")
        st.markdown(" ".join(pipeline.result_narrative(result)))
    st.markdown("### How the model behaves")
    behaviour_columns = st.columns(2)
    with behaviour_columns[0]:
        view = st.radio(
            "Confusion matrix view",
            ["Counts", "Row %"],
            horizontal=True,
            key="wz_confusion_view",
        )
        render_figure(
            plots.confusion_matrix_chart(
                result["confusion_matrix"],
                result["classes"],
                normalize=view == "Row %",
            )
        )
        st.caption("Rows are actual classes; columns are predictions.")
    with behaviour_columns[1]:
        if result.get("roc") is not None:
            render_figure(plots.roc_curve(result["roc"]))
            st.caption("The diagonal is random guessing; higher is better.")
        else:
            st.info(
                "ROC is shown for binary targets where the model exposes "
                "probabilities or decision scores."
            )
    if result.get("precision_recall") is not None or result.get("y_proba") is not None:
        st.markdown("### Ranking and confidence")
        ranking_columns = st.columns(2)
        with ranking_columns[0]:
            if result.get("precision_recall") is not None:
                render_figure(
                    plots.precision_recall_curve_chart(
                        result["precision_recall"],
                        result["precision_recall"]["baseline"],
                    )
                )
                st.caption(
                    "Precision-recall is often more informative than accuracy "
                    "when the positive class is uncommon."
                )
        with ranking_columns[1]:
            if result.get("y_proba") is not None:
                render_figure(
                    plots.probability_histogram(
                        result["y_test"],
                        result["y_proba"],
                        result["classes"],
                        target,
                        result.get("positive_class"),
                    )
                )
                if len(result["classes"]) == 2:
                    st.caption(
                        "Good separation means the two true-class distributions "
                        "sit on opposite sides of the 0.5 line."
                    )
                else:
                    st.caption(
                        "Each distribution shows the probability the model gave "
                        "to the actual class. Higher values are better."
                    )
    st.markdown("### Per-class performance")
    st.dataframe(
        per_class,
        hide_index=True,
        width="stretch",
        column_config={
            "precision": st.column_config.NumberColumn("Precision", format="%.3f"),
            "recall": st.column_config.NumberColumn("Recall", format="%.3f"),
            "f1": st.column_config.NumberColumn("F1", format="%.3f"),
            "support": st.column_config.NumberColumn("Support"),
        },
    )
    render_feature_importance(result, target)


def render_regression_results(result, target):
    metrics = result["metrics"]
    baseline = result["baseline"]
    metric_columns = st.columns(3)
    metric_columns[0].metric("R²", score_text(metrics["r2"]))
    metric_columns[1].metric("MAE", f"{metrics['mae']:,.0f}")
    metric_columns[2].metric("RMSE", f"{metrics['rmse']:,.0f}")
    with st.container(border=True):
        st.subheader("Baseline check")
        baseline_columns = st.columns(3)
        baseline_columns[0].metric("Mean-predictor R²", score_text(baseline["r2"]))
        baseline_columns[1].metric(
            "Model R²",
            score_text(metrics["r2"]),
            delta=f"{baseline['r2_uplift']:+.2f}",
        )
        baseline_columns[2].metric("MAE reduction", f"{baseline['mae_reduction_pct']:.1f}%")
        st.caption(
            f"R² is {baseline['r2_uplift']:+.2f} versus the mean predictor, "
            f"and MAE falls from {baseline['mae']:,.0f} to {metrics['mae']:,.0f}."
        )
    with st.container(border=True):
        st.subheader("What this means")
        st.markdown(" ".join(pipeline.result_narrative(result)))
    st.markdown("### How the model behaves")
    behaviour_columns = st.columns([1.05, 1])
    with behaviour_columns[0]:
        render_figure(
            plots.predicted_vs_actual(
                result["y_test"],
                result["y_pred"],
                metrics["r2"],
            )
        )
        st.caption("Points closer to the dashed line are more accurate.")
    with behaviour_columns[1]:
        st.markdown("**Largest test misses**")
        errors = result["largest_errors"].copy()
        errors["actual"] = errors["actual"].map(lambda value: f"{value:,.0f}")
        errors["predicted"] = errors["predicted"].map(lambda value: f"{value:,.0f}")
        errors["absolute_error"] = errors["absolute_error"].map(lambda value: f"{value:,.0f}")
        errors = errors.rename(
            columns={
                "row": "Row",
                "actual": "Actual",
                "predicted": "Predicted",
                "absolute_error": "Absolute error",
            }
        )
        st.dataframe(errors, hide_index=True, width="stretch")
    render_figure(plots.residuals_plot(result["y_test"], result["y_pred"]))
    st.caption(
        "Residual = actual − predicted. A funnel or curve suggests the model "
        "misses a pattern rather than just random noise."
    )
    render_feature_importance(result, target)


def render_run_details(result):
    with st.expander("Run details and settings", expanded=False):
        detail_columns = st.columns(3)
        detail_columns[0].metric("Train rows", f"{result['train_rows']:,}")
        detail_columns[1].metric("Test rows", f"{result['test_rows']:,}")
        detail_columns[2].metric("Rows dropped", f"{result['rows_dropped']:,}")
        if result.get("rows_dropped"):
            st.info(
                f"{result['rows_dropped']:,} rows were dropped by target handling, "
                "missing-row removal, or outlier removal before the split."
            )
        if result.get("best_params") is not None:
            st.markdown("**Best cross-validation settings**")
            st.json(result["best_params"])
            st.caption(
                f"Best CV score: {result['best_cv_score']:.4f} · "
                f"{result['search_combinations']} combinations searched."
            )
        st.markdown("**Selected configuration**")
        st.json(
            {
                "target": wizard.get("target"),
                "task": result.get("task"),
                "model": result.get("model_name"),
                "features": wizard.get("features", []),
                "split": result.get("split", {}),
                "params": result.get("params"),
            }
        )
        st.markdown("**Preprocessing recipe**")
        st.code(preprocess.recipe_to_json(wizard.get("recipe", {})), language="json")
        st.download_button(
            "Download recipe JSON",
            data=preprocess.recipe_to_json(wizard.get("recipe", {})).encode("utf-8"),
            file_name="preprocessing_recipe.json",
            mime="application/json",
            key="wz_results_recipe_download",
        )
        st.caption(
            f"Trained at {wizard.get('trained_at', 'unknown time')} "
            f"in {result.get('elapsed_seconds', 0):.2f} seconds."
        )


def render_downloads(result):
    model_slug = slugify(result["model_name"])
    model_buffer = io.BytesIO()
    joblib.dump(result["model"], model_buffer)
    st.markdown("### Take the run with you")
    download_columns = st.columns(2)
    with download_columns[0]:
        st.download_button(
            "Download trained model (.joblib)",
            data=model_buffer.getvalue(),
            file_name=f"{model_slug}.joblib",
            mime="application/octet-stream",
            width="stretch",
            key="wz_download_model",
        )
    with download_columns[1]:
        st.download_button(
            "Download metrics summary (.csv)",
            data=metrics_frame(result).to_csv(index=False).encode("utf-8"),
            file_name=f"{model_slug}_metrics.csv",
            mime="text/csv",
            width="stretch",
            key="wz_download_metrics",
        )


def render_cv_panel(result):
    st.markdown("### Cross-validation on the training split")
    st.caption(
        "Cross-validation reuses only the training rows, so it never touches the "
        "held-out test set. It is the more reliable signal when the test set is small."
    )
    split = wizard.get("split")
    df = wizard.get("df")
    recipe = wizard.get("recipe")
    features = wizard.get("features", [])
    target = wizard.get("target")
    task = wizard.get("task")
    if split is None or df is None or recipe is None:
        st.info("A saved split and applied recipe are required for cross-validation.")
        return
    try:
        X, y, summary, target_meta, encoded = preprocess.prepare_xy(
            df,
            features,
            target,
            task,
            recipe,
        )
    except Exception as exc:
        st.error(f"Could not prepare cross-validation data: {exc}")
        return
    folds = st.slider(
        "Cross-validation folds",
        min_value=3,
        max_value=10,
        value=5,
        step=1,
        key="wz_cv_folds",
    )
    if st.button("Run cross-validation", key="wz_run_cv"):
        train_indices = np.asarray(split["train"])
        X_train = X.iloc[train_indices].reset_index(drop=True)
        y_train = pd.Series(encoded).iloc[train_indices].reset_index(drop=True)
        try:
            cv_result = pipeline.cross_validate_recipe(
                X_train,
                y_train,
                task=task,
                model_name=result["model_name"],
                params=result.get("params") or {},
                recipe=recipe,
                random_state=int(split["random_state"]),
                folds=int(folds),
            )
        except PipelineError as exc:
            st.error(str(exc))
        else:
            wizard.put("cv_result", cv_result)
            st.rerun()

    cv_result = wizard.get("cv_result")
    if cv_result is None:
        st.info("Run cross-validation to add fold-level stability to the test score.")
        if result.get("test_rows", 0) < 50:
            st.warning(
                f"The test set has only {result['test_rows']} rows. "
                "Cross-validation will give a steadier estimate."
            )
        return
    metric_columns = st.columns(3)
    metric_columns[0].metric(f"Mean {cv_result['metric']}", f"{cv_result['mean']:.3f}")
    metric_columns[1].metric("Std. dev.", f"{cv_result['std']:.3f}")
    metric_columns[2].metric("Folds used", f"{cv_result['used_folds']}")
    if cv_result.get("capped"):
        st.warning(
            f"Requested {cv_result['requested_folds']} folds, but "
            f"{cv_result['used_folds']} were used because of class counts, row "
            "counts, or the training rows × folds cap."
        )
    st.caption(
        f"{cv_result['rows']:,} training rows · {cv_result['used_folds']} folds · "
        f"mean {cv_result['mean']:.3f} ± {cv_result['std']:.3f}."
    )
    render_figure(plots.cv_score_chart(cv_result))


def render_results_step():
    wizard.render_header(
        5,
        "Results",
        "Read the held-out metrics, inspect errors and drivers, check fold stability, and download the fitted pipeline.",
    )
    result = wizard.get("result")
    if result is None:
        st.info("Train a model in the Model step to unlock results.")
        wizard.render_footer(5, False, "Train a model first.", final=True)
        return
    target = wizard.get("target")
    st.caption(
        f"**{result['model_name']}** · {result['task']} · "
        f"{result['test_rows']:,} held-out rows · trained in "
        f"{result.get('elapsed_seconds', 0):.2f}s"
    )
    if result.get("rows_dropped"):
        st.info(
            f"{result['rows_dropped']:,} rows were dropped before the split by "
            "target, missing-value, or outlier rules."
        )
    if result["task"] == "classification":
        render_classification_results(result, target)
    else:
        render_regression_results(result, target)
    render_cv_panel(result)
    render_run_details(result)
    render_downloads(result)
    wizard.summary_card(
        [
            ("Model", result["model_name"]),
            ("Test rows", f"{result['test_rows']:,}"),
            ("Trained", wizard.get("trained_at", "n/a")),
            ("Stable split", "Yes" if wizard.get("split") else "No"),
        ]
    )
    wizard.render_footer(5, False, final=True)


begin()
