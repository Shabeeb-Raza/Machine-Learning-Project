import streamlit as st

STEPS = [
    (1, "Data"),
    (2, "Preprocess"),
    (3, "Split"),
    (4, "Model"),
    (5, "Results"),
]
PREFIX = "wz_"


def key(name):
    return f"{PREFIX}{name}"


def init_state():
    defaults = {
        "step": 1,
        "completed": [],
        "flash": None,
        "df": None,
        "source_name": None,
        "source_key": None,
        "original_rows": 0,
        "row_notice": None,
        "target": None,
        "task": None,
        "features": [],
        "recipe": None,
        "recipe_applied": False,
        "clean_summary": None,
        "target_meta": None,
        "split": None,
        "result": None,
        "result_signature": None,
        "cv_result": None,
        "model_name": None,
        "trained_at": None,
        "data_applied": None,
        "sample_kind_value": "Classification",
        "data_mode_value": "Upload CSV",
        "delimiter_value": "Auto",
        "encoding_value": "Auto",
    }
    for name, value in defaults.items():
        st.session_state.setdefault(key(name), value)


def get(name, default=None):
    return st.session_state.get(key(name), default)


def put(name, value):
    st.session_state[key(name)] = value


def go(step):
    put("step", int(step))


def complete(step):
    completed = set(get("completed", []))
    completed.add(int(step))
    put("completed", sorted(completed))


def can_enter(step):
    step = int(step)
    if step == 1:
        return True
    completed = set(get("completed", []))
    return step in completed or (step - 1) in completed


def mark_incomplete_from(step):
    put("completed", [value for value in get("completed", []) if value < step])


def clear_recipe():
    put("recipe", None)
    put("recipe_applied", False)
    put("clean_summary", None)
    put("target_meta", None)


def clear_split():
    put("split", None)


def clear_results():
    put("result", None)
    put("result_signature", None)
    put("cv_result", None)
    put("trained_at", None)


def invalidate_from(step, message=None):
    step = int(step)
    mark_incomplete_from(step)
    if step <= 2:
        clear_recipe()
    if step <= 3:
        clear_split()
    if step <= 4:
        clear_results()
    if message:
        set_flash(message)


def reset_all(message=None):
    for name in (
        "df",
        "source_name",
        "source_key",
        "original_rows",
        "row_notice",
        "target",
        "task",
        "features",
        "recipe",
        "recipe_applied",
        "clean_summary",
        "target_meta",
        "split",
        "result",
        "result_signature",
        "cv_result",
        "model_name",
        "trained_at",
        "data_applied",
    ):
        put(name, [] if name == "features" else 0 if name == "original_rows" else None)
    put("completed", [])
    go(1)
    if message:
        set_flash(message)


def set_flash(message, level="info"):
    put("flash", {"message": message, "level": level})


def render_flash():
    flash = get("flash")
    if not flash:
        return
    message = flash.get("message", "")
    level = flash.get("level", "info")
    if level == "warning":
        st.warning(message)
    elif level == "error":
        st.error(message)
    elif level == "success":
        st.success(message)
    else:
        st.info(message)
    put("flash", None)


def apply_page_style():
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {
            display: none;
        }
        .block-container {padding-top: 2.2rem; max-width: 1180px;}
        div[data-testid="stMetric"] {
            background: #FFFFFF;
            border: 1px solid #E5E7EB;
            border-radius: 12px;
            padding: 0.75rem 0.9rem;
        }
        .wz-step {
            display:inline-block; padding:0.18rem 0.55rem; border-radius:999px;
            background:#E8F1F8; color:#0072B2; font-weight:600; font-size:0.78rem;
            letter-spacing:0.04em; text-transform:uppercase;
        }
        .wz-muted {color:#6B7280;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_stepper():
    active = int(get("step", 1))
    columns = st.columns(len(STEPS))
    for column, (number, label) in zip(columns, STEPS):
        enabled = can_enter(number) or number == active
        with column:
            clicked = st.button(
                f"{number} · {label}",
                key=key(f"nav_{number}"),
                type="primary" if number == active else "secondary",
                disabled=not enabled,
                width="stretch",
            )
            if clicked and number != active:
                go(number)
                st.rerun()


def render_header(step, title, description):
    st.markdown(f'<span class="wz-step">Step {step} of 5</span>', unsafe_allow_html=True)
    st.title(title)
    st.caption(description)


def summary_card(items):
    with st.container(border=True):
        st.markdown("**Step summary**")
        columns = st.columns(max(len(items), 1))
        for column, (label, value) in zip(columns, items):
            column.caption(label)
            column.markdown(f"**{value}**")


def render_footer(step, can_continue, reason=None, next_label="Continue", final=False):
    st.divider()
    columns = st.columns([1, 1, 2])
    with columns[0]:
        if step > 1:
            if st.button("Back", width="stretch", key=key(f"back_{step}")):
                go(step - 1)
                st.rerun()
    with columns[1]:
        if final:
            if st.button("Start over", width="stretch", key=key("finish_restart")):
                put("confirm_restart", True)
                st.rerun()
        else:
            clicked = st.button(
                next_label,
                type="primary",
                width="stretch",
                disabled=not can_continue,
                key=key(f"next_{step}"),
            )
            if clicked and can_continue:
                complete(step)
                go(step + 1)
                st.rerun()
    with columns[2]:
        if not can_continue and reason:
            st.caption(reason)
