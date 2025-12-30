# routing.py
import hashlib
import re
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from algorithms_registry import HyperparameterSpec
from utils import (
    BASE_GAMMA_SPEC,
    BASE_N_SPEC,
    clamp_value,
    clear_grid_cache_entry,
    get_tau_grid,
    slider_for_param,
    value_index,
)


def init_session_state():
    st.session_state.setdefault("ui_phase", "config")
    st.session_state.setdefault("selected_algorithm", None)
    st.session_state.setdefault("range_store", {})
    st.session_state.setdefault("other_params_store", {})
    st.session_state.setdefault("pending_settings", None)
    st.session_state.setdefault("active_settings", None)
    st.session_state.setdefault("other_editor_open", False)


def reset_for_algorithm_change(algo_key: str):
    st.session_state["selected_algorithm"] = algo_key
    st.session_state["ui_phase"] = "config"
    st.session_state["pending_settings"] = None
    st.session_state["active_settings"] = None
    st.session_state["other_editor_open"] = False


def render_range_inputs(label: str, base: HyperparameterSpec, stored: dict) -> dict:
    defaults = {
        "min": stored.get("min", base.min_value) if stored else base.min_value,
        "max": stored.get("max", base.max_value) if stored else base.max_value,
        "step": stored.get("step", base.step) if stored else base.step,
    }
    col1, col2, col3 = st.columns(3)
    if base.value_type == "int":
        min_value = col1.number_input(f"{label} min", value=int(defaults["min"]), step=1)
        max_value = col2.number_input(f"{label} max", value=int(defaults["max"]), step=1)
        step_value = col3.number_input(
            f"{label} step",
            min_value=1,
            value=max(int(defaults["step"]), 1),
            step=1,
        )
        return {"min": int(min_value), "max": int(max_value), "step": int(step_value)}

    min_value = col1.number_input(
        f"{label} min",
        value=float(defaults["min"]),
        step=float(base.step),
        format="%.4f",
    )
    max_value = col2.number_input(
        f"{label} max",
        value=float(defaults["max"]),
        step=float(base.step),
        format="%.4f",
    )
    step_value = col3.number_input(
        f"{label} step",
        min_value=1e-6,
        value=float(defaults["step"]),
        step=float(base.step),
        format="%.4f",
    )
    return {"min": float(min_value), "max": float(max_value), "step": float(step_value)}


def render_config_phase(algo_key: str, spec):
    st.subheader("Configuration")
    with st.container(border=True):
        st.write("Set gamma/n ranges and choose constant values for other parameters.")

        range_store = st.session_state["range_store"]
        algo_ranges = range_store.setdefault(algo_key, {})
        gamma_settings = render_range_inputs("gamma", BASE_GAMMA_SPEC, algo_ranges.get("gamma", {}))
        n_settings = render_range_inputs("n", BASE_N_SPEC, algo_ranges.get("n", {}))
        algo_ranges["gamma"] = gamma_settings
        algo_ranges["n"] = n_settings

        params_store = st.session_state["other_params_store"]
        current_params = params_store.setdefault(
            algo_key,
            {param.name: param.default for param in spec.hyperparameters},
        )
    with st.container(border=True):
        st.write("Other parameters")
        if not spec.hyperparameters:
            st.caption("This algorithm has no additional parameters.")
        else:
            for param in spec.hyperparameters:
                current_params[param.name] = slider_for_param(
                    param,
                    value=current_params.get(param.name, param.default),
                    key=f"config-{algo_key}-{param.name}",
                )
            stale_keys = set(current_params) - {p.name for p in spec.hyperparameters}
            for key in stale_keys:
                current_params.pop(key, None)

    if st.button("Plot"):
        errors = []
        if gamma_settings["max"] <= gamma_settings["min"]:
            errors.append("gamma max must be greater than gamma min.")
        if gamma_settings["step"] <= 0:
            errors.append("gamma step must be positive.")
        if n_settings["max"] <= n_settings["min"]:
            errors.append("n max must be greater than n min.")
        if n_settings["step"] <= 0:
            errors.append("n step must be positive.")
        if errors:
            for error in errors:
                st.error(error)
            return
        gamma_spec = HyperparameterSpec(
            name="gamma",
            label="gamma",
            min_value=float(gamma_settings["min"]),
            max_value=float(gamma_settings["max"]),
            default=float(gamma_settings["min"]),
            step=float(gamma_settings["step"]),
            value_type=BASE_GAMMA_SPEC.value_type,
        )
        n_spec = HyperparameterSpec(
            name="n",
            label="n",
            min_value=float(n_settings["min"]),
            max_value=float(n_settings["max"]),
            default=float(n_settings["min"]),
            step=float(n_settings["step"]),
            value_type=BASE_N_SPEC.value_type,
        )
        st.session_state["pending_settings"] = {
            "algo_key": algo_key,
            "gamma_spec": gamma_spec,
            "n_spec": n_spec,
            "other_params": dict(current_params),
        }
        st.session_state["ui_phase"] = "loading"
        st.rerun()


def render_loading_phase(algo_key: str, spec):
    pending = st.session_state.get("pending_settings")
    if not pending or pending["algo_key"] != algo_key:
        st.session_state["ui_phase"] = "config"
        st.rerun()

    gamma_spec = pending["gamma_spec"]
    n_spec = pending["n_spec"]
    st.subheader(f"Computing tau values for `{spec.name}`")
    st.caption(f"Algorithm: {spec.description}")
    other_display = ", ".join(f"{name}={value}" for name, value in pending["other_params"].items()) or "None"
    st.info(
        f"gamma: [{gamma_spec.min_value}, {gamma_spec.max_value}], step_size={gamma_spec.step}  \n"
        f"n: [{n_spec.min_value}, {n_spec.max_value}], step_size={n_spec.step}  \n"
        f"other params: {other_display}"
    )

    result = get_tau_grid(
        algo_key,
        gamma_spec,
        n_spec,
        pending["other_params"],
        show_progress=True,
    )
    if result is None:
        st.error("Unable to compute tau grid.")
        st.session_state["ui_phase"] = "config"
        return

    st.session_state["active_settings"] = pending
    st.session_state["pending_settings"] = None
    st.session_state["ui_phase"] = "results"
    st.session_state[f"dual_selection_{algo_key}"] = {}
    st.session_state[f"gamma_slider_{algo_key}"] = float(gamma_spec.min_value)
    st.session_state[f"n_slider_{algo_key}"] = float(n_spec.min_value)
    st.rerun()


def render_other_params_editor(algo_key: str, spec, settings):
    if not spec.hyperparameters:
        return
    editor_open = st.session_state.get("other_editor_open", False)
    params_store = st.session_state["other_params_store"]
    params_store.setdefault(algo_key, dict(settings["other_params"]))

    if not editor_open:
        if st.button("Change other parameters"):
            st.session_state["other_editor_open"] = True
            st.rerun()
        return
    with st.container(border=True):
        st.write("Adjust other parameters")
        with st.form("other-params-edit"):
            new_values = {}
            for param in spec.hyperparameters:
                new_values[param.name] = slider_for_param(
                    param,
                    value=params_store[algo_key].get(param.name, param.default),
                    key=f"edit-{algo_key}-{param.name}",
                )
            submitted = st.form_submit_button("Replot")
        if submitted:
            params_store[algo_key] = new_values
            clear_grid_cache_entry(
                algo_key,
                settings["gamma_spec"],
                settings["n_spec"],
                new_values,
            )
            st.session_state["pending_settings"] = {
                "algo_key": algo_key,
                "gamma_spec": settings["gamma_spec"],
                "n_spec": settings["n_spec"],
                "other_params": dict(new_values),
            }
            st.session_state["other_editor_open"] = False
            st.session_state["ui_phase"] = "loading"
            st.rerun()
    if st.button("Cancel"):
        st.session_state["other_editor_open"] = False
        st.rerun()


def render_results_phase(algo_key: str, spec):
    settings = st.session_state.get("active_settings")
    if not settings or settings["algo_key"] != algo_key:
        st.session_state["ui_phase"] = "config"
        st.rerun()

    cached = get_tau_grid(
        algo_key,
        settings["gamma_spec"],
        settings["n_spec"],
        settings["other_params"],
        show_progress=False,
    )
    if cached is None:
        st.session_state["pending_settings"] = settings
        st.session_state["ui_phase"] = "loading"
        st.rerun()

    gamma_values, n_values, tau_grid, cached_warnings, duals_grid, dual_fluctuations = cached
    gamma_spec = settings["gamma_spec"]
    n_spec = settings["n_spec"]

    st.subheader(f"Results for `{spec.name}`")
    st.caption(f"Algorithm: {spec.description}")
    if st.button("Change gamma/n settings"):
        st.session_state["ui_phase"] = "config"
        st.rerun()

    render_other_params_editor(algo_key, spec, settings)

    gamma_slider_key = f"gamma_slider_{algo_key}"
    n_slider_key = f"n_slider_{algo_key}"
    st.session_state.setdefault(gamma_slider_key, float(gamma_spec.min_value))
    st.session_state.setdefault(n_slider_key, float(n_spec.min_value))

    st.session_state[gamma_slider_key] = clamp_value(float(st.session_state[gamma_slider_key]), gamma_spec)
    st.session_state[n_slider_key] = clamp_value(float(st.session_state[n_slider_key]), n_spec)
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            gamma_value = st.slider(
                "gamma",
                float(gamma_spec.min_value),
                float(gamma_spec.max_value),
                step=float(gamma_spec.step),
                key=gamma_slider_key,
            )
    with col2:
        with st.container(border=True):
            if n_spec.value_type == "int":
                st.session_state[n_slider_key] = int(round(st.session_state[n_slider_key]))
                n_value_raw = st.slider(
                    "n",
                    int(n_spec.min_value),
                    int(n_spec.max_value),
                    step=int(max(1, n_spec.step)),
                    key=n_slider_key,
                )
                n_value = float(n_value_raw)
            else:
                n_value = st.slider(
                    "n",
                    float(n_spec.min_value),
                    float(n_spec.max_value),
                    step=float(n_spec.step),
                    key=n_slider_key,
                )

    gamma_idx = value_index(float(gamma_value), gamma_spec)
    n_idx = value_index(float(n_value), n_spec)
    tau_gamma = tau_grid[:, n_idx]
    tau_n = tau_grid[gamma_idx, :]
    current_tau = tau_grid[gamma_idx, n_idx]
    warning_messages = set(cached_warnings)

    gamma_fig = go.Figure()
    gamma_fig.add_trace(
        go.Scatter(
            x=gamma_values,
            y=tau_gamma,
            mode="lines",
            hovertemplate="gamma=%{x:.3f}<br>tau=%{y:.3e}<extra></extra>",
        )
    )
    if current_tau is not None and np.isfinite(current_tau):
        gamma_fig.add_trace(
            go.Scatter(
                x=[gamma_value],
                y=[current_tau],
                mode="markers",
                marker={"size": 12},
                hovertemplate="gamma=%{x:.3f}<br>tau=%{y:.3e}<extra></extra>",
                name="current",
            )
        )
    gamma_fig.update_layout(
        showlegend=False,
        title="Tau vs gamma",
        xaxis_title="gamma",
        yaxis_title="tau",
    )

    n_fig = go.Figure()
    n_fig.add_trace(
        go.Scatter(
            x=n_values,
            y=tau_n,
            mode="lines",
            hovertemplate="n=%{x:.3f}<br>tau=%{y:.3e}<extra></extra>",
        )
    )
    if current_tau is not None and np.isfinite(current_tau):
        n_fig.add_trace(
            go.Scatter(
                x=[n_value],
                y=[current_tau],
                mode="markers",
                marker={"size": 12},
                hovertemplate="n=%{x:.3f}<br>tau=%{y:.3e}<extra></extra>",
            )
        )
    n_fig.update_layout(
        showlegend=False,
        title="Tau vs n",
        xaxis_title="n",
        yaxis_title="tau",
    )

    with col1:
        with st.container(border=True):
            st.plotly_chart(gamma_fig)
    with col2:
        with st.container(border=True):
            st.plotly_chart(n_fig)

    if warning_messages:
        warning_text = "\n".join(sorted(warning_messages))
        st.warning(
            "Some parameter combinations could not be solved; missing points are shown as gaps.\n" + warning_text
        )

    render_dual_values_panel(
        algo_key,
        duals_grid,
        dual_fluctuations,
        gamma_idx,
        n_idx,
    )


def _jet_color(value: float) -> str:
    value = max(0.0, min(1.0, float(value)))
    r = max(0.0, min(1.0, 1.5 - abs(4 * value - 3)))
    g = max(0.0, min(1.0, 1.5 - abs(4 * value - 2)))
    b = max(0.0, min(1.0, 1.5 - abs(4 * value - 1)))
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def _css_escape(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace("'", "\\'")
    return re.sub(r"[\r\n]+", " ", escaped)


def _text_color_for_bg(hex_color: str) -> str:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return "#0b0b0b"
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#f7f7f7" if luminance < 0.55 else "#0b0b0b"


def _marker_class(value: str) -> str:
    digest = hashlib.md5(value.encode("utf-8")).hexdigest()[:10]
    return f"dual-marker-{digest}"


def _dual_selection_state(algo_key: str) -> dict:
    return st.session_state.setdefault(f"dual_selection_{algo_key}", {})


def _toggle_dual_selection(algo_key: str, key: str) -> None:
    selected = _dual_selection_state(algo_key)
    selected[key] = not selected.get(key, False)


def render_dual_values_panel(
    algo_key: str,
    duals_grid: list[list[dict]],
    dual_fluctuations: dict,
    gamma_idx: int,
    n_idx: int,
) -> None:
    st.subheader("Dual values")
    if not dual_fluctuations:
        st.caption("No dual values available for these settings.")
        return

    selected = _dual_selection_state(algo_key)
    current_duals = duals_grid[gamma_idx][n_idx] if duals_grid else {}
    selected_values = []

    for constraint, fluct_map in sorted(dual_fluctuations.items()):
        if not fluct_map:
            continue
        st.markdown(f"**{constraint}**")
        max_fluct = max(fluct_map.values()) if fluct_map else 0.0
        max_fluct = max(max_fluct, 1e-12)
        keys = sorted(fluct_map.keys(), key=lambda k: fluct_map[k], reverse=True)
        cols = st.columns(4)
        for idx, dual_key in enumerate(keys):
            fluct = fluct_map[dual_key]
            color = _jet_color(fluct / max_fluct)
            text_color = _text_color_for_bg(color)
            button_key = f"dual-btn-{algo_key}-{constraint}-{dual_key}"
            marker = _marker_class(button_key)
            selection_key = f"{constraint}::{dual_key}"
            is_selected = selected.get(selection_key, False)
            with cols[idx % 4].form(key=f"{button_key}-form"):
                st.markdown(
                    f"<div class='{_css_escape(marker)}'></div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<style>"
                    f"div[data-testid='stForm']:has(.{_css_escape(marker)}) button {{"
                    f"background-color: {color} !important;"
                    f"border-color: {color} !important;"
                    f"color: {text_color} !important;"
                    f"}}"
                    f"div[data-testid='stForm']:has(.{_css_escape(marker)}) button:hover {{"
                    f"filter: brightness(0.95);"
                    f"}}"
                    f"</style>",
                    unsafe_allow_html=True,
                )
                submitted = st.form_submit_button(
                    dual_key,
                    type="primary" if is_selected else "secondary",
                )
            if submitted:
                _toggle_dual_selection(algo_key, selection_key)
            if is_selected:
                value = current_duals.get(constraint, {}).get(dual_key)
                selected_values.append((constraint, dual_key, value))

    st.subheader("Selected dual values")
    if not selected_values:
        st.caption("Click a dual value button to list its current value.")
        return
    for constraint, dual_key, value in selected_values:
        st.write(f"{constraint} | {dual_key} = {value}")
