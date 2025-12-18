# app.py
import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from functions_registry import (
    DEFAULT_HYPERPARAMETERS,
    FUNCTIONS,
    FunctionEvaluationError,
    HyperparameterSpec,
)


def slider_for_param(
    param: HyperparameterSpec,
    *,
    value: float | None = None,
    key: str | None = None,
):
    slider_value = param.default if value is None else value
    if param.value_type == "int":
        return int(
            st.slider(
                param.label,
                int(param.min_value),
                int(param.max_value),
                int(slider_value),
                step=int(param.step),
                key=key,
            )
        )
    return float(
        st.slider(
            param.label,
            float(param.min_value),
            float(param.max_value),
            float(slider_value),
            step=float(param.step),
            key=key,
        )
    )


def discrete_values(param: HyperparameterSpec) -> np.ndarray:
    steps = int(round((param.max_value - param.min_value) / param.step))
    values = np.array([param.min_value + i * param.step for i in range(steps + 1)], dtype=float)
    if param.value_type == "int":
        values = np.round(values).astype(int)
    return values


def make_cache_key(
    function_key: str,
    gamma_spec: HyperparameterSpec,
    n_spec: HyperparameterSpec,
    other_params: dict,
) -> tuple:
    return (
        function_key,
        (
            gamma_spec.name,
            gamma_spec.label,
            gamma_spec.min_value,
            gamma_spec.max_value,
            gamma_spec.default,
            gamma_spec.step,
            gamma_spec.value_type,
        ),
        (
            n_spec.name,
            n_spec.label,
            n_spec.min_value,
            n_spec.max_value,
            n_spec.default,
            n_spec.step,
            n_spec.value_type,
        ),
        tuple(sorted(other_params.items())),
    )


def clear_grid_cache_entry(
    function_key: str,
    gamma_spec: HyperparameterSpec,
    n_spec: HyperparameterSpec,
    other_params: dict,
):
    cache = st.session_state.get("tau_grid_cache")
    if cache is None:
        return
    cache.pop(make_cache_key(function_key, gamma_spec, n_spec, other_params), None)


def get_tau_grid(
    function_key: str,
    gamma_spec: HyperparameterSpec,
    n_spec: HyperparameterSpec,
    other_params: dict,
):
    cache = st.session_state.setdefault("tau_grid_cache", {})
    cache_key = make_cache_key(function_key, gamma_spec, n_spec, other_params)

    if cache_key in cache:
        return cache[cache_key]

    spec = FUNCTIONS[function_key]
    gamma_values = discrete_values(gamma_spec)
    n_values = discrete_values(n_spec)
    tau_grid = np.full((len(gamma_values), len(n_values)), np.nan)
    warnings: set[str] = set()

    total = max(len(gamma_values) * len(n_values), 1)
    completed = 0
    progress_bar = st.progress(0.0)
    status_placeholder = st.empty()
    start = time.perf_counter()
    update_every = max(total // 100, 1)

    for i, gamma_value in enumerate(gamma_values):
        for j, n_value in enumerate(n_values):
            try:
                raw = spec.func(float(gamma_value), float(n_value), **other_params)
                tau_grid[i, j] = float(np.asarray(raw).reshape(-1)[0])
            except FunctionEvaluationError as exc:
                warnings.add(f"{spec.name}: {exc}")
            except Exception as exc:
                warnings.add(f"{spec.name}: unexpected error - {exc}")
            completed += 1
            if completed % update_every == 0 or completed == total:
                fraction = completed / total
                elapsed = time.perf_counter() - start
                eta = (elapsed / fraction) - elapsed if fraction > 0 else 0.0
                progress_bar.progress(fraction)
                status_placeholder.write(f"Computing ... {completed}/{total} (eta {eta:.1f}s)")

    progress_bar.empty()
    status_placeholder.empty()

    cache[cache_key] = (
        gamma_values,
        n_values,
        tau_grid,
        tuple(sorted(warnings)),
    )
    return cache[cache_key]


def value_index(value: float, spec: HyperparameterSpec) -> int:
    idx = int(round((value - spec.min_value) / spec.step))
    total = int(round((spec.max_value - spec.min_value) / spec.step))
    return int(min(max(idx, 0), total))


st.title("Interactive tau explorer")

function_key = st.selectbox(
    "Function",
    options=list(FUNCTIONS.keys()),
    format_func=lambda key: FUNCTIONS[key].name,
)
spec = FUNCTIONS[function_key]
st.caption(f"Selected function: `{spec.description}`")

gamma_spec = next(param for param in DEFAULT_HYPERPARAMETERS if param.name == "gamma")
n_spec = next(param for param in DEFAULT_HYPERPARAMETERS if param.name == "n")

if st.session_state.get("last_function") != function_key:
    st.session_state["last_function"] = function_key
    st.session_state["other_editor_open"] = False

params_store = st.session_state.setdefault("other_params_store", {})
if function_key not in params_store:
    params_store[function_key] = {param.name: param.default for param in spec.hyperparameters}
else:
    for param in spec.hyperparameters:
        params_store[function_key].setdefault(param.name, param.default)
    valid_keys = {param.name for param in spec.hyperparameters}
    for existing_key in list(params_store[function_key].keys()):
        if existing_key not in valid_keys:
            params_store[function_key].pop(existing_key)

other_params = dict(params_store[function_key])

if spec.hyperparameters:
    editor_open = st.session_state.get("other_editor_open", False)
    if not editor_open:
        if st.button("Change other parameters"):
            st.session_state["other_editor_open"] = True
            st.rerun()
    else:
        with st.container(border=True):
            with st.form("other-params-form"):
                st.caption("Adjust values then click Recompute to update the grid.")
                new_values = {}
                for param in spec.hyperparameters:
                    new_values[param.name] = slider_for_param(
                        param,
                        value=other_params.get(param.name, param.default),
                        key=f"other-param-{function_key}-{param.name}",
                    )
                submitted = st.form_submit_button("Recompute")
            if submitted:
                params_store[function_key] = new_values
                clear_grid_cache_entry(function_key, gamma_spec, n_spec, new_values)
                st.session_state["other_editor_open"] = False
                st.rerun()
            if st.button("Close other parameters"):
                st.session_state["other_editor_open"] = False
                st.rerun()
else:
    st.caption("This function has no additional parameters.")

gamma_values, n_values, tau_grid, cached_warnings = get_tau_grid(
    function_key,
    gamma_spec,
    n_spec,
    other_params,
)
col1, col2 = st.columns(2)

gamma_slider_key = f"gamma_value_{function_key}"
gamma_default_value = float(st.session_state.get(gamma_slider_key, gamma_spec.default))
with col1:
    gamma_value = float(
        st.slider(
            "gamma",
            float(gamma_spec.min_value),
            float(gamma_spec.max_value),
            value=gamma_default_value,
            step=float(gamma_spec.step),
            key=gamma_slider_key,
        )
    )

n_slider_key = f"n_value_{function_key}"
n_default_value = st.session_state.get(n_slider_key, n_spec.default)
with col2:
    if n_spec.value_type == "int":
        n_value_raw = st.slider(
            "n",
            int(n_spec.min_value),
            int(n_spec.max_value),
            value=int(n_default_value),
            step=int(n_spec.step),
            key=n_slider_key,
        )
    else:
        n_value_raw = st.slider(
            "n",
            float(n_spec.min_value),
            float(n_spec.max_value),
            value=float(n_default_value),
            step=float(n_spec.step),
            key=n_slider_key,
        )
n_value = float(n_value_raw)

gamma_idx = value_index(gamma_value, gamma_spec)
n_idx = value_index(n_value, n_spec)
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
if np.isfinite(current_tau):
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
gamma_fig.update_layout(showlegend=False, xaxis_title="gamma", yaxis_title="tau")

n_fig = go.Figure()
n_fig.add_trace(
    go.Scatter(
        x=n_values,
        y=tau_n,
        mode="lines",
        hovertemplate="n=%{x:.3f}<br>tau=%{y:.3e}<extra></extra>",
    )
)
if np.isfinite(current_tau):
    n_fig.add_trace(
        go.Scatter(
            x=[n_value],
            y=[current_tau],
            mode="markers",
            marker={"size": 12},
            hovertemplate="n=%{x:.3f}<br>tau=%{y:.3e}<extra></extra>",
            name="current",
        )
    )
n_fig.update_layout(showlegend=False, xaxis_title="n", yaxis_title="tau")


with col1:
    st.subheader("tau vs gamma")
    st.plotly_chart(gamma_fig, use_container_width=True)
with col2:
    st.subheader("tau vs n")
    st.plotly_chart(n_fig, use_container_width=True)

if warning_messages:
    warning_text = "\n".join(sorted(warning_messages))
    st.warning("Some parameter combinations could not be solved; missing points are shown as gaps.\n" + warning_text)
