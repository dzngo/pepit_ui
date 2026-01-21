# routing.py
import json
import re
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from streamlit_ace import st_ace

from algorithms_registry import (
    AlgorithmSpec,
    HyperparameterSpec,
    ALGORITHMS,
    FUNCTIONS,
    INITIAL_CONDITIONS,
    PERFORMANCE_METRICS,
    compile_steps_for_test,
    get_algorithm_steps_code,
    get_base_algorithm_name,
    register_custom_algorithm,
    remove_custom_algorithm,
    CUSTOM_ALGORITHMS,
    run_algorithm,
)
from utils import (
    BASE_GAMMA_SPEC,
    BASE_N_SPEC,
    _evaluate_pattern_expression,
    _float_text_default,
    _parse_float_input,
    _parse_float_list,
    _random_pattern_example,
    build_dual_series_data,
    build_dual_section_html,
    clamp_value,
    dual_ranking_by_slice,
    compute,
    value_index,
)


def init_session_state():
    st.session_state.setdefault("ui_phase", "config")
    st.session_state.setdefault("selected_algorithm", None)
    st.session_state.setdefault("range_store", {})
    st.session_state.setdefault("pending_settings", None)
    st.session_state.setdefault("active_settings", None)
    st.session_state.setdefault("rerun_nan_caches", False)
    st.session_state.setdefault("function_store", {})
    st.session_state.setdefault("function_params_store", {})
    st.session_state.setdefault("initial_condition_store", {})
    st.session_state.setdefault("performance_metric_store", {})


def reset_for_algorithm_change(algo_key: str):
    st.session_state["selected_algorithm"] = algo_key
    st.session_state["ui_phase"] = "config"
    st.session_state["pending_settings"] = None
    st.session_state["active_settings"] = None


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


def _steps_source(spec: AlgorithmSpec) -> str:
    return get_algorithm_steps_code(spec.name)


def _editor_steps_source(spec: AlgorithmSpec) -> str:
    code = _steps_source(spec)
    pattern = r"^def\s+\w+\s*\("
    if re.search(pattern, code, flags=re.MULTILINE):
        return re.sub(pattern, "def customized_steps(", code, count=1, flags=re.MULTILINE)
    return code


def _render_steps_editor(
    *,
    algo_key: str,
    spec: AlgorithmSpec,
    context: str,
    test_context: dict | None = None,
) -> None:
    open_key = f"customize-open-{context}-{algo_key}"
    code_key = f"customize-code-{context}-{algo_key}"
    name_key = f"customize-name-{context}-{algo_key}"
    editor_key = f"customize-editor-{context}-{algo_key}"

    if st.session_state.get(open_key, False):
        st.session_state.setdefault(code_key, _editor_steps_source(spec))
        updated = st_ace(
            value=st.session_state.get(code_key, ""),
            language="python",
            key=editor_key,
            height=320,
            show_gutter=True,
            wrap=True,
        )
        if isinstance(updated, str):
            st.session_state[code_key] = updated
        st.text_input("Custom algorithm name", key=name_key)
        col1, col2, col3 = st.columns(3)
        with col1:
            save_clicked = st.button("Save", key="btn-save-config")
        with col2:
            test_clicked = st.button("Test", key="btn-test-config")
        with col3:
            cancel_clicked = st.button("Cancel", key="btn-cancel-config")
        if save_clicked:
            name = str(st.session_state.get(name_key, "")).strip()
            steps_code = st.session_state.get(code_key, "")
            base_algo = get_base_algorithm_name(spec.name)
            try:
                if not name:
                    raise ValueError("Custom algorithm name is required.")
                register_custom_algorithm(
                    name=name,
                    steps_code=str(steps_code),
                    base_algo=base_algo,
                )
            except Exception as exc:
                st.error(str(exc))
            else:
                st.success(f"Saved custom algorithm '{name}'.")
                st.session_state[open_key] = False
                st.session_state["pending_algorithm_select"] = name
                st.session_state["selected_algorithm"] = None
                st.session_state["ui_phase"] = "config"
                st.rerun()
        if cancel_clicked:
            st.session_state[open_key] = False
            st.rerun()
        if test_context and test_clicked:
            if test_context["function_param_errors"]:
                st.error("; ".join(test_context["function_param_errors"]))
            else:
                try:
                    steps_code = st.session_state.get(code_key, "")
                    steps = compile_steps_for_test(steps_code)
                    temp_spec = AlgorithmSpec(
                        name=spec.name,
                        steps=steps,
                        function_slots=list(spec.function_slots),
                        default_function_keys=dict(spec.default_function_keys),
                        default_initial_condition=spec.default_initial_condition,
                        default_performance_metric=spec.default_performance_metric,
                    )
                    run_algorithm(
                        algo_spec=temp_spec,
                        function_config=test_context["function_config"],
                        initial_condition_key=test_context["initial_condition_key"],
                        performance_metric_key=test_context["performance_metric_key"],
                        algo_params={
                            "gamma": float(test_context["gamma_min"]),
                            "n": float(test_context["n_min"]),
                        },
                    )
                except Exception as exc:
                    st.error(f"Test failed: {exc}")
                else:
                    st.success("Test succeeded.")
    else:
        st.code(_steps_source(spec), language="python")
        if st.button("Customize", key="btn-customize-config"):
            st.session_state[open_key] = True
            st.session_state.setdefault(code_key, _editor_steps_source(spec))
            st.session_state.setdefault(name_key, "")
            st.rerun()


def render_config_phase(algo_key: str, spec: AlgorithmSpec):
    css_path = Path(__file__).resolve().parent / "ui" / "config_panel.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)
    st.subheader("Configuration")

    sections = st.columns(2)
    with sections[0]:
        with st.container(border=True):
            st.write("Set gamma/n ranges")

            range_store = st.session_state["range_store"]
            algo_ranges = range_store.setdefault(algo_key, {})
            gamma_settings = render_range_inputs("gamma", BASE_GAMMA_SPEC, algo_ranges.get("gamma", {}))
            n_settings = render_range_inputs("n", BASE_N_SPEC, algo_ranges.get("n", {}))
            algo_ranges["gamma"] = gamma_settings
            algo_ranges["n"] = n_settings

        with st.container(border=True):
            st.write("Functions")
            function_store = st.session_state["function_store"]
            function_params_store = st.session_state["function_params_store"]
            algo_functions = function_store.setdefault(algo_key, {})
            algo_function_params = function_params_store.setdefault(algo_key, {})
            function_param_errors: list[str] = []
            for slot in spec.function_slots:
                default_function = spec.default_function_keys.get(slot.key)
                selected_function = algo_functions.get(slot.key, default_function)
                function_names = sorted(FUNCTIONS.keys())
                if selected_function not in function_names:
                    selected_function = function_names[0]
                selected_function = st.selectbox(
                    f"{slot.key} function",
                    options=function_names,
                    index=function_names.index(selected_function) if selected_function in function_names else 0,
                    key=f"function-{algo_key}-{slot.key}",
                )
                algo_functions[slot.key] = selected_function

                function_spec = FUNCTIONS[selected_function]
                slot_params = algo_function_params.setdefault(
                    slot.key,
                    {param.name: param.default for param in function_spec.parameters},
                )
                if not function_spec.parameters:
                    st.caption(f"{slot.key} has no required parameters.")
                else:
                    columns = st.columns(3)
                    for idx, param in enumerate(function_spec.parameters):
                        with columns[idx % 3]:
                            with st.container(border=True):
                                input_key = f"function-param-{algo_key}-{slot.key}-{param.name}"
                                param_context = f"{slot.key} ({function_spec.cls.__name__}), parameter {param.name}"
                                if param.param_type == "float":
                                    default_text = _float_text_default(slot_params.get(param.name, param.default))
                                    st.session_state.setdefault(input_key, default_text)
                                    raw_value = st.text_input(param.name, key=input_key)
                                    parsed_value, error = _parse_float_input(raw_value)
                                    if error:
                                        function_param_errors.append(f"{param_context}: {error}")
                                    else:
                                        if param.required and parsed_value is None:
                                            function_param_errors.append(f"{param_context}: value required.")
                                        elif parsed_value is not None and parsed_value < 0:
                                            function_param_errors.append(f"{param_context}: value must be >= 0.")
                                        else:
                                            slot_params[param.name] = parsed_value
                                    if param.description:
                                        st.caption(param.description)
                                elif param.param_type == "BlockPartition":
                                    d_value = st.number_input(
                                        f"{param.name} (d)",
                                        min_value=0,
                                        step=1,
                                        value=int(slot_params.get(param.name, 1) or 1),
                                        key=input_key,
                                    )
                                    slot_params[param.name] = int(d_value)
                                    desc_parts = []
                                    if param.description:
                                        desc_parts.append(param.description)
                                    desc_parts.append(
                                        "Partition will be created via `problem.declare_block_partition(d=...)`."
                                    )
                                    st.caption(" ".join(desc_parts))
                                elif param.param_type == "Point":
                                    checked = st.checkbox(
                                        param.name,
                                        value=bool(slot_params.get(param.name, False)),
                                        key=input_key,
                                    )
                                    slot_params[param.name] = bool(checked)
                                    desc_parts = []
                                    if param.description:
                                        desc_parts.append(param.description)
                                    desc_parts.append("When checked, a Point is created and passed as `center`.")
                                    st.caption(" ".join(desc_parts))
                                elif param.param_type == "list":
                                    desc = param.description
                                    if desc:
                                        desc += " "
                                    desc += "Enter list values separated by ','"
                                    existing = slot_params.get(param.name, param.default)
                                    if isinstance(existing, list):
                                        default_text = ", ".join(str(value) for value in existing)
                                    else:
                                        default_text = ""
                                    raw_value = st.text_input(
                                        param.name,
                                        value=st.session_state.get(input_key, default_text),
                                        key=input_key,
                                    )
                                    parsed_list, error = _parse_float_list(raw_value)
                                    if error:
                                        function_param_errors.append(f"{param_context}: {error}")
                                    else:
                                        if param.required and not parsed_list:
                                            function_param_errors.append(f"{param_context}: value required.")
                                        else:
                                            slot_params[param.name] = parsed_list
                                    st.caption(desc)
                                else:
                                    raw_value = st.text_input(
                                        param.name,
                                        value=str(slot_params.get(param.name, param.default) or ""),
                                        key=input_key,
                                    )
                                    slot_params[param.name] = raw_value
                    stale_params = set(slot_params) - {p.name for p in function_spec.parameters}
                    for key in stale_params:
                        slot_params.pop(key, None)

        with st.container(border=True):
            st.write("Initial condition and performance metric")
            ic_store = st.session_state["initial_condition_store"]
            pm_store = st.session_state["performance_metric_store"]
            ic_options = list(INITIAL_CONDITIONS.keys())
            pm_options = list(PERFORMANCE_METRICS.keys())
            ic_default = ic_store.get(algo_key, spec.default_initial_condition)
            pm_default = pm_store.get(algo_key, spec.default_performance_metric)
            ic_selection = st.selectbox(
                "Initial condition",
                options=ic_options,
                format_func=lambda key: INITIAL_CONDITIONS[key].label,
                index=ic_options.index(ic_default) if ic_default in ic_options else 0,
                key=f"ic-{algo_key}",
            )
            pm_selection = st.selectbox(
                "Performance metric",
                options=pm_options,
                format_func=lambda key: PERFORMANCE_METRICS[key].label,
                index=pm_options.index(pm_default) if pm_default in pm_options else 0,
                key=f"pm-{algo_key}",
            )
            ic_store[algo_key] = ic_selection
            pm_store[algo_key] = pm_selection
    with sections[1]:
        with st.container(border=True):
            st.write("Algorithm")
            function_config = {
                slot.key: {
                    "function_key": st.session_state["function_store"][algo_key][slot.key],
                    "function_params": dict(st.session_state["function_params_store"][algo_key][slot.key]),
                }
                for slot in spec.function_slots
            }
            ic_key = st.session_state["initial_condition_store"][algo_key]
            pm_key = st.session_state["performance_metric_store"][algo_key]
            _render_steps_editor(
                algo_key=algo_key,
                spec=spec,
                context="config",
                test_context={
                    "function_config": function_config,
                    "function_param_errors": list(function_param_errors),
                    "initial_condition_key": ic_key,
                    "performance_metric_key": pm_key,
                    "gamma_min": gamma_settings["min"],
                    "n_min": n_settings["min"],
                },
            )

        with st.container(border=True):
            st.write("Remove customized algorithm")
            custom_names = sorted(CUSTOM_ALGORITHMS.keys())
            if not custom_names:
                st.caption("No customized algorithms saved.")
            else:
                selected_custom = st.selectbox(
                    "Custom algorithms",
                    options=custom_names,
                    key="remove-custom-algorithm",
                )
                remove_clicked = st.button("Remove", key="btn-remove-config")
                if remove_clicked:
                    try:
                        remove_custom_algorithm(selected_custom)
                    except Exception as exc:
                        st.error(str(exc))
                    else:
                        st.success(f"Removed '{selected_custom}'.")
                        if st.session_state.get("selected_algorithm") == selected_custom:
                            st.session_state["selected_algorithm"] = None
                            st.session_state["pending_algorithm_select"] = next(iter(ALGORITHMS.keys()), None)
                        st.rerun()

    st.checkbox("Rerun Nan caches", key="rerun_nan_caches")

    plot_clicked = st.button("Plot", key="btn-plot-config")
    if plot_clicked:
        errors = []
        if gamma_settings["max"] <= gamma_settings["min"]:
            errors.append("gamma max must be greater than gamma min.")
        if gamma_settings["step"] <= 0:
            errors.append("gamma step must be positive.")
        if n_settings["max"] <= n_settings["min"]:
            errors.append("n max must be greater than n min.")
        if n_settings["step"] <= 0:
            errors.append("n step must be positive.")
        errors.extend(function_param_errors)
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
            "function_config": {
                slot.key: {
                    "function_key": st.session_state["function_store"][algo_key][slot.key],
                    "function_params": dict(st.session_state["function_params_store"][algo_key][slot.key]),
                }
                for slot in spec.function_slots
            },
            "initial_condition_key": st.session_state["initial_condition_store"][algo_key],
            "performance_metric_key": st.session_state["performance_metric_store"][algo_key],
            "rerun_nan_caches": bool(st.session_state.get("rerun_nan_caches", False)),
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

    with st.container(border=True):
        summary_lines = [
            f"**Algorithm**: `{spec.name}`",
            f"**gamma**: [{gamma_spec.min_value}, {gamma_spec.max_value}], step_size={gamma_spec.step}",
            f"**n**: [{n_spec.min_value}, {n_spec.max_value}], step_size={n_spec.step}",
            f"**Initial condition**: {INITIAL_CONDITIONS[pending['initial_condition_key']].label}",
            f"**Performance metric**: {PERFORMANCE_METRICS[pending['performance_metric_key']].label}",
        ]
        st.markdown("<br>".join(summary_lines), unsafe_allow_html=True)
        st.markdown("**Steps**")
        st.code(_steps_source(spec), language="python")
        st.markdown("**Functions**")
        for slot_key, slot_config in sorted(pending["function_config"].items()):
            st.markdown(f"{slot_key}: `{slot_config['function_key']}`")
            if slot_config["function_params"]:
                params_line = ", ".join(f"{name}={value}" for name, value in slot_config["function_params"].items())
                st.markdown(f"*params*: {params_line}")
            else:
                st.markdown("*params*: `{}`")

    result = compute(
        algo_key,
        gamma_spec,
        n_spec,
        pending["function_config"],
        pending["initial_condition_key"],
        pending["performance_metric_key"],
        show_progress=True,
        rerun_nan_cache=bool(pending.get("rerun_nan_caches", False)),
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


def render_results_phase(algo_key: str, spec):
    settings = st.session_state.get("active_settings")
    if not settings or settings["algo_key"] != algo_key:
        st.session_state["ui_phase"] = "config"
        st.rerun()

    result = compute(
        algo_key,
        settings["gamma_spec"],
        settings["n_spec"],
        settings["function_config"],
        settings["initial_condition_key"],
        settings["performance_metric_key"],
        show_progress=False,
        rerun_nan_cache=bool(settings.get("rerun_nan_caches", False)),
    )
    if result is None:
        st.session_state["pending_settings"] = settings
        st.session_state["ui_phase"] = "loading"
        st.rerun()

    gamma_values, n_values, tau_grid, cached_warnings, duals_grid = result
    gamma_spec = settings["gamma_spec"]
    n_spec = settings["n_spec"]

    st.subheader(f"Results for `{spec.name}`")
    if st.button("Change gamma/n settings"):
        st.session_state["ui_phase"] = "config"
        st.rerun()

    with st.expander("Configuration details"):
        summary_lines = [
            f"**Algorithm**: `{spec.name}`",
            f"**Initial condition**: {INITIAL_CONDITIONS[settings['initial_condition_key']].label}",
            f"**Performance metric**: {PERFORMANCE_METRICS[settings['performance_metric_key']].label}",
        ]
        st.markdown("<br>".join(summary_lines), unsafe_allow_html=True)
        st.markdown("**Steps**")
        st.code(_steps_source(spec), language="python")
        st.markdown("**Functions**")
        for slot_key, slot_config in sorted(settings["function_config"].items()):
            st.markdown(f"{slot_key}: `{slot_config['function_key']}`")
            if slot_config["function_params"]:
                params_line = ", ".join(f"{name}={value}" for name, value in slot_config["function_params"].items())
                st.markdown(f"*params*: {params_line}")
            else:
                st.markdown("*params*: `{}`")

    gamma_slider_key = f"gamma_slider_{algo_key}"
    n_slider_key = f"n_slider_{algo_key}"
    st.session_state.setdefault(gamma_slider_key, float(gamma_spec.min_value))
    st.session_state.setdefault(n_slider_key, float(n_spec.min_value))

    st.session_state[gamma_slider_key] = clamp_value(float(st.session_state[gamma_slider_key]), gamma_spec)
    st.session_state[n_slider_key] = clamp_value(float(st.session_state[n_slider_key]), n_spec)
    col1, col2 = st.columns(2)

    pattern_gamma_key = f"tau_pattern_gamma_{algo_key}"
    pattern_n_key = f"tau_pattern_n_{algo_key}"
    st.session_state.setdefault(pattern_gamma_key, "")
    st.session_state.setdefault(pattern_n_key, "")
    gamma_overlay_values = None
    n_overlay_values = None

    with col1:
        with st.container(border=True):
            # gamma slider
            gamma_value = st.slider(
                "gamma",
                float(gamma_spec.min_value),
                float(gamma_spec.max_value),
                step=float(gamma_spec.step),
                key=gamma_slider_key,
            )

            # tau vs gamma hypothesis
            gamma_pattern = st.text_input(
                "pattern hypothesis",
                key=pattern_gamma_key,
                placeholder=_random_pattern_example(),
            )
            gamma_overlay_values, gamma_error = _evaluate_pattern_expression(gamma_pattern, gamma_values)
            if gamma_error:
                st.error(gamma_error)
    with col2:
        with st.container(border=True):
            # n slider
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

            # tau vs n hypothesis
            n_pattern = st.text_input(
                "pattern hypothesis",
                key=pattern_n_key,
                placeholder=_random_pattern_example(),
            )
            n_overlay_values, n_error = _evaluate_pattern_expression(n_pattern, n_values)
            if n_error:
                st.error(n_error)

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
    if gamma_overlay_values is not None and not gamma_error:
        gamma_fig.add_trace(
            go.Scatter(
                x=gamma_values,
                y=gamma_overlay_values,
                mode="lines",
                line={"color": "#ff8a00", "width": 2},
                hovertemplate="gamma=%{x:.3f}<br>pattern=%{y:.3e}<extra></extra>",
                showlegend=False,
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
        title={"text": "Tau vs gamma", "pad": {"t": 6, "b": 0}},
        xaxis_title="gamma",
        yaxis_title="tau",
        height=320,
        margin={"t": 30, "l": 50, "r": 20, "b": 10},
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
    if n_overlay_values is not None and not n_error:
        n_fig.add_trace(
            go.Scatter(
                x=n_values,
                y=n_overlay_values,
                mode="lines",
                line={"color": "#ff8a00", "width": 2},
                hovertemplate="n=%{x:.3f}<br>pattern=%{y:.3e}<extra></extra>",
                showlegend=False,
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
        title={"text": "Tau vs n", "pad": {"t": 6, "b": 0}},
        xaxis_title="n",
        yaxis_title="tau",
        height=320,
        margin={"t": 30, "l": 50, "r": 20, "b": 10},
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
        gamma_values,
        n_values,
        gamma_idx,
        n_idx,
    )


DUAL_BUTTON_MIN_WIDTH = 70
DUAL_BUTTON_MAX_WIDTH = 100
DUAL_BUTTON_ROW_HEIGHT = 52
DUAL_SECTION_PADDING = 70
DUAL_PLOT_HEIGHT = 100
DUAL_PLOT_COLUMNS = 7
DUAL_PLOT_MIN_WIDTH = 220

UI_DIR = Path(__file__).resolve().parent / "ui"
DUAL_PANEL_HTML = (UI_DIR / "dual_panel.html").read_text()
DUAL_PANEL_CSS = (UI_DIR / "dual_panel.css").read_text()
DUAL_PANEL_JS = (UI_DIR / "dual_panel.js").read_text()


def render_dual_values_panel(
    algo_key: str,
    duals_grid: list[list[dict]],
    gamma_values: np.ndarray,
    n_values: np.ndarray,
    gamma_idx: int,
    n_idx: int,
) -> None:
    st.subheader("Dual values")
    if not duals_grid:
        st.caption("No dual values available for these settings.")
        return
    metric_labels = {
        "non_zero_pct": "Non-zero %",
        "non_zero_pct_with_none": "Non-zero % (None=0)",
        "std": "Std dev",
        "std_with_none": "Std dev (None=0)",
        "median_abs": "Median |x|",
        "median_abs_with_none": "Median |x| (None=0)",
        "mean_abs": "Average |x|",
        "mean_abs_with_none": "Average |x| (None=0)",
    }
    metric_col, _ = st.columns([1, 5])
    with metric_col:
        metric = st.selectbox(
            "Ranking metric",
            list(metric_labels.keys()),
            format_func=lambda key: metric_labels.get(key, key),
            index=list(metric_labels.keys()).index("non_zero_pct_with_none"),
            key=f"dual-ranking-metric-{algo_key}",
        )

    current_duals = duals_grid[gamma_idx][n_idx] if duals_grid else {}
    gamma_slice = [row[n_idx] for row in duals_grid]
    n_slice = list(duals_grid[gamma_idx])
    gamma_ranking = dual_ranking_by_slice(gamma_slice, metric=metric)
    n_ranking = dual_ranking_by_slice(n_slice, metric=metric)
    series_data = build_dual_series_data(
        duals_grid,
        gamma_values,
        n_values,
        gamma_idx,
        n_idx,
    )
    series_json = json.dumps(series_data).replace("</", "<\\/")

    gamma_ranking_title = f"Ranking vs gamma (n = {n_values[n_idx]})"
    n_ranking_title = f"Ranking vs n (gamma = {gamma_values[gamma_idx]})"
    gamma_html, gamma_count = build_dual_section_html(
        section_id=f"{algo_key}-gamma",
        section_key="gamma",
        title=gamma_ranking_title,
        dual_ranking=gamma_ranking,
        current_duals=current_duals,
        min_width=DUAL_BUTTON_MIN_WIDTH,
    )
    n_html, n_count = build_dual_section_html(
        section_id=f"{algo_key}-n",
        section_key="n",
        title=n_ranking_title,
        dual_ranking=n_ranking,
        current_duals=current_duals,
        min_width=DUAL_BUTTON_MIN_WIDTH,
    )
    gamma_plot_title = f"Dual value vs gamma (n = {n_values[n_idx]})"
    n_plot_title = f"Dual value vs n (gamma = {gamma_values[gamma_idx]})"
    total_buttons = gamma_count + n_count
    plot_rows = (max(total_buttons, 1) + DUAL_PLOT_COLUMNS - 1) // DUAL_PLOT_COLUMNS
    component_height = 140 + DUAL_SECTION_PADDING * 2 + DUAL_PLOT_HEIGHT * plot_rows * 2
    component_height = max(component_height, 700)
    plot_card_title_px = max(11, min(14, DUAL_PLOT_HEIGHT // 15))

    css = DUAL_PANEL_CSS
    css = css.replace("{{PLOT_MIN_WIDTH}}", str(DUAL_PLOT_MIN_WIDTH))
    css = css.replace("{{PLOT_HEIGHT}}", str(DUAL_PLOT_HEIGHT))
    css = css.replace("{{PLOT_CARD_TITLE_PX}}", str(plot_card_title_px))
    css = css.replace("{{BUTTON_MIN_WIDTH}}", str(DUAL_BUTTON_MIN_WIDTH))
    css = css.replace("{{BUTTON_MAX_WIDTH}}", str(DUAL_BUTTON_MAX_WIDTH))

    html = DUAL_PANEL_HTML
    html = html.replace("{{CSS}}", css)
    html = html.replace("{{JS}}", DUAL_PANEL_JS)
    html = html.replace("{{SERIES_JSON}}", series_json)
    html = html.replace("{{GAMMA_HTML}}", gamma_html)
    html = html.replace("{{N_HTML}}", n_html)
    html = html.replace("{{PLOT_TITLE_GAMMA}}", gamma_plot_title)
    html = html.replace("{{PLOT_TITLE_N}}", n_plot_title)

    components.html(html, height=component_height, scrolling=True)
