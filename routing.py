# routing.py
import re
from pathlib import Path

import numpy as np
import streamlit as st
import streamlit.components.v2 as components_v2
from streamlit_ace import st_ace

from algorithms_registry import (
    ALGORITHMS,
    CUSTOM_ALGORITHMS,
    FUNCTIONS,
    AlgorithmSpec,
    HyperparameterSpec,
    _compile_steps,
    get_algorithm_steps_code,
    get_base_algorithm_name,
    register_custom_algorithm,
    remove_custom_algorithm,
    run_algorithm,
)
from utils import (
    BASE_GAMMA_SPEC,
    BASE_N_SPEC,
    _build_pattern_param_values,
    _float_text_default,
    _parse_float_input,
    _parse_float_list,
    build_dual_section_html,
    build_dual_series_data,
    clear_algorithm_caches,
    compute,
    dual_ranking_by_slice,
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
        return re.sub(pattern, "def customized_algorithm(", code, count=1, flags=re.MULTILINE)
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
        st.session_state[code_key] = _editor_steps_source(spec)
        updated = st_ace(
            value=st.session_state.get(code_key, ""),
            language="python",
            key=editor_key,
            height=320,
            show_gutter=False,
            wrap=False,
            theme="github",
            auto_update=True,
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
                    steps = _compile_steps(steps_code)
                    temp_spec = AlgorithmSpec(
                        name=spec.name,
                        algo=steps,
                        function_slots=list(spec.function_slots),
                        default_function_keys=dict(spec.default_function_keys),
                    )
                    run_algorithm(
                        algo_spec=temp_spec,
                        function_config=test_context["function_config"],
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
    with sections[1]:
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

    with sections[0]:
        with st.container(border=True):
            st.write("Algorithm")
            function_config = {
                slot.key: {
                    "function_key": st.session_state["function_store"][algo_key][slot.key],
                    "function_params": dict(st.session_state["function_params_store"][algo_key][slot.key]),
                }
                for slot in spec.function_slots
            }
            _render_steps_editor(
                algo_key=algo_key,
                spec=spec,
                context="config",
                test_context={
                    "function_config": function_config,
                    "function_param_errors": list(function_param_errors),
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
                        clear_algorithm_caches(selected_custom)
                        remove_custom_algorithm(selected_custom)
                    except Exception as exc:
                        st.error(str(exc))
                    else:
                        st.success(f"Removed '{selected_custom}'.")
                        if st.session_state.get("selected_algorithm") == selected_custom:
                            st.session_state["selected_algorithm"] = None
                            st.session_state["pending_algorithm_select"] = next(iter(ALGORITHMS.keys()), None)
                        st.rerun()
    st.divider()
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
    st.session_state[_runs_key(algo_key)] = []
    st.session_state[_run_counter_key(algo_key)] = 0
    st.session_state.pop(_last_recompute_event_key(algo_key), None)
    st.session_state.pop(_last_cursor_event_key(algo_key), None)
    st.session_state.pop(_last_metric_event_key(algo_key), None)
    st.session_state.pop(_last_remove_event_key(algo_key), None)
    st.session_state[f"dual_selected_{algo_key}"] = []
    st.session_state[f"gamma_slider_{algo_key}"] = float(gamma_spec.min_value)
    st.session_state[f"n_slider_{algo_key}"] = float(n_spec.min_value)
    st.session_state[f"gamma_idx_{algo_key}"] = 0
    st.session_state[f"n_idx_{algo_key}"] = 0
    st.session_state[f"tau_local_n_idx_for_gamma_{algo_key}"] = 0
    st.session_state[f"tau_local_gamma_idx_for_n_{algo_key}"] = 0
    st.session_state[f"tau_pattern_gamma_{algo_key}"] = ""
    st.session_state[f"tau_pattern_n_{algo_key}"] = ""
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
    pattern_gamma_key = f"tau_pattern_gamma_{algo_key}"
    pattern_n_key = f"tau_pattern_n_{algo_key}"

    st.subheader(f"Results for `{spec.name}`")
    if st.button("Change gamma/n settings"):
        st.session_state["ui_phase"] = "config"
        st.rerun()

    with st.expander("Configuration details"):
        summary_lines = [
            f"**Algorithm**: `{spec.name}`",
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
    gamma_idx_key = f"gamma_idx_{algo_key}"
    n_idx_key = f"n_idx_{algo_key}"
    tau_local_n_idx_key = f"tau_local_n_idx_for_gamma_{algo_key}"
    tau_local_gamma_idx_key = f"tau_local_gamma_idx_for_n_{algo_key}"
    gamma_idx = int(st.session_state.get(gamma_idx_key, 0))
    n_idx = int(st.session_state.get(n_idx_key, 0))
    tau_local_n_idx_for_gamma = int(st.session_state.get(tau_local_n_idx_key, n_idx))
    tau_local_gamma_idx_for_n = int(st.session_state.get(tau_local_gamma_idx_key, gamma_idx))
    gamma_idx = max(0, min(gamma_idx, len(gamma_values) - 1))
    n_idx = max(0, min(n_idx, len(n_values) - 1))
    tau_local_n_idx_for_gamma = max(0, min(tau_local_n_idx_for_gamma, len(n_values) - 1))
    tau_local_gamma_idx_for_n = max(0, min(tau_local_gamma_idx_for_n, len(gamma_values) - 1))
    st.session_state[gamma_idx_key] = gamma_idx
    st.session_state[n_idx_key] = n_idx
    st.session_state[tau_local_n_idx_key] = tau_local_n_idx_for_gamma
    st.session_state[tau_local_gamma_idx_key] = tau_local_gamma_idx_for_n
    st.session_state.setdefault(pattern_gamma_key, "")
    st.session_state.setdefault(pattern_n_key, "")
    base_param_values, invalid_params, conflict_params = _build_pattern_param_values(settings["function_config"])
    warning_messages = set(cached_warnings)
    runs = st.session_state.setdefault(_runs_key(algo_key), [])
    run_results: dict[str, tuple] = {}
    for run in runs:
        selected_series_ids = tuple(sorted(set(run.get("selected_series_ids", []))))
        if not selected_series_ids:
            continue
        run_result = compute(
            algo_key,
            settings["gamma_spec"],
            settings["n_spec"],
            settings["function_config"],
            show_progress=False,
            rerun_nan_cache=bool(settings.get("rerun_nan_caches", False)),
            selected_dual_series_ids=selected_series_ids,
        )
        run_results[run["id"]] = run_result
        if isinstance(run_result, tuple) and len(run_result) >= 4:
            for warning in run_result[3]:
                warning_messages.add(f"{run['name']}: {warning}")

    tau_grid_payload = [
        [float(value) if value is not None and np.isfinite(value) else None for value in row] for row in tau_grid
    ]

    if warning_messages:
        warning_text = "\n".join(sorted(warning_messages))
        st.warning(
            "Some parameter combinations could not be solved; missing points are shown as gaps.\n" + warning_text
        )

    event = render_dual_values_panel(
        algo_key,
        duals_grid,
        gamma_values,
        n_values,
        gamma_idx,
        n_idx,
        runs,
        run_results,
        tau_payload={
            "gamma_values": [float(value) for value in gamma_values],
            "n_values": [float(value) for value in n_values],
            "gamma_spec": {
                "min": float(gamma_spec.min_value),
                "max": float(gamma_spec.max_value),
                "step": float(gamma_spec.step),
                "value_type": str(gamma_spec.value_type),
            },
            "n_spec": {
                "min": float(n_spec.min_value),
                "max": float(n_spec.max_value),
                "step": float(n_spec.step),
                "value_type": str(n_spec.value_type),
            },
            "tau_grid": tau_grid_payload,
            "default_gamma_idx": gamma_idx,
            "default_n_idx": n_idx,
            "default_local_n_idx_for_gamma": tau_local_n_idx_for_gamma,
            "default_local_gamma_idx_for_n": tau_local_gamma_idx_for_n,
            "pattern_gamma": str(st.session_state.get(pattern_gamma_key, "")),
            "pattern_n": str(st.session_state.get(pattern_n_key, "")),
            "pattern_params": {name: float(value) for name, value in sorted(base_param_values.items())},
            "pattern_invalid_params": [str(name) for name in invalid_params],
            "pattern_conflict_params": [str(name) for name in conflict_params],
        },
    )
    if event:
        event_type = str(event.get("type", ""))
        event_id = str(event.get("request_id", ""))
        last_event_key = _last_recompute_event_key(algo_key)
        if event_type == "cursor":
            cursor_event_key = _last_cursor_event_key(algo_key)
            if event_id and st.session_state.get(cursor_event_key) != event_id:
                next_gamma_idx = int(event.get("gamma_idx", gamma_idx))
                next_n_idx = int(event.get("n_idx", n_idx))
                next_local_n_idx_for_gamma = int(event.get("local_n_idx_for_gamma", tau_local_n_idx_for_gamma))
                next_local_gamma_idx_for_n = int(event.get("local_gamma_idx_for_n", tau_local_gamma_idx_for_n))
                st.session_state[pattern_gamma_key] = str(event.get("pattern_gamma", ""))
                st.session_state[pattern_n_key] = str(event.get("pattern_n", ""))
                st.session_state[gamma_idx_key] = max(0, min(next_gamma_idx, len(gamma_values) - 1))
                st.session_state[n_idx_key] = max(0, min(next_n_idx, len(n_values) - 1))
                st.session_state[tau_local_n_idx_key] = max(0, min(next_local_n_idx_for_gamma, len(n_values) - 1))
                st.session_state[tau_local_gamma_idx_key] = max(
                    0, min(next_local_gamma_idx_for_n, len(gamma_values) - 1)
                )
                st.session_state[cursor_event_key] = event_id
                st.rerun()
        elif event_type == "metric":
            metric_event_key = _last_metric_event_key(algo_key)
            if event_id and st.session_state.get(metric_event_key) != event_id:
                requested_metric = str(event.get("metric", ""))
                allowed_metrics = {
                    "non_zero_pct",
                    "non_zero_pct_with_none",
                    "std",
                    "std_with_none",
                    "median_abs",
                    "median_abs_with_none",
                    "mean_abs",
                    "mean_abs_with_none",
                }
                if requested_metric in allowed_metrics:
                    st.session_state[f"dual-ranking-metric-{algo_key}"] = requested_metric
                st.session_state[metric_event_key] = event_id
                st.rerun()
        elif event_type == "remove_run":
            remove_event_key = _last_remove_event_key(algo_key)
            if event_id and st.session_state.get(remove_event_key) != event_id:
                run_id = str(event.get("run_id", ""))
                st.session_state[pattern_gamma_key] = str(event.get("pattern_gamma", ""))
                st.session_state[pattern_n_key] = str(event.get("pattern_n", ""))
                if run_id:
                    runs[:] = [run for run in runs if str(run.get("id", "")) != run_id]
                st.session_state[remove_event_key] = event_id
                st.rerun()
        elif event_id and st.session_state.get(last_event_key) != event_id:
            if event_type == "recompute":
                active_series_ids = tuple(sorted(set(event.get("selected_series_ids", []))))
                deactivated_series_ids = list(dict.fromkeys(event.get("deactivated_series_ids", [])))
                deactivated_labels = list(event.get("deactivated_labels", []))
                next_local_n_idx_for_gamma = int(event.get("local_n_idx_for_gamma", tau_local_n_idx_for_gamma))
                next_local_gamma_idx_for_n = int(event.get("local_gamma_idx_for_n", tau_local_gamma_idx_for_n))
                st.session_state[pattern_gamma_key] = str(event.get("pattern_gamma", ""))
                st.session_state[pattern_n_key] = str(event.get("pattern_n", ""))
                st.session_state[tau_local_n_idx_key] = max(0, min(next_local_n_idx_for_gamma, len(n_values) - 1))
                st.session_state[tau_local_gamma_idx_key] = max(
                    0, min(next_local_gamma_idx_for_n, len(gamma_values) - 1)
                )
                with st.spinner("Recomputing tau grid with selected dual values..."):
                    recompute_result = compute(
                        algo_key,
                        settings["gamma_spec"],
                        settings["n_spec"],
                        settings["function_config"],
                        show_progress=True,
                        rerun_nan_cache=bool(settings.get("rerun_nan_caches", False)),
                        selected_dual_series_ids=active_series_ids,
                    )
                if recompute_result is None:
                    st.error("Unable to recompute tau grid for selected dual values.")
                else:
                    next_index = int(st.session_state.setdefault(_run_counter_key(algo_key), 0)) + 1
                    st.session_state[_run_counter_key(algo_key)] = next_index
                    runs.append(
                        {
                            "id": f"run-{next_index}",
                            "name": f"Run {next_index}",
                            "selected_series_ids": list(active_series_ids),
                            "deactivated_series_ids": deactivated_series_ids,
                            "deactivated_labels": deactivated_labels,
                            "selected_labels": deactivated_labels,
                            "visible": True,
                        }
                    )
                st.session_state[last_event_key] = event_id
                st.rerun()


UI_DIR = Path(__file__).resolve().parent / "ui"
DUAL_PANEL_CSS = (UI_DIR / "dual_panel.css").read_text()
DUAL_PANEL_V2_JS = (UI_DIR / "dual_panel.js").read_text()


DUAL_PANEL_COMPONENT = components_v2.component(
    "dual_panel_component_v2",
    html="<div id='dual-panel-v2-root'></div>",
    js=DUAL_PANEL_V2_JS,
)


def _runs_key(algo_key: str) -> str:
    return f"recompute_runs_{algo_key}"


def _run_counter_key(algo_key: str) -> str:
    return f"recompute_counter_{algo_key}"


def _last_recompute_event_key(algo_key: str) -> str:
    return f"recompute_event_{algo_key}"


def _last_cursor_event_key(algo_key: str) -> str:
    return f"cursor_event_{algo_key}"


def _last_metric_event_key(algo_key: str) -> str:
    return f"metric_event_{algo_key}"


def _last_remove_event_key(algo_key: str) -> str:
    return f"remove_run_event_{algo_key}"


def render_dual_values_panel(
    algo_key: str,
    duals_grid: list[list[dict]],
    gamma_values: np.ndarray,
    n_values: np.ndarray,
    gamma_idx: int,
    n_idx: int,
    runs: list[dict],
    run_results: dict[str, tuple],
    tau_payload: dict,
) -> dict | None:
    if not duals_grid:
        st.caption("No dual values available for these settings.")
        return None
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
    metric_state_key = f"dual-ranking-metric-{algo_key}"
    st.session_state.setdefault(metric_state_key, "non_zero_pct_with_none")
    metric = str(st.session_state.get(metric_state_key, "non_zero_pct_with_none"))
    if metric not in metric_labels:
        metric = "non_zero_pct_with_none"
        st.session_state[metric_state_key] = metric

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

    gamma_ranking_title = f"Ranking vs gamma (n = {n_values[n_idx]})"
    n_ranking_title = f"Ranking vs n (gamma = {gamma_values[gamma_idx]})"
    gamma_html, _ = build_dual_section_html(
        section_id=f"{algo_key}-gamma",
        section_key="gamma",
        title=gamma_ranking_title,
        dual_ranking=gamma_ranking,
        current_duals=current_duals,
    )
    n_html, _ = build_dual_section_html(
        section_id=f"{algo_key}-n",
        section_key="n",
        title=n_ranking_title,
        dual_ranking=n_ranking,
        current_duals=current_duals,
    )
    gamma_plot_title = f"Dual value vs gamma (n = {n_values[n_idx]})"
    n_plot_title = f"Dual value vs n (gamma = {gamma_values[gamma_idx]})"
    selected_series_ids = st.session_state.get(f"dual_selected_{algo_key}", [])
    dual_runs_data: list[dict] = []
    for run in runs:
        run_result = run_results.get(run["id"])
        if run_result is None:
            continue
        run_duals_grid = run_result[4]
        run_series_data = build_dual_series_data(
            run_duals_grid,
            gamma_values,
            n_values,
            gamma_idx,
            n_idx,
        )
        dual_runs_data.append(
            {
                "id": run["id"],
                "name": run["name"],
                "visible": bool(run.get("visible", True)),
                "selected_labels": [
                    str(label) for label in run.get("deactivated_labels", run.get("selected_labels", []))
                ],
                "tau_grid": [
                    [float(value) if value is not None and np.isfinite(value) else None for value in row]
                    for row in run_result[2]
                ],
                "series_data": run_series_data,
            }
        )
    result = DUAL_PANEL_COMPONENT(
        key=f"dual-panel-{algo_key}",
        data={
            "css": DUAL_PANEL_CSS,
            "tau_payload": tau_payload,
            "series_data": series_data,
            "dual_runs": dual_runs_data,
            "gamma_html": gamma_html,
            "n_html": n_html,
            "plot_title_gamma": gamma_plot_title,
            "plot_title_n": n_plot_title,
            "selected_series_ids": selected_series_ids,
            "metric": metric,
            "metric_labels": metric_labels,
        },
        height="content",
        isolate_styles=False,
        on_recompute_change=lambda: None,
    )
    if result is None:
        return None
    cursor = result.get("cursor")
    if isinstance(cursor, dict):
        return {"type": "cursor", **cursor}
    metric_change = result.get("metric")
    if isinstance(metric_change, dict):
        return {"type": "metric", **metric_change}
    remove_run = result.get("remove_run")
    if isinstance(remove_run, dict):
        return {"type": "remove_run", **remove_run}
    recompute = result.get("recompute")
    if isinstance(recompute, dict):
        return {"type": "recompute", **recompute}
    return None
