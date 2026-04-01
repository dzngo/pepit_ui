# routing.py
import re
from math import isfinite, prod
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v2 as components_v2
from streamlit_ace import st_ace

from algorithm.algorithm_custom import (
    ALGORITHMS,
    CUSTOM_ALGORITHMS,
    compile_algorithm_body,
    get_algorithm_steps_code,
    get_base_algorithm_name,
    register_custom_algorithm,
    remove_custom_algorithm,
)
from algorithm.function_registry import FUNCTIONS
from algorithm.runtime import run_algorithm
from algorithm.types import (
    AlgorithmSpec,
    HyperparameterSpec,
    default_gamma_n_hyperparameters,
)
from utils import (
    _build_pattern_param_values,
    _float_text_default,
    _parse_float_input,
    _parse_float_list,
    build_dual_section_html,
    build_dual_series_by_param,
    build_dual_slice_by_param,
    build_tau_series_by_param,
    clear_algorithm_caches,
    compute,
    discrete_values,
    dual_ranking_by_slice,
)


def init_session_state():
    st.session_state.setdefault("ui_phase", "config")
    st.session_state.setdefault("selected_algorithm", None)
    st.session_state.setdefault("hyperparameter_store", {})
    st.session_state.setdefault("pending_settings", None)
    st.session_state.setdefault("active_settings", None)
    st.session_state.setdefault("rerun_nan_caches", False)
    st.session_state.setdefault("function_rows_store", {})
    st.session_state.setdefault("function_row_counter_store", {})


def reset_for_algorithm_change(algo_key: str):
    st.session_state["selected_algorithm"] = algo_key
    st.session_state["ui_phase"] = "config"
    st.session_state["pending_settings"] = None
    st.session_state["active_settings"] = None


_HYPERPARAM_COLUMNS = ("name", "label", "value_type", "min", "max", "step", "default")
_HYPERPARAM_RESERVED_NAMES = {"x", "pi", "e"}
_FUNCTION_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_RUNTIME_RESERVED_NAMES = {
    "problem",
    "funcs",
    "params",
    "customized_algorithm",
    "PEP",
    "Point",
    "Function",
    "Dict",
    "np",
    "sqrt",
}


def _hyperparameter_rows_from_specs(specs: list[HyperparameterSpec]) -> list[dict]:
    return [
        {
            "name": hp.name,
            "label": hp.label,
            "value_type": hp.value_type,
            "min": float(hp.min_value),
            "max": float(hp.max_value),
            "step": float(hp.step),
            "default": float(hp.default),
        }
        for hp in specs
    ]


def _render_hyperparameter_editor(algo_key: str, spec: AlgorithmSpec) -> tuple[list[HyperparameterSpec], list[str]]:
    store = st.session_state["hyperparameter_store"]
    if algo_key not in store:
        store[algo_key] = _hyperparameter_rows_from_specs(spec.default_hyperparameters)
    # Keep editor widget identity separate from persisted config to avoid
    # st.data_editor "every second edit reverts" behavior.
    editor_version_key = f"hyperparam-editor-version-{algo_key}"
    editor_version = int(st.session_state.get(editor_version_key, 0))

    def _bump_editor_version() -> None:
        # Rotating the widget key forces a fresh editor instance after preset/reset.
        st.session_state[editor_version_key] = int(st.session_state.get(editor_version_key, 0)) + 1

    top_left, top_right = st.columns([1, 1])
    with top_left:
        if st.button("Use gamma/n quick preset", key=f"btn-hp-preset-{algo_key}"):
            store[algo_key] = _hyperparameter_rows_from_specs(default_gamma_n_hyperparameters())
            _bump_editor_version()
            st.rerun()
    with top_right:
        if st.button("Reset to algorithm defaults", key=f"btn-hp-reset-{algo_key}"):
            store[algo_key] = _hyperparameter_rows_from_specs(spec.default_hyperparameters)
            _bump_editor_version()
            st.rerun()

    rows = store.get(algo_key, [])
    df = pd.DataFrame(rows, columns=list(_HYPERPARAM_COLUMNS))
    editor_widget_key = f"hyperparam-editor-{algo_key}-v{editor_version}"
    edited = st.data_editor(
        df,
        key=editor_widget_key,
        num_rows="dynamic",
        hide_index=True,
        width="stretch",
        column_config={
            "name": st.column_config.TextColumn("Name", required=True, help="Unique parameter id used in code."),
            "label": st.column_config.TextColumn("Label", required=True),
            "value_type": st.column_config.SelectboxColumn("Type", options=["float", "int"], required=True),
            "min": st.column_config.NumberColumn("Min", format="%.8g", required=True),
            "max": st.column_config.NumberColumn("Max", format="%.8g", required=True),
            "step": st.column_config.NumberColumn("Step", format="%.8g", required=True),
            "default": st.column_config.NumberColumn("Default", format="%.8g", required=True),
        },
    )
    edited_rows = [{col: row.get(col) for col in _HYPERPARAM_COLUMNS} for row in edited.to_dict(orient="records")]
    # store[algo_key] = edited_rows

    return _parse_hyperparameter_specs(edited_rows)


def _parse_hyperparameter_specs(rows: list[dict]) -> tuple[list[HyperparameterSpec], list[str]]:
    def _is_blank(value: object) -> bool:
        if value is None:
            return True
        if isinstance(value, str) and not value.strip():
            return True
        try:
            return bool(pd.isna(value))
        except Exception:
            return False

    errors: list[str] = []
    specs: list[HyperparameterSpec] = []
    seen_names: set[str] = set()

    for row_idx, row in enumerate(rows, start=1):
        raw_name = str(row.get("name") or "").strip()
        raw_label = str(row.get("label") or "").strip()
        raw_type = str(row.get("value_type") or "float").strip().lower()
        row_is_empty = all(_is_blank(row.get(col)) for col in ("name", "label", "min", "max", "step", "default"))
        if row_is_empty:
            continue
        if not raw_name:
            errors.append(f"Hyperparameter row {row_idx}: name is required.")
            continue
        if raw_name in seen_names:
            errors.append(f"Hyperparameter row {row_idx}: duplicate name '{raw_name}'.")
            continue
        seen_names.add(raw_name)
        if raw_name in _HYPERPARAM_RESERVED_NAMES:
            errors.append(f"Hyperparameter row {row_idx}: '{raw_name}' is reserved.")
            continue
        if raw_name in _RUNTIME_RESERVED_NAMES:
            errors.append(f"Hyperparameter row {row_idx}: '{raw_name}' is reserved.")
            continue
        if raw_type not in {"float", "int"}:
            errors.append(f"Hyperparameter row {row_idx}: type must be 'float' or 'int'.")
            continue

        numeric: dict[str, float] = {}
        numeric_ok = True
        for key in ("min", "max", "step", "default"):
            try:
                numeric[key] = float(row.get(key))
            except (TypeError, ValueError):
                errors.append(f"Hyperparameter row {row_idx}: invalid numeric value for '{key}'.")
                numeric_ok = False
                break
        if not numeric_ok:
            continue
        if numeric["max"] <= numeric["min"]:
            errors.append(f"Hyperparameter row {row_idx}: max must be greater than min.")
            continue
        if numeric["step"] <= 0:
            errors.append(f"Hyperparameter row {row_idx}: step must be positive.")
            continue
        if not (numeric["min"] <= numeric["default"] <= numeric["max"]):
            errors.append(f"Hyperparameter row {row_idx}: default must be between min and max.")
            continue
        if raw_type == "int":
            int_fields = ("min", "max", "step", "default")
            if any(abs(numeric[field] - round(numeric[field])) > 1e-9 for field in int_fields):
                errors.append(f"Hyperparameter row {row_idx}: int type requires integer min/max/step/default.")
                continue

        label = raw_label or raw_name
        specs.append(
            HyperparameterSpec(
                name=raw_name,
                label=label,
                min_value=float(int(numeric["min"])) if raw_type == "int" else float(numeric["min"]),
                max_value=float(int(numeric["max"])) if raw_type == "int" else float(numeric["max"]),
                default=float(int(numeric["default"])) if raw_type == "int" else float(numeric["default"]),
                step=float(int(numeric["step"])) if raw_type == "int" else float(numeric["step"]),
                value_type=raw_type,
            )
        )

    if not specs:
        errors.append("At least one hyperparameter is required.")
    return specs, errors


def _cursor_indices_key(algo_key: str) -> str:
    return f"cursor_indices_by_param_{algo_key}"


def _local_cursor_indices_by_axis_key(algo_key: str) -> str:
    return f"local_cursor_indices_by_axis_{algo_key}"


def _patterns_by_param_key(algo_key: str) -> str:
    return f"tau_patterns_by_param_{algo_key}"


def _default_index_for_spec(spec: HyperparameterSpec) -> int:
    total = int(round((spec.max_value - spec.min_value) / spec.step))
    idx = int(round((spec.default - spec.min_value) / spec.step))
    return int(min(max(idx, 0), total))


def _default_cursor_indices(specs: list[HyperparameterSpec]) -> dict[str, int]:
    return {hp.name: _default_index_for_spec(hp) for hp in specs}


def _clamp_cursor_indices(indices: dict[str, int], specs: list[HyperparameterSpec]) -> dict[str, int]:
    clamped: dict[str, int] = {}
    for hp in specs:
        values = discrete_values(hp)
        max_idx = max(len(values) - 1, 0)
        raw_idx = int(indices.get(hp.name, _default_index_for_spec(hp)))
        clamped[hp.name] = max(0, min(raw_idx, max_idx))
    return clamped


def _param_values_by_name(specs: list[HyperparameterSpec]) -> dict[str, list[float]]:
    return {hp.name: [float(v) for v in discrete_values(hp)] for hp in specs}


def _function_row_id_key(algo_key: str) -> str:
    return f"function-row-{algo_key}-"


def _next_function_row_id(algo_key: str) -> str:
    counters = st.session_state["function_row_counter_store"]
    next_id = int(counters.get(algo_key, 0)) + 1
    counters[algo_key] = next_id
    return f"r{next_id}"


def _default_function_rows_from_spec(spec: AlgorithmSpec) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for slot in spec.function_slots:
        rows.append(
            {
                "id": f"slot-{slot.key}",
                "name": slot.key,
                "function_key": spec.default_function_keys.get(slot.key, ""),
                "function_params": {},
            }
        )
    return rows


def _sanitize_function_rows(
    algo_key: str, rows: list[dict[str, object]], spec: AlgorithmSpec
) -> list[dict[str, object]]:
    function_names = sorted(FUNCTIONS.keys())
    fallback_function_key = function_names[0] if function_names else ""
    normalized: list[dict[str, object]] = []
    for idx, row in enumerate(rows):
        row_id = str(row.get("id") or "").strip()
        if not row_id:
            row_id = _next_function_row_id(algo_key)
        name = str(row.get("name") or "").strip()
        if not name:
            name = f"f{idx + 1}"
        function_key = str(row.get("function_key") or "").strip()
        if function_key not in FUNCTIONS:
            function_key = fallback_function_key
        function_params = row.get("function_params")
        if not isinstance(function_params, dict):
            function_params = {}
        normalized.append(
            {
                "id": row_id,
                "name": name,
                "function_key": function_key,
                "function_params": dict(function_params),
            }
        )

    if not normalized:
        normalized = _default_function_rows_from_spec(spec)
        normalized = _sanitize_function_rows(algo_key, normalized, spec)
    return normalized


def _get_function_rows(algo_key: str, spec: AlgorithmSpec) -> list[dict[str, object]]:
    rows_store = st.session_state["function_rows_store"]
    if algo_key not in rows_store:
        rows_store[algo_key] = _default_function_rows_from_spec(spec)
    rows_store[algo_key] = _sanitize_function_rows(algo_key, list(rows_store.get(algo_key, [])), spec)
    return rows_store[algo_key]


def _suggest_new_function_name(rows: list[dict[str, object]]) -> str:
    used = {str(row.get("name") or "").strip() for row in rows}
    candidate = "f"
    if candidate not in used:
        return candidate
    idx = 1
    while True:
        candidate = f"f{idx}"
        if candidate not in used:
            return candidate
        idx += 1


def _validate_function_rows(rows: list[dict[str, object]]) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    for idx, row in enumerate(rows, start=1):
        name = str(row.get("name") or "").strip()
        function_key = str(row.get("function_key") or "").strip()
        if not name:
            errors.append(f"Function row {idx}: name is required.")
        elif not _FUNCTION_NAME_PATTERN.fullmatch(name):
            errors.append(f"Function row {idx}: invalid name '{name}'. Use letters, numbers, and '_' only.")
        elif name in _RUNTIME_RESERVED_NAMES:
            errors.append(f"Function row {idx}: '{name}' is reserved.")
        elif name in seen:
            errors.append(f"Function row {idx}: duplicate name '{name}'.")
        else:
            seen.add(name)
        if not function_key:
            errors.append(f"Function row {idx}: function type is required.")
        elif function_key not in FUNCTIONS:
            errors.append(f"Function row {idx}: unknown function type '{function_key}'.")
    return errors


def _build_function_config(algo_key: str, spec: AlgorithmSpec) -> dict[str, dict[str, object]]:
    rows = _get_function_rows(algo_key, spec)
    function_config: dict[str, dict[str, object]] = {}
    for row in rows:
        row_id = str(row.get("id") or _next_function_row_id(algo_key))
        name = str(row.get("name") or "").strip()
        function_config[row_id] = {
            "function_key": str(row.get("function_key") or "").strip(),
            "function_params": dict(row.get("function_params") or {}),
            "alias": name,
        }
    return function_config


def _default_local_cursor_indices_by_axis(
    specs: list[HyperparameterSpec],
    cursor_indices: dict[str, int],
) -> dict[str, dict[str, int]]:
    names = [hp.name for hp in specs]
    return {
        axis: {
            name: int(cursor_indices.get(name, _default_index_for_spec(next_hp)))
            for name, next_hp in ((hp.name, hp) for hp in specs)
            if name != axis
        }
        for axis in names
    }


def _clamp_local_cursor_indices_by_axis(
    local_by_axis: dict[str, dict[str, int]],
    specs: list[HyperparameterSpec],
    cursor_indices: dict[str, int],
) -> dict[str, dict[str, int]]:
    names = [hp.name for hp in specs]
    clamped_cursor = _clamp_cursor_indices(cursor_indices, specs)
    clamped: dict[str, dict[str, int]] = {}
    for axis in names:
        incoming_axis = local_by_axis.get(axis, {})
        axis_map: dict[str, int] = {}
        for hp in specs:
            if hp.name == axis:
                continue
            values = discrete_values(hp)
            max_idx = max(len(values) - 1, 0)
            fallback = int(clamped_cursor.get(hp.name, _default_index_for_spec(hp)))
            raw_idx = int(incoming_axis.get(hp.name, fallback))
            axis_map[hp.name] = max(0, min(raw_idx, max_idx))
        clamped[axis] = axis_map
    return clamped


def _steps_source(spec: AlgorithmSpec) -> str:
    return get_algorithm_steps_code(spec.name)


def _editor_steps_source(spec: AlgorithmSpec) -> str:
    return _steps_source(spec)


def _run_steps_smoke_test(
    *,
    spec: AlgorithmSpec,
    context: str,
    algo_key: str,
    test_context: dict,
) -> str | None:
    if test_context["function_param_errors"]:
        return "; ".join(test_context["function_param_errors"])
    if test_context.get("function_row_errors"):
        return "; ".join(test_context["function_row_errors"])
    if test_context["runtime_param_errors"]:
        return "; ".join(test_context["runtime_param_errors"])
    try:
        open_key = f"customize-open-{context}-{algo_key}"
        code_key = f"customize-code-{context}-{algo_key}"
        is_custom_editor_open = bool(st.session_state.get(open_key, False))
        if is_custom_editor_open:
            default_steps = _editor_steps_source(spec)
            steps_code = st.session_state.get(code_key, default_steps)
            steps = compile_algorithm_body(steps_code)
        else:
            # For non-customized flow, use the registered algorithm callable directly.
            steps = spec.algo
        temp_spec = AlgorithmSpec(
            name=spec.name,
            algo=steps,
            function_slots=list(spec.function_slots),
            default_function_keys=dict(spec.default_function_keys),
            default_hyperparameters=list(spec.default_hyperparameters),
        )
        configured_hyperparameters: list[HyperparameterSpec] = list(test_context.get("hyperparameter_specs", []))
        algo_params: dict[str, float | int] = {}
        for hp in configured_hyperparameters:
            value = hp.default
            algo_params[hp.name] = int(round(value)) if hp.value_type == "int" else float(value)
        run_algorithm(
            algo_spec=temp_spec,
            function_config=test_context["function_config"],
            algo_params=algo_params,
        )
    except KeyError as exc:
        missing_key = exc.args[0] if exc.args else "<unknown>"
        return f"key not found: {missing_key!r}"
    except Exception as exc:
        return str(exc)
    return None


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
                hyperparameter_store = st.session_state.get("hyperparameter_store", {})
                if isinstance(hyperparameter_store, dict):
                    copied_rows: list[dict] = []
                    if test_context:
                        configured_specs = list(test_context.get("hyperparameter_specs", []))
                        if configured_specs:
                            copied_rows = _hyperparameter_rows_from_specs(configured_specs)
                    if not copied_rows:
                        current_rows = hyperparameter_store.get(algo_key, [])
                        copied_rows = [dict(row) for row in current_rows]
                    hyperparameter_store[name] = copied_rows
                # Preserve function configuration when switching to the newly
                # saved custom algorithm key.
                function_rows_store = st.session_state.get("function_rows_store", {})
                if isinstance(function_rows_store, dict):
                    source_rows = function_rows_store.get(algo_key, [])
                    copied_function_rows: list[dict] = []
                    for row in source_rows:
                        copied_row = dict(row)
                        copied_row["function_params"] = dict(row.get("function_params", {}))
                        copied_function_rows.append(copied_row)
                    function_rows_store[name] = copied_function_rows
                function_row_counter_store = st.session_state.get("function_row_counter_store", {})
                if isinstance(function_row_counter_store, dict):
                    function_row_counter_store[name] = int(function_row_counter_store.get(algo_key, 0))
                st.session_state[open_key] = False
                st.session_state["pending_algorithm_select"] = name
                st.session_state["selected_algorithm"] = None
                st.session_state["ui_phase"] = "config"
                st.rerun()
        if cancel_clicked:
            st.session_state[open_key] = False
            st.rerun()
        if test_context and test_clicked:
            error_message = _run_steps_smoke_test(
                spec=spec,
                context=context,
                algo_key=algo_key,
                test_context=test_context,
            )
            if error_message:
                st.error(f"Test failed: {error_message}")
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
            st.write("Hyperparameter config")
            hyperparameter_specs, hyperparameter_errors = _render_hyperparameter_editor(algo_key, spec)
            for error in hyperparameter_errors:
                st.error(error)

            runtime_param_errors: list[str] = []
        with st.container(border=True):
            st.write("Functions")
            function_rows = _get_function_rows(algo_key, spec)
            function_names = sorted(FUNCTIONS.keys())
            default_function_key = function_names[0] if function_names else ""
            function_param_errors: list[str] = []
            if not function_names:
                st.error("No function types are registered.")
            else:
                st.caption("Each block defines `funcs[name]` and its function class.")

                remove_row_id: str | None = None
                for idx, row in enumerate(function_rows, start=1):
                    row_id = str(row.get("id") or _next_function_row_id(algo_key))
                    row["id"] = row_id
                    with st.container(border=True):
                        name_col, function_type_col = st.columns([1, 3])
                        with name_col:
                            name_value = st.text_input(
                                "name",
                                value=str(row.get("name", "")),
                                key=f"{_function_row_id_key(algo_key)}name-{row_id}",
                            )
                        with function_type_col:
                            selected_function = str(row.get("function_key") or "")
                            if selected_function not in function_names:
                                selected_function = default_function_key
                            selected_function = st.selectbox(
                                "function type",
                                options=function_names,
                                index=(
                                    function_names.index(selected_function)
                                    if selected_function in function_names
                                    else 0
                                ),
                                key=f"{_function_row_id_key(algo_key)}type-{row_id}",
                            )

                        row["name"] = name_value.strip()
                        row["function_key"] = selected_function

                        function_spec = FUNCTIONS[selected_function]
                        slot_params = row.setdefault("function_params", {})
                        if not isinstance(slot_params, dict):
                            slot_params = {}
                            row["function_params"] = slot_params
                        display_name = row["name"] or f"row {idx}"
                        if not function_spec.parameters:
                            st.caption(f"{display_name} has no required parameters.")
                        else:
                            columns = st.columns(3)
                            for param_idx, param in enumerate(function_spec.parameters):
                                with columns[param_idx % 3]:
                                    with st.container(border=True):
                                        input_key = f"{_function_row_id_key(algo_key)}param-{row_id}-{param.name}"
                                        param_context = (
                                            f"{display_name} ({function_spec.cls.__name__}), parameter {param.name}"
                                        )
                                        if param.param_type == "float":
                                            default_text = _float_text_default(
                                                slot_params.get(param.name, param.default)
                                            )
                                            st.session_state.setdefault(input_key, default_text)
                                            raw_value = st.text_input(param.name, key=input_key)
                                            parsed_value, error = _parse_float_input(raw_value)
                                            if error:
                                                function_param_errors.append(f"{param_context}: {error}")
                                            else:
                                                if param.required and parsed_value is None:
                                                    function_param_errors.append(f"{param_context}: value required.")
                                                elif parsed_value is not None and parsed_value < 0:
                                                    function_param_errors.append(
                                                        f"{param_context}: value must be >= 0."
                                                    )
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
                                                "Partition will be created via "
                                                "`problem.declare_block_partition(d=...)`."
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
                                            desc_parts.append(
                                                "When checked, a Point is created and passed as `center`."
                                            )
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

                        if st.button("Remove", key=f"{_function_row_id_key(algo_key)}remove-{row_id}"):
                            remove_row_id = row_id

                if remove_row_id is not None:
                    updated_rows = [row for row in function_rows if str(row.get("id")) != remove_row_id]
                    st.session_state["function_rows_store"][algo_key] = updated_rows
                    st.rerun()

            if st.button("Add function", key=f"btn-add-function-{algo_key}"):
                function_rows.append(
                    {
                        "id": _next_function_row_id(algo_key),
                        "name": _suggest_new_function_name(function_rows),
                        "function_key": default_function_key,
                        "function_params": {},
                    }
                )
                st.rerun()

            function_row_errors = _validate_function_rows(function_rows)
            for error in function_row_errors:
                st.error(error)

    with sections[0]:
        with st.container(border=True):
            st.write("Algorithm")
            function_config = _build_function_config(algo_key, spec)
            _render_steps_editor(
                algo_key=algo_key,
                spec=spec,
                context="config",
                test_context={
                    "function_config": function_config,
                    "function_param_errors": list(function_param_errors),
                    "function_row_errors": list(function_row_errors),
                    "runtime_param_errors": runtime_param_errors,
                    "hyperparameter_specs": list(hyperparameter_specs),
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
        errors = list(hyperparameter_errors)
        errors.extend(runtime_param_errors)
        errors.extend(function_param_errors)
        errors.extend(function_row_errors)
        if errors:
            for error in errors:
                st.error(error)
            return
        function_config = _build_function_config(algo_key, spec)
        plot_test_context = {
            "function_config": function_config,
            "function_param_errors": list(function_param_errors),
            "function_row_errors": list(function_row_errors),
            "runtime_param_errors": runtime_param_errors,
            "hyperparameter_specs": list(hyperparameter_specs),
        }
        test_error = _run_steps_smoke_test(
            spec=spec,
            context="config",
            algo_key=algo_key,
            test_context=plot_test_context,
        )
        if test_error:
            st.error(f"Algorithm test failed: {test_error}")
            return
        st.session_state["pending_settings"] = {
            "algo_key": algo_key,
            "hyperparameter_specs": list(hyperparameter_specs),
            "function_config": function_config,
            "rerun_nan_caches": bool(st.session_state.get("rerun_nan_caches", False)),
        }
        st.session_state.pop(_loading_progress_key(algo_key), None)
        st.session_state["ui_phase"] = "loading"
        st.rerun()


def render_loading_phase(algo_key: str, spec):
    pending = st.session_state.get("pending_settings")
    if not pending or pending["algo_key"] != algo_key:
        st.session_state.pop(_loading_progress_key(algo_key), None)
        st.session_state["ui_phase"] = "config"
        st.rerun()

    hyperparameter_specs = list(pending.get("hyperparameter_specs", []))
    st.subheader(f"Computing tau values for `{spec.name}`")

    with st.container(border=True):
        summary_lines = [
            f"**Algorithm**: `{spec.name}`",
        ]
        for hp in hyperparameter_specs:
            summary_lines.append(
                f"**{hp.name}**: [{hp.min_value}, {hp.max_value}], "
                f"step_size={hp.step}, default={hp.default}, type={hp.value_type}"
            )
        st.markdown("<br>".join(summary_lines), unsafe_allow_html=True)
        st.markdown("**Steps**")
        st.code(_steps_source(spec), language="python")
        st.markdown("**Functions**")
        for slot_key, slot_config in sorted(pending["function_config"].items()):
            alias = str(slot_config.get("alias", slot_key) or slot_key)
            st.markdown(f"{alias}: `{slot_config['function_key']}`")
            if slot_config["function_params"]:
                params_line = ", ".join(f"{name}={value}" for name, value in slot_config["function_params"].items())
                st.markdown(f"*params*: {params_line}")
            else:
                st.markdown("*params*: `{}`")

    progress_key = _loading_progress_key(algo_key)
    # Keep batches small enough so the fragment remains responsive to "Interrupt".
    batch_size = 24
    total_estimate = prod(len(discrete_values(hp)) for hp in hyperparameter_specs) if hyperparameter_specs else 0

    # Fragment reruns independently, so loading UI/progress can update without full-page rerun churn.
    @st.fragment(run_every=0.25)
    def _loading_fragment() -> None:
        current_pending = st.session_state.get("pending_settings")
        if not current_pending or current_pending.get("algo_key") != algo_key:
            return

        controls_col, progress_col = st.columns([1, 3])
        with controls_col:
            if st.button("Interrupt", key=f"btn-interrupt-loading-{algo_key}"):
                st.session_state.pop(progress_key, None)
                st.session_state["pending_settings"] = None
                st.session_state["ui_phase"] = "config"
                st.rerun()

        # Render progress from the previous batch before computing the next one.
        progress_state = st.session_state.get(progress_key, {})
        done = int(progress_state.get("done", 0))
        total = int(progress_state.get("total", total_estimate))
        if total > 0:
            fraction = min(max(done / total, 0.0), 1.0)
            with progress_col:
                st.progress(fraction)
                st.caption(f"Computing grid… {done}/{total}")

        result = compute(
            algo_key,
            current_pending["function_config"],
            hyperparameter_specs,
            show_progress=False,
            rerun_nan_cache=bool(current_pending.get("rerun_nan_caches", False)),
            # Process one batch per fragment tick; return None while work remains.
            batch_size=batch_size,
            progress_state_key=progress_key,
        )

        if result is None:
            return

        st.session_state.pop(progress_key, None)
        st.session_state["active_settings"] = current_pending
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
        cursor_indices = _default_cursor_indices(hyperparameter_specs)
        local_cursor_indices_by_axis = _default_local_cursor_indices_by_axis(hyperparameter_specs, cursor_indices)
        patterns_by_param = {hp.name: "" for hp in hyperparameter_specs}
        st.session_state[_cursor_indices_key(algo_key)] = cursor_indices
        st.session_state[_local_cursor_indices_by_axis_key(algo_key)] = local_cursor_indices_by_axis
        st.session_state[_patterns_by_param_key(algo_key)] = patterns_by_param
        st.rerun()

    _loading_fragment()


def render_results_phase(algo_key: str, spec):
    settings = st.session_state.get("active_settings")
    if not settings or settings["algo_key"] != algo_key:
        st.session_state["ui_phase"] = "config"
        st.rerun()

    result = compute(
        algo_key,
        settings["function_config"],
        list(settings.get("hyperparameter_specs", [])),
        show_progress=False,
        rerun_nan_cache=bool(settings.get("rerun_nan_caches", False)),
    )
    if result is None:
        st.session_state["pending_settings"] = settings
        st.session_state["ui_phase"] = "loading"
        st.rerun()

    _, _, cached_warnings, _ = result
    hyperparameter_specs = list(settings.get("hyperparameter_specs", []))
    specs_by_name = {hp.name: hp for hp in hyperparameter_specs}
    param_values_by_name = _param_values_by_name(hyperparameter_specs)

    st.subheader(f"Results for `{spec.name}`")
    if st.button("Change hyperparameter settings"):
        st.session_state["ui_phase"] = "config"
        st.rerun()
    metric_state_key = f"dual-ranking-metric-{algo_key}"
    metric = str(st.session_state.get(metric_state_key, "non_zero_pct_with_none"))

    with st.expander("Configuration details"):
        summary_lines = [
            f"**Algorithm**: `{spec.name}`",
        ]
        for hp in settings.get("hyperparameter_specs", []):
            summary_lines.append(
                f"**{hp.name}**: [{hp.min_value}, {hp.max_value}], "
                f"step_size={hp.step}, default={hp.default}, type={hp.value_type}"
            )
        st.markdown("<br>".join(summary_lines), unsafe_allow_html=True)
        st.markdown("**Steps**")
        st.code(_steps_source(spec), language="python")
        st.markdown("**Functions**")
        for slot_key, slot_config in sorted(settings["function_config"].items()):
            alias = str(slot_config.get("alias", slot_key) or slot_key)
            st.markdown(f"{alias}: `{slot_config['function_key']}`")
            if slot_config["function_params"]:
                params_line = ", ".join(f"{name}={value}" for name, value in slot_config["function_params"].items())
                st.markdown(f"*params*: {params_line}")
            else:
                st.markdown("*params*: `{}`")
    cursor_state_key = _cursor_indices_key(algo_key)
    local_axis_state_key = _local_cursor_indices_by_axis_key(algo_key)
    pattern_state_key = _patterns_by_param_key(algo_key)
    default_cursor = _default_cursor_indices(hyperparameter_specs)
    cursor_indices = _clamp_cursor_indices(
        dict(st.session_state.get(cursor_state_key, default_cursor)),
        hyperparameter_specs,
    )
    local_cursor_indices_by_axis = _clamp_local_cursor_indices_by_axis(
        dict(st.session_state.get(local_axis_state_key, {})),
        hyperparameter_specs,
        cursor_indices,
    )
    patterns_by_param = dict(st.session_state.get(pattern_state_key, {}))
    for hp in hyperparameter_specs:
        patterns_by_param[hp.name] = str(patterns_by_param.get(hp.name, ""))

    cursor_indices = _clamp_cursor_indices(cursor_indices, hyperparameter_specs)
    st.session_state[cursor_state_key] = cursor_indices
    st.session_state[local_axis_state_key] = local_cursor_indices_by_axis
    st.session_state[pattern_state_key] = patterns_by_param
    base_param_values, invalid_params, conflict_params = _build_pattern_param_values(settings["function_config"])
    warning_messages = set(cached_warnings)
    runs = st.session_state.setdefault(_runs_key(algo_key), [])
    run_tau_series_by_param: dict[str, dict[str, dict[str, object]]] = {}
    run_series_data_by_param: dict[str, dict] = {}
    for run in runs:
        selected_series_ids = tuple(sorted(set(run.get("selected_series_ids", []))))
        if not selected_series_ids:
            continue
        run_result = compute(
            algo_key,
            settings["function_config"],
            hyperparameter_specs,
            show_progress=False,
            rerun_nan_cache=bool(settings.get("rerun_nan_caches", False)),
            selected_dual_series_ids=selected_series_ids,
        )
        if run_result is not None:
            run_param_values, run_tau_nd, _, run_duals_nd = run_result
            run_tau_series_by_param[run["id"]] = build_tau_series_by_param(
                hyperparameter_specs,
                run_param_values,
                run_tau_nd,
                local_cursor_indices_by_axis,
            )
            run_series_data_by_param[run["id"]] = build_dual_series_by_param(
                run_duals_nd,
                run_param_values,
                hyperparameter_specs,
                cursor_indices,
            )
            for warning in run_result[2]:
                warning_messages.add(f"{run['name']}: {warning}")

    if warning_messages:
        warning_text = "\n".join(sorted(warning_messages))
        st.warning(
            "Some parameter combinations could not be solved; missing points are shown as gaps.\n" + warning_text
        )

    nd_result = compute(
        algo_key,
        settings["function_config"],
        hyperparameter_specs,
        show_progress=False,
        rerun_nan_cache=bool(settings.get("rerun_nan_caches", False)),
    )
    tau_series_by_param: dict[str, dict[str, object]] = {}
    series_data_by_param: dict[str, dict] = {}
    sections_html_by_param: dict[str, str] = {}
    plot_titles_by_param: dict[str, str] = {}
    if nd_result is not None:
        param_values_nd, tau_nd, _, duals_nd = nd_result
        tau_series_by_param = build_tau_series_by_param(
            hyperparameter_specs,
            param_values_nd,
            tau_nd,
            local_cursor_indices_by_axis,
        )
        series_data_by_param = build_dual_series_by_param(
            duals_nd,
            param_values_nd,
            hyperparameter_specs,
            cursor_indices,
        )
        axis_index = {hp.name: idx for idx, hp in enumerate(hyperparameter_specs)}
        base_idx = []
        for hp in hyperparameter_specs:
            values = param_values_nd.get(hp.name, [])
            max_idx = max(len(values) - 1, 0)
            base_idx.append(max(0, min(int(cursor_indices.get(hp.name, 0)), max_idx)))
        current_duals = duals_nd[tuple(base_idx)] if hyperparameter_specs else {}
        for hp in hyperparameter_specs:
            slice_duals = build_dual_slice_by_param(
                duals_nd,
                hyperparameter_specs,
                cursor_indices,
                hp.name,
            )
            ranking = dual_ranking_by_slice(slice_duals, metric=metric)
            fixed_parts = []
            for other_hp in hyperparameter_specs:
                if other_hp.name == hp.name:
                    continue
                other_axis = axis_index[other_hp.name]
                other_idx = base_idx[other_axis]
                other_values = param_values_nd[other_hp.name]
                if len(other_values):
                    fixed_parts.append(f"{other_hp.name} = {other_values[other_idx]}")
            fixed_text = ", ".join(fixed_parts) if fixed_parts else "no fixed parameters"
            title = f"Ranking vs {hp.name} ({fixed_text})"
            html_value, _ = build_dual_section_html(
                section_id=f"{algo_key}-{hp.name}",
                section_key=hp.name,
                title=title,
                dual_ranking=ranking,
                current_duals=current_duals if isinstance(current_duals, dict) else {},
            )
            sections_html_by_param[hp.name] = html_value
            plot_titles_by_param[hp.name] = f"Dual value vs {hp.name} ({fixed_text})"

    event = render_dual_values_panel(
        algo_key,
        runs,
        run_tau_series_by_param,
        run_series_data_by_param,
        tau_payload={
            "tau_series_by_param": tau_series_by_param,
            "param_order": [hp.name for hp in hyperparameter_specs],
            "param_values_by_name": param_values_by_name,
            "cursor_indices_by_param": {name: int(idx) for name, idx in cursor_indices.items()},
            "local_cursor_indices_by_axis": {
                axis: {name: int(idx) for name, idx in axis_map.items()}
                for axis, axis_map in local_cursor_indices_by_axis.items()
            },
            "patterns_by_param": {name: str(value) for name, value in patterns_by_param.items()},
            "sections_html_by_param": sections_html_by_param,
            "plot_titles_by_param": plot_titles_by_param,
            "series_data_by_param": series_data_by_param,
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
                next_cursor_indices = dict(cursor_indices)
                incoming_cursor = event.get("cursor_indices_by_param")
                if isinstance(incoming_cursor, dict):
                    for name in specs_by_name:
                        if name in incoming_cursor:
                            next_cursor_indices[name] = int(incoming_cursor[name])
                next_local_cursor_indices_by_axis = dict(local_cursor_indices_by_axis)
                incoming_local_cursor_by_axis = event.get("local_cursor_indices_by_axis")
                if isinstance(incoming_local_cursor_by_axis, dict):
                    for axis_name in specs_by_name:
                        axis_payload = incoming_local_cursor_by_axis.get(axis_name)
                        if not isinstance(axis_payload, dict):
                            continue
                        next_axis = dict(next_local_cursor_indices_by_axis.get(axis_name, {}))
                        for name in specs_by_name:
                            if name == axis_name:
                                continue
                            if name in axis_payload:
                                next_axis[name] = int(axis_payload[name])
                        next_local_cursor_indices_by_axis[axis_name] = next_axis

                next_patterns_by_param = dict(patterns_by_param)
                incoming_patterns = event.get("patterns_by_param")
                if isinstance(incoming_patterns, dict):
                    for name in specs_by_name:
                        if name in incoming_patterns:
                            next_patterns_by_param[name] = str(incoming_patterns[name])

                next_cursor_indices = _clamp_cursor_indices(next_cursor_indices, hyperparameter_specs)
                next_local_cursor_indices_by_axis = _clamp_local_cursor_indices_by_axis(
                    next_local_cursor_indices_by_axis,
                    hyperparameter_specs,
                    next_cursor_indices,
                )
                st.session_state[cursor_state_key] = next_cursor_indices
                st.session_state[local_axis_state_key] = next_local_cursor_indices_by_axis
                st.session_state[pattern_state_key] = next_patterns_by_param
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
                next_patterns_by_param = dict(patterns_by_param)
                incoming_patterns = event.get("patterns_by_param")
                if isinstance(incoming_patterns, dict):
                    for name in specs_by_name:
                        if name in incoming_patterns:
                            next_patterns_by_param[name] = str(incoming_patterns[name])
                st.session_state[pattern_state_key] = next_patterns_by_param
                if run_id:
                    runs[:] = [run for run in runs if str(run.get("id", "")) != run_id]
                st.session_state[remove_event_key] = event_id
                st.rerun()
        elif event_id and st.session_state.get(last_event_key) != event_id:
            if event_type == "recompute":
                active_series_ids = tuple(sorted(set(event.get("selected_series_ids", []))))
                deactivated_series_ids = list(dict.fromkeys(event.get("deactivated_series_ids", [])))
                deactivated_labels = list(event.get("deactivated_labels", []))
                next_local_cursor_indices_by_axis = dict(local_cursor_indices_by_axis)
                incoming_local_cursor_by_axis = event.get("local_cursor_indices_by_axis")
                if isinstance(incoming_local_cursor_by_axis, dict):
                    for axis_name in specs_by_name:
                        axis_payload = incoming_local_cursor_by_axis.get(axis_name)
                        if not isinstance(axis_payload, dict):
                            continue
                        next_axis = dict(next_local_cursor_indices_by_axis.get(axis_name, {}))
                        for name in specs_by_name:
                            if name == axis_name:
                                continue
                            if name in axis_payload:
                                next_axis[name] = int(axis_payload[name])
                        next_local_cursor_indices_by_axis[axis_name] = next_axis

                next_patterns_by_param = dict(patterns_by_param)
                incoming_patterns = event.get("patterns_by_param")
                if isinstance(incoming_patterns, dict):
                    for name in specs_by_name:
                        if name in incoming_patterns:
                            next_patterns_by_param[name] = str(incoming_patterns[name])
                next_local_cursor_indices_by_axis = _clamp_local_cursor_indices_by_axis(
                    next_local_cursor_indices_by_axis,
                    hyperparameter_specs,
                    cursor_indices,
                )
                st.session_state[local_axis_state_key] = next_local_cursor_indices_by_axis
                st.session_state[pattern_state_key] = next_patterns_by_param
                with st.spinner("Recomputing tau grid with selected dual values..."):
                    recompute_result = compute(
                        algo_key,
                        settings["function_config"],
                        list(settings.get("hyperparameter_specs", [])),
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
DUAL_PANEL_JS = (UI_DIR / "dual_panel.js").read_text()


DUAL_PANEL_COMPONENT = components_v2.component(
    "dual_panel_component_v2",
    html="<div id='dual-panel-v2-root'></div>",
    js=DUAL_PANEL_JS,
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


def _loading_progress_key(algo_key: str) -> str:
    return f"loading_progress_{algo_key}"


def render_dual_values_panel(
    algo_key: str,
    runs: list[dict],
    run_tau_series_by_param: dict[str, dict[str, dict[str, object]]],
    run_series_data_by_param: dict[str, dict],
    tau_payload: dict,
) -> dict | None:
    def _json_safe(value: object) -> object:
        if isinstance(value, dict):
            return {str(k): _json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_json_safe(v) for v in value]
        if isinstance(value, bool) or value is None:
            return value
        if isinstance(value, (int, float)):
            if isinstance(value, float) and not isfinite(value):
                return None
            return value
        if isinstance(value, (pd.Timestamp,)):
            return str(value)
        if hasattr(value, "item"):
            try:
                scalar = value.item()
            except Exception:
                return str(value)
            return _json_safe(scalar)
        return value

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

    selected_series_ids = st.session_state.get(f"dual_selected_{algo_key}", [])
    dual_runs_data: list[dict] = []
    for run in runs:
        dual_runs_data.append(
            {
                "id": run["id"],
                "name": run["name"],
                "visible": bool(run.get("visible", True)),
                "selected_labels": [
                    str(label) for label in run.get("deactivated_labels", run.get("selected_labels", []))
                ],
                "tau_series_by_param": run_tau_series_by_param.get(run["id"], {}),
                "series_data_by_param": run_series_data_by_param.get(run["id"], {}),
            }
        )
    payload = {
        "css": DUAL_PANEL_CSS,
        "tau_payload": tau_payload,
        "series_data_by_param": tau_payload.get("series_data_by_param", {}),
        "dual_runs": dual_runs_data,
        "selected_series_ids": selected_series_ids,
        "metric": metric,
        "metric_labels": metric_labels,
    }
    result = DUAL_PANEL_COMPONENT(
        key=f"dual-panel-{algo_key}",
        data=_json_safe(payload),
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
