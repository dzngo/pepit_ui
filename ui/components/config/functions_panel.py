import streamlit as st

from algorithm.function_registry import FUNCTIONS
from algorithm.types import AlgorithmSpec, HyperparameterSpec
from core.compute import float_text_default, parse_float_input, parse_float_list
from core.config.hyperparams import parse_hyperparameter_specs
from ui.state.config_state import (
    function_row_id_key,
    get_function_rows,
    next_function_row_id,
    suggest_new_function_name,
    validate_function_rows_with_rules,
)

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


def render_functions_panel(
    *,
    algo_key: str,
    spec: AlgorithmSpec,
) -> tuple[list[str], list[dict[str, object]], list[str], list[str], list[HyperparameterSpec], list[str]]:
    function_names = sorted(FUNCTIONS.keys())
    function_rows = get_function_rows(
        algo_key,
        spec,
        valid_function_keys=function_names,
    )
    default_function_key = function_names[0] if function_names else ""
    function_param_errors: list[str] = []

    if not function_names:
        st.error("No function types are registered.")
    else:
        remove_row_id: str | None = None
        for idx, row in enumerate(function_rows, start=1):
            row_id = str(row.get("id") or next_function_row_id(algo_key))
            row["id"] = row_id
            with st.container(border=True):
                name_col, function_type_col = st.columns([1, 3])
                with name_col:
                    name_value = st.text_input(
                        "name",
                        value=str(row.get("name", "")),
                        key=f"{function_row_id_key(algo_key)}name-{row_id}",
                    )
                with function_type_col:
                    selected_function = str(row.get("function_key") or "")
                    if selected_function not in function_names:
                        selected_function = default_function_key
                    selected_function = st.selectbox(
                        "function type",
                        options=function_names,
                        index=(function_names.index(selected_function) if selected_function in function_names else 0),
                        key=f"{function_row_id_key(algo_key)}type-{row_id}",
                    )

                row["name"] = name_value.strip()
                row["function_key"] = selected_function

                function_spec = FUNCTIONS[selected_function]
                slot_params = row.setdefault("function_params", {})
                if not isinstance(slot_params, dict):
                    slot_params = {}
                    row["function_params"] = slot_params
                slot_vary = row.setdefault("function_param_vary", {})
                if not isinstance(slot_vary, dict):
                    slot_vary = {}
                    row["function_param_vary"] = slot_vary
                display_name = row["name"] or f"row {idx}"
                if not function_spec.parameters:
                    st.caption(f"{display_name} has no required parameters.")
                else:
                    columns = st.columns(1)
                    for param_idx, param in enumerate(function_spec.parameters):
                        with columns[param_idx % len(columns)]:
                            with st.container(border=True):
                                input_key = f"{function_row_id_key(algo_key)}param-{row_id}-{param.name}"
                                param_context = f"{display_name} ({function_spec.cls.__name__}), parameter {param.name}"
                                if param.param_type == "float":
                                    vary_toggle_key = (
                                        f"{function_row_id_key(algo_key)}vary-toggle-{row_id}-{param.name}"
                                    )
                                    header_left, header_right = st.columns([5, 2])
                                    with header_right:
                                        st.caption("Vary")
                                        vary_enabled = st.checkbox(
                                            "Vary",
                                            key=vary_toggle_key,
                                            value=param.name in slot_vary,
                                            label_visibility="collapsed",
                                        )
                                    default_text = float_text_default(slot_params.get(param.name, param.default))
                                    st.session_state.setdefault(input_key, default_text)
                                    parsed_value, _ = parse_float_input(st.session_state.get(input_key, default_text))
                                    if not vary_enabled:
                                        with header_left:
                                            raw_value = st.text_input(param.name, key=input_key)
                                        parsed_value, error = parse_float_input(raw_value)
                                        if error:
                                            function_param_errors.append(f"{param_context}: {error}")
                                        else:
                                            if param.required and parsed_value is None:
                                                function_param_errors.append(f"{param_context}: value required.")
                                            elif parsed_value is not None and parsed_value < 0:
                                                function_param_errors.append(f"{param_context}: value must be >= 0.")
                                            else:
                                                slot_params[param.name] = parsed_value
                                    else:
                                        with header_left:
                                            st.markdown(f"**{param.name}**")
                                    if vary_enabled:
                                        vary_prefix = f"{function_row_id_key(algo_key)}vary-{row_id}-{param.name}"
                                        vary_defaults = slot_vary.get(param.name, {})
                                        min_col, max_col, step_col, default_col, type_col = st.columns(5)
                                        with min_col:
                                            min_v = st.number_input(
                                                "min",
                                                value=float(
                                                    vary_defaults.get(
                                                        "min", parsed_value if parsed_value is not None else 0.0
                                                    )
                                                ),
                                                key=f"{vary_prefix}-min",
                                            )
                                        with max_col:
                                            max_v = st.number_input(
                                                "max",
                                                value=float(
                                                    vary_defaults.get(
                                                        "max", parsed_value if parsed_value is not None else 1.0
                                                    )
                                                ),
                                                key=f"{vary_prefix}-max",
                                            )
                                        with step_col:
                                            step_v = st.number_input(
                                                "step",
                                                min_value=1e-12,
                                                value=float(vary_defaults.get("step", 0.1)),
                                                key=f"{vary_prefix}-step",
                                            )
                                        with default_col:
                                            default_v = st.number_input(
                                                "default",
                                                value=float(
                                                    vary_defaults.get(
                                                        "default", parsed_value if parsed_value is not None else min_v
                                                    )
                                                ),
                                                key=f"{vary_prefix}-default",
                                            )
                                        with type_col:
                                            vary_type = st.selectbox(
                                                "type",
                                                options=["float", "int"],
                                                index=0
                                                if str(vary_defaults.get("value_type", "float")) != "int"
                                                else 1,
                                                key=f"{vary_prefix}-type",
                                            )
                                        slot_vary[param.name] = {
                                            "min": float(min_v),
                                            "max": float(max_v),
                                            "step": float(step_v),
                                            "default": float(default_v),
                                            "value_type": str(vary_type),
                                        }
                                    else:
                                        slot_vary.pop(param.name, None)
                                    if param.description:
                                        st.caption(param.description)
                                elif param.param_type == "BlockPartition":
                                    vary_toggle_key = (
                                        f"{function_row_id_key(algo_key)}vary-toggle-{row_id}-{param.name}"
                                    )
                                    header_left, header_right = st.columns([5, 2])
                                    with header_right:
                                        st.caption("Vary")
                                        vary_enabled = st.checkbox(
                                            "Vary",
                                            key=vary_toggle_key,
                                            value=param.name in slot_vary,
                                            label_visibility="collapsed",
                                        )
                                    d_value = int(slot_params.get(param.name, 1) or 1)
                                    if not vary_enabled:
                                        with header_left:
                                            d_value = st.number_input(
                                                f"{param.name} (d)",
                                                min_value=0,
                                                step=1,
                                                value=d_value,
                                                key=input_key,
                                            )
                                        slot_params[param.name] = int(d_value)
                                    else:
                                        with header_left:
                                            st.markdown(f"**{param.name} (d)**")
                                    if vary_enabled:
                                        vary_prefix = f"{function_row_id_key(algo_key)}vary-{row_id}-{param.name}"
                                        vary_defaults = slot_vary.get(param.name, {})
                                        min_col, max_col, step_col, default_col, type_col = st.columns(5)
                                        with min_col:
                                            min_v = st.number_input(
                                                "min",
                                                step=1,
                                                value=int(vary_defaults.get("min", int(d_value))),
                                                key=f"{vary_prefix}-min",
                                            )
                                        with max_col:
                                            max_v = st.number_input(
                                                "max",
                                                step=1,
                                                value=int(vary_defaults.get("max", int(d_value) + 1)),
                                                key=f"{vary_prefix}-max",
                                            )
                                        with step_col:
                                            step_v = st.number_input(
                                                "step",
                                                min_value=1,
                                                step=1,
                                                value=int(vary_defaults.get("step", 1)),
                                                key=f"{vary_prefix}-step",
                                            )
                                        with default_col:
                                            default_v = st.number_input(
                                                "default",
                                                step=1,
                                                value=int(vary_defaults.get("default", int(d_value))),
                                                key=f"{vary_prefix}-default",
                                            )
                                        with type_col:
                                            st.caption("type")
                                            st.text_input(
                                                "type",
                                                value="int",
                                                key=f"{vary_prefix}-type-readonly",
                                                disabled=True,
                                                label_visibility="collapsed",
                                            )
                                        slot_vary[param.name] = {
                                            "min": float(min_v),
                                            "max": float(max_v),
                                            "step": float(step_v),
                                            "default": float(default_v),
                                            "value_type": "int",
                                        }
                                    else:
                                        slot_vary.pop(param.name, None)
                                    desc_parts = []
                                    if param.description:
                                        desc_parts.append(param.description)
                                    desc_parts.append(
                                        "Partition will be created via " "`problem.declare_block_partition(d=...)`."
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
                                    parsed_list, error = parse_float_list(raw_value)
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
                    stale_vary = set(slot_vary) - {p.name for p in function_spec.parameters}
                    for key in stale_vary:
                        slot_vary.pop(key, None)

                if len(function_rows) >= 2 and st.button(
                    "Remove",
                    key=f"{function_row_id_key(algo_key)}remove-{row_id}",
                ):
                    remove_row_id = row_id

        if remove_row_id is not None:
            updated_rows = [row for row in function_rows if str(row.get("id")) != remove_row_id]
            st.session_state["function_rows_store"][algo_key] = updated_rows
            st.rerun()

    if st.button("Add function", key=f"btn-add-function-{algo_key}"):
        function_rows.append(
            {
                "id": next_function_row_id(algo_key),
                "name": suggest_new_function_name(function_rows),
                "function_key": default_function_key,
                "function_params": {},
                "function_param_vary": {},
            }
        )
        st.rerun()

    function_row_errors = validate_function_rows_with_rules(
        function_rows,
        reserved_names=_RUNTIME_RESERVED_NAMES,
        valid_function_keys=function_names,
    )
    for error in function_row_errors:
        st.error(error)

    vary_rows: list[dict[str, object]] = []
    for row in function_rows:
        alias = str(row.get("name") or "").strip()
        vary_map = row.get("function_param_vary", {})
        if not alias or not isinstance(vary_map, dict):
            continue
        for param_name, vary in vary_map.items():
            if not isinstance(vary, dict):
                continue
            vary_rows.append(
                {
                    "name": f"{alias}.{param_name}",
                    "label": f"{alias}.{param_name}",
                    "value_type": str(vary.get("value_type", "float")),
                    "min": vary.get("min"),
                    "max": vary.get("max"),
                    "step": vary.get("step"),
                    "default": vary.get("default"),
                }
            )
    if vary_rows:
        vary_specs, vary_errors = parse_hyperparameter_specs(
            vary_rows,
            reserved_names=_RUNTIME_RESERVED_NAMES,
        )
    else:
        vary_specs, vary_errors = [], []
    vary_errors = [f"Function vary: {msg}" for msg in vary_errors]

    return function_names, function_rows, function_param_errors, function_row_errors, vary_specs, vary_errors
