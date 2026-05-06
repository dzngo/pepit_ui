import streamlit as st
from streamlit_ace import st_ace

from algorithm.algorithm_compiler import list_supported_primitive_steps
from algorithm.algorithm_custom import compile_algorithm_body, get_algorithm_steps_code
from algorithm.runtime import run_algorithm
from algorithm.types import AlgorithmSpec, HyperparameterSpec
from service.config_service import (
    register_custom_algorithm_bundle,
    sync_custom_algorithm_defaults_in_state,
)


def steps_source(spec: AlgorithmSpec) -> str:
    return get_algorithm_steps_code(spec.name)


def editor_steps_source(spec: AlgorithmSpec) -> str:
    return steps_source(spec)


def run_steps_smoke_test(
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
            default_steps = editor_steps_source(spec)
            steps_code = st.session_state.get(code_key, default_steps)
            steps = compile_algorithm_body(steps_code)
        else:
            steps = spec.algo
        temp_spec = AlgorithmSpec(
            name=spec.name,
            algo=steps,
            default_hyperparameters=list(spec.default_hyperparameters),
            default_function_rows=list(spec.default_function_rows),
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


def render_steps_editor(
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
        st.session_state[code_key] = editor_steps_source(spec)
        updated = st_ace(
            value=st.session_state.get(code_key, ""),
            language="python",
            key=editor_key,
            height=None,
            show_gutter=False,
            wrap=False,
            theme="github",
            auto_update=True,
        )
        if isinstance(updated, str):
            st.session_state[code_key] = updated
        primitive_steps = list_supported_primitive_steps()
        st.caption("Algorithm body can directly use configured function/hyperparameter names.")
        st.caption("Access function parameters directly from function objects (e.g. `f.L`, `f.mu`).")
        with st.expander("Supported primitive steps", expanded=False):
            if primitive_steps:
                tags_html = "".join(
                    f"<span style='display:inline-block;padding:3px 8px;margin:4px 6px 0 0;"
                    f"border:1px solid #d4d4d8;border-radius:999px;background:#f8fafc;"
                    f"font-family:monospace;font-size:12px;color:#0f172a;'>{name}</span>"
                    for name in primitive_steps
                )
                st.markdown(f"<div>{tags_html}</div>", unsafe_allow_html=True)
            else:
                st.caption("No primitive steps detected.")
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
            try:
                if not name:
                    raise ValueError("Custom algorithm name is required.")
                register_custom_algorithm_bundle(
                    name=name,
                    steps_code=str(steps_code),
                    source_algorithm_name=spec.name,
                    default_hyperparameters=list(test_context.get("hyperparameter_specs", []))
                    if test_context
                    else None,
                    default_function_rows=list(test_context.get("function_rows", [])) if test_context else None,
                )
            except Exception as exc:
                st.error(str(exc))
            else:
                st.success(f"Saved custom algorithm '{name}'.")
                sync_custom_algorithm_defaults_in_state(
                    st.session_state,
                    source_algo_key=algo_key,
                    target_algo_name=name,
                    hyperparameter_specs=list(test_context.get("hyperparameter_specs", [])) if test_context else None,
                )
                st.session_state[open_key] = False
                st.session_state["pending_algorithm_select"] = name
                st.session_state["selected_algorithm"] = None
                st.session_state["ui_phase"] = "config"
                st.rerun()
        if cancel_clicked:
            st.session_state[open_key] = False
            st.rerun()
        if test_context and test_clicked:
            error_message = run_steps_smoke_test(
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
        st.code(steps_source(spec), language="python")
        if st.button("Customize", key="btn-customize-config"):
            st.session_state[open_key] = True
            st.session_state.setdefault(code_key, editor_steps_source(spec))
            st.session_state.setdefault(name_key, "")
            st.rerun()
