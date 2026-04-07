import streamlit as st

from algorithm.algorithm_custom import (
    ALGORITHMS,
    CUSTOM_ALGORITHMS,
    remove_custom_algorithm,
)
from algorithm.function_registry import FUNCTIONS
from core.compute import clear_algorithm_caches
from services.config_service import (
    build_pending_settings,
    build_steps_test_context,
    collect_config_errors,
)
from ui.components.algorithm_editor import run_steps_smoke_test as _run_steps_smoke_test
from ui.state.config_state import build_function_config as _cfg_build_function_config
from ui.state.state_utils import loading_progress_key as _loading_progress_key


def render_remove_custom_algorithms_panel() -> None:
    st.write("Remove customized algorithm")
    custom_names = sorted(CUSTOM_ALGORITHMS.keys())
    if not custom_names:
        st.caption("No customized algorithms saved.")
        return

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


def handle_plot_action(
    *,
    algo_key: str,
    spec,
    hyperparameter_specs: list,
    hyperparameter_errors: list[str],
    runtime_param_errors: list[str],
    function_param_errors: list[str],
    function_row_errors: list[str],
) -> None:
    plot_clicked = st.button("Plot", key="btn-plot-config")
    if not plot_clicked:
        return

    errors = collect_config_errors(
        hyperparameter_errors=hyperparameter_errors,
        runtime_param_errors=runtime_param_errors,
        function_param_errors=function_param_errors,
        function_row_errors=function_row_errors,
    )
    if errors:
        for error in errors:
            st.error(error)
        return

    function_config = _cfg_build_function_config(
        algo_key,
        spec,
        valid_function_keys=sorted(FUNCTIONS.keys()),
    )
    plot_test_context = build_steps_test_context(
        function_config=function_config,
        function_param_errors=function_param_errors,
        function_row_errors=function_row_errors,
        runtime_param_errors=runtime_param_errors,
        hyperparameter_specs=hyperparameter_specs,
    )
    test_error = _run_steps_smoke_test(
        spec=spec,
        context="config",
        algo_key=algo_key,
        test_context=plot_test_context,
    )
    if test_error:
        st.error(f"Algorithm test failed: {test_error}")
        return

    st.session_state["pending_settings"] = build_pending_settings(
        algo_key=algo_key,
        hyperparameter_specs=hyperparameter_specs,
        function_config=function_config,
        rerun_nan_caches=bool(st.session_state.get("rerun_nan_caches", False)),
    )
    st.session_state.pop(_loading_progress_key(algo_key), None)
    st.session_state["ui_phase"] = "loading"
    st.rerun()
