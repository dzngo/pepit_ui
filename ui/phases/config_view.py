from pathlib import Path

import streamlit as st

from algorithm.types import AlgorithmSpec
from services.config_service import build_steps_test_context
from ui.components.algorithm_editor import render_steps_editor as _render_steps_editor
from ui.components.config import (
    handle_plot_action,
    render_functions_panel,
    render_hyperparameter_editor,
    render_remove_custom_algorithms_panel,
)
from ui.state.config_state import build_function_config as _cfg_build_function_config


def render_config_phase(algo_key: str, spec: AlgorithmSpec):
    css_path = Path(__file__).resolve().parents[1] / "assets" / "css" / "config_panel.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)
    st.subheader("Configuration")

    sections = st.columns(2)
    runtime_param_errors: list[str] = []

    with sections[1]:
        with st.container(border=True):
            st.write("Hyperparameter config")
            hyperparameter_specs, hyperparameter_errors = render_hyperparameter_editor(algo_key, spec)
            for error in hyperparameter_errors:
                st.error(error)
        with st.container(border=True):
            st.write("Functions")
            (function_names, function_rows, function_param_errors, function_row_errors,) = render_functions_panel(
                algo_key=algo_key,
                spec=spec,
            )

    with sections[0]:
        with st.container(border=True):
            st.write("Algorithm")
            function_config = _cfg_build_function_config(
                algo_key,
                spec,
                valid_function_keys=function_names,
            )
            _render_steps_editor(
                algo_key=algo_key,
                spec=spec,
                context="config",
                test_context=build_steps_test_context(
                    function_config=function_config,
                    function_param_errors=function_param_errors,
                    function_row_errors=function_row_errors,
                    runtime_param_errors=runtime_param_errors,
                    hyperparameter_specs=hyperparameter_specs,
                    function_rows=function_rows,
                ),
            )

        with st.container(border=True):
            render_remove_custom_algorithms_panel()

    st.divider()
    st.checkbox("Rerun Nan caches", key="rerun_nan_caches")

    handle_plot_action(
        algo_key=algo_key,
        spec=spec,
        hyperparameter_specs=hyperparameter_specs,
        hyperparameter_errors=hyperparameter_errors,
        runtime_param_errors=runtime_param_errors,
        function_param_errors=function_param_errors,
        function_row_errors=function_row_errors,
    )


__all__ = ["render_config_phase"]
