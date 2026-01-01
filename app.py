# app.py
import streamlit as st

from algorithms_registry import ALGORITHMS
from routing import (
    init_session_state,
    render_config_phase,
    render_loading_phase,
    render_results_phase,
    reset_for_algorithm_change,
)


def main():
    init_session_state()
    st.title("Interactive tau explorer")
    st.set_page_config(layout="wide")

    algorithm_key = st.selectbox(
        "Algorithm",
        options=list(ALGORITHMS.keys()),
        format_func=lambda key: ALGORITHMS[key].name,
    )
    spec = ALGORITHMS[algorithm_key]

    if st.session_state["selected_algorithm"] != algorithm_key:
        reset_for_algorithm_change(algorithm_key)

    phase = st.session_state["ui_phase"]
    if phase == "config":
        render_config_phase(algorithm_key, spec)
    elif phase == "loading":
        render_loading_phase(algorithm_key, spec)
    elif phase == "results":
        render_results_phase(algorithm_key, spec)
    else:
        st.session_state["ui_phase"] = "config"
        st.rerun()


if __name__ == "__main__":
    main()
