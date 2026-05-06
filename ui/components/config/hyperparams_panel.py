import pandas as pd
import streamlit as st

from algorithm.types import (
    AlgorithmSpec,
    HyperparameterSpec,
    default_gamma_n_hyperparameters,
)
from core.config import (
    HYPERPARAM_COLUMNS,
    hyperparameter_rows_from_specs,
    parse_hyperparameter_specs,
)

_HYPERPARAM_RESERVED_NAMES = {"x", "pi", "e"}
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


def render_hyperparameter_editor(algo_key: str, spec: AlgorithmSpec) -> tuple[list[HyperparameterSpec], list[str]]:
    store = st.session_state["hyperparameter_store"]
    if algo_key not in store:
        store[algo_key] = hyperparameter_rows_from_specs(spec.default_hyperparameters)
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
            store[algo_key] = hyperparameter_rows_from_specs(default_gamma_n_hyperparameters())
            _bump_editor_version()
            st.rerun()
    with top_right:
        if st.button("Reset to algorithm defaults", key=f"btn-hp-reset-{algo_key}"):
            store[algo_key] = hyperparameter_rows_from_specs(spec.default_hyperparameters)
            _bump_editor_version()
            st.rerun()

    rows = store.get(algo_key, [])
    df = pd.DataFrame(rows, columns=list(HYPERPARAM_COLUMNS))
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
    edited_rows = [{col: row.get(col) for col in HYPERPARAM_COLUMNS} for row in edited.to_dict(orient="records")]

    return parse_hyperparameter_specs(
        edited_rows,
        reserved_names=_HYPERPARAM_RESERVED_NAMES.union(_RUNTIME_RESERVED_NAMES),
        allow_equal_bounds=True,
    )
