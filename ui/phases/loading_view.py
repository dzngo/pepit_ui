from math import prod

import streamlit as st

from core.compute import discrete_values
from service.loading_service import (
    clear_stale_loading_state,
    finalize_loading_success,
    interrupt_loading,
    progress_from_state,
    run_loading_batch,
)
from ui.components.algorithm_editor import steps_source as _steps_source
from ui.state.state_utils import cursor_indices_key as _cursor_indices_key
from ui.state.state_utils import default_cursor_indices as _default_cursor_indices
from ui.state.state_utils import (
    default_local_cursor_indices_by_axis as _default_local_cursor_indices_by_axis,
)
from ui.state.state_utils import last_cursor_event_key as _last_cursor_event_key
from ui.state.state_utils import last_metric_event_key as _last_metric_event_key
from ui.state.state_utils import last_recompute_event_key as _last_recompute_event_key
from ui.state.state_utils import last_remove_event_key as _last_remove_event_key
from ui.state.state_utils import loading_progress_key as _loading_progress_key
from ui.state.state_utils import (
    local_cursor_indices_by_axis_key as _local_cursor_indices_by_axis_key,
)
from ui.state.state_utils import patterns_by_param_key as _patterns_by_param_key
from ui.state.state_utils import run_counter_key as _run_counter_key
from ui.state.state_utils import runs_key as _runs_key


def render_loading_phase(algo_key: str, spec):
    pending = st.session_state.get("pending_settings")
    if not pending or pending["algo_key"] != algo_key:
        clear_stale_loading_state(
            st.session_state,
            progress_key=_loading_progress_key(algo_key),
        )
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
                interrupt_loading(
                    st.session_state,
                    progress_key=progress_key,
                )
                st.rerun()

        # Render progress from the previous batch before computing the next one.
        progress_state = st.session_state.get(progress_key, {})
        done, total, fraction = progress_from_state(progress_state, total_estimate)
        if total > 0:
            with progress_col:
                st.progress(fraction)
                st.caption(f"Computing grid… {done}/{total}")

        result = run_loading_batch(
            algo_key=algo_key,
            current_pending=current_pending,
            hyperparameter_specs=hyperparameter_specs,
            progress_state_key=progress_key,
            batch_size=batch_size,
        )

        if result is None:
            return

        cursor_indices = _default_cursor_indices(hyperparameter_specs)
        local_cursor_indices_by_axis = _default_local_cursor_indices_by_axis(hyperparameter_specs, cursor_indices)
        patterns_by_param = {hp.name: "" for hp in hyperparameter_specs}
        finalize_loading_success(
            st.session_state,
            algo_key=algo_key,
            current_pending=current_pending,
            progress_key=progress_key,
            runs_key=_runs_key(algo_key),
            run_counter_key=_run_counter_key(algo_key),
            last_recompute_event_key=_last_recompute_event_key(algo_key),
            last_cursor_event_key=_last_cursor_event_key(algo_key),
            last_metric_event_key=_last_metric_event_key(algo_key),
            last_remove_event_key=_last_remove_event_key(algo_key),
            cursor_state_key=_cursor_indices_key(algo_key),
            local_axis_state_key=_local_cursor_indices_by_axis_key(algo_key),
            pattern_state_key=_patterns_by_param_key(algo_key),
            cursor_indices=cursor_indices,
            local_cursor_indices_by_axis=local_cursor_indices_by_axis,
            patterns_by_param=patterns_by_param,
        )
        st.rerun()

    _loading_fragment()


__all__ = ["render_loading_phase"]
