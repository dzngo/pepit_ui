import streamlit as st

from core.compute import build_pattern_param_values
from infrastructure.compute_runner import run_compute_grid
from service.results_service import (
    append_recompute_run,
    build_results_artifacts,
    handle_results_event,
)
from ui.components.algorithm_editor import steps_source
from ui.components.results_panel import render_dual_values_panel
from ui.state.state_utils import (
    algo_state_key,
    clamp_cursor_indices,
    clamp_local_cursor_indices_by_axis,
    default_cursor_indices,
    param_values_by_name,
)


def render_results_phase(algo_key: str, spec):
    settings = st.session_state.get("active_settings")
    if not settings or settings["algo_key"] != algo_key:
        st.session_state["ui_phase"] = "config"
        st.rerun()

    result = run_compute_grid(
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
    param_values_map = param_values_by_name(hyperparameter_specs)

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
        st.code(steps_source(spec), language="python")
        st.markdown("**Functions**")
        for slot_key, slot_config in sorted(settings["function_config"].items()):
            alias = str(slot_config.get("alias", slot_key) or slot_key)
            st.markdown(f"{alias}: `{slot_config['function_key']}`")
            if slot_config["function_params"]:
                params_line = ", ".join(f"{name}={value}" for name, value in slot_config["function_params"].items())
                st.markdown(f"*params*: {params_line}")
            else:
                st.markdown("*params*: `{}`")
    cursor_state_key = algo_state_key("cursor_indices_by_param", algo_key)
    local_axis_state_key = algo_state_key("local_cursor_indices_by_axis", algo_key)
    pattern_state_key = algo_state_key("tau_patterns_by_param", algo_key)
    default_cursor = default_cursor_indices(hyperparameter_specs)
    cursor_indices = clamp_cursor_indices(
        dict(st.session_state.get(cursor_state_key, default_cursor)),
        hyperparameter_specs,
    )
    local_cursor_indices_by_axis = clamp_local_cursor_indices_by_axis(
        dict(st.session_state.get(local_axis_state_key, {})),
        hyperparameter_specs,
        cursor_indices,
    )
    patterns_by_param = dict(st.session_state.get(pattern_state_key, {}))
    for hp in hyperparameter_specs:
        patterns_by_param[hp.name] = str(patterns_by_param.get(hp.name, ""))

    cursor_indices = clamp_cursor_indices(cursor_indices, hyperparameter_specs)
    st.session_state[cursor_state_key] = cursor_indices
    st.session_state[local_axis_state_key] = local_cursor_indices_by_axis
    st.session_state[pattern_state_key] = patterns_by_param
    base_param_values, invalid_params, conflict_params = build_pattern_param_values(settings["function_config"])
    runs = st.session_state.setdefault(algo_state_key("recompute_runs", algo_key), [])
    artifacts = build_results_artifacts(
        algo_key=algo_key,
        settings=settings,
        hyperparameter_specs=hyperparameter_specs,
        cursor_indices=cursor_indices,
        local_cursor_indices_by_axis=local_cursor_indices_by_axis,
        metric=metric,
        runs=runs,
        cached_warnings=cached_warnings,
    )
    warning_messages = artifacts["warning_messages"]
    run_tau_series_by_param = artifacts["run_tau_series_by_param"]
    run_series_data_by_param = artifacts["run_series_data_by_param"]
    tau_series_by_param = artifacts["tau_series_by_param"]
    series_data_by_param = artifacts["series_data_by_param"]
    sections_html_by_param = artifacts["sections_html_by_param"]
    plot_titles_by_param = artifacts["plot_titles_by_param"]

    if warning_messages:
        warning_text = "\n".join(sorted(warning_messages))
        st.warning(
            "Some parameter combinations could not be solved; missing points are shown as gaps.\n" + warning_text
        )

    event = render_dual_values_panel(
        algo_key,
        runs,
        run_tau_series_by_param,
        run_series_data_by_param,
        tau_payload={
            "tau_series_by_param": tau_series_by_param,
            "param_order": [hp.name for hp in hyperparameter_specs],
            "param_values_by_name": param_values_map,
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
        outcome = handle_results_event(
            event=event,
            specs_by_name=specs_by_name,
            hyperparameter_specs=hyperparameter_specs,
            cursor_indices=cursor_indices,
            local_cursor_indices_by_axis=local_cursor_indices_by_axis,
            patterns_by_param=patterns_by_param,
            runs=runs,
            current_metric=str(st.session_state.get(f"dual-ranking-metric-{algo_key}", metric)),
            last_seen_ids={
                "cursor": str(st.session_state.get(algo_state_key("cursor_event", algo_key), "")),
                "metric": str(st.session_state.get(algo_state_key("metric_event", algo_key), "")),
                "remove": str(st.session_state.get(algo_state_key("remove_run_event", algo_key), "")),
                "recompute": str(st.session_state.get(algo_state_key("recompute_event", algo_key), "")),
            },
        )
        if outcome.get("applied"):
            next_seen = outcome.get("next_last_seen", {})
            st.session_state[algo_state_key("cursor_event", algo_key)] = next_seen.get("cursor", "")
            st.session_state[algo_state_key("metric_event", algo_key)] = next_seen.get("metric", "")
            st.session_state[algo_state_key("remove_run_event", algo_key)] = next_seen.get("remove", "")
            st.session_state[algo_state_key("recompute_event", algo_key)] = next_seen.get("recompute", "")

            kind = str(outcome.get("kind", "noop"))
            if kind == "cursor":
                st.session_state[cursor_state_key] = outcome["next_cursor_indices"]
                st.session_state[local_axis_state_key] = outcome["next_local_cursor_indices_by_axis"]
                st.session_state[pattern_state_key] = outcome["next_patterns_by_param"]
                st.rerun()
            elif kind == "metric":
                st.session_state[f"dual-ranking-metric-{algo_key}"] = outcome["next_metric"]
                st.rerun()
            elif kind == "remove_run":
                st.session_state[pattern_state_key] = outcome["next_patterns_by_param"]
                runs[:] = outcome["next_runs"]
                st.rerun()
            elif kind == "recompute":
                st.session_state[local_axis_state_key] = outcome["next_local_cursor_indices_by_axis"]
                st.session_state[pattern_state_key] = outcome["next_patterns_by_param"]
                with st.spinner("Recomputing tau grid with selected dual values..."):
                    recompute_result = run_compute_grid(
                        algo_key,
                        settings["function_config"],
                        list(settings.get("hyperparameter_specs", [])),
                        show_progress=True,
                        rerun_nan_cache=bool(settings.get("rerun_nan_caches", False)),
                        selected_dual_series_ids=outcome["active_series_ids"],
                    )
                if recompute_result is None:
                    st.error("Unable to recompute tau grid for selected dual values.")
                else:
                    counter_key = algo_state_key("recompute_counter", algo_key)
                    next_index = int(st.session_state.setdefault(counter_key, 0)) + 1
                    st.session_state[counter_key] = next_index
                    runs[:] = append_recompute_run(
                        runs=runs,
                        next_index=next_index,
                        active_series_ids=outcome["active_series_ids"],
                        deactivated_series_ids=outcome["deactivated_series_ids"],
                        deactivated_labels=outcome["deactivated_labels"],
                    )
                st.rerun()


__all__ = ["render_results_phase"]
