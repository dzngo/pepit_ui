from core.compute import (
    build_dual_section_html,
    build_dual_series_by_param,
    build_dual_slice_by_param,
    build_tau_series_by_param,
    dual_ranking_by_slice,
)
from infrastructure.compute_runner import run_compute_grid
from ui.state.state_utils import (
    clamp_cursor_indices,
    clamp_local_cursor_indices_by_axis,
)


def build_results_artifacts(
    *,
    algo_key: str,
    settings: dict,
    hyperparameter_specs: list,
    plot_hyperparameter_specs: list | None = None,
    cursor_indices: dict[str, int],
    local_cursor_indices_by_axis: dict[str, dict[str, int]],
    metric: str,
    runs: list[dict],
    cached_warnings: tuple[str, ...] | list[str],
) -> dict:
    plot_specs = list(plot_hyperparameter_specs if plot_hyperparameter_specs is not None else hyperparameter_specs)
    warning_messages = set(cached_warnings)
    run_tau_series_by_param: dict[str, dict[str, dict[str, object]]] = {}
    run_series_data_by_param: dict[str, dict] = {}

    for run in runs:
        selected_series_ids = tuple(sorted(set(run.get("selected_series_ids", []))))
        if not selected_series_ids:
            continue
        run_result = run_compute_grid(
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
                plot_specs=plot_specs,
            )
            run_series_data_by_param[run["id"]] = build_dual_series_by_param(
                run_duals_nd,
                run_param_values,
                hyperparameter_specs,
                cursor_indices,
                plot_specs=plot_specs,
            )
            for warning in run_result[2]:
                warning_messages.add(f"{run['name']}: {warning}")

    nd_result = run_compute_grid(
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
            plot_specs=plot_specs,
        )
        series_data_by_param = build_dual_series_by_param(
            duals_nd,
            param_values_nd,
            hyperparameter_specs,
            cursor_indices,
            plot_specs=plot_specs,
        )
        axis_index = {hp.name: idx for idx, hp in enumerate(hyperparameter_specs)}
        base_idx = []
        for hp in hyperparameter_specs:
            values = param_values_nd.get(hp.name, [])
            max_idx = max(len(values) - 1, 0)
            base_idx.append(max(0, min(int(cursor_indices.get(hp.name, 0)), max_idx)))
        current_duals = duals_nd[tuple(base_idx)] if hyperparameter_specs else {}
        for hp in plot_specs:
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

    return {
        "warning_messages": warning_messages,
        "run_tau_series_by_param": run_tau_series_by_param,
        "run_series_data_by_param": run_series_data_by_param,
        "tau_series_by_param": tau_series_by_param,
        "series_data_by_param": series_data_by_param,
        "sections_html_by_param": sections_html_by_param,
        "plot_titles_by_param": plot_titles_by_param,
    }


ALLOWED_METRICS = {
    "non_zero_pct",
    "non_zero_pct_with_none",
    "std",
    "std_with_none",
    "median_abs",
    "median_abs_with_none",
    "mean_abs",
    "mean_abs_with_none",
}


def apply_cursor_event(
    *,
    event: dict,
    specs_by_name: dict,
    hyperparameter_specs: list,
    cursor_indices: dict[str, int],
    local_cursor_indices_by_axis: dict[str, dict[str, int]],
    patterns_by_param: dict[str, str],
) -> tuple[dict[str, int], dict[str, dict[str, int]], dict[str, str]]:
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

    next_cursor_indices = clamp_cursor_indices(next_cursor_indices, hyperparameter_specs)
    next_local_cursor_indices_by_axis = clamp_local_cursor_indices_by_axis(
        next_local_cursor_indices_by_axis,
        hyperparameter_specs,
        next_cursor_indices,
    )
    return next_cursor_indices, next_local_cursor_indices_by_axis, next_patterns_by_param


def apply_metric_event(*, event: dict, current_metric: str) -> str:
    requested_metric = str(event.get("metric", ""))
    if requested_metric in ALLOWED_METRICS:
        return requested_metric
    return current_metric


def apply_remove_run_event(
    *,
    event: dict,
    specs_by_name: dict,
    patterns_by_param: dict[str, str],
    runs: list[dict],
) -> tuple[dict[str, str], list[dict]]:
    next_patterns_by_param = dict(patterns_by_param)
    incoming_patterns = event.get("patterns_by_param")
    if isinstance(incoming_patterns, dict):
        for name in specs_by_name:
            if name in incoming_patterns:
                next_patterns_by_param[name] = str(incoming_patterns[name])
    run_id = str(event.get("run_id", ""))
    next_runs = list(runs)
    if run_id:
        next_runs = [run for run in next_runs if str(run.get("id", "")) != run_id]
    return next_patterns_by_param, next_runs


def prepare_recompute_event(
    *,
    event: dict,
    specs_by_name: dict,
    hyperparameter_specs: list,
    cursor_indices: dict[str, int],
    local_cursor_indices_by_axis: dict[str, dict[str, int]],
    patterns_by_param: dict[str, str],
) -> tuple[tuple[str, ...], list[str], list[str], dict[str, dict[str, int]], dict[str, str]]:
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
    next_local_cursor_indices_by_axis = clamp_local_cursor_indices_by_axis(
        next_local_cursor_indices_by_axis,
        hyperparameter_specs,
        cursor_indices,
    )
    return (
        active_series_ids,
        deactivated_series_ids,
        deactivated_labels,
        next_local_cursor_indices_by_axis,
        next_patterns_by_param,
    )


def append_recompute_run(
    *,
    runs: list[dict],
    next_index: int,
    active_series_ids: tuple[str, ...],
    deactivated_series_ids: list[str],
    deactivated_labels: list[str],
) -> list[dict]:
    next_runs = list(runs)
    next_runs.append(
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
    return next_runs


def handle_results_event(
    *,
    event: dict | None,
    specs_by_name: dict,
    hyperparameter_specs: list,
    cursor_indices: dict[str, int],
    local_cursor_indices_by_axis: dict[str, dict[str, int]],
    patterns_by_param: dict[str, str],
    runs: list[dict],
    current_metric: str,
    last_seen_ids: dict[str, str],
) -> dict:
    if not event:
        return {"kind": "noop", "applied": False}

    event_type = str(event.get("type", ""))
    event_id = str(event.get("request_id", ""))
    if not event_id:
        return {"kind": "noop", "applied": False}

    key_by_type = {
        "cursor": "cursor",
        "metric": "metric",
        "remove_run": "remove",
        "recompute": "recompute",
    }
    seen_key = key_by_type.get(event_type, "recompute")
    if last_seen_ids.get(seen_key) == event_id:
        return {"kind": "noop", "applied": False}

    next_last_seen = dict(last_seen_ids)
    next_last_seen[seen_key] = event_id

    if event_type == "cursor":
        (next_cursor_indices, next_local_cursor_indices_by_axis, next_patterns_by_param,) = apply_cursor_event(
            event=event,
            specs_by_name=specs_by_name,
            hyperparameter_specs=hyperparameter_specs,
            cursor_indices=cursor_indices,
            local_cursor_indices_by_axis=local_cursor_indices_by_axis,
            patterns_by_param=patterns_by_param,
        )
        return {
            "kind": "cursor",
            "applied": True,
            "event_id": event_id,
            "next_last_seen": next_last_seen,
            "next_cursor_indices": next_cursor_indices,
            "next_local_cursor_indices_by_axis": next_local_cursor_indices_by_axis,
            "next_patterns_by_param": next_patterns_by_param,
        }

    if event_type == "metric":
        return {
            "kind": "metric",
            "applied": True,
            "event_id": event_id,
            "next_last_seen": next_last_seen,
            "next_metric": apply_metric_event(
                event=event,
                current_metric=current_metric,
            ),
        }

    if event_type == "remove_run":
        next_patterns_by_param, next_runs = apply_remove_run_event(
            event=event,
            specs_by_name=specs_by_name,
            patterns_by_param=patterns_by_param,
            runs=runs,
        )
        return {
            "kind": "remove_run",
            "applied": True,
            "event_id": event_id,
            "next_last_seen": next_last_seen,
            "next_patterns_by_param": next_patterns_by_param,
            "next_runs": next_runs,
        }

    if event_type == "recompute":
        (
            active_series_ids,
            deactivated_series_ids,
            deactivated_labels,
            next_local_cursor_indices_by_axis,
            next_patterns_by_param,
        ) = prepare_recompute_event(
            event=event,
            specs_by_name=specs_by_name,
            hyperparameter_specs=hyperparameter_specs,
            cursor_indices=cursor_indices,
            local_cursor_indices_by_axis=local_cursor_indices_by_axis,
            patterns_by_param=patterns_by_param,
        )
        return {
            "kind": "recompute",
            "applied": True,
            "event_id": event_id,
            "next_last_seen": next_last_seen,
            "active_series_ids": active_series_ids,
            "deactivated_series_ids": deactivated_series_ids,
            "deactivated_labels": deactivated_labels,
            "next_local_cursor_indices_by_axis": next_local_cursor_indices_by_axis,
            "next_patterns_by_param": next_patterns_by_param,
        }

    return {"kind": "noop", "applied": False}
