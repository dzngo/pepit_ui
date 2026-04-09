from collections.abc import MutableMapping

from infrastructure.compute_runner import run_compute_grid


def clear_stale_loading_state(
    session_state: MutableMapping[str, object],
    *,
    progress_key: str,
) -> None:
    session_state.pop(progress_key, None)
    session_state["ui_phase"] = "config"


def interrupt_loading(
    session_state: MutableMapping[str, object],
    *,
    progress_key: str,
) -> None:
    session_state.pop(progress_key, None)
    session_state["pending_settings"] = None
    session_state["ui_phase"] = "config"


def progress_from_state(progress_state: dict, total_estimate: int) -> tuple[int, int, float]:
    done = int(progress_state.get("done", 0))
    total = int(progress_state.get("total", total_estimate))
    fraction = min(max(done / total, 0.0), 1.0) if total > 0 else 0.0
    return done, total, fraction


def run_loading_batch(
    *,
    algo_key: str,
    current_pending: dict,
    hyperparameter_specs: list,
    progress_state_key: str,
    batch_size: int,
):
    return run_compute_grid(
        algo_key,
        current_pending["function_config"],
        hyperparameter_specs,
        show_progress=False,
        rerun_nan_cache=bool(current_pending.get("rerun_nan_caches", False)),
        batch_size=batch_size,
        progress_state_key=progress_state_key,
    )


def finalize_loading_success(
    session_state: MutableMapping[str, object],
    *,
    algo_key: str,
    current_pending: dict,
    progress_key: str,
    runs_key: str,
    run_counter_key: str,
    last_recompute_event_key: str,
    last_cursor_event_key: str,
    last_metric_event_key: str,
    last_remove_event_key: str,
    cursor_state_key: str,
    local_axis_state_key: str,
    pattern_state_key: str,
    cursor_indices: dict[str, int],
    local_cursor_indices_by_axis: dict[str, dict[str, int]],
    patterns_by_param: dict[str, str],
) -> None:
    session_state.pop(progress_key, None)
    session_state["active_settings"] = current_pending
    session_state["pending_settings"] = None
    session_state["ui_phase"] = "results"
    session_state[f"dual_selection_{algo_key}"] = {}
    session_state[runs_key] = []
    session_state[run_counter_key] = 0
    session_state.pop(last_recompute_event_key, None)
    session_state.pop(last_cursor_event_key, None)
    session_state.pop(last_metric_event_key, None)
    session_state.pop(last_remove_event_key, None)
    session_state[f"dual_selected_{algo_key}"] = []
    session_state[cursor_state_key] = cursor_indices
    session_state[local_axis_state_key] = local_cursor_indices_by_axis
    session_state[pattern_state_key] = patterns_by_param
