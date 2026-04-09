from pathlib import Path

import streamlit as st

from core.compute.engine import (
    compute_grid,
    discrete_values,
    make_cache_key,
    normalize_function_config,
    normalize_hyperparameter_specs,
    normalize_params,
    round_value,
)
from infrastructure.cache import PicklePointCache, SessionGridCache
from infrastructure.execution import ProcessPoolPointExecutor

POINT_CACHE_PATH = Path(__file__).resolve().parents[2] / ".tau_point_cache.pkl"


def run_compute_grid(
    algo_key: str,
    function_config: dict[str, dict[str, object]],
    hyperparameter_specs: list,
    *,
    show_progress: bool,
    rerun_nan_cache: bool = False,
    selected_dual_series_ids: tuple[str, ...] | None = None,
    batch_size: int | None = None,
    progress_state_key: str | None = None,
):
    session_grid_cache = SessionGridCache(st.session_state, key="tau_grid_cache_nd")
    persistent_point_cache = PicklePointCache(POINT_CACHE_PATH)
    executor = ProcessPoolPointExecutor()

    progress_bar = st.progress(0.0) if show_progress and batch_size is None else None
    status_placeholder = st.empty() if show_progress and batch_size is None else None

    def _progress_callback(completed: int, total: int, eta_seconds: float) -> None:
        if progress_bar is not None:
            progress_bar.progress(completed / max(total, 1))
        if status_placeholder is not None:
            status_placeholder.write(f"Computing grid… {completed}/{total} (eta {eta_seconds:.1f}s)")

    try:
        return compute_grid(
            algo_key=algo_key,
            function_config=function_config,
            hyperparameter_specs=hyperparameter_specs,
            rerun_nan_cache=rerun_nan_cache,
            selected_dual_series_ids=selected_dual_series_ids,
            show_progress=show_progress,
            batch_size=batch_size,
            session_grid_cache=session_grid_cache,
            persistent_point_cache=persistent_point_cache,
            executor=executor,
            progress_store=st.session_state,
            progress_state_key=progress_state_key,
            progress_callback=_progress_callback if (show_progress and batch_size is None) else None,
        )
    finally:
        if progress_bar is not None:
            progress_bar.empty()
        if status_placeholder is not None:
            status_placeholder.empty()


def clear_compute_caches(algo_key: str) -> None:
    session_grid_cache = SessionGridCache(st.session_state, key="tau_grid_cache_nd")
    session_grid_cache.remove_by_algo(algo_key)

    persistent_point_cache = PicklePointCache(POINT_CACHE_PATH)
    persistent_point_cache.remove_by_algo(algo_key)
    persistent_point_cache.flush()


__all__ = [
    "run_compute_grid",
    "clear_compute_caches",
    "discrete_values",
    "POINT_CACHE_PATH",
    "round_value",
    "normalize_params",
    "normalize_function_config",
    "normalize_hyperparameter_specs",
    "make_cache_key",
]
