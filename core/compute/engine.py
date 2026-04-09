import time
from decimal import ROUND_HALF_UP, Decimal
from itertools import product
from typing import Callable

import numpy as np

from algorithm.types import HyperparameterSpec
from core.ports import GridCachePort, PointCachePort, PointExecutorPort, ProgressStore


def discrete_values(param: HyperparameterSpec) -> np.ndarray:
    min_value = Decimal(str(param.min_value))
    max_value = Decimal(str(param.max_value))
    step = Decimal(str(param.step))
    steps = int(((max_value - min_value) / step).to_integral_value(rounding=ROUND_HALF_UP))
    values = np.array([float(min_value + Decimal(i) * step) for i in range(steps + 1)], dtype=float)
    if param.value_type == "int":
        values = np.round(values).astype(int)
    return values


def round_value(value: float, *, digits: int = 12) -> float:
    return float(round(float(value), digits))


def normalize_param_value(value: object) -> object:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return ("int", int(value))
    if isinstance(value, (float, np.floating)):
        return ("float", round_value(float(value)))
    if isinstance(value, np.ndarray):
        return tuple(normalize_param_value(item) for item in value.tolist())
    if isinstance(value, (list, tuple, set)):
        return tuple(normalize_param_value(item) for item in value)
    if isinstance(value, dict):
        return tuple((key, normalize_param_value(val)) for key, val in sorted(value.items()))
    return value


def normalize_params(params: dict[str, object]) -> tuple[tuple[str, object], ...]:
    normalized = []
    for name, value in params.items():
        if value is None:
            continue
        normalized.append((name, normalize_param_value(value)))
    return tuple(sorted(normalized, key=lambda item: item[0]))


def normalize_function_config(function_config: dict[str, dict[str, object]]) -> tuple:
    normalized = []
    for slot_key, config in sorted(function_config.items()):
        function_key = config["function_key"]
        function_params = config["function_params"]
        normalized.append((slot_key, function_key, normalize_params(function_params)))
    return tuple(normalized)


def normalize_hyperparameter_specs(hyperparameter_specs: list[HyperparameterSpec] | None) -> tuple:
    specs = hyperparameter_specs or []
    return tuple(
        (
            spec.name,
            spec.label,
            round_value(spec.min_value),
            round_value(spec.max_value),
            round_value(spec.default),
            round_value(spec.step),
            spec.value_type,
        )
        for spec in specs
    )


def make_cache_key(
    *,
    scope: str,
    algo_key: str,
    function_config: dict[str, dict[str, object]],
    selected_dual_series_ids: tuple[str, ...] | None = None,
    hyperparameter_specs: list[HyperparameterSpec] | None = None,
    param_assignment: dict[str, object] | None = None,
) -> tuple:
    """Build a stable cache key for either full-grid or single-point caching.

    - `scope="session_grid"`: key for the full N-D result cache.
    - `scope="point"`: key for one concrete parameter assignment.
    """
    selected = tuple(selected_dual_series_ids or ())
    normalized_function_config = normalize_function_config(function_config)

    if scope == "session_grid":
        return (
            "session_grid",
            algo_key,
            normalize_hyperparameter_specs(hyperparameter_specs),
            normalized_function_config,
            selected,
        )

    if scope == "point":
        if param_assignment is None:
            raise ValueError("param_assignment is required for point cache keys.")
        return (
            "point",
            algo_key,
            normalized_function_config,
            normalize_params(param_assignment),
            selected,
        )

    raise ValueError(f"Unsupported cache-key scope: {scope!r}")


def value_for_spec(spec: HyperparameterSpec, value: float) -> object:
    if spec.value_type == "int":
        return int(round(value))
    return float(value)


def _split_warning_lines(message: str | None) -> list[str]:
    if not message:
        return []
    return [line.strip() for line in str(message).splitlines() if line.strip()]


def compute_grid(
    *,
    algo_key: str,
    function_config: dict[str, dict[str, object]],
    hyperparameter_specs: list[HyperparameterSpec],
    rerun_nan_cache: bool,
    selected_dual_series_ids: tuple[str, ...] | None,
    show_progress: bool,
    batch_size: int | None,
    session_grid_cache: GridCachePort,
    persistent_point_cache: PointCachePort,
    executor: PointExecutorPort,
    progress_store: ProgressStore | None = None,
    progress_state_key: str | None = None,
    progress_callback: Callable[[int, int, float], None] | None = None,
) -> tuple | None:
    """Compute (or resume) an N-D tau/dual grid.

    Caching layers are intentionally split:
    - `session_grid_cache`: full-grid cache for fast same-request hits in the
      current UI session.
    - `persistent_point_cache`: point-level cache for incremental reuse/resume
      across reruns and restarts.
    """
    session_grid_key = make_cache_key(
        scope="session_grid",
        algo_key=algo_key,
        function_config=function_config,
        selected_dual_series_ids=selected_dual_series_ids,
        hyperparameter_specs=hyperparameter_specs,
    )
    cached_session_grid = session_grid_cache.get(session_grid_key)
    if isinstance(cached_session_grid, tuple) and len(cached_session_grid) == 4:
        if not rerun_nan_cache:
            return cached_session_grid
        cached_tau_nd = cached_session_grid[1]
        if isinstance(cached_tau_nd, np.ndarray) and not np.isnan(cached_tau_nd).any():
            return cached_session_grid

    param_values = {hp.name: discrete_values(hp) for hp in hyperparameter_specs}
    shape = tuple(len(param_values[hp.name]) for hp in hyperparameter_specs)
    tau_nd = np.full(shape, np.nan, dtype=float)
    duals_nd = np.empty(shape, dtype=object)
    warnings: set[str] = set()
    missing: list[tuple[tuple[int, ...], dict[str, object], tuple]] = []

    for idx_tuple in product(*[range(size) for size in shape]):
        algo_params = {
            hp.name: value_for_spec(hp, float(param_values[hp.name][idx_tuple[pos]]))
            for pos, hp in enumerate(hyperparameter_specs)
        }
        point_key = make_cache_key(
            scope="point",
            algo_key=algo_key,
            function_config=function_config,
            selected_dual_series_ids=selected_dual_series_ids,
            param_assignment=algo_params,
        )
        cached_point = persistent_point_cache.get(point_key)
        if cached_point is None or not isinstance(cached_point, tuple):
            duals_nd[idx_tuple] = {}
            missing.append((idx_tuple, algo_params, point_key))
            continue

        if len(cached_point) == 2:
            cached_tau, cached_warning = cached_point
            cached_duals = {}
        else:
            cached_tau, cached_warning, cached_duals = cached_point

        if rerun_nan_cache:
            try:
                if cached_tau is None or not np.isfinite(float(cached_tau)):
                    duals_nd[idx_tuple] = {}
                    missing.append((idx_tuple, algo_params, point_key))
                    continue
            except (TypeError, ValueError):
                duals_nd[idx_tuple] = {}
                missing.append((idx_tuple, algo_params, point_key))
                continue

        tau_nd[idx_tuple] = np.nan if cached_tau is None else float(cached_tau)
        duals_nd[idx_tuple] = cached_duals or {}
        for line in _split_warning_lines(cached_warning):
            warnings.add(line)

    total_points = int(np.prod(shape)) if shape else 0
    resolved_points = total_points - len(missing)
    if progress_store is not None and progress_state_key:
        progress_store[progress_state_key] = {
            "done": int(resolved_points),
            "total": int(total_points),
            "remaining": int(len(missing)),
        }

    if missing and not show_progress and batch_size is None:
        return None

    if missing:
        work_items = missing
        if batch_size is not None:
            work_items = missing[: max(1, int(batch_size))]

        total = max(len(work_items), 1)
        completed = 0
        start = time.perf_counter()
        update_every = max(total // 100, 1)
        active_dual_series_ids = tuple(selected_dual_series_ids or ())

        for idx_tuple, point_key, tau_value, duals, warning_message, should_cache in executor.compute_points(
            work_items=work_items,
            algo_key=algo_key,
            function_config=function_config,
            active_dual_series_ids=active_dual_series_ids,
        ):
            tau_nd[idx_tuple] = tau_value
            duals_nd[idx_tuple] = duals
            for line in _split_warning_lines(warning_message):
                warnings.add(line)
            if should_cache:
                persistent_point_cache.set(point_key, (tau_value, warning_message, duals))
            completed += 1
            if progress_callback and (completed % update_every == 0 or completed == total):
                fraction = completed / total
                elapsed = time.perf_counter() - start
                eta = (elapsed / fraction) - elapsed if fraction > 0 else 0.0
                progress_callback(completed, total, eta)

        persistent_point_cache.flush()

        remaining_after = len(missing) - len(work_items)
        done_after = resolved_points + len(work_items)
        if progress_store is not None and progress_state_key:
            progress_store[progress_state_key] = {
                "done": int(done_after),
                "total": int(total_points),
                "remaining": int(max(remaining_after, 0)),
            }
        if batch_size is not None and remaining_after > 0:
            return None

    result = (param_values, tau_nd, tuple(sorted(warnings)), duals_nd)
    session_grid_cache.set(session_grid_key, result)
    return result
