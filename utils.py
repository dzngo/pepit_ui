# utils.py
import html
import pickle
import random
import time
from decimal import ROUND_HALF_UP, Decimal
from itertools import product
from math import isfinite
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import streamlit as st
import sympy as sp

from algorithm.algorithm_custom import ALGORITHMS
from algorithm.runtime import run_algorithm
from algorithm.types import AlgorithmEvaluationError, HyperparameterSpec


def slider_for_param(
    param: HyperparameterSpec,
    *,
    value: float | None = None,
    key: str | None = None,
) -> float:
    slider_value = param.default if value is None else value
    if param.value_type == "int":
        return float(
            st.slider(
                param.label,
                int(param.min_value),
                int(param.max_value),
                int(slider_value),
                step=int(param.step),
                key=key,
            )
        )
    return float(
        st.slider(
            param.label,
            float(param.min_value),
            float(param.max_value),
            float(slider_value),
            step=float(param.step),
            key=key,
        )
    )


def discrete_values(param: HyperparameterSpec) -> np.ndarray:
    min_value = Decimal(str(param.min_value))
    max_value = Decimal(str(param.max_value))
    step = Decimal(str(param.step))
    steps = int(((max_value - min_value) / step).to_integral_value(rounding=ROUND_HALF_UP))
    values = np.array([float(min_value + Decimal(i) * step) for i in range(steps + 1)], dtype=float)
    if param.value_type == "int":
        values = np.round(values).astype(int)
    return values


POINT_CACHE_PATH = Path(__file__).resolve().parent / ".tau_point_cache.pkl"
POINT_CACHE_KEY = "tau_point_cache"


def _round_value(value: float, *, digits: int = 12) -> float:
    return float(round(float(value), digits))


def _normalize_param_value(value: object) -> object:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return _round_value(float(value))
    if isinstance(value, np.ndarray):
        return tuple(_normalize_param_value(item) for item in value.tolist())
    if isinstance(value, (list, tuple, set)):
        return tuple(_normalize_param_value(item) for item in value)
    if isinstance(value, dict):
        return tuple((key, _normalize_param_value(val)) for key, val in sorted(value.items()))
    return value


def _normalize_params(params: Dict[str, object]) -> Tuple[Tuple[str, object], ...]:
    normalized = []
    for name, value in params.items():
        if value is None:
            continue
        normalized.append((name, _normalize_param_value(value)))
    return tuple(sorted(normalized, key=lambda item: item[0]))


def _load_point_cache() -> Dict[Tuple, Tuple[float, str | None, Dict[str, Dict[str, float]]]]:
    cached = st.session_state.get(POINT_CACHE_KEY)
    if isinstance(cached, dict):
        return cached
    if POINT_CACHE_PATH.exists():
        try:
            with POINT_CACHE_PATH.open("rb") as handle:
                cached = pickle.load(handle)
        except Exception:
            cached = {}
    else:
        cached = {}
    if not isinstance(cached, dict):
        cached = {}
    st.session_state[POINT_CACHE_KEY] = cached
    return cached


def _save_point_cache(cache: Dict[Tuple, Tuple[float, str | None, Dict[str, Dict[str, float]]]]) -> None:
    tmp_path = POINT_CACHE_PATH.with_suffix(".tmp")
    try:
        with tmp_path.open("wb") as handle:
            pickle.dump(cache, handle)
        tmp_path.replace(POINT_CACHE_PATH)
    except Exception:
        return


def _normalize_function_config(function_config: Dict[str, Dict[str, object]]) -> Tuple:
    normalized = []
    for slot_key, config in sorted(function_config.items()):
        function_key = config["function_key"]
        function_params = config["function_params"]
        normalized.append((slot_key, function_key, _normalize_params(function_params)))
    return tuple(normalized)


def _normalize_hyperparameter_specs(hyperparameter_specs: list[HyperparameterSpec] | None) -> Tuple:
    specs = hyperparameter_specs or []
    return tuple(
        (
            spec.name,
            spec.label,
            _round_value(spec.min_value),
            _round_value(spec.max_value),
            _round_value(spec.default),
            _round_value(spec.step),
            spec.value_type,
        )
        for spec in specs
    )


def _point_cache_key(
    algo_key: str,
    function_config: Dict[str, Dict[str, object]],
    param_assignment: Dict[str, object],
    hyperparameter_specs: list[HyperparameterSpec] | None = None,
    selected_dual_series_ids: tuple[str, ...] | None = None,
) -> Tuple:
    return (
        algo_key,
        _normalize_function_config(function_config),
        _normalize_params(param_assignment),
        _normalize_hyperparameter_specs(hyperparameter_specs),
        tuple(selected_dual_series_ids or ()),
    )


def make_cache_key(
    algo_key: str,
    gamma_spec: HyperparameterSpec,
    n_spec: HyperparameterSpec,
    function_config: Dict[str, Dict[str, object]],
    hyperparameter_specs: list[HyperparameterSpec] | None = None,
    selected_dual_series_ids: tuple[str, ...] | None = None,
) -> Tuple:
    return (
        algo_key,
        (
            gamma_spec.min_value,
            gamma_spec.max_value,
            gamma_spec.step,
            gamma_spec.value_type,
        ),
        (
            n_spec.min_value,
            n_spec.max_value,
            n_spec.step,
            n_spec.value_type,
        ),
        _normalize_hyperparameter_specs(hyperparameter_specs),
        _normalize_function_config(function_config),
        tuple(selected_dual_series_ids or ()),
    )


def _value_for_spec(spec: HyperparameterSpec, value: float) -> object:
    if spec.value_type == "int":
        return int(round(value))
    return float(value)


def _slice_index_by_defaults(
    hyperparameter_specs: list[HyperparameterSpec],
    *,
    axis_overrides: Dict[str, int] | None = None,
) -> tuple[int, ...]:
    overrides = axis_overrides or {}
    indices: list[int] = []
    for spec in hyperparameter_specs:
        idx = int(overrides.get(spec.name, value_index(float(spec.default), spec)))
        total = int(round((spec.max_value - spec.min_value) / spec.step))
        indices.append(int(min(max(idx, 0), total)))
    return tuple(indices)


def _compute_nd(
    algo_key: str,
    function_config: Dict[str, Dict[str, object]],
    hyperparameter_specs: list[HyperparameterSpec],
    *,
    show_progress: bool,
    rerun_nan_cache: bool = False,
    selected_dual_series_ids: tuple[str, ...] | None = None,
):
    nd_cache = st.session_state.setdefault("tau_grid_cache_nd", {})
    nd_key = (
        algo_key,
        _normalize_hyperparameter_specs(hyperparameter_specs),
        _normalize_function_config(function_config),
        tuple(selected_dual_series_ids or ()),
    )
    if nd_key in nd_cache:
        cached = nd_cache[nd_key]
        if isinstance(cached, tuple) and len(cached) == 4:
            if not rerun_nan_cache:
                return cached
            cached_tau_nd = cached[1]
            if isinstance(cached_tau_nd, np.ndarray) and not np.isnan(cached_tau_nd).any():
                return cached
            nd_cache.pop(nd_key, None)
        else:
            nd_cache.pop(nd_key, None)

    spec = ALGORITHMS[algo_key]
    point_cache = _load_point_cache()
    param_values = {hp.name: discrete_values(hp) for hp in hyperparameter_specs}
    shape = tuple(len(param_values[hp.name]) for hp in hyperparameter_specs)
    tau_nd = np.full(shape, np.nan, dtype=float)
    duals_nd = np.empty(shape, dtype=object)
    warnings: set[str] = set()
    missing: list[tuple[tuple[int, ...], Dict[str, object], Tuple]] = []

    for idx_tuple in product(*[range(size) for size in shape]):
        algo_params = {
            hp.name: _value_for_spec(hp, float(param_values[hp.name][idx_tuple[pos]]))
            for pos, hp in enumerate(hyperparameter_specs)
        }
        point_key = _point_cache_key(
            algo_key,
            function_config,
            algo_params,
            hyperparameter_specs,
            selected_dual_series_ids,
        )
        cached_point = point_cache.get(point_key)
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
        if cached_warning:
            warnings.add(cached_warning)

    if missing and not show_progress:
        return None

    if missing:
        total = max(len(missing), 1)
        completed = 0
        progress_bar = st.progress(0.0)
        status_placeholder = st.empty()
        start = time.perf_counter()
        update_every = max(total // 100, 1)

        for idx_tuple, algo_params, point_key in missing:
            try:
                raw = run_algorithm(
                    algo_spec=spec,
                    function_config=function_config,
                    algo_params=algo_params,
                    active_dual_series_ids=set(selected_dual_series_ids or ()),
                )
                if isinstance(raw, tuple) and len(raw) == 2:
                    tau_raw, duals = raw
                else:
                    tau_raw, duals = raw, {}
                tau_value = float(np.asarray(tau_raw).reshape(-1)[0])
                tau_nd[idx_tuple] = tau_value
                duals_nd[idx_tuple] = duals or {}
                point_cache[point_key] = (tau_value, None, duals or {})
            except AlgorithmEvaluationError as exc:
                message = f"{spec.name}: {exc}"
                warnings.add(message)
                tau_nd[idx_tuple] = np.nan
                duals_nd[idx_tuple] = {}
                point_cache[point_key] = (np.nan, message, {})
            except Exception as exc:
                message = f"{spec.name}: unexpected error - {exc}"
                warnings.add(message)
                tau_nd[idx_tuple] = np.nan
                duals_nd[idx_tuple] = {}
            completed += 1
            if completed % update_every == 0 or completed == total:
                fraction = completed / total
                elapsed = time.perf_counter() - start
                eta = (elapsed / fraction) - elapsed if fraction > 0 else 0.0
                progress_bar.progress(fraction)
                status_placeholder.write(f"Computing grid… {completed}/{total} (eta {eta:.1f}s)")

        progress_bar.empty()
        status_placeholder.empty()
        _save_point_cache(point_cache)

    nd_cache[nd_key] = (param_values, tau_nd, tuple(sorted(warnings)), duals_nd)
    return nd_cache[nd_key]


def compute_nd(
    algo_key: str,
    function_config: Dict[str, Dict[str, object]],
    hyperparameter_specs: list[HyperparameterSpec],
    *,
    show_progress: bool,
    rerun_nan_cache: bool = False,
    selected_dual_series_ids: tuple[str, ...] | None = None,
):
    return _compute_nd(
        algo_key,
        function_config,
        hyperparameter_specs,
        show_progress=show_progress,
        rerun_nan_cache=rerun_nan_cache,
        selected_dual_series_ids=selected_dual_series_ids,
    )


def build_tau_series_by_param(
    hyperparameter_specs: list[HyperparameterSpec],
    param_values: dict[str, np.ndarray],
    tau_nd: np.ndarray,
    local_cursor_indices_by_axis: dict[str, dict[str, int]],
) -> dict[str, dict[str, object]]:
    if not hyperparameter_specs:
        return {}
    axis_index = {hp.name: idx for idx, hp in enumerate(hyperparameter_specs)}
    series_by_param: dict[str, dict[str, object]] = {}
    for hp in hyperparameter_specs:
        base_indices: list[int] = []
        axis_locals = local_cursor_indices_by_axis.get(hp.name, {})
        for base_hp in hyperparameter_specs:
            values = np.asarray(param_values[base_hp.name])
            max_idx = max(len(values) - 1, 0)
            default_idx = value_index(float(base_hp.default), base_hp)
            if base_hp.name == hp.name:
                idx = default_idx
            else:
                idx = int(axis_locals.get(base_hp.name, default_idx))
            base_indices.append(max(0, min(idx, max_idx)))
        axis = axis_index[hp.name]
        values = np.asarray(param_values[hp.name], dtype=float)
        y_values: list[float | None] = []
        for i in range(len(values)):
            idx_tuple = list(base_indices)
            idx_tuple[axis] = i
            tau_val = float(tau_nd[tuple(idx_tuple)])
            y_values.append(tau_val if np.isfinite(tau_val) else None)
        series_by_param[hp.name] = {
            "x_values": [float(v) for v in values],
            "y_values": y_values,
            "cursor_idx": int(base_indices[axis]),
            "cursor_value": float(values[base_indices[axis]]) if len(values) else None,
        }
    return series_by_param


def build_dual_slice_by_param(
    duals_nd: np.ndarray,
    hyperparameter_specs: list[HyperparameterSpec],
    cursor_indices: dict[str, int],
    param_name: str,
) -> list[dict]:
    if not hyperparameter_specs:
        return []
    axis_index = {hp.name: idx for idx, hp in enumerate(hyperparameter_specs)}
    if param_name not in axis_index:
        return []
    base_indices: list[int] = []
    for hp in hyperparameter_specs:
        total = int(round((hp.max_value - hp.min_value) / hp.step))
        idx = int(cursor_indices.get(hp.name, value_index(float(hp.default), hp)))
        base_indices.append(int(min(max(idx, 0), total)))
    axis = axis_index[param_name]
    length = duals_nd.shape[axis]
    out: list[dict] = []
    for i in range(length):
        idx_tuple = list(base_indices)
        idx_tuple[axis] = i
        value = duals_nd[tuple(idx_tuple)]
        out.append(value if isinstance(value, dict) else {})
    return out


def build_dual_series_by_param(
    duals_nd: np.ndarray,
    param_values: dict[str, np.ndarray],
    hyperparameter_specs: list[HyperparameterSpec],
    cursor_indices: dict[str, int],
) -> dict:
    if not hyperparameter_specs:
        return {}
    axis_index = {hp.name: idx for idx, hp in enumerate(hyperparameter_specs)}
    base_indices: list[int] = []
    for hp in hyperparameter_specs:
        values = np.asarray(param_values[hp.name])
        max_idx = max(len(values) - 1, 0)
        idx = int(cursor_indices.get(hp.name, value_index(float(hp.default), hp)))
        base_indices.append(max(0, min(idx, max_idx)))

    series_meta: dict[str, tuple[str, str]] = {}
    for idx_tuple in np.ndindex(duals_nd.shape):
        point = duals_nd[idx_tuple]
        if not isinstance(point, dict):
            continue
        for constraint, values in point.items():
            for dual_key in values.keys():
                series_meta[_dual_series_id(constraint, dual_key)] = (constraint, dual_key)

    series_data: dict[str, dict] = {}
    for series_id, (constraint, dual_key) in series_meta.items():
        by_param: dict[str, dict[str, object]] = {}
        for hp in hyperparameter_specs:
            axis = axis_index[hp.name]
            x_vals = np.asarray(param_values[hp.name], dtype=float)
            y_vals: list[float | None] = []
            for i in range(len(x_vals)):
                idx_tuple = list(base_indices)
                idx_tuple[axis] = i
                point = duals_nd[tuple(idx_tuple)]
                if not isinstance(point, dict):
                    y_vals.append(None)
                    continue
                val = point.get(constraint, {}).get(dual_key)
                if val is None or not np.isfinite(val):
                    y_vals.append(None)
                else:
                    y_vals.append(float(val))
            clean = [v for v in y_vals if v is not None and np.isfinite(v)]
            by_param[hp.name] = {
                "x_values": [float(v) for v in x_vals],
                "y_values": y_vals,
                "all_zero": bool(clean) and all(abs(v) <= 1e-12 for v in clean),
            }
        series_data[series_id] = {
            "constraint": constraint,
            "dual_key": dual_key,
            "label": f"{constraint} | {dual_key}",
            "by_param": by_param,
        }
    return series_data


def compute(
    algo_key: str,
    gamma_spec: HyperparameterSpec,
    n_spec: HyperparameterSpec,
    function_config: Dict[str, Dict[str, object]],
    hyperparameter_specs: list[HyperparameterSpec] | None = None,
    *,
    show_progress: bool,
    rerun_nan_cache: bool = False,
    selected_dual_series_ids: tuple[str, ...] | None = None,
):
    effective_hyperparameter_specs = list(hyperparameter_specs or [gamma_spec, n_spec])
    grid_cache = st.session_state.setdefault("tau_grid_cache", {})
    key = make_cache_key(
        algo_key,
        gamma_spec,
        n_spec,
        function_config,
        effective_hyperparameter_specs,
        selected_dual_series_ids,
    )
    if key in grid_cache:
        cached = grid_cache[key]
        if isinstance(cached, tuple) and len(cached) == 5:
            if not rerun_nan_cache:
                return cached
            cached_tau_grid = cached[2]
            if isinstance(cached_tau_grid, np.ndarray) and not np.isnan(cached_tau_grid).any():
                return cached
            grid_cache.pop(key, None)
        else:
            grid_cache.pop(key, None)
    nd_result = _compute_nd(
        algo_key,
        function_config,
        effective_hyperparameter_specs,
        show_progress=show_progress,
        rerun_nan_cache=rerun_nan_cache,
        selected_dual_series_ids=selected_dual_series_ids,
    )
    if nd_result is None:
        return None

    param_values, tau_nd, warnings_tuple, duals_nd = nd_result
    first_name = effective_hyperparameter_specs[0].name
    if len(effective_hyperparameter_specs) > 1:
        second_name = effective_hyperparameter_specs[1].name
    else:
        second_name = first_name

    gamma_values = np.asarray(param_values[first_name], dtype=float)
    n_values = np.asarray(param_values[second_name], dtype=float)
    gamma_axis = 0
    n_axis = 1 if len(effective_hyperparameter_specs) > 1 else 0

    tau_grid = np.full((len(gamma_values), len(n_values)), np.nan, dtype=float)
    duals_grid = [[{} for _ in range(len(n_values))] for _ in range(len(gamma_values))]

    base_index = list(_slice_index_by_defaults(effective_hyperparameter_specs))
    for i in range(len(gamma_values)):
        for j in range(len(n_values)):
            base_index[gamma_axis] = i
            base_index[n_axis] = j
            idx_tuple = tuple(base_index)
            tau_grid[i, j] = float(tau_nd[idx_tuple]) if np.isfinite(tau_nd[idx_tuple]) else np.nan
            dual_value = duals_nd[idx_tuple]
            duals_grid[i][j] = dual_value if isinstance(dual_value, dict) else {}

    grid_cache[key] = (
        gamma_values,
        n_values,
        tau_grid,
        tuple(sorted(set(warnings_tuple))),
        duals_grid,
    )
    return grid_cache[key]


def clear_algorithm_caches(algo_key: str) -> None:
    grid_cache = st.session_state.get("tau_grid_cache")
    if isinstance(grid_cache, dict):
        keys_to_remove = [key for key in grid_cache.keys() if key and key[0] == algo_key]
        for key in keys_to_remove:
            grid_cache.pop(key, None)

    point_cache = _load_point_cache()
    if isinstance(point_cache, dict):
        point_keys = [key for key in point_cache.keys() if key and key[0] == algo_key]
        for key in point_keys:
            point_cache.pop(key, None)
        _save_point_cache(point_cache)


def value_index(value: float, spec: HyperparameterSpec) -> int:
    idx = int(round((value - spec.min_value) / spec.step))
    total = int(round((spec.max_value - spec.min_value) / spec.step))
    return int(min(max(idx, 0), total))


def clamp_value(value: float, spec: HyperparameterSpec) -> float:
    return float(min(max(value, spec.min_value), spec.max_value))


def _dual_series_id(constraint: str, dual_key: str) -> str:
    return f"{constraint}||{dual_key}"


def jet_color(value: float) -> str:
    value = max(0.0, min(1.0, float(value)))
    r = max(0.0, min(1.0, 1.5 - abs(4 * value - 3)))
    g = max(0.0, min(1.0, 1.5 - abs(4 * value - 2)))
    b = max(0.0, min(1.0, 1.5 - abs(4 * value - 1)))
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def text_color_for_bg(hex_color: str) -> str:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return "#0b0b0b"
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#f7f7f7" if luminance < 0.55 else "#0b0b0b"


def html_escape(value: str) -> str:
    return html.escape(str(value), quote=True)


def format_dual_value(value: float | None) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float) and not np.isfinite(value):
        return "N/A"
    return f"{value:.6g}"


def dual_ranking_by_slice(
    slice_duals: list[dict],
    *,
    metric: str = "std",
) -> dict:
    include_none = metric.endswith("_with_none")
    base_metric = metric.replace("_with_none", "", 1) if include_none else metric
    dual_values: dict = {}
    if include_none:
        keys: set[tuple[str, str]] = set()
        for point_duals in slice_duals:
            for constraint, values in point_duals.items():
                for dual_key in values.keys():
                    keys.add((constraint, dual_key))
        for constraint, dual_key in keys:
            values: list[float] = []
            for point_duals in slice_duals:
                dual_value = point_duals.get(constraint, {}).get(dual_key, 0.0)
                if dual_value is None or not np.isfinite(dual_value):
                    dual_value = 0.0
                values.append(float(dual_value))
            dual_values.setdefault(constraint, {})[dual_key] = values
    else:
        for point_duals in slice_duals:
            for constraint, values in point_duals.items():
                for dual_key, dual_value in values.items():
                    if dual_value is None or not np.isfinite(dual_value):
                        continue
                    dual_values.setdefault(constraint, {}).setdefault(dual_key, []).append(float(dual_value))
    ranking: dict = {}
    for constraint, key_values in dual_values.items():
        ranking_map: dict = {}
        for dual_key, values in key_values.items():
            arr = np.asarray(values, dtype=float)
            if arr.size == 0:
                continue
            if base_metric == "std":
                ranking_map[dual_key] = float(np.std(arr))
            elif base_metric == "non_zero_pct":
                non_zero = float(np.sum(np.abs(arr) > 1e-12))
                ranking_map[dual_key] = 100.0 * non_zero / float(arr.size)
            elif base_metric == "median_abs":
                ranking_map[dual_key] = float(np.median(np.abs(arr)))
            elif base_metric == "mean_abs":
                ranking_map[dual_key] = float(np.mean(np.abs(arr)))
            else:
                raise NotImplementedError
        if ranking_map:
            ranking[constraint] = ranking_map
    return ranking


def build_dual_series_data(
    duals_grid: list[list[dict]],
    gamma_values: np.ndarray,
    n_values: np.ndarray,
    gamma_idx: int,
    n_idx: int,
) -> dict:
    gamma_len = len(gamma_values)
    n_len = len(n_values)
    series_meta: dict[str, tuple[str, str]] = {}
    gamma_keys: set[str] = set()
    n_keys: set[str] = set()

    for i in range(gamma_len):
        point = duals_grid[i][n_idx]
        for constraint, values in point.items():
            for dual_key in values.keys():
                key = _dual_series_id(constraint, dual_key)
                series_meta[key] = (constraint, dual_key)
                gamma_keys.add(key)

    for j in range(n_len):
        point = duals_grid[gamma_idx][j]
        for constraint, values in point.items():
            for dual_key in values.keys():
                key = _dual_series_id(constraint, dual_key)
                series_meta[key] = (constraint, dual_key)
                n_keys.add(key)

    all_keys = gamma_keys | n_keys
    gamma_series = {key: [None] * gamma_len for key in all_keys}
    n_series = {key: [None] * n_len for key in all_keys}

    for i in range(gamma_len):
        point = duals_grid[i][n_idx]
        for constraint, values in point.items():
            for dual_key, value in values.items():
                key = _dual_series_id(constraint, dual_key)
                if key not in gamma_series:
                    continue
                if value is None or not np.isfinite(value):
                    continue
                gamma_series[key][i] = float(value)

    for j in range(n_len):
        point = duals_grid[gamma_idx][j]
        for constraint, values in point.items():
            for dual_key, value in values.items():
                key = _dual_series_id(constraint, dual_key)
                if key not in n_series:
                    continue
                if value is None or not np.isfinite(value):
                    continue
                n_series[key][j] = float(value)

    series_data = {}
    gamma_list = [float(value) for value in gamma_values]
    n_list = [float(value) for value in n_values]
    for key in all_keys:
        constraint, dual_key = series_meta.get(key, ("", ""))
        gamma_dual = gamma_series[key]
        n_dual = n_series[key]
        gamma_values_clean = [value for value in gamma_dual if value is not None and np.isfinite(value)]
        n_values_clean = [value for value in n_dual if value is not None and np.isfinite(value)]
        all_zero_gamma = bool(gamma_values_clean) and all(abs(value) <= 1e-12 for value in gamma_values_clean)
        all_zero_n = bool(n_values_clean) and all(abs(value) <= 1e-12 for value in n_values_clean)
        series_data[key] = {
            "constraint": constraint,
            "dual_key": dual_key,
            "label": f"{constraint} | {dual_key}",
            "gamma_values": gamma_list,
            "gamma_dual": gamma_dual,
            "n_values": n_list,
            "n_dual": n_dual,
            "all_zero_gamma": all_zero_gamma,
            "all_zero_n": all_zero_n,
        }
    return series_data


def build_dual_section_html(
    *,
    section_id: str,
    section_key: str,
    title: str,
    dual_ranking: dict,
    current_duals: dict,
) -> tuple[str, int]:
    if not dual_ranking:
        return f"<div class='dual-section-title'>{html_escape(title)}</div><div>No data.</div>", 0

    section_html = [f"<div class='dual-section-title'>{html_escape(title)}</div>"]
    total_buttons = 0
    for constraint, ranking_map in sorted(dual_ranking.items()):
        if not ranking_map:
            continue
        max_ranking = max(ranking_map.values()) if ranking_map else 0.0
        max_ranking = max(max_ranking, 1e-12)
        section_html.append(
            f"<div class='dual-constraint-title' data-constraint='{html_escape(constraint)}'>"
            f"{html_escape(constraint)}</div>"
        )
        section_html.append(
            "<div class='dual-grid' "
            f"data-constraint='{html_escape(constraint)}' "
            f"data-section='{html_escape(section_key)}'>"
        )
        for dual_key, ranking in sorted(ranking_map.items(), key=lambda item: item[1], reverse=True):
            color = jet_color(ranking / max_ranking)
            text_color = text_color_for_bg(color)
            label = f"{constraint} | {dual_key}"
            data_id = f"{section_id}::{constraint}::{dual_key}"
            series_id = _dual_series_id(constraint, dual_key)
            button_label = _format_dual_key_label(dual_key)
            section_html.append(
                f"<button class='dual-button' data-id='{html_escape(data_id)}' "
                f"data-series-id='{html_escape(series_id)}' "
                f"data-section='{html_escape(section_key)}' "
                f"data-ranking='{html_escape(ranking)}' "
                f"data-label='{html_escape(label)}'"
                f"ranking-legend='score: {html_escape(format_dual_value(ranking))}' "
                f"style='background:{html_escape(color)};color:{html_escape(text_color)}'>"
                f"{button_label}</button>"
            )
            total_buttons += 1
        section_html.append("</div>")
    return "".join(section_html), total_buttons


def _format_dual_key_label(text: str) -> str:
    if not text:
        return ""
    return _subscript_to_html(text.strip())


def _subscript_to_html(text: str) -> str:
    parts: list[str] = []
    i = 0
    length = len(text)
    while i < length:
        ch = text[i]
        if ch != "_":
            parts.append(html_escape(ch))
            i += 1
            continue
        if i + 1 >= length:
            parts.append(html_escape(ch))
            i += 1
            continue
        start = i + 1
        end = start
        while end < length and (text[end].isalnum() or text[end] == "*"):
            end += 1
        if end == start:
            sub_text = text[i + 1]
            parts.append(f"<sub>{html_escape(sub_text)}</sub>")
            i += 2
            continue
        sub_text = text[start:end]
        parts.append(f"<sub>{html_escape(sub_text)}</sub>")
        i = end
    return "".join(parts)


def _float_default(param_default: object | None) -> float:
    if isinstance(param_default, (int, float)) and isfinite(float(param_default)):
        return float(param_default)
    return 1.0


def _float_text_default(param_default: object | None) -> str:
    if isinstance(param_default, (int, float)):
        value = float(param_default)
        if isfinite(value):
            return str(value)
        return "inf"
    return ""


def _parse_float_input(raw: str) -> tuple[float | None, str | None]:
    text = raw.strip().lower()
    if not text:
        return None, None
    if text in {"inf", "infinity", "+inf", "+infinity", "np.inf"}:
        return float("inf"), None
    try:
        return float(text), None
    except ValueError:
        return None, f"Invalid float value: {raw!r}"


def _parse_float_list(raw: str) -> tuple[list[float], str | None]:
    text = raw.strip()
    if not text:
        return [], None
    values: list[float] = []
    for part in text.split(","):
        if not part.strip():
            continue
        try:
            values.append(float(part.strip()))
        except ValueError:
            return [], f"Invalid list value: {part.strip()!r}"
    return values, None


_PATTERN_EXAMPLES = (
    "1/log(x)",
    "sin(x)",
    "2*x^2 + 3",
    "log(x)",
    "sqrt(x)",
    "exp(-x)",
)

_X_SYMBOL = sp.symbols("x")


def _random_pattern_example() -> str:
    return f"example: {random.choice(_PATTERN_EXAMPLES)}"


def _build_pattern_param_values(function_config: dict) -> tuple[dict[str, float], list[str], list[str]]:
    param_values: dict[str, float] = {}
    invalid: list[str] = []
    conflicts: list[str] = []
    for _, slot_config in sorted(function_config.items()):
        for name, raw in (slot_config.get("function_params") or {}).items():
            if name in ("x", "pi", "e"):
                conflicts.append(name)
                continue
            value = None
            if isinstance(raw, (int, float, np.integer, np.floating)):
                value = float(raw)
            elif isinstance(raw, str) and raw.strip():
                try:
                    value = float(raw)
                except ValueError:
                    invalid.append(name)
            if value is None:
                if name not in invalid:
                    invalid.append(name)
                continue
            if name in param_values and not np.isclose(param_values[name], value, rtol=1e-6, atol=1e-12):
                conflicts.append(name)
                continue
            param_values[name] = value
    return param_values, sorted(set(invalid)), sorted(set(conflicts))


def _evaluate_pattern_expression(
    expression: str,
    x_values: np.ndarray,
    param_values: dict[str, float],
    variable_names: tuple[str, ...],
) -> tuple[np.ndarray | None, str | None]:
    if not expression:
        return None, None
    cleaned = expression.strip()
    if not cleaned:
        return None, None
    conflicts = set(param_values).intersection(variable_names)
    if conflicts:
        names = ", ".join(sorted(conflicts))
        return None, f"Parameter name conflicts with variable names: {names}"
    locals_map = dict(param_values)
    for name in variable_names:
        locals_map[name] = _X_SYMBOL
    try:
        parsed = sp.sympify(cleaned, locals=locals_map)
    except (sp.SympifyError, TypeError, ValueError) as err:
        return None, f"Invalid expression: {err}"
    remaining_symbols = parsed.free_symbols - {_X_SYMBOL}
    if remaining_symbols:
        names = ", ".join(sorted(sym.name for sym in remaining_symbols))
        return None, f"Unknown parameters: {names}"
    try:
        fn = sp.lambdify(_X_SYMBOL, parsed, "numpy")
        values = np.asarray(fn(x_values), dtype=float)
    except Exception:
        return None, "Expression could not be evaluated for the current axis values."
    if not np.any(np.isfinite(values)):
        return values, "Expression did not produce finite values."
    return values, None
