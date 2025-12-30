# utils.py
import pickle
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import streamlit as st

from algorithms_registry import (
    DEFAULT_HYPERPARAMETERS,
    ALGORITHMS,
    AlgorithmEvaluationError,
    HyperparameterSpec,
)


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
    steps = int(round((param.max_value - param.min_value) / param.step))
    values = np.array([param.min_value + i * param.step for i in range(steps + 1)], dtype=float)
    if param.value_type == "int":
        values = np.round(values).astype(int)
    return values


POINT_CACHE_PATH = Path(__file__).resolve().parent / ".tau_point_cache.pkl"
POINT_CACHE_KEY = "tau_point_cache"


def _round_value(value: float, *, digits: int = 12) -> float:
    return float(round(float(value), digits))


def _normalize_other_params(other_params: Dict[str, float]) -> Tuple[Tuple[str, float], ...]:
    return tuple(sorted((name, _round_value(value)) for name, value in other_params.items()))


def _quantize_value(value: float, spec: HyperparameterSpec) -> float:
    idx = value_index(float(value), spec)
    quantized = spec.min_value + idx * spec.step
    if spec.value_type == "int":
        quantized = int(round(quantized))
    return _round_value(float(quantized))


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


def _save_point_cache(
    cache: Dict[Tuple, Tuple[float, str | None, Dict[str, Dict[str, float]]]]
) -> None:
    tmp_path = POINT_CACHE_PATH.with_suffix(".tmp")
    try:
        with tmp_path.open("wb") as handle:
            pickle.dump(cache, handle)
        tmp_path.replace(POINT_CACHE_PATH)
    except Exception:
        return


def _point_cache_key(
    algo_key: str,
    other_params: Dict[str, float],
    gamma_value: float,
    n_value: float,
) -> Tuple:
    return (
        algo_key,
        _normalize_other_params(other_params),
        _round_value(gamma_value),
        _round_value(n_value),
    )


def make_cache_key(
    algo_key: str,
    gamma_spec: HyperparameterSpec,
    n_spec: HyperparameterSpec,
    other_params: Dict[str, float],
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
        _normalize_other_params(other_params),
    )


def clear_grid_cache_entry(
    algo_key: str,
    gamma_spec: HyperparameterSpec,
    n_spec: HyperparameterSpec,
    other_params: Dict[str, float],
) -> None:
    cache = st.session_state.get("tau_grid_cache")
    if cache is not None:
        cache.pop(make_cache_key(algo_key, gamma_spec, n_spec, other_params), None)


def get_tau_grid(
    algo_key: str,
    gamma_spec: HyperparameterSpec,
    n_spec: HyperparameterSpec,
    other_params: Dict[str, float],
    *,
    show_progress: bool,
):
    grid_cache = st.session_state.setdefault("tau_grid_cache", {})
    key = make_cache_key(algo_key, gamma_spec, n_spec, other_params)
    if key in grid_cache:
        cached = grid_cache[key]
        if isinstance(cached, tuple) and len(cached) == 6:
            return cached
        grid_cache.pop(key, None)

    spec = ALGORITHMS[algo_key]
    point_cache = _load_point_cache()
    gamma_values = discrete_values(gamma_spec)
    n_values = discrete_values(n_spec)
    tau_grid = np.full((len(gamma_values), len(n_values)), np.nan)
    warnings: set[str] = set()
    duals_grid = [[{} for _ in range(len(n_values))] for _ in range(len(gamma_values))]
    missing: list[tuple[int, int, float, float, Tuple]] = []

    for i, gamma_value in enumerate(gamma_values):
        gamma_key = _quantize_value(float(gamma_value), gamma_spec)
        for j, n_value in enumerate(n_values):
            n_key = _quantize_value(float(n_value), n_spec)
            point_key = _point_cache_key(algo_key, other_params, gamma_key, n_key)
            cached_point = point_cache.get(point_key)
            if cached_point is None or not isinstance(cached_point, tuple):
                missing.append((i, j, float(gamma_value), float(n_value), point_key))
                continue
            if len(cached_point) == 2:
                cached_tau, cached_warning = cached_point
                cached_duals = {}
            else:
                cached_tau, cached_warning, cached_duals = cached_point
            tau_grid[i, j] = float(cached_tau)
            duals_grid[i][j] = cached_duals or {}
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

        for i, j, gamma_value, n_value, point_key in missing:
            try:
                raw = spec.algo(
                    gamma=float(gamma_value),
                    n=float(n_value),
                    **other_params,
                )
                if isinstance(raw, tuple) and len(raw) == 2:
                    tau_raw, duals = raw
                else:
                    tau_raw, duals = raw, {}
                tau_value = float(np.asarray(tau_raw).reshape(-1)[0])
                duals_grid[i][j] = duals or {}
                tau_grid[i, j] = tau_value
                point_cache[point_key] = (tau_value, None, duals or {})
            except AlgorithmEvaluationError as exc:
                message = f"{spec.name}: {exc}"
                warnings.add(message)
                point_cache[point_key] = (float("nan"), message, {})
            except Exception as exc:
                message = f"{spec.name}: unexpected error - {exc}"
                warnings.add(message)
                point_cache[point_key] = (float("nan"), message, {})
            completed += 1
            if completed % update_every == 0 or completed == total:
                fraction = completed / total
                elapsed = time.perf_counter() - start
                eta = (elapsed / fraction) - elapsed if fraction > 0 else 0.0
                progress_bar.progress(fraction)
                status_placeholder.write(f"Computing grid… {completed}/{total} (eta {eta:.1f}s)")

        progress_bar.empty()
        status_placeholder.empty()
    if missing:
        _save_point_cache(point_cache)

    dual_fluctuations: Dict[str, Dict[str, float]] = {}
    dual_values: Dict[str, Dict[str, list[float]]] = {}
    for row in duals_grid:
        for point_duals in row:
            for constraint, values in point_duals.items():
                for dual_key, dual_value in values.items():
                    if dual_value is None or not np.isfinite(dual_value):
                        continue
                    dual_values.setdefault(constraint, {}).setdefault(dual_key, []).append(float(dual_value))
    for constraint, key_values in dual_values.items():
        fluct_map: Dict[str, float] = {}
        for dual_key, values in key_values.items():
            arr = np.asarray(values, dtype=float)
            if arr.size == 0:
                continue
            rms = float(np.sqrt(np.mean(arr**2)))
            if rms <= 0:
                fluct_map[dual_key] = 0.0
            else:
                fluct_map[dual_key] = float(np.std(arr) / rms)
        if fluct_map:
            dual_fluctuations[constraint] = fluct_map

    grid_cache[key] = (
        gamma_values,
        n_values,
        tau_grid,
        tuple(sorted(warnings)),
        duals_grid,
        dual_fluctuations,
    )
    return grid_cache[key]


def value_index(value: float, spec: HyperparameterSpec) -> int:
    idx = int(round((value - spec.min_value) / spec.step))
    total = int(round((spec.max_value - spec.min_value) / spec.step))
    return int(min(max(idx, 0), total))


def clamp_value(value: float, spec: HyperparameterSpec) -> float:
    return float(min(max(value, spec.min_value), spec.max_value))


def base_spec(name: str) -> HyperparameterSpec:
    return next(param for param in DEFAULT_HYPERPARAMETERS if param.name == name)


BASE_GAMMA_SPEC = base_spec("gamma")
BASE_N_SPEC = base_spec("n")
