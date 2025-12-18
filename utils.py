# utils.py
import time
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
        tuple(sorted(other_params.items())),
    )


def clear_grid_cache_entry(
    algo_key: str,
    gamma_spec: HyperparameterSpec,
    n_spec: HyperparameterSpec,
    other_params: Dict[str, float],
) -> None:
    cache = st.session_state.get("tau_grid_cache")
    if cache is None:
        return
    cache.pop(make_cache_key(algo_key, gamma_spec, n_spec, other_params), None)


def get_tau_grid(
    algo_key: str,
    gamma_spec: HyperparameterSpec,
    n_spec: HyperparameterSpec,
    other_params: Dict[str, float],
    *,
    show_progress: bool,
):
    cache = st.session_state.setdefault("tau_grid_cache", {})
    key = make_cache_key(algo_key, gamma_spec, n_spec, other_params)
    if key in cache:
        return cache[key]
    if not show_progress:
        return None

    spec = ALGORITHMS[algo_key]
    gamma_values = discrete_values(gamma_spec)
    n_values = discrete_values(n_spec)
    tau_grid = np.full((len(gamma_values), len(n_values)), np.nan)
    warnings: set[str] = set()

    total = max(len(gamma_values) * len(n_values), 1)
    completed = 0
    progress_bar = st.progress(0.0)
    status_placeholder = st.empty()
    start = time.perf_counter()
    update_every = max(total // 100, 1)

    for i, gamma_value in enumerate(gamma_values):
        for j, n_value in enumerate(n_values):
            try:
                raw = spec.algo(float(gamma_value), float(n_value), **other_params)
                tau_grid[i, j] = float(np.asarray(raw).reshape(-1)[0])
            except AlgorithmEvaluationError as exc:
                warnings.add(f"{spec.name}: {exc}")
            except Exception as exc:
                warnings.add(f"{spec.name}: unexpected error - {exc}")
            completed += 1
            if completed % update_every == 0 or completed == total:
                fraction = completed / total
                elapsed = time.perf_counter() - start
                eta = (elapsed / fraction) - elapsed if fraction > 0 else 0.0
                progress_bar.progress(fraction)
                status_placeholder.write(f"Computing grid… {completed}/{total} (eta {eta:.1f}s)")

    progress_bar.empty()
    status_placeholder.empty()

    cache[key] = (gamma_values, n_values, tau_grid, tuple(sorted(warnings)))
    return cache[key]


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
