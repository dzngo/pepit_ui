# utils.py
import html
from math import isfinite
import pickle
import time
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import streamlit as st
import sympy as sp

from algorithms_registry import (
    DEFAULT_HYPERPARAMETERS,
    ALGORITHMS,
    AlgorithmEvaluationError,
    HyperparameterSpec,
    run_algorithm,
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


def _normalize_params(params: Dict[str, float]) -> Tuple[Tuple[str, float], ...]:
    return tuple(sorted((name, _round_value(value)) for name, value in params.items()))


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


def _save_point_cache(cache: Dict[Tuple, Tuple[float, str | None, Dict[str, Dict[str, float]]]]) -> None:
    tmp_path = POINT_CACHE_PATH.with_suffix(".tmp")
    try:
        with tmp_path.open("wb") as handle:
            pickle.dump(cache, handle)
        tmp_path.replace(POINT_CACHE_PATH)
    except Exception:
        return


def _normalize_function_config(function_config: Dict[str, Dict[str, float]]) -> Tuple:
    normalized = []
    for slot_key, config in sorted(function_config.items()):
        function_key = config["function_key"]
        function_params = config["function_params"]
        normalized.append((slot_key, function_key, _normalize_params(function_params)))
    return tuple(normalized)


def _point_cache_key(
    algo_key: str,
    function_config: Dict[str, Dict[str, float]],
    initial_condition_key: str,
    performance_metric_key: str,
    gamma_value: float,
    n_value: float,
) -> Tuple:
    return (
        algo_key,
        _normalize_function_config(function_config),
        initial_condition_key,
        performance_metric_key,
        _round_value(gamma_value),
        _round_value(n_value),
    )


def make_cache_key(
    algo_key: str,
    gamma_spec: HyperparameterSpec,
    n_spec: HyperparameterSpec,
    function_config: Dict[str, Dict[str, float]],
    initial_condition_key: str,
    performance_metric_key: str,
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
        _normalize_function_config(function_config),
        initial_condition_key,
        performance_metric_key,
    )


def clear_grid_cache_entry(
    algo_key: str,
    gamma_spec: HyperparameterSpec,
    n_spec: HyperparameterSpec,
    function_config: Dict[str, Dict[str, float]],
    initial_condition_key: str,
    performance_metric_key: str,
) -> None:
    cache = st.session_state.get("tau_grid_cache")
    if cache is not None:
        cache.pop(
            make_cache_key(
                algo_key,
                gamma_spec,
                n_spec,
                function_config,
                initial_condition_key,
                performance_metric_key,
            ),
            None,
        )


def compute(
    algo_key: str,
    gamma_spec: HyperparameterSpec,
    n_spec: HyperparameterSpec,
    function_config: Dict[str, Dict[str, float]],
    initial_condition_key: str,
    performance_metric_key: str,
    *,
    show_progress: bool,
    rerun_nan_cache: bool = False,
):
    grid_cache = st.session_state.setdefault("tau_grid_cache", {})
    key = make_cache_key(
        algo_key,
        gamma_spec,
        n_spec,
        function_config,
        initial_condition_key,
        performance_metric_key,
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
            point_key = _point_cache_key(
                algo_key,
                function_config,
                initial_condition_key,
                performance_metric_key,
                gamma_key,
                n_key,
            )
            cached_point = point_cache.get(point_key)
            if cached_point is None or not isinstance(cached_point, tuple):
                missing.append((i, j, float(gamma_value), float(n_value), point_key))
                continue
            if len(cached_point) == 2:
                cached_tau, cached_warning = cached_point
                cached_duals = {}
            else:
                cached_tau, cached_warning, cached_duals = cached_point
            if rerun_nan_cache:
                try:
                    if cached_tau is None or not np.isfinite(float(cached_tau)):
                        missing.append((i, j, float(gamma_value), float(n_value), point_key))
                        continue
                except (TypeError, ValueError):
                    missing.append((i, j, float(gamma_value), float(n_value), point_key))
                    continue
            tau_grid[i, j] = np.nan if cached_tau is None else float(cached_tau)
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
                raw = run_algorithm(
                    algo_spec=spec,
                    function_config=function_config,
                    initial_condition_key=initial_condition_key,
                    performance_metric_key=performance_metric_key,
                    algo_params={
                        "gamma": float(gamma_value),
                        "n": float(n_value),
                    },
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
                point_cache[point_key] = (np.nan, message, {})
            except Exception as exc:
                message = f"{spec.name}: unexpected error - {exc}"
                warnings.add(message)
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

    grid_cache[key] = (
        gamma_values,
        n_values,
        tau_grid,
        tuple(sorted(warnings)),
        duals_grid,
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
    min_width: int,
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
            value = current_duals.get(constraint, {}).get(dual_key)
            label = f"{constraint} | {dual_key}"
            data_id = f"{section_id}::{constraint}::{dual_key}"
            series_id = _dual_series_id(constraint, dual_key)
            section_html.append(
                f"<button class='dual-button' data-id='{html_escape(data_id)}' "
                f"data-series-id='{html_escape(series_id)}' "
                f"data-section='{html_escape(section_key)}' "
                f"data-ranking='{html_escape(ranking)}' "
                f"data-label='{html_escape(label)}' data-value='{html_escape(format_dual_value(value))}' "
                f"style='background:{html_escape(color)};color:{html_escape(text_color)}'>"
                f"{html_escape(dual_key)}</button>"
            )
            total_buttons += 1
        section_html.append("</div>")
    return "".join(section_html), total_buttons


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


def _random_pattern_example() -> str:
    return f"example: {random.choice(_PATTERN_EXAMPLES)}"


def _evaluate_pattern_expression(expression: str, x_values: np.ndarray) -> tuple[np.ndarray | None, str | None]:
    if not expression:
        return None, None
    cleaned = expression.strip()
    if not cleaned:
        return None, None
    try:
        parsed = sp.sympify(cleaned)
    except (sp.SympifyError, TypeError, ValueError) as err:
        return None, f"Invalid expression: {err}"
    if parsed.free_symbols - {sp.symbols("x")}:
        return None, "Expression must use x only."
    try:
        fn = sp.lambdify(sp.symbols("x"), parsed, "numpy")
        values = np.asarray(fn(x_values), dtype=float)
    except Exception:
        return None, "Expression could not be evaluated for the current axis values."
    if not np.any(np.isfinite(values)):
        return values, "Expression did not produce finite values."
    return values, None
