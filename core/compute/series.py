import html
import re

import numpy as np

from algorithm.types import HyperparameterSpec


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
                if dual_value is None:
                    values.append(0.0)
                else:
                    values.append(float(dual_value))
            dual_values.setdefault(constraint, {})[dual_key] = values
    else:
        for point_duals in slice_duals:
            for constraint, values in point_duals.items():
                for dual_key, dual_value in values.items():
                    if dual_value is None:
                        continue
                    dual_values.setdefault(constraint, {}).setdefault(dual_key, []).append(float(dual_value))

    ranking: dict = {}
    for constraint, values in dual_values.items():
        for dual_key, series in values.items():
            arr = np.asarray(series, dtype=float)
            if arr.size == 0:
                score = 0.0
            elif base_metric == "non_zero_pct":
                score = float(np.count_nonzero(np.abs(arr) > 1e-12) / arr.size)
            elif base_metric == "mean_abs":
                score = float(np.mean(np.abs(arr)))
            elif base_metric == "median_abs":
                score = float(np.median(np.abs(arr)))
            else:
                score = float(np.std(arr))
            ranking.setdefault(constraint, {})[dual_key] = score
    return ranking


def build_dual_section_html(
    *,
    section_id: str,
    section_key: str,
    title: str,
    dual_ranking: dict,
    current_duals: dict[str, dict[str, float]],
) -> tuple[str, list[str]]:
    items: list[tuple[str, str, float]] = []
    for constraint, values in dual_ranking.items():
        for dual_key, score in values.items():
            items.append((constraint, dual_key, float(score)))
    if not items:
        return (
            f"<section id='{html_escape(section_id)}' class='dual-section'>"
            f"<h4>{html_escape(title)}</h4><p class='dual-empty'>No dual variables detected.</p></section>",
            [],
        )

    scores = np.array([item[2] for item in items], dtype=float)
    min_score = float(np.min(scores))
    max_score = float(np.max(scores))
    spread = max(max_score - min_score, 1e-12)

    rows: list[str] = []
    selected_ids: list[str] = []
    for constraint, dual_key, score in sorted(items, key=lambda item: item[2], reverse=True):
        normalized = (score - min_score) / spread
        bg = jet_color(normalized)
        fg = text_color_for_bg(bg)
        dual_id = _dual_series_id(constraint, dual_key)
        selected_ids.append(dual_id)
        dual_label = _format_dual_key_label(dual_key)
        dual_title = f"{constraint} | {dual_key}"
        ranking_legend = f"ranking score: {score:.4g}"
        rows.append(
            f"<button type='button' class='dual-button'"
            f" data-series-id='{html_escape(dual_id)}'"
            f" data-section='{html_escape(section_key)}'"
            f" data-label='{html_escape(dual_title)}'"
            f" ranking-legend='{html_escape(ranking_legend)}'"
            f" style='background:{bg};color:{fg}'"
            f" title='{html_escape(dual_title)}'>"
            f"{dual_label}"
            "</button>"
        )

    html_value = (
        f"<section id='{html_escape(section_id)}' class='dual-section' data-section-key='{html_escape(section_key)}'>"
        f"<h4>{html_escape(title)}</h4>"
        "<div class='dual-grid'>" + "".join(rows) + "</div></section>"
    )
    return html_value, selected_ids


def _format_dual_key_label(text: str) -> str:
    if "|" not in text:
        return _subscript_to_html(text)
    left, right = text.split("|", 1)
    return f"{html_escape(left.strip())} | {_subscript_to_html(right.strip())}"


def _subscript_to_html(text: str) -> str:
    if not text:
        return ""
    out = []
    i = 0
    while i < len(text):
        if text[i] != "_":
            out.append(html_escape(text[i]))
            i += 1
            continue
        if i + 1 < len(text) and text[i + 1] == "{":
            end = text.find("}", i + 2)
            if end != -1:
                out.append(f"<sub>{html_escape(text[i + 2:end])}</sub>")
                i = end + 1
                continue
        j = i + 1
        while j < len(text) and re.match(r"[A-Za-z0-9*]", text[j]):
            j += 1
        if j == i + 1:
            out.append(html_escape(text[i]))
            i += 1
            continue
        out.append(f"<sub>{html_escape(text[i + 1:j])}</sub>")
        i = j
    return "".join(out)
