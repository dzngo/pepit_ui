from typing import Literal

from algorithm.types import HyperparameterSpec
from core.compute import discrete_values

AlgoStatePrefix = Literal[
    "cursor_indices_by_param",
    "local_cursor_indices_by_axis",
    "tau_patterns_by_param",
    "recompute_runs",
    "recompute_counter",
    "recompute_event",
    "cursor_event",
    "metric_event",
    "remove_run_event",
    "loading_progress",
]


def algo_state_key(prefix: AlgoStatePrefix, algo_key: str) -> str:
    """Build session-state keys scoped by algorithm key.

    Example: `algo_state_key("recompute_runs", "gradient_descent")`
    returns `"recompute_runs_gradient_descent"`.
    """
    return f"{prefix}_{algo_key}"


def default_index_for_spec(spec: HyperparameterSpec) -> int:
    total = int(round((spec.max_value - spec.min_value) / spec.step))
    idx = int(round((spec.default - spec.min_value) / spec.step))
    return int(min(max(idx, 0), total))


def default_cursor_indices(specs: list[HyperparameterSpec]) -> dict[str, int]:
    return {hp.name: default_index_for_spec(hp) for hp in specs}


def clamp_cursor_indices(indices: dict[str, int], specs: list[HyperparameterSpec]) -> dict[str, int]:
    clamped: dict[str, int] = {}
    for hp in specs:
        values = discrete_values(hp)
        max_idx = max(len(values) - 1, 0)
        raw_idx = int(indices.get(hp.name, default_index_for_spec(hp)))
        clamped[hp.name] = max(0, min(raw_idx, max_idx))
    return clamped


def param_values_by_name(specs: list[HyperparameterSpec]) -> dict[str, list[float]]:
    return {hp.name: [float(v) for v in discrete_values(hp)] for hp in specs}


def default_local_cursor_indices_by_axis(
    specs: list[HyperparameterSpec],
    cursor_indices: dict[str, int],
) -> dict[str, dict[str, int]]:
    names = [hp.name for hp in specs]
    return {
        axis: {
            name: int(cursor_indices.get(name, default_index_for_spec(next_hp)))
            for name, next_hp in ((hp.name, hp) for hp in specs)
            if name != axis
        }
        for axis in names
    }


def clamp_local_cursor_indices_by_axis(
    local_by_axis: dict[str, dict[str, int]],
    specs: list[HyperparameterSpec],
    cursor_indices: dict[str, int],
) -> dict[str, dict[str, int]]:
    names = [hp.name for hp in specs]
    clamped_cursor = clamp_cursor_indices(cursor_indices, specs)
    clamped: dict[str, dict[str, int]] = {}
    for axis in names:
        incoming_axis = local_by_axis.get(axis, {})
        axis_map: dict[str, int] = {}
        for hp in specs:
            if hp.name == axis:
                continue
            values = discrete_values(hp)
            max_idx = max(len(values) - 1, 0)
            fallback = int(clamped_cursor.get(hp.name, default_index_for_spec(hp)))
            raw_idx = int(incoming_axis.get(hp.name, fallback))
            axis_map[hp.name] = max(0, min(raw_idx, max_idx))
        clamped[axis] = axis_map
    return clamped
