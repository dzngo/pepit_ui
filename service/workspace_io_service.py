from __future__ import annotations

import pickle
from collections.abc import MutableMapping
from datetime import datetime, timezone
from typing import Any

from algorithm.algorithm_custom import (
    ALGORITHMS,
    CUSTOM_ALGORITHMS,
    register_custom_algorithm,
)
from algorithm.types import HyperparameterSpec
from ui.state.state_utils import algo_state_key

WORKSPACE_FILE_VERSION = 1
SESSION_GRID_CACHE_KEY = "tau_grid_cache_nd"


def _spec_to_dict(spec: HyperparameterSpec) -> dict[str, object]:
    return {
        "name": spec.name,
        "label": spec.label,
        "min_value": float(spec.min_value),
        "max_value": float(spec.max_value),
        "default": float(spec.default),
        "step": float(spec.step),
        "value_type": spec.value_type,
    }


def _spec_from_dict(value: object) -> HyperparameterSpec:
    if not isinstance(value, dict):
        raise ValueError("Invalid hyperparameter spec entry.")
    return HyperparameterSpec(
        name=str(value["name"]),
        label=str(value.get("label", value["name"])),
        min_value=float(value["min_value"]),
        max_value=float(value["max_value"]),
        default=float(value["default"]),
        step=float(value["step"]),
        value_type=str(value.get("value_type", "float")),
    )


def _serialize_active_settings(settings: dict[str, object]) -> dict[str, object]:
    hyperparameter_specs = list(settings.get("hyperparameter_specs", []))
    return {
        "algo_key": str(settings["algo_key"]),
        "hyperparameter_specs": [_spec_to_dict(spec) for spec in hyperparameter_specs],
        "function_config": dict(settings.get("function_config", {})),
        "rerun_nan_caches": bool(settings.get("rerun_nan_caches", False)),
    }


def _deserialize_active_settings(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("Missing active settings.")
    if "algo_key" not in value:
        raise ValueError("Missing algorithm key in active settings.")
    specs_payload = value.get("hyperparameter_specs", [])
    if not isinstance(specs_payload, list):
        raise ValueError("Invalid hyperparameter specs payload.")
    function_config = value.get("function_config", {})
    if not isinstance(function_config, dict):
        raise ValueError("Invalid function configuration payload.")
    return {
        "algo_key": str(value["algo_key"]),
        "hyperparameter_specs": [_spec_from_dict(item) for item in specs_payload],
        "function_config": dict(function_config),
        "rerun_nan_caches": bool(value.get("rerun_nan_caches", False)),
    }


def _is_session_grid_key_for_algo(key: object, algo_key: str) -> bool:
    return isinstance(key, tuple) and len(key) >= 2 and key[0] == "session_grid" and key[1] == algo_key


def _extract_session_grid_cache_subset(session_state: MutableMapping[str, object], algo_key: str) -> dict[tuple, tuple]:
    cache = session_state.get(SESSION_GRID_CACHE_KEY, {})
    if not isinstance(cache, dict):
        return {}
    return {key: value for key, value in cache.items() if _is_session_grid_key_for_algo(key, algo_key)}


def _remove_session_grid_cache_for_algo(session_state: MutableMapping[str, object], algo_key: str) -> None:
    cache = session_state.setdefault(SESSION_GRID_CACHE_KEY, {})
    if not isinstance(cache, dict):
        cache = {}
        session_state[SESSION_GRID_CACHE_KEY] = cache
    keys = [key for key in cache if _is_session_grid_key_for_algo(key, algo_key)]
    for key in keys:
        cache.pop(key, None)


def _custom_algorithm_payload_for(algo_key: str) -> dict[str, object] | None:
    payload = CUSTOM_ALGORITHMS.get(algo_key)
    if not isinstance(payload, dict):
        return None
    return dict(payload)


def build_work_checkpoint_bytes(
    session_state: MutableMapping[str, object],
    *,
    algo_key: str,
) -> tuple[bytes, str]:
    settings = session_state.get("active_settings")
    if not isinstance(settings, dict) or str(settings.get("algo_key", "")) != algo_key:
        raise ValueError("No active results state available for this algorithm.")

    snapshot = {
        "version": WORKSPACE_FILE_VERSION,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "algo_key": algo_key,
        "active_settings": _serialize_active_settings(settings),
        "custom_algorithm": _custom_algorithm_payload_for(algo_key),
        "ui_state": {
            "metric": str(session_state.get(f"dual-ranking-metric-{algo_key}", "non_zero_pct_with_none")),
            "cursor_indices_by_param": dict(session_state.get(algo_state_key("cursor_indices_by_param", algo_key), {})),
            "local_cursor_indices_by_axis": dict(
                session_state.get(algo_state_key("local_cursor_indices_by_axis", algo_key), {})
            ),
            "tau_patterns_by_param": dict(session_state.get(algo_state_key("tau_patterns_by_param", algo_key), {})),
            "recompute_runs": list(session_state.get(algo_state_key("recompute_runs", algo_key), [])),
            "recompute_counter": int(session_state.get(algo_state_key("recompute_counter", algo_key), 0)),
            "event_ids": {
                "recompute": str(session_state.get(algo_state_key("recompute_event", algo_key), "")),
                "cursor": str(session_state.get(algo_state_key("cursor_event", algo_key), "")),
                "metric": str(session_state.get(algo_state_key("metric_event", algo_key), "")),
                "remove": str(session_state.get(algo_state_key("remove_run_event", algo_key), "")),
            },
        },
        "session_grid_cache_subset": _extract_session_grid_cache_subset(session_state, algo_key),
        "stores": {
            "hyperparameter_rows": dict(session_state.get("hyperparameter_store", {})).get(algo_key, []),
            "function_rows": dict(session_state.get("function_rows_store", {})).get(algo_key, []),
            "function_row_counter": int(dict(session_state.get("function_row_counter_store", {})).get(algo_key, 0)),
        },
    }
    payload = pickle.dumps(snapshot, protocol=pickle.HIGHEST_PROTOCOL)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"{algo_key}-work-{timestamp}.pepit-work.bin"
    return payload, filename


def _ensure_custom_algorithm_available(algo_key: str, payload: dict[str, object] | None) -> None:
    if payload is None:
        if algo_key not in ALGORITHMS:
            raise ValueError(f"Algorithm '{algo_key}' is unavailable in this app.")
        return
    existing = CUSTOM_ALGORITHMS.get(algo_key)
    if existing is not None:
        if dict(existing) != dict(payload):
            raise ValueError(f"Custom algorithm '{algo_key}' already exists with different definition.")
        return
    if algo_key in ALGORITHMS:
        # Base algorithm with same name already exists and cannot be overwritten.
        raise ValueError(f"Algorithm name conflict for '{algo_key}'.")

    default_specs_payload = payload.get("default_hyperparameters", [])
    if not isinstance(default_specs_payload, list):
        default_specs_payload = []
    default_hyperparameters = [_spec_from_dict(item) for item in default_specs_payload]
    default_function_rows = payload.get("default_function_rows", [])
    if not isinstance(default_function_rows, list):
        default_function_rows = []
    register_custom_algorithm(
        name=algo_key,
        steps_code=str(payload.get("steps_code", "")),
        base_algo=str(payload.get("base_algo", "")),
        default_hyperparameters=default_hyperparameters,
        default_function_rows=[dict(row) for row in default_function_rows if isinstance(row, dict)],
    )


def _validate_snapshot(raw: object) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("Invalid work file format.")
    version = raw.get("version")
    if version != WORKSPACE_FILE_VERSION:
        raise ValueError(f"Unsupported work file version: {version}.")
    if "algo_key" not in raw:
        raise ValueError("Missing algorithm key.")
    algo_key = str(raw["algo_key"])
    active_settings = _deserialize_active_settings(raw.get("active_settings"))
    if str(active_settings.get("algo_key", "")) != algo_key:
        raise ValueError("Inconsistent algorithm key in active settings.")
    ui_state = raw.get("ui_state", {})
    if not isinstance(ui_state, dict):
        raise ValueError("Invalid UI state block.")
    session_grid_cache_subset = raw.get("session_grid_cache_subset", {})
    if not isinstance(session_grid_cache_subset, dict):
        raise ValueError("Invalid session grid cache block.")
    stores = raw.get("stores", {})
    if not isinstance(stores, dict):
        stores = {}
    custom_algorithm = raw.get("custom_algorithm")
    if custom_algorithm is not None and not isinstance(custom_algorithm, dict):
        raise ValueError("Invalid custom algorithm payload.")
    return {
        "algo_key": algo_key,
        "active_settings": active_settings,
        "ui_state": ui_state,
        "session_grid_cache_subset": session_grid_cache_subset,
        "stores": stores,
        "custom_algorithm": custom_algorithm,
    }


def load_work_checkpoint(
    session_state: MutableMapping[str, object],
    *,
    payload_bytes: bytes,
) -> tuple[bool, str]:
    try:
        raw = pickle.loads(payload_bytes)
    except Exception:
        return False, "Invalid or corrupted work file."
    try:
        parsed = _validate_snapshot(raw)
        algo_key = parsed["algo_key"]
        _ensure_custom_algorithm_available(algo_key, parsed["custom_algorithm"])

        active_settings = parsed["active_settings"]
        ui_state = parsed["ui_state"]
        session_grid_cache_subset = parsed["session_grid_cache_subset"]
        stores = parsed["stores"]

        # Apply atomically after all validation passes.
        session_state["selected_algorithm"] = algo_key
        session_state["pending_algorithm_select"] = algo_key
        session_state["active_settings"] = active_settings
        session_state["pending_settings"] = None
        session_state["ui_phase"] = "results"
        session_state["loaded_rerun_nan_caches"] = bool(active_settings.get("rerun_nan_caches", False))

        session_state[f"dual-ranking-metric-{algo_key}"] = str(ui_state.get("metric", "non_zero_pct_with_none"))
        session_state[algo_state_key("cursor_indices_by_param", algo_key)] = dict(
            ui_state.get("cursor_indices_by_param", {})
        )
        session_state[algo_state_key("local_cursor_indices_by_axis", algo_key)] = dict(
            ui_state.get("local_cursor_indices_by_axis", {})
        )
        session_state[algo_state_key("tau_patterns_by_param", algo_key)] = dict(
            ui_state.get("tau_patterns_by_param", {})
        )
        session_state[algo_state_key("recompute_runs", algo_key)] = list(ui_state.get("recompute_runs", []))
        session_state[algo_state_key("recompute_counter", algo_key)] = int(ui_state.get("recompute_counter", 0))

        event_ids = ui_state.get("event_ids", {})
        if not isinstance(event_ids, dict):
            event_ids = {}
        session_state[algo_state_key("recompute_event", algo_key)] = str(event_ids.get("recompute", ""))
        session_state[algo_state_key("cursor_event", algo_key)] = str(event_ids.get("cursor", ""))
        session_state[algo_state_key("metric_event", algo_key)] = str(event_ids.get("metric", ""))
        session_state[algo_state_key("remove_run_event", algo_key)] = str(event_ids.get("remove", ""))

        _remove_session_grid_cache_for_algo(session_state, algo_key)
        cache = session_state.setdefault(SESSION_GRID_CACHE_KEY, {})
        if not isinstance(cache, dict):
            cache = {}
            session_state[SESSION_GRID_CACHE_KEY] = cache
        for key, value in session_grid_cache_subset.items():
            cache[key] = value

        hyperparameter_store = session_state.setdefault("hyperparameter_store", {})
        if isinstance(hyperparameter_store, dict):
            hyperparameter_store[algo_key] = list(stores.get("hyperparameter_rows", []))
        function_rows_store = session_state.setdefault("function_rows_store", {})
        if isinstance(function_rows_store, dict):
            function_rows_store[algo_key] = list(stores.get("function_rows", []))
        function_row_counter_store = session_state.setdefault("function_row_counter_store", {})
        if isinstance(function_row_counter_store, dict):
            function_row_counter_store[algo_key] = int(stores.get("function_row_counter", 0))

    except Exception as exc:
        return False, str(exc)
    return True, "Work loaded successfully."


__all__ = [
    "build_work_checkpoint_bytes",
    "load_work_checkpoint",
    "WORKSPACE_FILE_VERSION",
]
