from algorithm.types import AlgorithmSpec
from core.config import default_function_rows_from_spec, validate_function_rows


def function_row_id_key(algo_key: str) -> str:
    return f"function-row-{algo_key}-"


def next_function_row_id(algo_key: str) -> str:
    import streamlit as st

    counters = st.session_state["function_row_counter_store"]
    next_id = int(counters.get(algo_key, 0)) + 1
    counters[algo_key] = next_id
    return f"r{next_id}"


def sanitize_function_rows(
    algo_key: str,
    rows: list[dict[str, object]],
    spec: AlgorithmSpec,
    *,
    valid_function_keys: list[str],
) -> list[dict[str, object]]:
    fallback_function_key = valid_function_keys[0] if valid_function_keys else ""
    normalized: list[dict[str, object]] = []
    for idx, row in enumerate(rows):
        row_id = str(row.get("id") or "").strip()
        if not row_id:
            row_id = next_function_row_id(algo_key)
        name = str(row.get("name") or "").strip()
        if not name:
            name = f"f{idx + 1}"
        function_key = str(row.get("function_key") or "").strip()
        if function_key not in valid_function_keys:
            function_key = fallback_function_key
        function_params = row.get("function_params")
        if not isinstance(function_params, dict):
            function_params = {}
        normalized.append(
            {
                "id": row_id,
                "name": name,
                "function_key": function_key,
                "function_params": dict(function_params),
            }
        )

    if not normalized:
        normalized = default_function_rows_from_spec(spec)
        normalized = sanitize_function_rows(
            algo_key,
            normalized,
            spec,
            valid_function_keys=valid_function_keys,
        )
    return normalized


def get_function_rows(
    algo_key: str,
    spec: AlgorithmSpec,
    *,
    valid_function_keys: list[str],
) -> list[dict[str, object]]:
    import streamlit as st

    rows_store = st.session_state["function_rows_store"]
    if algo_key not in rows_store:
        rows_store[algo_key] = default_function_rows_from_spec(spec)
    rows_store[algo_key] = sanitize_function_rows(
        algo_key,
        list(rows_store.get(algo_key, [])),
        spec,
        valid_function_keys=valid_function_keys,
    )
    return rows_store[algo_key]


def suggest_new_function_name(rows: list[dict[str, object]]) -> str:
    used = {str(row.get("name") or "").strip() for row in rows}
    candidate = "f"
    if candidate not in used:
        return candidate
    idx = 1
    while True:
        candidate = f"f{idx}"
        if candidate not in used:
            return candidate
        idx += 1


def validate_function_rows_with_rules(
    rows: list[dict[str, object]],
    *,
    reserved_names: set[str],
    valid_function_keys: list[str],
) -> list[str]:
    return validate_function_rows(
        rows,
        reserved_names=reserved_names,
        valid_function_keys=valid_function_keys,
    )


def build_function_config(
    algo_key: str,
    spec: AlgorithmSpec,
    *,
    valid_function_keys: list[str],
) -> dict[str, dict[str, object]]:
    rows = get_function_rows(algo_key, spec, valid_function_keys=valid_function_keys)
    function_config: dict[str, dict[str, object]] = {}
    for row in rows:
        row_id = str(row.get("id") or next_function_row_id(algo_key))
        name = str(row.get("name") or "").strip()
        function_config[row_id] = {
            "function_key": str(row.get("function_key") or "").strip(),
            "function_params": dict(row.get("function_params") or {}),
            "alias": name,
        }
    return function_config
