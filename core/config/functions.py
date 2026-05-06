import re
from typing import Iterable

from algorithm.types import AlgorithmSpec

FUNCTION_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def default_function_rows_from_spec(spec: AlgorithmSpec) -> list[dict[str, object]]:
    if not spec.default_function_rows:
        return []
    rows: list[dict[str, object]] = []
    for row in spec.default_function_rows:
        rows.append(
            {
                "id": str(row.get("id", "")),
                "name": str(row.get("name", "")),
                "function_key": str(row.get("function_key", "")),
                "function_params": dict(row.get("function_params", {})),
                "function_param_vary": dict(row.get("function_param_vary", {})),
            }
        )
    return rows


def validate_function_rows(
    rows: list[dict[str, object]],
    *,
    reserved_names: Iterable[str],
    valid_function_keys: Iterable[str],
) -> list[str]:
    reserved = set(reserved_names)
    valid_function_key_set = set(valid_function_keys)
    errors: list[str] = []
    seen: set[str] = set()
    for idx, row in enumerate(rows, start=1):
        name = str(row.get("name") or "").strip()
        function_key = str(row.get("function_key") or "").strip()
        if not name:
            errors.append(f"Function row {idx}: name is required.")
        elif not FUNCTION_NAME_PATTERN.fullmatch(name):
            errors.append(f"Function row {idx}: invalid name '{name}'. Use letters, numbers, and '_' only.")
        elif name in reserved:
            errors.append(f"Function row {idx}: '{name}' is reserved.")
        elif name in seen:
            errors.append(f"Function row {idx}: duplicate name '{name}'.")
        else:
            seen.add(name)
        if not function_key:
            errors.append(f"Function row {idx}: function type is required.")
        elif function_key not in valid_function_key_set:
            errors.append(f"Function row {idx}: unknown function type '{function_key}'.")
    return errors
