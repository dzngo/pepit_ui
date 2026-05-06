import random
import re
from math import isfinite

import numpy as np
import sympy as sp


def float_text_default(param_default: object | None) -> str:
    if isinstance(param_default, (int, float)):
        value = float(param_default)
        if isfinite(value):
            return str(value)
        return "inf"
    return ""


def parse_float_input(raw: str) -> tuple[float | None, str | None]:
    text = raw.strip().lower()
    if not text:
        return None, None
    if text in {"inf", "infinity", "+inf", "+infinity", "np.inf"}:
        return float("inf"), None
    try:
        return float(text), None
    except ValueError:
        return None, f"Invalid float value: {raw!r}"


def parse_float_list(raw: str) -> tuple[list[float], str | None]:
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


_NON_IDENTIFIER = re.compile(r"[^A-Za-z0-9_]")


def _pattern_param_key(alias: str, param_name: str) -> str:
    safe_alias = _NON_IDENTIFIER.sub("_", alias.strip()) or "f"
    safe_param = _NON_IDENTIFIER.sub("_", param_name.strip()) or "p"
    return f"{safe_alias}_{safe_param}"


def build_pattern_param_values(function_config: dict) -> tuple[dict[str, float], list[str], list[str]]:
    param_values: dict[str, float] = {}
    invalid: list[str] = []
    conflicts: list[str] = []
    for slot_key, slot_config in sorted(function_config.items()):
        alias = str(slot_config.get("alias", slot_key) or slot_key)
        for name, raw in (slot_config.get("function_params") or {}).items():
            pattern_name = _pattern_param_key(alias, str(name))
            if pattern_name in ("x", "pi", "e"):
                conflicts.append(pattern_name)
                continue
            value = None
            if isinstance(raw, (int, float, np.integer, np.floating)):
                value = float(raw)
            elif isinstance(raw, str) and raw.strip():
                try:
                    value = float(raw)
                except ValueError:
                    invalid.append(pattern_name)
            if value is None:
                if pattern_name not in invalid:
                    invalid.append(pattern_name)
                continue
            if pattern_name in param_values and not np.isclose(
                param_values[pattern_name], value, rtol=1e-6, atol=1e-12
            ):
                conflicts.append(pattern_name)
                continue
            param_values[pattern_name] = value
    return param_values, sorted(set(invalid)), sorted(set(conflicts))


_X_SYMBOL = sp.symbols("x")


def evaluate_pattern_expression(
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


_PATTERN_EXAMPLES = (
    "1/log(x)",
    "sin(x)",
    "2*x^2 + 3",
    "log(x)",
    "sqrt(x)",
    "exp(-x)",
)


def random_pattern_example() -> str:
    return f"example: {random.choice(_PATTERN_EXAMPLES)}"
