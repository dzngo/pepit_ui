from typing import Iterable

import pandas as pd

from algorithm.types import HyperparameterSpec

HYPERPARAM_COLUMNS = ("name", "label", "value_type", "min", "max", "step", "default")


def hyperparameter_rows_from_specs(specs: list[HyperparameterSpec]) -> list[dict]:
    return [
        {
            "name": hp.name,
            "label": hp.label,
            "value_type": hp.value_type,
            "min": float(hp.min_value),
            "max": float(hp.max_value),
            "step": float(hp.step),
            "default": float(hp.default),
        }
        for hp in specs
    ]


def parse_hyperparameter_specs(
    rows: list[dict],
    *,
    reserved_names: Iterable[str],
    allow_equal_bounds: bool = False,
) -> tuple[list[HyperparameterSpec], list[str]]:
    reserved = set(reserved_names)

    def _is_blank(value: object) -> bool:
        if value is None:
            return True
        if isinstance(value, str) and not value.strip():
            return True
        try:
            return bool(pd.isna(value))
        except Exception:
            return False

    errors: list[str] = []
    specs: list[HyperparameterSpec] = []
    seen_names: set[str] = set()

    for row_idx, row in enumerate(rows, start=1):
        raw_name = str(row.get("name") or "").strip()
        raw_label = str(row.get("label") or "").strip()
        raw_type = str(row.get("value_type") or "float").strip().lower()
        row_is_empty = all(_is_blank(row.get(col)) for col in ("name", "label", "min", "max", "step", "default"))
        if row_is_empty:
            continue
        if not raw_name:
            errors.append(f"Hyperparameter row {row_idx}: name is required.")
            continue
        if raw_name in seen_names:
            errors.append(f"Hyperparameter row {row_idx}: duplicate name '{raw_name}'.")
            continue
        seen_names.add(raw_name)
        if raw_name in reserved:
            errors.append(f"Hyperparameter row {row_idx}: '{raw_name}' is reserved.")
            continue
        if raw_type not in {"float", "int"}:
            errors.append(f"Hyperparameter row {row_idx}: type must be 'float' or 'int'.")
            continue

        numeric: dict[str, float] = {}
        numeric_ok = True
        for key in ("min", "max", "step", "default"):
            try:
                numeric[key] = float(row.get(key))
            except (TypeError, ValueError):
                errors.append(f"Hyperparameter row {row_idx}: invalid numeric value for '{key}'.")
                numeric_ok = False
                break
        if not numeric_ok:
            continue
        if numeric["max"] < numeric["min"]:
            errors.append(f"Hyperparameter row {row_idx}: max must be greater than or equal to min.")
            continue
        if numeric["max"] == numeric["min"] and not allow_equal_bounds:
            errors.append(f"Hyperparameter row {row_idx}: max must be greater than min.")
            continue
        if numeric["step"] <= 0:
            errors.append(f"Hyperparameter row {row_idx}: step must be positive.")
            continue
        if not (numeric["min"] <= numeric["default"] <= numeric["max"]):
            errors.append(f"Hyperparameter row {row_idx}: default must be between min and max.")
            continue
        if raw_type == "int":
            int_fields = ("min", "max", "step", "default")
            if any(abs(numeric[field] - round(numeric[field])) > 1e-9 for field in int_fields):
                errors.append(f"Hyperparameter row {row_idx}: int type requires integer min/max/step/default.")
                continue

        label = raw_label or raw_name
        specs.append(
            HyperparameterSpec(
                name=raw_name,
                label=label,
                min_value=float(int(numeric["min"])) if raw_type == "int" else float(numeric["min"]),
                max_value=float(int(numeric["max"])) if raw_type == "int" else float(numeric["max"]),
                default=float(int(numeric["default"])) if raw_type == "int" else float(numeric["default"]),
                step=float(int(numeric["step"])) if raw_type == "int" else float(numeric["step"]),
                value_type=raw_type,
            )
        )

    if not specs:
        errors.append("At least one hyperparameter is required.")
    return specs, errors
