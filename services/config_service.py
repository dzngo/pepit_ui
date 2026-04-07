from collections.abc import MutableMapping

from algorithm.algorithm_custom import (
    get_base_algorithm_name,
    register_custom_algorithm,
)
from algorithm.types import HyperparameterSpec
from core.config import hyperparameter_rows_from_specs


def collect_config_errors(
    *,
    hyperparameter_errors: list[str],
    runtime_param_errors: list[str],
    function_param_errors: list[str],
    function_row_errors: list[str],
) -> list[str]:
    errors = list(hyperparameter_errors)
    errors.extend(runtime_param_errors)
    errors.extend(function_param_errors)
    errors.extend(function_row_errors)
    return errors


def build_steps_test_context(
    *,
    function_config: dict[str, dict[str, object]],
    function_param_errors: list[str],
    function_row_errors: list[str],
    runtime_param_errors: list[str],
    hyperparameter_specs: list[HyperparameterSpec],
    function_rows: list[dict[str, object]] | None = None,
) -> dict:
    context = {
        "function_config": function_config,
        "function_param_errors": list(function_param_errors),
        "function_row_errors": list(function_row_errors),
        "runtime_param_errors": list(runtime_param_errors),
        "hyperparameter_specs": list(hyperparameter_specs),
    }
    if function_rows is not None:
        context["function_rows"] = [
            {
                "id": str(row.get("id", "")),
                "name": str(row.get("name", "")),
                "function_key": str(row.get("function_key", "")),
                "function_params": dict(row.get("function_params", {})),
            }
            for row in function_rows
        ]
    return context


def build_pending_settings(
    *,
    algo_key: str,
    hyperparameter_specs: list[HyperparameterSpec],
    function_config: dict[str, dict[str, object]],
    rerun_nan_caches: bool,
) -> dict:
    return {
        "algo_key": algo_key,
        "hyperparameter_specs": list(hyperparameter_specs),
        "function_config": function_config,
        "rerun_nan_caches": bool(rerun_nan_caches),
    }


def register_custom_algorithm_bundle(
    *,
    name: str,
    steps_code: str,
    source_algorithm_name: str,
    default_hyperparameters: list[HyperparameterSpec] | None,
    default_function_rows: list[dict] | None,
):
    return register_custom_algorithm(
        name=name,
        steps_code=steps_code,
        base_algo=get_base_algorithm_name(source_algorithm_name),
        default_hyperparameters=default_hyperparameters,
        default_function_rows=default_function_rows,
    )


def sync_custom_algorithm_defaults_in_state(
    session_state: MutableMapping[str, object],
    *,
    source_algo_key: str,
    target_algo_name: str,
    hyperparameter_specs: list[HyperparameterSpec] | None,
) -> None:
    hyperparameter_store = session_state.get("hyperparameter_store", {})
    if isinstance(hyperparameter_store, dict):
        copied_rows: list[dict] = []
        if hyperparameter_specs:
            copied_rows = hyperparameter_rows_from_specs(list(hyperparameter_specs))
        if not copied_rows:
            current_rows = hyperparameter_store.get(source_algo_key, [])
            copied_rows = [dict(row) for row in current_rows]
        hyperparameter_store[target_algo_name] = copied_rows

    function_rows_store = session_state.get("function_rows_store", {})
    if isinstance(function_rows_store, dict):
        source_rows = function_rows_store.get(source_algo_key, [])
        copied_function_rows: list[dict] = []
        for row in source_rows:
            copied_row = dict(row)
            copied_row["function_params"] = dict(row.get("function_params", {}))
            copied_function_rows.append(copied_row)
        function_rows_store[target_algo_name] = copied_function_rows

    function_row_counter_store = session_state.get("function_row_counter_store", {})
    if isinstance(function_row_counter_store, dict):
        function_row_counter_store[target_algo_name] = int(function_row_counter_store.get(source_algo_key, 0))
