import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

from algorithm.algorithm_compiler import compile_algorithm_body
from algorithm.algorithm_templates import BASE_ALGORITHM_BODIES, BASE_ALGORITHMS
from algorithm.types import AlgorithmSpec, HyperparameterSpec

CUSTOM_ALGORITHMS_PATH = Path(__file__).resolve().parent.parent / "custom_algorithms.json"


def _load_custom_algorithms() -> Dict[str, dict]:
    if not CUSTOM_ALGORITHMS_PATH.exists():
        return {}
    try:
        data = json.loads(CUSTOM_ALGORITHMS_PATH.read_text())
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def _save_custom_algorithms(payload: Dict[str, dict]) -> None:
    CUSTOM_ALGORITHMS_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _custom_spec_from_payload(name: str, payload: dict) -> AlgorithmSpec | None:
    base_name = payload.get("base_algo")
    steps_code = payload.get("steps_code")
    if not isinstance(base_name, str) or not isinstance(steps_code, str):
        return None
    base_spec = BASE_ALGORITHMS.get(base_name)
    if base_spec is None:
        return None
    steps = compile_algorithm_body(steps_code)
    return AlgorithmSpec(
        name=name,
        algo=steps,
        function_slots=list(base_spec.function_slots),
        default_function_keys=dict(base_spec.default_function_keys),
        default_hyperparameters=[
            HyperparameterSpec(
                name=hp.name,
                label=hp.label,
                min_value=hp.min_value,
                max_value=hp.max_value,
                default=hp.default,
                step=hp.step,
                value_type=hp.value_type,
            )
            for hp in base_spec.default_hyperparameters
        ],
    )


CUSTOM_ALGORITHMS: Dict[str, dict] = _load_custom_algorithms()
CUSTOM_SPECS: Dict[str, AlgorithmSpec] = {}
for algo_name, payload in CUSTOM_ALGORITHMS.items():
    try:
        custom_spec = _custom_spec_from_payload(algo_name, payload)
    except Exception:
        custom_spec = None
    if custom_spec is not None:
        CUSTOM_SPECS[algo_name] = custom_spec

ALGORITHMS: Dict[str, AlgorithmSpec] = {
    **BASE_ALGORITHMS,
    **CUSTOM_SPECS,
}


def get_algorithm_steps_code(name: str) -> str:
    payload = CUSTOM_ALGORITHMS.get(name)
    if payload and isinstance(payload, dict):
        steps_code = payload.get("steps_code")
        if isinstance(steps_code, str):
            return steps_code
    return BASE_ALGORITHM_BODIES.get(name, "")


def get_base_algorithm_name(name: str) -> str:
    payload = CUSTOM_ALGORITHMS.get(name)
    if payload and isinstance(payload, dict):
        base_name = payload.get("base_algo")
        if isinstance(base_name, str):
            return base_name
    return name


def register_custom_algorithm(
    *,
    name: str,
    steps_code: str,
    base_algo: str,
) -> AlgorithmSpec:
    if name in ALGORITHMS:
        raise ValueError(f"Algorithm name '{name}' already exists.")
    base_spec = BASE_ALGORITHMS.get(base_algo)
    if base_spec is None:
        raise ValueError(f"Base algorithm '{base_algo}' not found.")
    steps = compile_algorithm_body(steps_code)
    spec = AlgorithmSpec(
        name=name,
        algo=steps,
        function_slots=list(base_spec.function_slots),
        default_function_keys=dict(base_spec.default_function_keys),
        default_hyperparameters=[
            HyperparameterSpec(
                name=hp.name,
                label=hp.label,
                min_value=hp.min_value,
                max_value=hp.max_value,
                default=hp.default,
                step=hp.step,
                value_type=hp.value_type,
            )
            for hp in base_spec.default_hyperparameters
        ],
    )
    CUSTOM_ALGORITHMS[name] = {
        "steps_code": steps_code,
        "base_algo": base_algo,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _save_custom_algorithms(CUSTOM_ALGORITHMS)
    ALGORITHMS[name] = spec
    return spec


def remove_custom_algorithm(name: str) -> None:
    if name in BASE_ALGORITHMS:
        raise ValueError(f"Cannot remove base algorithm '{name}'.")
    if name not in CUSTOM_ALGORITHMS:
        raise ValueError(f"Custom algorithm '{name}' not found.")
    CUSTOM_ALGORITHMS.pop(name, None)
    CUSTOM_SPECS.pop(name, None)
    ALGORITHMS.pop(name, None)
    _save_custom_algorithms(CUSTOM_ALGORITHMS)
