import re
from typing import Dict, Tuple

import numpy as np
from PEPit import PEP, Point

from .algorithm_custom import ALGORITHMS
from .function_registry import FUNCTIONS
from .types import AlgorithmEvaluationError, AlgorithmSpec


def _dual_key_label(key: str) -> str:
    return " | ".join(part.strip() for part in key.split(","))


def _extract_duals(problem: PEP) -> Dict[str, Dict[str, float]]:
    duals: Dict[str, Dict[str, float]] = {}
    pattern = r"^.*?_\d+_(?P<constraint>.+?)\((?P<xi>[^()]*)\)$"
    for constraint in problem._list_of_prepared_constraints:
        m = re.search(pattern, constraint.name)
        if not m:
            continue
        constraint_name = m.group("constraint")
        xi_id = m.group("xi")
        dual_value = float(constraint.eval_dual())
        duals.setdefault(constraint_name, {})[_dual_key_label(xi_id)] = dual_value
    return duals


def _constraint_series_id(constraint_name: str) -> str | None:
    pattern = r"^.*?_\d+_(?P<constraint>.+?)\((?P<xi>[^()]*)\)$"
    match = re.search(pattern, constraint_name)
    if not match:
        return None
    constraint = match.group("constraint")
    xi_id = _dual_key_label(match.group("xi"))
    return f"{constraint}||{xi_id}"


def run_algorithm(
    *,
    algo_spec: AlgorithmSpec,
    function_config: Dict[str, Dict[str, float]],
    algo_params: Dict[str, float],
    wrapper: str = "cvxpy",
    solver: str | None = None,
    active_dual_series_ids: set[str] | None = None,
) -> Tuple[float, Dict[str, Dict[str, float]]]:
    problem = PEP()
    funcs: Dict[str, object] = {}
    for slot_key, config in function_config.items():
        function_key = config["function_key"]
        function_params = config["function_params"]
        function_spec = FUNCTIONS[function_key]
        resolved_params: Dict[str, object] = {}
        for param in function_spec.parameters:
            if param.name in function_params:
                raw_value = function_params[param.name]
            elif param.default is not None:
                raw_value = param.default
            else:
                continue
            if param.param_type == "BlockPartition":
                if raw_value is None:
                    continue
                resolved_params[param.name] = problem.declare_block_partition(d=int(raw_value))
            elif param.param_type == "Point":
                resolved_params[param.name] = Point() if raw_value else None
            elif param.param_type == "list":
                if raw_value is None:
                    continue
                resolved_params[param.name] = list(raw_value)
            elif param.param_type == "float":
                if raw_value is None:
                    continue
                resolved_params[param.name] = float(raw_value)
            else:
                resolved_params[param.name] = raw_value
        func = problem.declare_function(function_spec.cls, **resolved_params)
        funcs[slot_key] = func

    algo_spec.algo(problem, funcs, algo_params)

    problem._prepare_constraints(verbose=0)
    if active_dual_series_ids:
        for constraint in problem._list_of_prepared_constraints:
            series_id = _constraint_series_id(constraint.name)
            if series_id is None:
                continue
            constraint.activated = series_id in active_dual_series_ids

    tau = problem.solve(wrapper=wrapper, solver=solver, verbose=0)
    if tau is None:
        raise AlgorithmEvaluationError(
            f"Solver failed to find a feasible tau for {algo_spec.name} with these hyperparameters."
        )
    duals: Dict[str, Dict[str, float]] = {}
    duals.update(_extract_duals(problem))

    return float(tau), duals


def compute_point_process(
    algo_key: str,
    function_config: Dict[str, Dict[str, object]],
    algo_params: Dict[str, object],
    active_dual_series_ids: tuple[str, ...],
) -> tuple[float, dict, str | None, bool]:
    spec = ALGORITHMS[algo_key]
    try:
        raw = run_algorithm(
            algo_spec=spec,
            function_config=function_config,
            algo_params=algo_params,
            active_dual_series_ids=set(active_dual_series_ids),
        )
        if isinstance(raw, tuple) and len(raw) == 2:
            tau_raw, duals = raw
        else:
            tau_raw, duals = raw, {}
        tau_value = float(np.asarray(tau_raw).reshape(-1)[0])
        return tau_value, duals or {}, None, True
    except AlgorithmEvaluationError as exc:
        message = f"{spec.name}: {exc}"
        return np.nan, {}, message, True
    except Exception as exc:
        message = f"{spec.name}: unexpected error - {exc}"
        # Preserve behavior: unexpected errors are not cached.
        return np.nan, {}, message, False
