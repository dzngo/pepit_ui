# functions_registry.py
from dataclasses import dataclass, field
import inspect
from math import isfinite
from math import sqrt
from typing import Callable, Dict, List, Tuple

from PEPit import PEP, Point
from PEPit.function import Function
import PEPit.functions as functions

from PEPit.primitive_steps import proximal_step


class AlgorithmEvaluationError(RuntimeError):
    """Raised when a solver-backed function cannot return tau."""


def _dual_key_label(key) -> str:
    if isinstance(key, tuple):
        return " | ".join(str(part) for part in key)
    return str(key)


def _extract_duals(func) -> Dict[str, Dict[str, float]]:
    duals: Dict[str, Dict[str, float]] = {}
    for name, df in func.get_class_constraints_duals().items():
        try:
            stacked = df.stack().to_dict()
        except Exception:
            stacked = {}
        duals[name] = {_dual_key_label(k): float(v) for k, v in stacked.items()}
    return duals


@dataclass
class HyperparameterSpec:
    name: str
    label: str
    min_value: float
    max_value: float
    default: float
    step: float
    value_type: str = "float"


@dataclass
class FunctionSpec:
    key: str
    cls: Function
    parameters: List["FunctionParamSpec"] = field(default_factory=list)


@dataclass
class FunctionParamSpec:
    name: str
    param_type: str
    description: str
    default: object | None = None
    required: bool = False


@dataclass
class FunctionSlot:
    key: str


@dataclass
class InitialConditionSpec:
    key: str
    label: str
    apply: Callable[[PEP, dict], None]
    hyperparameters: List[HyperparameterSpec] = field(default_factory=list)


@dataclass
class PerformanceMetricSpec:
    key: str
    label: str
    apply: Callable[[PEP, dict], None]
    hyperparameters: List[HyperparameterSpec] = field(default_factory=list)


@dataclass
class AlgorithmSpec:
    name: str
    steps: Callable[[PEP, Dict[str, object], Dict[str, float]], dict]
    function_slots: List[FunctionSlot]
    default_function_keys: Dict[str, str]
    default_initial_condition: str
    default_performance_metric: str


DEFAULT_HYPERPARAMETERS: List[HyperparameterSpec] = [
    HyperparameterSpec(
        name="gamma",
        label="gamma",
        min_value=0.0,
        max_value=5.0,
        default=2.0,
        step=0.1,
    ),
    HyperparameterSpec(
        name="n",
        label="n (iterations)",
        min_value=1,
        max_value=10,
        default=5,
        step=1,
        value_type="int",
    ),
]


def get_required_init_args(cls) -> List[str]:
    sig = inspect.signature(cls.__init__)
    return [
        name
        for name, param in sig.parameters.items()
        if name != "self"
        and param.default is inspect.Parameter.empty
        and param.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    ]


def list_classes_from_all(module) -> Dict[str, type]:
    return {
        name: getattr(module, name)
        for name in module.__all__
        if hasattr(module, name) and isinstance(getattr(module, name), type)
    }


def create_instance(class_name: str, *args, **kwargs):
    cls = getattr(functions, class_name)
    return cls(*args, **kwargs)


EXCLUDED_INIT_PARAMS = {"is_leaf", "decomposition_dict", "reuse_gradient", "name"}


def _parse_doc_section(doc: str, section: str) -> Dict[str, Dict[str, str]]:
    if not doc:
        return {}
    lines = doc.splitlines()
    entries: Dict[str, Dict[str, str]] = {}
    in_section = False
    current: str | None = None
    for line in lines:
        header = line.strip().lower()
        if header == f"{section.lower()}:":
            in_section = True
            current = None
            continue
        if in_section and header.endswith(":") and header != f"{section.lower()}:":
            break
        if not in_section or not line.strip():
            continue
        if line.lstrip() == line:
            if current:
                break
        if "(" in line and "):" in line:
            head, desc = line.split("):", 1)
            name, type_part = head.split("(", 1)
            param_name = name.strip()
            entries[param_name] = {
                "type": type_part.strip(),
                "desc": desc.strip(),
            }
            current = param_name
        elif current and (line.startswith(" " * 4) or line.startswith(" " * 8)):
            entries[current]["desc"] += " " + line.strip()
    return entries


def _normalize_param_type(type_str: str, default: object | None) -> str:
    if type_str:
        lowered = type_str.lower()
        if "blockpartition" in lowered:
            return "BlockPartition"
        if "point" in lowered:
            return "Point"
        if "list" in lowered:
            return "list"
        if "float" in lowered:
            return "float"
    if isinstance(default, (int, float)):
        return "float"
    if isinstance(default, list):
        return "list"
    return "unknown"


def _param_default(param: inspect.Parameter) -> object | None:
    if param.default is inspect.Parameter.empty:
        return None
    return param.default


def _build_param_specs(cls: type) -> List[FunctionParamSpec]:
    init_doc = inspect.getdoc(cls.__init__) or ""
    class_doc = inspect.getdoc(cls) or ""
    args_doc = _parse_doc_section(init_doc, "Args")
    attrs_doc = _parse_doc_section(class_doc, "Attributes")

    specs: List[FunctionParamSpec] = []
    signature = inspect.signature(cls.__init__)
    for name, param in signature.parameters.items():
        if name == "self" or name in EXCLUDED_INIT_PARAMS:
            continue
        doc_entry = attrs_doc.get(name) or args_doc.get(name) or {}
        type_str = doc_entry.get("type", "")
        desc = doc_entry.get("desc", "")
        default = _param_default(param)
        param_type = _normalize_param_type(type_str, default)
        required = param.default is inspect.Parameter.empty
        if param_type == "float" and isinstance(default, (int, float)) and not isfinite(float(default)):
            desc = (desc + ". Default is infinity.").strip()
        specs.append(
            FunctionParamSpec(
                name=name,
                param_type=param_type,
                description=desc,
                default=default,
                required=required,
            )
        )
    return specs


def build_function_spec(key: str, cls: Function) -> FunctionSpec:
    specs = _build_param_specs(cls)
    return FunctionSpec(key=key, cls=cls, parameters=specs)


FUNCTIONS: Dict[str, FunctionSpec] = {
    name: build_function_spec(name, cls) for name, cls in list_classes_from_all(functions).items()
}


def gradient_descent_steps(problem: PEP, funcs: Dict[str, Function], params: Dict[str, float]) -> dict:
    func = funcs["f"]
    xs = func.stationary_point()
    fs = func(xs)
    x0 = problem.set_initial_point()
    x = x0
    steps = int(params["n"])
    gamma = float(params["gamma"])
    for _ in range(steps):
        x = x - gamma * func.gradient(x)
    return {"x0": x0, "x": x, "xs": xs, "fs": fs, "funcs": funcs, "func": func}


def subgradient_method_steps(problem: PEP, funcs: Dict[str, Function], params: Dict[str, float]) -> dict:
    func = funcs["f"]
    xs = func.stationary_point()
    fs = func(xs)
    x0 = problem.set_initial_point()
    x = x0
    gx, fx = func.oracle(x)
    steps = int(params["n"])
    gamma = float(params["gamma"])
    for _ in range(steps):
        problem.set_performance_metric(fx - fs)
        x = x - gamma * gx
        gx, fx = func.oracle(x)
    return {"x0": x0, "x": x, "xs": xs, "fs": fs, "fx": fx, "funcs": funcs, "func": func}


def proximal_gradient_steps(problem: PEP, funcs: Dict[str, Function], params: Dict[str, float]) -> dict:
    f1 = funcs["f1"]
    f2 = funcs["f2"]
    func = f1 + f2
    xs = func.stationary_point()
    x0 = problem.set_initial_point()
    x = x0
    steps = int(params["n"])
    gamma = float(params["gamma"])
    for _ in range(steps):
        y = x - gamma * f1.gradient(x)
        x, _, _ = proximal_step(y, f2, gamma)
    return {"x0": x0, "x": x, "xs": xs, "funcs": funcs, "func": func, "f1": f1, "f2": f2}


def accelerated_proximal_point_steps(problem: PEP, funcs: Dict[str, Function], params: Dict[str, float]) -> dict:
    func = funcs["f"]
    xs = func.stationary_point()
    fs = func(xs)
    x0 = problem.set_initial_point()
    x = x0
    y = x0
    lam = 1
    steps = int(params["n"])
    gamma = float(params["gamma"])
    for _ in range(steps):
        lam_old = lam
        lam = (1 + sqrt(4 * lam_old**2 + 1)) / 2
        x_old = x
        x = y - gamma * func.gradient(y)
        y = x + (lam_old - 1) / lam * (x - x_old)
    return {"x0": x0, "x": x, "xs": xs, "fs": fs, "funcs": funcs, "func": func}


def ic_unit_distance(problem: PEP, state: dict) -> None:
    problem.set_initial_condition((state["x0"] - state["xs"]) ** 2 <= 1)


def pm_function_gap(problem: PEP, state: dict) -> None:
    problem.set_performance_metric(state["func"](state["x"]) - state["fs"])


def pm_subgradient_gap(problem: PEP, state: dict) -> None:
    problem.set_performance_metric(state["fx"] - state["fs"])


def pm_distance_to_opt(problem: PEP, state: dict) -> None:
    problem.set_performance_metric((state["x"] - state["xs"]) ** 2)


INITIAL_CONDITIONS: Dict[str, InitialConditionSpec] = {
    "unit_distance_to_opt": InitialConditionSpec(
        key="unit_distance_to_opt",
        label="||x0 - xs||^2 <= 1",
        apply=ic_unit_distance,
    ),
}


PERFORMANCE_METRICS: Dict[str, PerformanceMetricSpec] = {
    "function_gap": PerformanceMetricSpec(
        key="function_gap",
        label="func(x) - f*",
        apply=pm_function_gap,
    ),
    "subgradient_gap": PerformanceMetricSpec(
        key="subgradient_gap",
        label="fx - f*",
        apply=pm_subgradient_gap,
    ),
    "distance_to_opt": PerformanceMetricSpec(
        key="distance_to_opt",
        label="||x - x*||^2",
        apply=pm_distance_to_opt,
    ),
}


def run_algorithm(
    *,
    algo_spec: AlgorithmSpec,
    function_config: Dict[str, Dict[str, float]],
    initial_condition_key: str,
    performance_metric_key: str,
    algo_params: Dict[str, float],
    wrapper: str = "cvxpy",
    solver: str | None = None,
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

    state = algo_spec.steps(problem, funcs, algo_params)
    initial_spec = INITIAL_CONDITIONS[initial_condition_key]
    perf_spec = PERFORMANCE_METRICS[performance_metric_key]
    initial_spec.apply(problem, state)
    perf_spec.apply(problem, state)

    tau = problem.solve(wrapper=wrapper, solver=solver, verbose=0)
    if tau is None:
        raise AlgorithmEvaluationError(
            f"Solver failed to find a feasible tau for {algo_spec.name} with these hyperparameters."
        )
    duals: Dict[str, Dict[str, float]] = {}
    for func in funcs.values():
        duals.update(_extract_duals(func))
    return float(tau), duals


ALGORITHMS: Dict[str, AlgorithmSpec] = {
    "gradient_descent": AlgorithmSpec(
        name="gradient_descent",
        steps=gradient_descent_steps,
        function_slots=[FunctionSlot(key="f")],
        default_function_keys={"f": "SmoothConvexFunction"},
        default_initial_condition="unit_distance_to_opt",
        default_performance_metric="function_gap",
    ),
    "subgradient_method": AlgorithmSpec(
        name="subgradient_method",
        steps=subgradient_method_steps,
        function_slots=[FunctionSlot(key="f")],
        default_function_keys={"f": "ConvexLipschitzFunction"},
        default_initial_condition="unit_distance_to_opt",
        default_performance_metric="subgradient_gap",
    ),
    "proximal_gradient": AlgorithmSpec(
        name="proximal_gradient",
        steps=proximal_gradient_steps,
        function_slots=[FunctionSlot(key="f1"), FunctionSlot(key="f2")],
        default_function_keys={"f1": "SmoothStronglyConvexFunction", "f2": "ConvexFunction"},
        default_initial_condition="unit_distance_to_opt",
        default_performance_metric="distance_to_opt",
    ),
    "accelerated_proximal_point": AlgorithmSpec(
        name="accelerated_proximal_point",
        steps=accelerated_proximal_point_steps,
        function_slots=[FunctionSlot(key="f")],
        default_function_keys={"f": "SmoothStronglyConvexFunction"},
        default_initial_condition="unit_distance_to_opt",
        default_performance_metric="function_gap",
    ),
}
