from dataclasses import dataclass, field
from typing import Callable, Dict, List

from PEPit import PEP
from PEPit.function import Function


class AlgorithmEvaluationError(RuntimeError):
    """Raised when a solver-backed function cannot return tau."""


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
class FunctionParamSpec:
    name: str
    param_type: str
    description: str
    default: object | None = None
    required: bool = False


@dataclass
class FunctionSpec:
    key: str
    cls: Function
    parameters: List[FunctionParamSpec] = field(default_factory=list)


@dataclass
class FunctionSlot:
    key: str


@dataclass
class AlgorithmSpec:
    name: str
    algo: Callable[[PEP, Dict[str, object], Dict[str, float]], dict]
    function_slots: List[FunctionSlot]
    default_function_keys: Dict[str, str]
    default_hyperparameters: List[HyperparameterSpec] = field(default_factory=list)


def default_gamma_n_hyperparameters() -> List[HyperparameterSpec]:
    return [
        HyperparameterSpec(
            name="gamma",
            label="gamma",
            min_value=0.0,
            max_value=2.0,
            default=0.0,
            step=0.1,
            value_type="float",
        ),
        HyperparameterSpec(
            name="n",
            label="n (iterations)",
            min_value=1,
            max_value=3,
            default=1,
            step=1,
            value_type="int",
        ),
    ]
