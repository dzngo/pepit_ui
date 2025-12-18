# functions_registry.py
from dataclasses import dataclass
from typing import Callable, Dict, List

from PEPit import PEP
from PEPit.functions import SmoothConvexFunction


ArrayAlgo = Callable[..., float]


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
class AlgorithmSpec:
    name: str
    description: str
    algo: ArrayAlgo
    hyperparameters: List[HyperparameterSpec]


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


def gradient_descent(
    gamma: float,
    n: float,
    L: float,
    wrapper: str = "cvxpy",
    solver: str | None = None,
) -> float:
    problem = PEP()
    func = problem.declare_function(SmoothConvexFunction, L=L)

    # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
    xs = func.stationary_point()
    fs = func(xs)

    # Then define the starting point x0 of the algorithm
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the distance between x0 and x^*
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    # Run n steps of the GD method
    x = x0
    steps = int(n)
    for _ in range(steps):
        x = x - gamma * func.gradient(x)

    # Set the performance metric to the function values accuracy
    problem.set_performance_metric(func(x) - fs)

    # Solve the PEP
    tau = problem.solve(wrapper=wrapper, solver=solver, verbose=0)
    if tau is None:
        raise AlgorithmEvaluationError(
            "Solver failed to find a feasible tau for gradient_descent with these hyperparameters."
        )
    return float(tau)


ALGORITHMS: Dict[str, AlgorithmSpec] = {
    "gradient_descent": AlgorithmSpec(
        name="gradient_descent",
        description="x = x - gamma * func.gradient(x)",
        algo=gradient_descent,
        hyperparameters=[
            HyperparameterSpec(
                name="L",
                label="L (smoothness)",
                min_value=1,
                max_value=5,
                default=1,
                step=1,
                value_type="int",
            ),
        ],
    ),
}
