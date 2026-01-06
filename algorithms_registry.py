# functions_registry.py
from dataclasses import dataclass
from math import sqrt
from typing import Callable, Dict, List, Tuple

from PEPit import PEP
from PEPit.functions import SmoothConvexFunction, ConvexLipschitzFunction, SmoothStronglyConvexFunction, ConvexFunction
from PEPit.primitive_steps import proximal_step

ArrayAlgo = Callable[..., Tuple[float, Dict[str, Dict[str, float]]]]


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
) -> Tuple[float, Dict[str, Dict[str, float]]]:
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
    duals = _extract_duals(func)
    return float(tau), duals


def subgradient_method(M, n, gamma, wrapper="cvxpy", solver=None):
    # Instantiate PEP
    problem = PEP()

    # Declare a convex lipschitz function
    func = problem.declare_function(ConvexLipschitzFunction, M=M)

    # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
    xs = func.stationary_point()
    fs = func(xs)

    # Then define the starting point x0 of the algorithm
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the distance between x0 and xs
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    # Run n steps of the subgradient method
    x = x0
    gx, fx = func.oracle(x)

    for _ in range(int(n)):
        problem.set_performance_metric(fx - fs)
        x = x - gamma * gx
        gx, fx = func.oracle(x)

    # Set the performance metric to the function value accuracy
    problem.set_performance_metric(fx - fs)

    # Solve the PEP
    tau = problem.solve(wrapper=wrapper, solver=solver, verbose=0)
    if tau is None:
        raise AlgorithmEvaluationError(
            "Solver failed to find a feasible tau for gradient_descent with these hyperparameters."
        )
    duals = _extract_duals(func)
    return float(tau), duals


def proximal_gradient(L, mu, gamma, n, wrapper="cvxpy", solver=None):
    # Instantiate PEP
    problem = PEP()

    # Declare a strongly convex smooth function and a closed convex proper function
    f1 = problem.declare_function(SmoothStronglyConvexFunction, mu=mu, L=L)
    f2 = problem.declare_function(ConvexFunction)
    func = f1 + f2

    # Start by defining its unique optimal point xs = x_*
    xs = func.stationary_point()

    # Then define the starting point x0 of the algorithm
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the distance between x0 and x^*
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    # Run the proximal gradient method starting from x0
    x = x0
    for _ in range(int(n)):
        y = x - gamma * f1.gradient(x)
        x, _, _ = proximal_step(y, f2, gamma)

    # Set the performance metric to the distance between x and xs
    problem.set_performance_metric((x - xs) ** 2)

    # Solve the PEP
    tau = problem.solve(wrapper=wrapper, solver=solver, verbose=0)
    if tau is None:
        raise AlgorithmEvaluationError(
            "Solver failed to find a feasible tau for gradient_descent with these hyperparameters."
        )
    duals = _extract_duals(f1)
    duals.update(_extract_duals(f2))
    return float(tau), duals


def accelerated_proximal_point(gamma, n, mu, L, wrapper="cvxpy", solver=None):
    # Instantiate PEP
    problem = PEP()

    # Declare a strongly convex smooth function
    func = problem.declare_function(SmoothStronglyConvexFunction, mu=mu, L=L)

    # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
    xs = func.stationary_point()
    fs = func(xs)

    # Then define the starting point x0 of the algorithm
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the distance between x0 and x^*
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    # Run n steps of the fast gradient method
    x = x0
    y = x0
    lam = 1

    steps = int(n)
    for _ in range(steps):
        lam_old = lam
        lam = (1 + sqrt(4 * lam_old**2 + 1)) / 2
        x_old = x
        x = y - gamma * func.gradient(y)
        y = x + (lam_old - 1) / lam * (x - x_old)

    # Set the performance metric to the function value accuracy
    problem.set_performance_metric(func(x) - fs)

    # Solve the PEP
    tau = problem.solve(wrapper=wrapper, solver=solver, verbose=0)
    if tau is None:
        raise AlgorithmEvaluationError(
            "Solver failed to find a feasible tau for gradient_descent with these hyperparameters."
        )
    duals = _extract_duals(func)
    return float(tau), duals


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
    "subgradient_method": AlgorithmSpec(
        name="subgradient_method",
        description="x = x - gamma * func.oracle(x)",
        algo=subgradient_method,
        hyperparameters=[
            HyperparameterSpec(
                name="M",
                label="M (Lipschitz)",
                min_value=1,
                max_value=5,
                default=2,
                step=1,
                value_type="int",
            ),
        ],
    ),
    "proximal_gradient": AlgorithmSpec(
        name="proximal_gradient",
        description="y = x - gamma * f1.gradient(x), x = proximal_step(y, f2, gamma)",
        algo=proximal_gradient,
        hyperparameters=[
            HyperparameterSpec(
                name="mu",
                label="mu (strong convexity)",
                min_value=0.0,
                max_value=1.0,
                default=0.1,
                step=0.1,
            ),
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
    "accelerated_proximal_point": AlgorithmSpec(
        name="accelerated_proximal_point",
        description="lam = (1 + sqrt(4 * lam_old**2 + 1)) / 2, "
        "x = y - gamma * func.gradient(y), "
        "y = x + ((lam_old - 1)/lam) * (x - x_old)",
        algo=accelerated_proximal_point,
        hyperparameters=[
            HyperparameterSpec(
                name="mu",
                label="mu (strong convexity)",
                min_value=0.0,
                max_value=1.0,
                default=0.1,
                step=0.1,
            ),
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
