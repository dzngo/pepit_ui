from math import sqrt
from typing import Dict

from PEPit import PEP
from PEPit.function import Function
from PEPit.primitive_steps import epsilon_subgradient_step, proximal_step

from algorithm.types import (
    AlgorithmSpec,
    FunctionSlot,
    HyperparameterSpec,
    default_gamma_n_hyperparameters,
)


def gradient_descent(problem: PEP, funcs: Dict[str, Function], params: Dict[str, float]) -> dict:
    func = funcs["f"]
    xs = func.stationary_point()
    xs.set_name("x_*")
    fs = func(xs)
    x0 = problem.set_initial_point()
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)
    x = x0
    x.set_name("x_0")
    steps = int(params["n"])
    gamma = float(params["gamma"])
    for i in range(steps):
        x = x - gamma * func.gradient(x)
        x.set_name(f"x_{i+1}")
    problem.set_performance_metric(func(x) - fs)


def subgradient_method(problem: PEP, funcs: Dict[str, Function], params: Dict[str, float]) -> dict:
    func = funcs["f"]
    xs = func.stationary_point()
    xs.set_name("x_*")
    fs = func(xs)
    x0 = problem.set_initial_point()
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)
    x = x0
    x.set_name("x_0")
    gx, fx = func.oracle(x)
    steps = int(params["n"])
    gamma = float(params["gamma"])
    for i in range(steps):
        problem.set_performance_metric(fx - fs)
        x = x - gamma * gx
        gx, fx = func.oracle(x)
        x.set_name(f"x_{i+1}")
    problem.set_performance_metric(fx - fs)


def proximal_gradient(problem: PEP, funcs: Dict[str, Function], params: Dict[str, float]) -> dict:
    f1 = funcs["f1"]
    f2 = funcs["f2"]
    func = f1 + f2
    xs = func.stationary_point()
    xs.set_name("x_*")
    x0 = problem.set_initial_point()
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)
    x = x0
    x.set_name("x_0")
    steps = int(params["n"])
    gamma = float(params["gamma"])
    for i in range(steps):
        y = x - gamma * f1.gradient(x)
        x, _, _ = proximal_step(y, f2, gamma)
        x.set_name(f"x_{i+1}")
    problem.set_performance_metric((x - xs) ** 2)


def accelerated_proximal_point(problem: PEP, funcs: Dict[str, Function], params: Dict[str, float]) -> dict:
    func = funcs["f"]
    xs = func.stationary_point()
    xs.set_name("x_*")
    fs = func(xs)
    x0 = problem.set_initial_point()
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)
    x = x0
    x.set_name("x_0")
    y = x0
    lam = 1
    steps = int(params["n"])
    gamma = float(params["gamma"])
    for i in range(steps):
        lam_old = lam
        lam = (1 + sqrt(4 * lam_old**2 + 1)) / 2
        x_old = x
        x = y - gamma * func.gradient(y)
        y = x + (lam_old - 1) / lam * (x - x_old)
        x.set_name(f"x_{i+1}")
        y.set_name(f"y_{i+1}")
    problem.set_performance_metric(func(x) - fs)


def epsilon_subgradient(problem: PEP, funcs: Dict[str, Function], params: Dict[str, float]) -> dict:
    func = funcs["f"]
    M = float(params["M"])
    eps = float(params["eps"])
    R = float(params["R"])
    steps = int(params["n"])
    gamma = float(params["gamma"])

    xs = func.stationary_point()
    xs.set_name("x_*")
    fs = func(xs)
    x0 = problem.set_initial_point()
    problem.set_initial_condition((x0 - xs) ** 2 <= R**2)
    x = x0
    x.set_name("x_0")

    for i in range(steps):
        x, gx, fx, epsilon = epsilon_subgradient_step(x, func, gamma)
        x.set_name(f"x_{i+1}")
        problem.set_performance_metric(fx - fs)
        problem.add_constraint(epsilon <= eps)
        problem.add_constraint(gx**2 <= M**2)

    gx, fx = func.oracle(x)
    problem.add_constraint(gx**2 <= M**2)
    problem.set_performance_metric(fx - fs)


BASE_ALGORITHMS: Dict[str, AlgorithmSpec] = {
    "gradient_descent": AlgorithmSpec(
        name="gradient_descent",
        algo=gradient_descent,
        function_slots=[FunctionSlot(key="f")],
        default_function_keys={"f": "SmoothConvexFunction"},
        default_hyperparameters=default_gamma_n_hyperparameters(),
    ),
    "subgradient_method": AlgorithmSpec(
        name="subgradient_method",
        algo=subgradient_method,
        function_slots=[FunctionSlot(key="f")],
        default_function_keys={"f": "ConvexLipschitzFunction"},
        default_hyperparameters=default_gamma_n_hyperparameters(),
    ),
    "proximal_gradient": AlgorithmSpec(
        name="proximal_gradient",
        algo=proximal_gradient,
        function_slots=[FunctionSlot(key="f1"), FunctionSlot(key="f2")],
        default_function_keys={"f1": "SmoothStronglyConvexFunction", "f2": "ConvexFunction"},
        default_hyperparameters=default_gamma_n_hyperparameters(),
    ),
    "accelerated_proximal_point": AlgorithmSpec(
        name="accelerated_proximal_point",
        algo=accelerated_proximal_point,
        function_slots=[FunctionSlot(key="f")],
        default_function_keys={"f": "SmoothStronglyConvexFunction"},
        default_hyperparameters=default_gamma_n_hyperparameters(),
    ),
    "epsilon_subgradient": AlgorithmSpec(
        name="epsilon_subgradient",
        algo=epsilon_subgradient,
        function_slots=[FunctionSlot(key="f")],
        default_function_keys={"f": "ConvexFunction"},
        default_hyperparameters=[
            HyperparameterSpec(
                name="gamma",
                label="gamma",
                min_value=0.0,
                max_value=1.0,
                default=0.4,
                step=0.1,
                value_type="float",
            ),
            HyperparameterSpec(
                name="n",
                label="n (iterations)",
                min_value=5,
                max_value=7,
                default=6,
                step=1,
                value_type="int",
            ),
            HyperparameterSpec(
                name="M",
                label="M",
                min_value=1.5,
                max_value=2.5,
                default=2,
                step=0.5,
                value_type="float",
            ),
            HyperparameterSpec(
                name="eps",
                label="epsilon",
                min_value=0.0,
                max_value=0.5,
                default=0.1,
                step=0.1,
                value_type="float",
            ),
            HyperparameterSpec(
                name="R",
                label="R",
                min_value=0.5,
                max_value=1.5,
                default=1,
                step=0.5,
                value_type="float",
            ),
        ],
    ),
}
