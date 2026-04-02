from typing import Dict

from algorithm.algorithm_compiler import compile_algorithm_body
from algorithm.types import (
    AlgorithmSpec,
    HyperparameterSpec,
    default_gamma_n_hyperparameters,
)

BASE_ALGORITHM_BODIES: Dict[str, str] = {
    "gradient_descent": """
xs = f.stationary_point()
xs.set_name("x_*")
fs = f(xs)
x0 = problem.set_initial_point()
problem.set_initial_condition((x0 - xs) ** 2 <= 1)
x = x0
x.set_name("x_0")
for i in range(n):
    x = x - gamma * f.gradient(x)
    x.set_name(f"x_{i+1}")
problem.set_performance_metric(f(x) - fs)
""".strip(),
    "subgradient_method": """
xs = f.stationary_point()
xs.set_name("x_*")
fs = f(xs)
x0 = problem.set_initial_point()
problem.set_initial_condition((x0 - xs) ** 2 <= 1)
x = x0
x.set_name("x_0")
gx, fx = f.oracle(x)
for i in range(n):
    problem.set_performance_metric(fx - fs)
    x = x - gamma * gx
    gx, fx = f.oracle(x)
    x.set_name(f"x_{i+1}")
problem.set_performance_metric(fx - fs)
""".strip(),
    "proximal_gradient": """
func = f1 + f2
xs = func.stationary_point()
xs.set_name("x_*")
x0 = problem.set_initial_point()
problem.set_initial_condition((x0 - xs) ** 2 <= 1)
x = x0
x.set_name("x_0")
for i in range(n):
    y = x - gamma * f1.gradient(x)
    x, _, _ = proximal_step(y, f2, gamma)
    x.set_name(f"x_{i+1}")
problem.set_performance_metric((x - xs) ** 2)
""".strip(),
    "accelerated_gradient_convex": """
xs = f.stationary_point()
xs.set_name("x_*")
fs = f(xs)
x0 = problem.set_initial_point()
problem.set_initial_condition((x0 - xs) ** 2 <= 1)
x = x0
x.set_name("x_0")
y = x0
lam = 1
for i in range(n):
    lam_old = lam
    lam = (1 + sqrt(4 * lam_old**2 + 1)) / 2
    x_old = x
    x = y - gamma * f.gradient(y)
    y = x + (lam_old - 1) / lam * (x - x_old)
    x.set_name(f"x_{i+1}")
    y.set_name(f"y_{i+1}")
problem.set_performance_metric(f(x) - fs)
""".strip(),
    "epsilon_subgradient": """
xs = f.stationary_point()
xs.set_name("x_*")
fs = f(xs)
x0 = problem.set_initial_point()
problem.set_initial_condition((x0 - xs) ** 2 <= R**2)
x = x0
x.set_name("x_0")

for i in range(n):
    x, gx, fx, epsilon = epsilon_subgradient_step(x, f, gamma)
    x.set_name(f"x_{i+1}")
    problem.set_performance_metric(fx - fs)
    problem.add_constraint(epsilon <= eps)
    problem.add_constraint(gx**2 <= M**2)

gx, fx = f.oracle(x)
problem.add_constraint(gx**2 <= M**2)
problem.set_performance_metric(fx - fs)
""".strip(),
}


BASE_ALGORITHMS: Dict[str, AlgorithmSpec] = {
    "gradient_descent": AlgorithmSpec(
        name="gradient_descent",
        algo=compile_algorithm_body(BASE_ALGORITHM_BODIES["gradient_descent"]),
        default_hyperparameters=default_gamma_n_hyperparameters(),
        default_function_rows=[
            {"id": "slot-f", "name": "f", "function_key": "SmoothConvexFunction", "function_params": {}}
        ],
    ),
    "subgradient_method": AlgorithmSpec(
        name="subgradient_method",
        algo=compile_algorithm_body(BASE_ALGORITHM_BODIES["subgradient_method"]),
        default_hyperparameters=default_gamma_n_hyperparameters(),
        default_function_rows=[
            {"id": "slot-f", "name": "f", "function_key": "ConvexLipschitzFunction", "function_params": {}}
        ],
    ),
    "proximal_gradient": AlgorithmSpec(
        name="proximal_gradient",
        algo=compile_algorithm_body(BASE_ALGORITHM_BODIES["proximal_gradient"]),
        default_hyperparameters=default_gamma_n_hyperparameters(),
        default_function_rows=[
            {"id": "slot-f1", "name": "f1", "function_key": "SmoothStronglyConvexFunction", "function_params": {}},
            {"id": "slot-f2", "name": "f2", "function_key": "ConvexFunction", "function_params": {}},
        ],
    ),
    "accelerated_gradient_convex": AlgorithmSpec(
        name="accelerated_gradient_convex",
        algo=compile_algorithm_body(BASE_ALGORITHM_BODIES["accelerated_gradient_convex"]),
        default_hyperparameters=default_gamma_n_hyperparameters(),
        default_function_rows=[
            {"id": "slot-f", "name": "f", "function_key": "SmoothStronglyConvexFunction", "function_params": {}}
        ],
    ),
    "epsilon_subgradient": AlgorithmSpec(
        name="epsilon_subgradient",
        algo=compile_algorithm_body(BASE_ALGORITHM_BODIES["epsilon_subgradient"]),
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
        default_function_rows=[{"id": "slot-f", "name": "f", "function_key": "ConvexFunction", "function_params": {}}],
    ),
}
