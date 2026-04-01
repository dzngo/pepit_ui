from typing import Dict

from algorithm.algorithm_compiler import compile_algorithm_body
from algorithm.types import (
    AlgorithmSpec,
    FunctionSlot,
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
steps = int(n)
step_size = float(gamma)
for i in range(steps):
    x = x - step_size * f.gradient(x)
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
steps = int(n)
step_size = float(gamma)
for i in range(steps):
    problem.set_performance_metric(fx - fs)
    x = x - step_size * gx
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
steps = int(n)
step_size = float(gamma)
for i in range(steps):
    y = x - step_size * f1.gradient(x)
    x, _, _ = proximal_step(y, f2, step_size)
    x.set_name(f"x_{i+1}")
problem.set_performance_metric((x - xs) ** 2)
""".strip(),
    "accelerated_proximal_point": """
xs = f.stationary_point()
xs.set_name("x_*")
fs = f(xs)
x0 = problem.set_initial_point()
problem.set_initial_condition((x0 - xs) ** 2 <= 1)
x = x0
x.set_name("x_0")
y = x0
lam = 1
steps = int(n)
step_size = float(gamma)
for i in range(steps):
    lam_old = lam
    lam = (1 + sqrt(4 * lam_old**2 + 1)) / 2
    x_old = x
    x = y - step_size * f.gradient(y)
    y = x + (lam_old - 1) / lam * (x - x_old)
    x.set_name(f"x_{i+1}")
    y.set_name(f"y_{i+1}")
problem.set_performance_metric(f(x) - fs)
""".strip(),
    "epsilon_subgradient": """
M_value = float(M)
eps_value = float(eps)
R_value = float(R)
steps = int(n)
step_size = float(gamma)

xs = f.stationary_point()
xs.set_name("x_*")
fs = f(xs)
x0 = problem.set_initial_point()
problem.set_initial_condition((x0 - xs) ** 2 <= R_value**2)
x = x0
x.set_name("x_0")

for i in range(steps):
    x, gx, fx, epsilon = epsilon_subgradient_step(x, f, step_size)
    x.set_name(f"x_{i+1}")
    problem.set_performance_metric(fx - fs)
    problem.add_constraint(epsilon <= eps_value)
    problem.add_constraint(gx**2 <= M_value**2)

gx, fx = f.oracle(x)
problem.add_constraint(gx**2 <= M_value**2)
problem.set_performance_metric(fx - fs)
""".strip(),
}


BASE_ALGORITHMS: Dict[str, AlgorithmSpec] = {
    "gradient_descent": AlgorithmSpec(
        name="gradient_descent",
        algo=compile_algorithm_body(BASE_ALGORITHM_BODIES["gradient_descent"]),
        function_slots=[FunctionSlot(key="f")],
        default_function_keys={"f": "SmoothConvexFunction"},
        default_hyperparameters=default_gamma_n_hyperparameters(),
    ),
    "subgradient_method": AlgorithmSpec(
        name="subgradient_method",
        algo=compile_algorithm_body(BASE_ALGORITHM_BODIES["subgradient_method"]),
        function_slots=[FunctionSlot(key="f")],
        default_function_keys={"f": "ConvexLipschitzFunction"},
        default_hyperparameters=default_gamma_n_hyperparameters(),
    ),
    "proximal_gradient": AlgorithmSpec(
        name="proximal_gradient",
        algo=compile_algorithm_body(BASE_ALGORITHM_BODIES["proximal_gradient"]),
        function_slots=[FunctionSlot(key="f1"), FunctionSlot(key="f2")],
        default_function_keys={"f1": "SmoothStronglyConvexFunction", "f2": "ConvexFunction"},
        default_hyperparameters=default_gamma_n_hyperparameters(),
    ),
    "accelerated_proximal_point": AlgorithmSpec(
        name="accelerated_proximal_point",
        algo=compile_algorithm_body(BASE_ALGORITHM_BODIES["accelerated_proximal_point"]),
        function_slots=[FunctionSlot(key="f")],
        default_function_keys={"f": "SmoothStronglyConvexFunction"},
        default_hyperparameters=default_gamma_n_hyperparameters(),
    ),
    "epsilon_subgradient": AlgorithmSpec(
        name="epsilon_subgradient",
        algo=compile_algorithm_body(BASE_ALGORITHM_BODIES["epsilon_subgradient"]),
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
