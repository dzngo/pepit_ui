from math import sqrt
from typing import Callable, Dict

import numpy as np
from PEPit import PEP, Point, primitive_steps
from PEPit.function import Function


def _primitive_steps_namespace() -> dict[str, object]:
    exported_names = getattr(primitive_steps, "__all__", None)
    if exported_names is None:
        exported_names = [name for name in dir(primitive_steps) if not name.startswith("_")]
    out: dict[str, object] = {}
    for name in exported_names:
        attr = getattr(primitive_steps, name, None)
        if callable(attr):
            out[name] = attr
    return out


def list_supported_primitive_steps() -> list[str]:
    return sorted(_primitive_steps_namespace().keys())


def compile_algorithm_body(body_code: str) -> Callable[[PEP, Dict[str, object], Dict[str, float]], dict]:
    if any(line.lstrip().startswith("def ") for line in body_code.splitlines() if line.strip()):
        raise ValueError("Algorithm code must be function body only; do not include a 'def ...' wrapper.")
    code_obj = compile(body_code, "<algorithm_body>", "exec")
    base_namespace: dict[str, object] = {
        "PEP": PEP,
        "Point": Point,
        "sqrt": sqrt,
        "np": np,
        "Dict": Dict,
        "Function": Function,
    }
    primitive_step_namespace = _primitive_steps_namespace()
    collisions = set(base_namespace).intersection(primitive_step_namespace)
    if collisions:
        names = ", ".join(sorted(collisions))
        raise ValueError(f"Primitive step namespace collision: {names}")
    base_namespace.update(primitive_step_namespace)

    def _algo(
        problem: PEP,
        funcs: Dict[str, object],
        params: Dict[str, float],
        func_params: Dict[str, Dict[str, object]] | None = None,
    ) -> dict:
        namespace = dict(base_namespace)
        namespace["problem"] = problem
        namespace["func_params"] = dict(func_params or {})
        for name, value in funcs.items():
            if str(name).isidentifier():
                namespace[str(name)] = value
        for name, value in params.items():
            if str(name).isidentifier():
                namespace[str(name)] = value
        exec(code_obj, namespace, namespace)
        return {}

    return _algo
