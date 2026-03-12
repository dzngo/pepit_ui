import inspect
from math import isfinite
from typing import Dict, List

import PEPit.functions as functions
from PEPit.function import Function

from .types import FunctionParamSpec, FunctionSpec


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
