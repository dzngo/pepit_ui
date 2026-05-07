from math import isfinite
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v2 as components_v2

ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"
RESULTS_WORKSPACE_DIR = ASSETS_DIR / "results_workspace"


def _assemble_results_workspace_js() -> str:
    # No-build bundle assembly: concatenate feature modules in dependency order.
    ordered_files = [
        "constants.js",
        "payload.js",
        "state.js",
        "legend.js",
        "layout.js",
        "events.js",
        "tau_plots.js",
        "dual_plots.js",
        "runtime.js",
        "index.js",
    ]
    return "\n\n".join((RESULTS_WORKSPACE_DIR / name).read_text() for name in ordered_files)


RESULTS_WORKSPACE_CSS = (RESULTS_WORKSPACE_DIR / "results_workspace.css").read_text()
RESULTS_WORKSPACE_JS = _assemble_results_workspace_js()


RESULTS_WORKSPACE_COMPONENT = components_v2.component(
    "results_workspace_component_v22",
    html="<div id='results-workspace-root'></div>",
    js=RESULTS_WORKSPACE_JS,
)


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not isfinite(value):
            return None
        return value
    if isinstance(value, (pd.Timestamp,)):
        return str(value)
    if hasattr(value, "item"):
        try:
            scalar = value.item()
        except Exception:
            return str(value)
        return _json_safe(scalar)
    return value


def render_dual_values_panel(
    algo_key: str,
    runs: list[dict],
    run_tau_series_by_param: dict[str, dict[str, dict[str, object]]],
    run_series_data_by_param: dict[str, dict],
    tau_payload: dict,
) -> dict | None:
    metric_labels = {
        "non_zero_pct": "Non-zero %",
        "non_zero_pct_with_none": "Non-zero % (None=0)",
        "std": "Std dev",
        "std_with_none": "Std dev (None=0)",
        "median_abs": "Median |x|",
        "median_abs_with_none": "Median |x| (None=0)",
        "mean_abs": "Average |x|",
        "mean_abs_with_none": "Average |x| (None=0)",
    }
    metric_state_key = f"dual-ranking-metric-{algo_key}"
    st.session_state.setdefault(metric_state_key, "non_zero_pct_with_none")
    metric = str(st.session_state.get(metric_state_key, "non_zero_pct_with_none"))
    if metric not in metric_labels:
        metric = "non_zero_pct_with_none"
        st.session_state[metric_state_key] = metric

    selected_series_ids = st.session_state.get(f"dual_selected_{algo_key}", [])
    dual_runs_data: list[dict] = []
    for run in runs:
        dual_runs_data.append(
            {
                "id": run["id"],
                "name": run["name"],
                "visible": bool(run.get("visible", True)),
                "selected_labels": [
                    str(label) for label in run.get("deactivated_labels", run.get("selected_labels", []))
                ],
                "tau_series_by_param": run_tau_series_by_param.get(run["id"], {}),
                "series_data_by_param": run_series_data_by_param.get(run["id"], {}),
            }
        )
    payload = {
        "css": RESULTS_WORKSPACE_CSS,
        "tau_payload": tau_payload,
        "series_data_by_param": tau_payload.get("series_data_by_param", {}),
        "dual_runs": dual_runs_data,
        "selected_series_ids": selected_series_ids,
        "metric": metric,
        "metric_labels": metric_labels,
    }
    result = RESULTS_WORKSPACE_COMPONENT(
        key=f"results-workspace-{algo_key}",
        data=_json_safe(payload),
        height="content",
        isolate_styles=False,
        on_recompute_change=lambda: None,
    )
    if result is None:
        return None
    cursor = result.get("cursor")
    if isinstance(cursor, dict):
        return {"type": "cursor", **cursor}
    metric_change = result.get("metric")
    if isinstance(metric_change, dict):
        return {"type": "metric", **metric_change}
    remove_run = result.get("remove_run")
    if isinstance(remove_run, dict):
        return {"type": "remove_run", **remove_run}
    recompute = result.get("recompute")
    if isinstance(recompute, dict):
        return {"type": "recompute", **recompute}
    return None
