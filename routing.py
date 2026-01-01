# routing.py
import json
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from algorithms_registry import HyperparameterSpec
from utils import (
    BASE_GAMMA_SPEC,
    BASE_N_SPEC,
    build_dual_series_data,
    build_dual_section_html,
    clamp_value,
    clear_grid_cache_entry,
    dual_fluctuations_by_slice,
    get_tau_grid,
    slider_for_param,
    value_index,
)


def init_session_state():
    st.session_state.setdefault("ui_phase", "config")
    st.session_state.setdefault("selected_algorithm", None)
    st.session_state.setdefault("range_store", {})
    st.session_state.setdefault("other_params_store", {})
    st.session_state.setdefault("pending_settings", None)
    st.session_state.setdefault("active_settings", None)
    st.session_state.setdefault("other_editor_open", False)


def reset_for_algorithm_change(algo_key: str):
    st.session_state["selected_algorithm"] = algo_key
    st.session_state["ui_phase"] = "config"
    st.session_state["pending_settings"] = None
    st.session_state["active_settings"] = None
    st.session_state["other_editor_open"] = False


def render_range_inputs(label: str, base: HyperparameterSpec, stored: dict) -> dict:
    defaults = {
        "min": stored.get("min", base.min_value) if stored else base.min_value,
        "max": stored.get("max", base.max_value) if stored else base.max_value,
        "step": stored.get("step", base.step) if stored else base.step,
    }
    col1, col2, col3 = st.columns(3)
    if base.value_type == "int":
        min_value = col1.number_input(f"{label} min", value=int(defaults["min"]), step=1)
        max_value = col2.number_input(f"{label} max", value=int(defaults["max"]), step=1)
        step_value = col3.number_input(
            f"{label} step",
            min_value=1,
            value=max(int(defaults["step"]), 1),
            step=1,
        )
        return {"min": int(min_value), "max": int(max_value), "step": int(step_value)}

    min_value = col1.number_input(
        f"{label} min",
        value=float(defaults["min"]),
        step=float(base.step),
        format="%.4f",
    )
    max_value = col2.number_input(
        f"{label} max",
        value=float(defaults["max"]),
        step=float(base.step),
        format="%.4f",
    )
    step_value = col3.number_input(
        f"{label} step",
        min_value=1e-6,
        value=float(defaults["step"]),
        step=float(base.step),
        format="%.4f",
    )
    return {"min": float(min_value), "max": float(max_value), "step": float(step_value)}


def render_config_phase(algo_key: str, spec):
    st.subheader("Configuration")
    with st.container(border=True):
        st.write("Set gamma/n ranges and choose constant values for other parameters.")

        range_store = st.session_state["range_store"]
        algo_ranges = range_store.setdefault(algo_key, {})
        gamma_settings = render_range_inputs("gamma", BASE_GAMMA_SPEC, algo_ranges.get("gamma", {}))
        n_settings = render_range_inputs("n", BASE_N_SPEC, algo_ranges.get("n", {}))
        algo_ranges["gamma"] = gamma_settings
        algo_ranges["n"] = n_settings

        params_store = st.session_state["other_params_store"]
        current_params = params_store.setdefault(
            algo_key,
            {param.name: param.default for param in spec.hyperparameters},
        )
    with st.container(border=True):
        st.write("Other parameters")
        if not spec.hyperparameters:
            st.caption("This algorithm has no additional parameters.")
        else:
            for param in spec.hyperparameters:
                current_params[param.name] = slider_for_param(
                    param,
                    value=current_params.get(param.name, param.default),
                    key=f"config-{algo_key}-{param.name}",
                )
            stale_keys = set(current_params) - {p.name for p in spec.hyperparameters}
            for key in stale_keys:
                current_params.pop(key, None)

    if st.button("Plot"):
        errors = []
        if gamma_settings["max"] <= gamma_settings["min"]:
            errors.append("gamma max must be greater than gamma min.")
        if gamma_settings["step"] <= 0:
            errors.append("gamma step must be positive.")
        if n_settings["max"] <= n_settings["min"]:
            errors.append("n max must be greater than n min.")
        if n_settings["step"] <= 0:
            errors.append("n step must be positive.")
        if errors:
            for error in errors:
                st.error(error)
            return
        gamma_spec = HyperparameterSpec(
            name="gamma",
            label="gamma",
            min_value=float(gamma_settings["min"]),
            max_value=float(gamma_settings["max"]),
            default=float(gamma_settings["min"]),
            step=float(gamma_settings["step"]),
            value_type=BASE_GAMMA_SPEC.value_type,
        )
        n_spec = HyperparameterSpec(
            name="n",
            label="n",
            min_value=float(n_settings["min"]),
            max_value=float(n_settings["max"]),
            default=float(n_settings["min"]),
            step=float(n_settings["step"]),
            value_type=BASE_N_SPEC.value_type,
        )
        st.session_state["pending_settings"] = {
            "algo_key": algo_key,
            "gamma_spec": gamma_spec,
            "n_spec": n_spec,
            "other_params": dict(current_params),
        }
        st.session_state["ui_phase"] = "loading"
        st.rerun()


def render_loading_phase(algo_key: str, spec):
    pending = st.session_state.get("pending_settings")
    if not pending or pending["algo_key"] != algo_key:
        st.session_state["ui_phase"] = "config"
        st.rerun()

    gamma_spec = pending["gamma_spec"]
    n_spec = pending["n_spec"]
    st.subheader(f"Computing tau values for `{spec.name}`")
    st.caption(f"Algorithm: {spec.description}")
    other_display = ", ".join(f"{name}={value}" for name, value in pending["other_params"].items()) or "None"
    st.info(
        f"gamma: [{gamma_spec.min_value}, {gamma_spec.max_value}], step_size={gamma_spec.step}  \n"
        f"n: [{n_spec.min_value}, {n_spec.max_value}], step_size={n_spec.step}  \n"
        f"other params: {other_display}"
    )

    result = get_tau_grid(
        algo_key,
        gamma_spec,
        n_spec,
        pending["other_params"],
        show_progress=True,
    )
    if result is None:
        st.error("Unable to compute tau grid.")
        st.session_state["ui_phase"] = "config"
        return

    st.session_state["active_settings"] = pending
    st.session_state["pending_settings"] = None
    st.session_state["ui_phase"] = "results"
    st.session_state[f"dual_selection_{algo_key}"] = {}
    st.session_state[f"gamma_slider_{algo_key}"] = float(gamma_spec.min_value)
    st.session_state[f"n_slider_{algo_key}"] = float(n_spec.min_value)
    st.rerun()


def render_other_params_editor(algo_key: str, spec, settings):
    if not spec.hyperparameters:
        return
    editor_open = st.session_state.get("other_editor_open", False)
    params_store = st.session_state["other_params_store"]
    params_store.setdefault(algo_key, dict(settings["other_params"]))

    if not editor_open:
        if st.button("Change other parameters"):
            st.session_state["other_editor_open"] = True
            st.rerun()
        return
    with st.container(border=True):
        st.write("Adjust other parameters")
        with st.form("other-params-edit"):
            new_values = {}
            for param in spec.hyperparameters:
                new_values[param.name] = slider_for_param(
                    param,
                    value=params_store[algo_key].get(param.name, param.default),
                    key=f"edit-{algo_key}-{param.name}",
                )
            submitted = st.form_submit_button("Replot")
        if submitted:
            params_store[algo_key] = new_values
            clear_grid_cache_entry(
                algo_key,
                settings["gamma_spec"],
                settings["n_spec"],
                new_values,
            )
            st.session_state["pending_settings"] = {
                "algo_key": algo_key,
                "gamma_spec": settings["gamma_spec"],
                "n_spec": settings["n_spec"],
                "other_params": dict(new_values),
            }
    st.session_state["other_editor_open"] = False
    st.session_state["ui_phase"] = "loading"
    st.rerun()
    if st.button("Cancel"):
        st.session_state["other_editor_open"] = False
        st.rerun()


def render_results_phase(algo_key: str, spec):
    settings = st.session_state.get("active_settings")
    if not settings or settings["algo_key"] != algo_key:
        st.session_state["ui_phase"] = "config"
        st.rerun()

    cached = get_tau_grid(
        algo_key,
        settings["gamma_spec"],
        settings["n_spec"],
        settings["other_params"],
        show_progress=False,
    )
    if cached is None:
        st.session_state["pending_settings"] = settings
        st.session_state["ui_phase"] = "loading"
        st.rerun()

    gamma_values, n_values, tau_grid, cached_warnings, duals_grid = cached
    gamma_spec = settings["gamma_spec"]
    n_spec = settings["n_spec"]

    st.subheader(f"Results for `{spec.name}`")
    st.caption(f"Algorithm: {spec.description}")
    if st.button("Change gamma/n settings"):
        st.session_state["ui_phase"] = "config"
        st.rerun()

    render_other_params_editor(algo_key, spec, settings)

    gamma_slider_key = f"gamma_slider_{algo_key}"
    n_slider_key = f"n_slider_{algo_key}"
    st.session_state.setdefault(gamma_slider_key, float(gamma_spec.min_value))
    st.session_state.setdefault(n_slider_key, float(n_spec.min_value))

    st.session_state[gamma_slider_key] = clamp_value(float(st.session_state[gamma_slider_key]), gamma_spec)
    st.session_state[n_slider_key] = clamp_value(float(st.session_state[n_slider_key]), n_spec)
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            gamma_value = st.slider(
                "gamma",
                float(gamma_spec.min_value),
                float(gamma_spec.max_value),
                step=float(gamma_spec.step),
                key=gamma_slider_key,
            )
    with col2:
        with st.container(border=True):
            if n_spec.value_type == "int":
                st.session_state[n_slider_key] = int(round(st.session_state[n_slider_key]))
                n_value_raw = st.slider(
                    "n",
                    int(n_spec.min_value),
                    int(n_spec.max_value),
                    step=int(max(1, n_spec.step)),
                    key=n_slider_key,
                )
                n_value = float(n_value_raw)
            else:
                n_value = st.slider(
                    "n",
                    float(n_spec.min_value),
                    float(n_spec.max_value),
                    step=float(n_spec.step),
                    key=n_slider_key,
                )

    gamma_idx = value_index(float(gamma_value), gamma_spec)
    n_idx = value_index(float(n_value), n_spec)
    tau_gamma = tau_grid[:, n_idx]
    tau_n = tau_grid[gamma_idx, :]
    current_tau = tau_grid[gamma_idx, n_idx]
    warning_messages = set(cached_warnings)

    gamma_fig = go.Figure()
    gamma_fig.add_trace(
        go.Scatter(
            x=gamma_values,
            y=tau_gamma,
            mode="lines",
            hovertemplate="gamma=%{x:.3f}<br>tau=%{y:.3e}<extra></extra>",
        )
    )
    if current_tau is not None and np.isfinite(current_tau):
        gamma_fig.add_trace(
            go.Scatter(
                x=[gamma_value],
                y=[current_tau],
                mode="markers",
                marker={"size": 12},
                hovertemplate="gamma=%{x:.3f}<br>tau=%{y:.3e}<extra></extra>",
                name="current",
            )
        )
    gamma_fig.update_layout(
        showlegend=False,
        title="Tau vs gamma",
        xaxis_title="gamma",
        yaxis_title="tau",
    )

    n_fig = go.Figure()
    n_fig.add_trace(
        go.Scatter(
            x=n_values,
            y=tau_n,
            mode="lines",
            hovertemplate="n=%{x:.3f}<br>tau=%{y:.3e}<extra></extra>",
        )
    )
    if current_tau is not None and np.isfinite(current_tau):
        n_fig.add_trace(
            go.Scatter(
                x=[n_value],
                y=[current_tau],
                mode="markers",
                marker={"size": 12},
                hovertemplate="n=%{x:.3f}<br>tau=%{y:.3e}<extra></extra>",
            )
        )
    n_fig.update_layout(
        showlegend=False,
        title="Tau vs n",
        xaxis_title="n",
        yaxis_title="tau",
    )

    with col1:
        with st.container(border=True):
            st.plotly_chart(gamma_fig)
    with col2:
        with st.container(border=True):
            st.plotly_chart(n_fig)

    if warning_messages:
        warning_text = "\n".join(sorted(warning_messages))
        st.warning(
            "Some parameter combinations could not be solved; missing points are shown as gaps.\n" + warning_text
        )

    render_dual_values_panel(
        algo_key,
        duals_grid,
        gamma_values,
        n_values,
        gamma_idx,
        n_idx,
    )


DUAL_BUTTON_COLUMNS = 12
DUAL_BUTTON_MIN_WIDTH = 80
DUAL_BUTTON_ROW_HEIGHT = 52
DUAL_SECTION_PADDING = 70
DUAL_PLOT_HEIGHT = 320


def render_dual_values_panel(
    algo_key: str,
    duals_grid: list[list[dict]],
    gamma_values: np.ndarray,
    n_values: np.ndarray,
    gamma_idx: int,
    n_idx: int,
) -> None:
    st.subheader("Dual values")
    if not duals_grid:
        st.caption("No dual values available for these settings.")
        return

    current_duals = duals_grid[gamma_idx][n_idx] if duals_grid else {}
    gamma_slice = [row[n_idx] for row in duals_grid]
    n_slice = list(duals_grid[gamma_idx])
    gamma_fluctuations = dual_fluctuations_by_slice(gamma_slice)
    n_fluctuations = dual_fluctuations_by_slice(n_slice)
    series_data = build_dual_series_data(
        duals_grid,
        gamma_values,
        n_values,
        gamma_idx,
        n_idx,
    )
    series_json = json.dumps(series_data).replace("</", "<\\/")

    gamma_title = f"Fluctuation vs gamma (n = {n_values[n_idx]})"
    n_title = f"Fluctuation vs n (gamma = {gamma_values[gamma_idx]})"
    gamma_html, gamma_count = build_dual_section_html(
        section_id=f"{algo_key}-gamma",
        title=gamma_title,
        dual_fluctuations=gamma_fluctuations,
        current_duals=current_duals,
        columns=DUAL_BUTTON_COLUMNS,
        min_width=DUAL_BUTTON_MIN_WIDTH,
    )
    n_html, n_count = build_dual_section_html(
        section_id=f"{algo_key}-n",
        title=n_title,
        dual_fluctuations=n_fluctuations,
        current_duals=current_duals,
        columns=DUAL_BUTTON_COLUMNS,
        min_width=DUAL_BUTTON_MIN_WIDTH,
    )

    total_buttons = gamma_count + n_count
    rows = (max(total_buttons, 1) + DUAL_BUTTON_COLUMNS - 1) // DUAL_BUTTON_COLUMNS
    component_height = 140 + rows * DUAL_BUTTON_ROW_HEIGHT + DUAL_SECTION_PADDING * 2 + DUAL_PLOT_HEIGHT * 2

    components.html(
        f"""
<div class="dual-wrapper">
  <div class="dual-section">{gamma_html}</div>
  <div class="dual-section">{n_html}</div>
  <div class="dual-plot-actions">
    <button class="dual-plot-button" id="dual-plot">Plot dual values</button>
  </div>
  <div id="dual-plot-gamma" class="dual-plot"></div>
  <div id="dual-plot-n" class="dual-plot"></div>
  <div class="dual-selected-header">
    <div class="dual-selected-title">Selected dual values</div>
    <button class="dual-clear-button" id="dual-clear">Deselect all</button>
  </div>
  <div id="dual-selected-list" class="dual-selected-list">None</div>
</div>

<style>
  .dual-wrapper {{
    display: flex;
    flex-direction: column;
    gap: 18px;
  }}
  .dual-section-title {{
    font-weight: 600;
    margin: 8px 0 10px 0;
  }}
  .dual-constraint-title {{
    font-weight: 600;
    margin: 6px 0 6px 0;
  }}
  .dual-grid {{
    display: grid;
    gap: 10px;
    margin-bottom: 12px;
  }}
  .dual-plot-actions {{
    display: flex;
    justify-content: flex-start;
    margin: 6px 0 4px 0;
  }}
  .dual-plot-button {{
    border: 1px solid #111;
    background: #111;
    color: #fff;
    font-weight: 600;
    padding: 6px 12px;
    border-radius: 999px;
    cursor: pointer;
  }}
  .dual-plot-button:hover {{
    background: #2b2b2b;
  }}
  .dual-plot {{
    min-height: {DUAL_PLOT_HEIGHT}px;
  }}
  .dual-button {{
    border: none;
    font-weight: 600;
    padding: 8px 12px;
    border-radius: 8px;
    min-width: {DUAL_BUTTON_MIN_WIDTH}px;
    cursor: pointer;
    box-shadow: 0 1px 2px rgba(0,0,0,0.15);
  }}
  .dual-button.is-clicked {{
    outline: 2px solid #111;
    outline-offset: 2px;
  }}
  .dual-selected-header {{
    display: flex;
    align-items: center;
    gap: 12px;
  }}
  .dual-selected-title {{
    font-weight: 600;
  }}
  .dual-clear-button {{
    border: 1px solid #111;
    background: #fff;
    color: #111;
    font-weight: 600;
    padding: 4px 10px;
    border-radius: 999px;
    cursor: pointer;
  }}
  .dual-clear-button:hover {{
    background: #f2f2f2;
  }}
  .dual-selected-list {{
    font-family: monospace;
  }}
</style>

<script>
  const seriesData = {series_json};
  const selected = new Map();
  const listEl = document.getElementById('dual-selected-list');
  const clearBtn = document.getElementById('dual-clear');
  const plotBtn = document.getElementById('dual-plot');

  function updateList() {{
    if (!selected.size) {{
      listEl.textContent = 'None';
      return;
    }}
    const items = Array.from(selected.values());
    listEl.textContent = items.join(', ');
  }}

  function setButtonsSelected(seriesId, isSelected) {{
    const matcher = (btn) => btn.getAttribute('data-series-id') === seriesId;
    document.querySelectorAll('.dual-button').forEach((btn) => {{
      if (!matcher(btn)) {{
        return;
      }}
      if (isSelected) {{
        btn.classList.add('is-clicked');
      }} else {{
        btn.classList.remove('is-clicked');
      }}
    }});
  }}

  document.querySelectorAll('.dual-button').forEach((btn) => {{
    btn.addEventListener('click', () => {{
      const id = btn.getAttribute('data-id');
      const seriesId = btn.getAttribute('data-series-id');
      const label = btn.getAttribute('data-label');
      const value = btn.getAttribute('data-value');
      const display = `${{label}} = ${{value}}`;
      if (selected.has(seriesId)) {{
        selected.delete(seriesId);
        setButtonsSelected(seriesId, false);
      }} else {{
        selected.set(seriesId, display);
        setButtonsSelected(seriesId, true);
      }}
      updateList();
    }});
  }});

  function ensurePlotly(callback) {{
    if (window.Plotly) {{
      callback();
      return;
    }}
    const script = document.createElement('script');
    script.src = 'https://cdn.plot.ly/plotly-2.32.0.min.js';
    script.onload = callback;
    document.head.appendChild(script);
  }}

  function plotSelected() {{
    const seriesIds = Array.from(selected.keys());
    if (!seriesIds.length) {{
      const gammaEl = document.getElementById('dual-plot-gamma');
      const nEl = document.getElementById('dual-plot-n');
      gammaEl.textContent = 'Select dual values to plot.';
      nEl.textContent = 'Select dual values to plot.';
      return;
    }}
    const gammaTraces = [];
    const nTraces = [];
    seriesIds.forEach((seriesId) => {{
      const series = seriesData[seriesId];
      if (!series) {{
        return;
      }}
      gammaTraces.push({{
        x: series.gamma_values,
        y: series.gamma_dual,
        mode: 'lines+markers',
        name: series.label,
      }});
      nTraces.push({{
        x: series.n_values,
        y: series.n_dual,
        mode: 'lines+markers',
        name: series.label,
      }});
    }});
    Plotly.newPlot('dual-plot-gamma', gammaTraces, {{
      title: 'Dual values vs gamma',
      xaxis: {{ title: 'gamma' }},
      yaxis: {{ title: 'dual value' }},
      margin: {{ t: 40, l: 40, r: 20, b: 40 }},
    }}, {{ displayModeBar: false }});
    Plotly.newPlot('dual-plot-n', nTraces, {{
      title: 'Dual values vs n',
      xaxis: {{ title: 'n' }},
      yaxis: {{ title: 'dual value' }},
      margin: {{ t: 40, l: 40, r: 20, b: 40 }},
    }}, {{ displayModeBar: false }});
  }}

  plotBtn.addEventListener('click', () => {{
    ensurePlotly(plotSelected);
  }});

  clearBtn.addEventListener('click', () => {{
    selected.clear();
    document.querySelectorAll('.dual-button').forEach((btn) => {{
      btn.classList.remove('is-clicked');
    }});
    updateList();
  }});
</script>
""",
        height=component_height,
    )
