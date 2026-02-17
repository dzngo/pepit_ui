export default function(component) {
  const { data, parentElement, setTriggerValue } = component;

  const seriesData = (data && data.series_data) || {};
  const rawDualRuns = (data && data.dual_runs) || [];
  const tauPayload = (data && data.tau_payload) || {};
  const metricLabels = (data && data.metric_labels) || {};
  const metricKeys = Object.keys(metricLabels);
  const currentMetric = (data && data.metric) || 'non_zero_pct_with_none';

  const plotSelectionMap = new Map();
  const deactivatedForRecompute = new Map();
  const selectedPlotCards = { gamma: new Set(), n: new Set() };
  const overlayState = new Map();
  const overlayDebounce = new Map();
  const dualTraceRegistry = new Map();
  const tauTraceRegistry = { gamma: new Map(), n: new Map() };
  let hideZero = true;
  let actionTab = 'plot';

  const gammaValues = Array.isArray(tauPayload.gamma_values) ? tauPayload.gamma_values : [];
  const nValues = Array.isArray(tauPayload.n_values) ? tauPayload.n_values : [];
  const tauGrid = Array.isArray(tauPayload.tau_grid) ? tauPayload.tau_grid : [];
  const patternParams = (tauPayload && tauPayload.pattern_params) || {};
  const patternInvalidParams = Array.isArray(tauPayload.pattern_invalid_params) ? tauPayload.pattern_invalid_params : [];
  const patternConflictParams = Array.isArray(tauPayload.pattern_conflict_params) ? tauPayload.pattern_conflict_params : [];

  const RUN_COLORS = ['#0ea5e9', '#14b8a6', '#e11d48', '#f4ec0b', '#22c55e', '#ef4444', '#3b82f6', '#84cc16', '#a855f7'];

  function stableHash(text) {
    const value = String(text || '');
    let hash = 0;
    for (let i = 0; i < value.length; i += 1) {
      hash = (hash * 31 + value.charCodeAt(i)) >>> 0;
    }
    return hash;
  }

  function runColorFor(runId, fallbackIndex = 0) {
    if (!RUN_COLORS.length) return '#0ea5e9';
    const hash = stableHash(runId || fallbackIndex);
    return RUN_COLORS[hash % RUN_COLORS.length];
  }

  const dualRuns = rawDualRuns.map((run, idx) => ({
    ...run,
    color: run.color || runColorFor(run.id, idx),
  }));

  const runsById = new Map();
  dualRuns.forEach((run) => {
    runsById.set(run.id, { ...run, visible: run.visible !== false });
  });
  const runVisibility = new Map(Array.from(runsById.values()).map((run) => [run.id, run.visible !== false]));

  const overlayExamples = [
    '1/log(x)',
    'sin(x)',
    '2*x^2 + 3',
    'log(x)',
    'sqrt(x)',
    'exp(-x)',
  ];

  function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  }

  function formatSubscriptText(text) {
    if (!text) return '';
    const value = String(text);
    let out = '';
    let i = 0;
    while (i < value.length) {
      const ch = value[i];
      if (ch !== '_') {
        out += escapeHtml(ch);
        i += 1;
        continue;
      }
      if (i + 1 >= value.length) {
        out += escapeHtml(ch);
        i += 1;
        continue;
      }
      if (value[i + 1] === '{') {
        const end = value.indexOf('}', i + 2);
        if (end === -1) {
          out += escapeHtml(value.slice(i));
          break;
        }
        out += `<sub>${escapeHtml(value.slice(i + 2, end))}</sub>`;
        i = end + 1;
        continue;
      }
      let end = i + 1;
      while (end < value.length && /[a-zA-Z0-9*]/.test(value[end])) end += 1;
      if (end === i + 1) {
        out += `<sub>${escapeHtml(value[i + 1])}</sub>`;
        i += 2;
        continue;
      }
      out += `<sub>${escapeHtml(value.slice(i + 1, end))}</sub>`;
      i = end;
    }
    return out;
  }

  function formatDualLabel(text) {
    if (!text) return '';
    const value = String(text);
    const separator = ' | ';
    const splitIndex = value.indexOf(separator);
    if (splitIndex !== -1) {
      const left = value.slice(0, splitIndex);
      const right = value.slice(splitIndex + separator.length);
      return `${escapeHtml(left)}${separator}${formatSubscriptText(right)}`;
    }
    return formatSubscriptText(value);
  }

  function randomOverlayPlaceholder() {
    const example = overlayExamples[Math.floor(Math.random() * overlayExamples.length)];
    return `example: ${example}`;
  }

  function sanitizeId(value) {
    return value.replace(/[^a-zA-Z0-9_-]/g, '-');
  }

  function ensurePlotly(callback) {
    if (window.Plotly) {
      callback();
      return;
    }
    const script = document.createElement('script');
    script.src = 'https://cdn.plot.ly/plotly-2.32.0.min.js';
    script.onload = callback;
    document.head.appendChild(script);
  }

  function ensureMath(callback) {
    if (window.math) {
      callback();
      return;
    }
    const script = document.createElement('script');
    script.src = 'https://cdn.jsdelivr.net/npm/mathjs@12.4.2/lib/browser/math.js';
    script.onload = callback;
    document.head.appendChild(script);
  }

  function runRowsHtml() {
    if (!runsById.size) {
      return '<div class="dual-selected-list">No recompute runs yet.</div>';
    }
    const rows = [];
    rows.push(
      '<div class="run-row" style="margin-bottom:10px;">' +
        '<div style="display:flex;align-items:center;gap:8px;">' +
          '<span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:#9aa0a6;"></span>' +
          '<strong>Baseline</strong>' +
        '</div>' +
        '<div class="dual-selected-list" style="margin-top:8px;">All duals</div>' +
      '</div>'
    );
    Array.from(runsById.values()).forEach((run) => {
      const isVisible = runVisibility.get(run.id) !== false;
      const buttonText = isVisible ? 'Hide' : 'View';
      const buttonStateClass = isVisible ? 'is-active' : 'is-neutral';
      const labels = Array.isArray(run.selected_labels) ? run.selected_labels : [];
      const dualBadges = labels.length
        ? labels.map((label) => `<span class="dual-badge">${formatDualLabel(String(label))}</span>`).join('')
        : '<span class="dual-badge">No dual values selected.</span>';
      rows.push(
        `<div class="run-row" data-run-row="${escapeHtml(run.id)}">` +
          `<div style="display:flex;justify-content:space-between;align-items:center;gap:12px;">` +
            `<div><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:${escapeHtml(run.color || '#f97316')};margin-right:8px;"></span><strong>${escapeHtml(run.name || run.id)}</strong></div>` +
            `<div style="display:flex;align-items:center;gap:8px;">` +
              `<button type="button" class="dual-toggle-button run-toggle ${buttonStateClass}" data-run-id="${escapeHtml(run.id)}">${buttonText}</button>` +
              `<button type="button" class="dual-remove-button run-remove" data-run-id="${escapeHtml(run.id)}">Remove</button>` +
            `</div>` +
          `</div>` +
          `<details style="margin-top:8px;">` +
            `<summary style="cursor:pointer;font-weight:600;">View deactivated duals (${labels.length})</summary>` +
            `<div class="dual-selected-list" style="margin-top:8px;">${dualBadges}</div>` +
          `</details>` +
        `</div>`
      );
    });
    return rows.join('');
  }

  function metricOptionsHtml() {
    if (!metricKeys.length) return '';
    return metricKeys.map((key) => {
      const selected = key === currentMetric ? ' selected' : '';
      const label = metricLabels[key] || key;
      return `<option value="${escapeHtml(key)}"${selected}>${escapeHtml(label)}</option>`;
    }).join('');
  }

  parentElement.innerHTML = `
    <style>${data.css || ''}</style>
    <div class="dual-wrapper">
      <div class="dual-section-title">Worst-case guarantee</div>
      <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:12px;">
        <div class="dual-panel">
          <div class="dual-plot-section">
            <div style="display:flex;align-items:center;gap:8px;">
              <strong>gamma</strong>
              <input type="range" id="tau-gamma-slider" min="0" max="${Math.max(gammaValues.length - 1, 0)}" step="1" value="${Math.min(Math.max(Number(tauPayload.default_gamma_idx || 0), 0), Math.max(gammaValues.length - 1, 0))}" style="flex:1;" />
              <span id="tau-gamma-value"></span>
            </div>
            <div id="tau-plot-gamma" class="dual-plot-chart" style="min-height:320px;"></div>
            <input class="dual-overlay-input" id="tau-pattern-gamma" type="text" value="${escapeHtml(String(tauPayload.pattern_gamma || ''))}" placeholder="${randomOverlayPlaceholder()}" style="display:block;">
            <div id="tau-pattern-gamma-hint" style="font-size:12px;color:#666;"></div>
            <div id="tau-pattern-gamma-error" style="font-size:12px;color:#cb0000;min-height:16px;"></div>
          </div>
        </div>
        <div class="dual-panel">
          <div class="dual-plot-section">
            <div style="display:flex;align-items:center;gap:8px;">
              <strong>n</strong>
              <input type="range" id="tau-n-slider" min="0" max="${Math.max(nValues.length - 1, 0)}" step="1" value="${Math.min(Math.max(Number(tauPayload.default_n_idx || 0), 0), Math.max(nValues.length - 1, 0))}" style="flex:1;" />
              <span id="tau-n-value"></span>
            </div>
            <div id="tau-plot-n" class="dual-plot-chart" style="min-height:320px;"></div>
            <input class="dual-overlay-input" id="tau-pattern-n" type="text" value="${escapeHtml(String(tauPayload.pattern_n || ''))}" placeholder="${randomOverlayPlaceholder()}" style="display:block;">
            <div id="tau-pattern-n-hint" style="font-size:12px;color:#666;"></div>
            <div id="tau-pattern-n-error" style="font-size:12px;color:#cb0000;min-height:16px;"></div>
          </div>
        </div>
      </div>
      <div class="dual-section-title">Recompute Runs</div>
      <div class="dual-panel">
        ${runRowsHtml()}
      </div>
      <div class="dual-section-title">Dual values</div>
      <div style="display:flex;align-items:center;gap:10px;justify-content:space-between;flex-wrap:wrap;">
        <div style="display:flex;align-items:center;gap:10px;">
        <label for="dual-ranking-metric" style="font-weight:600;">Ranking metric</label>
        <select id="dual-ranking-metric" style="width:280px;max-width:100%;padding:8px 10px;border-radius:8px;border:1px solid rgba(49,51,63,0.2);">
          ${metricOptionsHtml()}
        </select>
        <button type="button" class="dual-toggle-button is-active" id="dual-toggle-zero">Show all-zero duals</button>
      </div>
      </div>
      <div class="dual-panel"><div class="dual-section">${data.gamma_html || ''}</div></div>
      <div class="dual-panel"><div class="dual-section">${data.n_html || ''}</div></div>
      <div class="dual-tabs" role="tablist" aria-label="Dual action tabs">
        <button
          type="button"
          class="dual-tab is-active"
          id="tab-plot"
          role="tab"
          aria-selected="true"
          aria-controls="tab-panel-plot"
        >
          Plot
        </button>
        <button
          type="button"
          class="dual-tab"
          id="tab-recompute"
          role="tab"
          aria-selected="false"
          aria-controls="tab-panel-recompute"
          tabindex="-1"
        >
          Recompute
        </button>
      </div>
      <div id="tab-panel-plot" class="dual-tabpanel" role="tabpanel" aria-labelledby="tab-plot">
        <div class="dual-plot-actions">
          <button type="button" class="dual-plot-button" id="dual-plot">Plot dual values</button>
          <button type="button" class="dual-overlay-button" id="dual-overlay" style="display:none;">Overlay plot</button>
          <button type="button" class="dual-clear-button" id="dual-clear">Select all</button>
        </div>
      </div>
      <div id="tab-panel-recompute" class="dual-tabpanel" role="tabpanel" aria-labelledby="tab-recompute" hidden>
        <div class="dual-selected-header"><div class="dual-selected-title">Deactivated dual values</div></div>
        <div id="dual-deactivated-list" class="dual-selected-list">None</div>
        <div class="dual-plot-actions dual-plot-actions-inline dual-recompute-actions">
          <button type="button" class="dual-plot-button" id="dual-recompute" disabled>Recompute</button>
          <button type="button" class="dual-clear-button" id="dual-activate-all">Activate all</button>
          <button type="button" class="dual-toggle-button is-active" id="dual-deactivate-all">Deactivate all</button>
        </div>
      </div>
      <div class="dual-panel">
        <div class="dual-plot-section">
          <div class="dual-plot-title">${escapeHtml(data.plot_title_gamma || '')}</div>
          <div id="dual-plot-gamma" class="dual-plot-sections"></div>
          <div class="dual-plot-actions dual-plot-actions-inline">
            <button type="button" class="dual-remove-button" id="dual-remove-gamma" style="display:none;">Remove selected dual values</button>
          </div>
        </div>
      </div>
      <div class="dual-panel">
        <div class="dual-plot-section">
          <div class="dual-plot-title">${escapeHtml(data.plot_title_n || '')}</div>
          <div id="dual-plot-n" class="dual-plot-sections"></div>
          <div class="dual-plot-actions dual-plot-actions-inline">
            <button type="button" class="dual-remove-button" id="dual-remove-n" style="display:none;">Remove selected dual values</button>
          </div>
        </div>
      </div>
    </div>
  `;

  const root = parentElement;
  const deactivatedListEl = root.querySelector('#dual-deactivated-list');
  const toggleBtn = root.querySelector('#dual-toggle-zero');
  const tabPlotBtn = root.querySelector('#tab-plot');
  const tabRecomputeBtn = root.querySelector('#tab-recompute');
  const tabPanelPlot = root.querySelector('#tab-panel-plot');
  const tabPanelRecompute = root.querySelector('#tab-panel-recompute');
  const plotBtn = root.querySelector('#dual-plot');
  const overlayBtn = root.querySelector('#dual-overlay');
  const clearBtn = root.querySelector('#dual-clear');
  const removeGammaBtn = root.querySelector('#dual-remove-gamma');
  const removeNBtn = root.querySelector('#dual-remove-n');
  const recomputeBtn = root.querySelector('#dual-recompute');
  const activateAllBtn = root.querySelector('#dual-activate-all');
  const deactivateAllBtn = root.querySelector('#dual-deactivate-all');
  const tauGammaSlider = root.querySelector('#tau-gamma-slider');
  const tauNSlider = root.querySelector('#tau-n-slider');
  const tauGammaValue = root.querySelector('#tau-gamma-value');
  const tauNValue = root.querySelector('#tau-n-value');
  const tauPatternGammaInput = root.querySelector('#tau-pattern-gamma');
  const tauPatternNInput = root.querySelector('#tau-pattern-n');
  const tauPatternGammaHint = root.querySelector('#tau-pattern-gamma-hint');
  const tauPatternNHint = root.querySelector('#tau-pattern-n-hint');
  const tauPatternGammaError = root.querySelector('#tau-pattern-gamma-error');
  const tauPatternNError = root.querySelector('#tau-pattern-n-error');
  const metricSelect = root.querySelector('#dual-ranking-metric');

  let gammaIdx = tauGammaSlider ? Number(tauGammaSlider.value) : 0;
  let nIdx = tauNSlider ? Number(tauNSlider.value) : 0;
  let lastCursorEventKey = `${gammaIdx}:${nIdx}`;

  function visibleButtons() {
    return Array.from(root.querySelectorAll('.dual-button:not(.is-hidden):not(.is-recompute-hidden)'));
  }

  function updateDeactivatedList() {
    if (!deactivatedListEl) return;
    if (!deactivatedForRecompute.size) {
      deactivatedListEl.textContent = 'None';
      if (recomputeBtn) {
        recomputeBtn.disabled = true;
        recomputeBtn.style.display = 'none';
      }
      return;
    }
    deactivatedListEl.innerHTML = Array.from(deactivatedForRecompute.entries()).map(([seriesId, label]) => (
      `<button type="button" class="dual-badge dual-reactivate-button" data-series-id="${escapeHtml(seriesId)}">${formatDualLabel(label)}</button>`
    )).join('');
    if (recomputeBtn) {
      recomputeBtn.disabled = false;
      recomputeBtn.style.display = 'inline-flex';
    }
  }

  function syncDualButtonState() {
    root.querySelectorAll('.dual-button').forEach((btn) => {
      const seriesId = btn.getAttribute('data-series-id');
      const hideByTab = actionTab === 'recompute' && deactivatedForRecompute.has(seriesId);
      const selectedForPlot = actionTab === 'plot' && plotSelectionMap.has(seriesId);
      btn.classList.toggle('is-clicked', selectedForPlot);
      btn.classList.toggle('is-recompute-hidden', hideByTab);
    });
  }

  function setActionTab(nextTab) {
    actionTab = nextTab === 'recompute' ? 'recompute' : 'plot';
    if (tabPlotBtn) {
      const active = actionTab === 'plot';
      tabPlotBtn.classList.toggle('is-active', active);
      tabPlotBtn.setAttribute('aria-selected', active ? 'true' : 'false');
      tabPlotBtn.tabIndex = active ? 0 : -1;
    }
    if (tabRecomputeBtn) {
      const active = actionTab === 'recompute';
      tabRecomputeBtn.classList.toggle('is-active', active);
      tabRecomputeBtn.setAttribute('aria-selected', active ? 'true' : 'false');
      tabRecomputeBtn.tabIndex = active ? 0 : -1;
    }
    if (tabPanelPlot) tabPanelPlot.hidden = actionTab !== 'plot';
    if (tabPanelRecompute) tabPanelRecompute.hidden = actionTab !== 'recompute';
    syncDualButtonState();
    updateClearButton();
  }

  function updateClearButton() {
    if (!clearBtn) return;
    const buttons = visibleButtons();
    if (!buttons.length) {
      clearBtn.textContent = 'Select all';
      return;
    }
    const visibleSeries = new Set(buttons.map((btn) => btn.getAttribute('data-series-id')));
    const allSelected = Array.from(visibleSeries).every((id) => plotSelectionMap.has(id));
    clearBtn.textContent = allSelected ? 'Deselect all' : 'Select all';
  }

  function updateRemoveButtons() {
    const hasSelection = selectedPlotCards.gamma.size || selectedPlotCards.n.size;
    const display = hasSelection ? 'inline-flex' : 'none';
    if (removeGammaBtn) removeGammaBtn.style.display = display;
    if (removeNBtn) removeNBtn.style.display = display;
  }

  function updateOverlayButtonVisibility(show) {
    if (!overlayBtn) return;
    overlayBtn.style.display = show ? 'inline-flex' : 'none';
    const wrapperEl = root.querySelector('.dual-wrapper');
    if (!show && wrapperEl) {
      wrapperEl.classList.remove('dual-show-overlay');
      overlayBtn.classList.remove('is-active');
    }
  }

  function normalizeExpression(raw) {
    if (!raw) return '';
    const trimmed = raw.trim();
    if (!trimmed) return '';
    return trimmed.replace(/^y\s*=\s*/i, '');
  }

  function buildOverlayYValues(expr, xValues, scopeBuilder) {
    let compiled;
    try {
      compiled = window.math.compile(expr);
    } catch (err) {
      return { yValues: xValues.map(() => null), hasError: true };
    }
    let hasValid = false;
    const yValues = xValues.map((x, idx) => {
      if (!Number.isFinite(x)) return null;
      const scope = typeof scopeBuilder === 'function' ? scopeBuilder(x, idx) : { x };
      try {
        const value = compiled.evaluate(scope);
        const numeric = typeof value === 'number' ? value : Number(value);
        if (Number.isFinite(numeric)) {
          hasValid = true;
          return numeric;
        }
      } catch (err) {
        return null;
      }
      return null;
    });
    return { yValues, hasError: !hasValid };
  }

  function updateOverlayTrace(plotId, xValues, yValues, expr) {
    const plotDiv = root.querySelector(`#${plotId}`);
    if (!plotDiv || !window.Plotly) return;
    const prevState = overlayState.get(plotId);
    if (!expr) {
      if (prevState) {
        Plotly.deleteTraces(plotDiv, [prevState.traceIndex]);
        overlayState.delete(plotId);
      }
      return;
    }
    const traceName = `Overlay: ${expr}`;
    if (prevState && plotDiv.data && plotDiv.data[prevState.traceIndex]) {
      Plotly.restyle(plotDiv, { x: [xValues], y: [yValues], name: [traceName] }, [prevState.traceIndex]);
      return;
    }
    const traceIndex = plotDiv.data ? plotDiv.data.length : 0;
    Plotly.addTraces(plotDiv, {
      x: xValues,
      y: yValues,
      mode: 'lines',
      name: traceName,
      showlegend: false,
      line: { color: '#ff8a00', width: 2 },
    }).then(() => {
      overlayState.set(plotId, { traceIndex });
    });
  }

  function handleOverlayInput(event) {
    const input = event.target;
    if (!input || !window.math) return;
    const axis = input.getAttribute('data-axis');
    const plotId = input.getAttribute('data-plot-id');
    const seriesId = input.getAttribute('data-series-id');
    const series = seriesData[seriesId];
    if (!series || !axis || !plotId) return;
    const rawExpr = normalizeExpression(input.value);
    if (!rawExpr) {
      input.classList.remove('is-error');
      updateOverlayTrace(plotId, [], [], '');
      return;
    }
    const xValues = axis === 'gamma' ? series.gamma_values : series.n_values;
    const result = buildOverlayYValues(rawExpr, xValues, (x) => {
      const scope = { x };
      Object.entries(patternParams).forEach(([name, value]) => {
        const numeric = Number(value);
        if (Number.isFinite(numeric)) scope[name] = numeric;
      });
      if (axis === 'gamma') {
        scope.gamma = Number(x);
        scope.n = Number(nValues[nIdx]);
      } else {
        scope.n = Number(x);
        scope.gamma = Number(gammaValues[gammaIdx]);
      }
      return scope;
    });
    updateOverlayTrace(plotId, xValues, result.yValues, rawExpr);
    input.classList.toggle('is-error', result.hasError);
  }

  function setRunToggleButton(runId) {
    const button = root.querySelector(`.run-toggle[data-run-id="${runId}"]`);
    if (!button) return;
    const isVisible = runVisibility.get(runId) !== false;
    button.textContent = isVisible ? 'Hide' : 'View';
    button.classList.toggle('is-active', isVisible);
    button.classList.toggle('is-neutral', !isVisible);
  }

  function applyRunVisibilityToTau() {
    if (!window.Plotly) return;
    ['gamma', 'n'].forEach((axis) => {
      const plotId = axis === 'gamma' ? 'tau-plot-gamma' : 'tau-plot-n';
      const plotDiv = root.querySelector(`#${plotId}`);
      if (!plotDiv) return;
      tauTraceRegistry[axis].forEach((traceIndex, runId) => {
        const visible = runVisibility.get(runId) !== false;
        Plotly.restyle(plotDiv, { visible: visible ? true : 'legendonly' }, [traceIndex]);
      });
    });
  }

  function applyRunVisibilityToDual() {
    if (!window.Plotly) return;
    dualTraceRegistry.forEach((runMap, plotId) => {
      const plotDiv = root.querySelector(`#${plotId}`);
      if (!plotDiv) return;
      runMap.forEach((traceIndex, runId) => {
        const visible = runVisibility.get(runId) !== false;
        Plotly.restyle(plotDiv, { visible: visible ? true : 'legendonly' }, [traceIndex]);
      });
    });
  }

  function toggleRunVisibility(runId) {
    const current = runVisibility.get(runId) !== false;
    runVisibility.set(runId, !current);
    setRunToggleButton(runId);
    applyRunVisibilityToTau();
    applyRunVisibilityToDual();
  }

  function tauColumn(grid, columnIdx) {
    return grid.map((row) => (Array.isArray(row) ? row[columnIdx] ?? null : null));
  }

  function tauRow(grid, rowIdx) {
    if (!Array.isArray(grid[rowIdx])) return [];
    return grid[rowIdx].map((value) => value ?? null);
  }

  function updateTauLabels() {
    if (tauGammaValue) {
      tauGammaValue.textContent = gammaValues[gammaIdx] !== undefined ? String(gammaValues[gammaIdx]) : '';
    }
    if (tauNValue) {
      tauNValue.textContent = nValues[nIdx] !== undefined ? String(nValues[nIdx]) : '';
    }
  }

  function buildPatternHintText(primaryAxisName) {
    const entries = Object.entries(patternParams).filter(([, value]) => Number.isFinite(Number(value)));
    const paramsHint = entries.length
      ? entries.map(([name, value]) => `${name}=${Number(value).toString()}`).join(', ')
      : 'none';
    const parts = [`Available parameters: ${paramsHint}.`];
    if (patternInvalidParams.length) parts.push(`Ignored (non-scalar): ${patternInvalidParams.join(', ')}.`);
    if (patternConflictParams.length) parts.push(`Conflicts: ${patternConflictParams.join(', ')}.`);
    if (primaryAxisName === 'gamma') parts.push('Variables: x, gamma, n (current slider value).');
    else parts.push('Variables: x, n, gamma (current slider value).');
    return parts.join(' ');
  }

  function tauPatternOverlay(axis, exprRaw) {
    const expr = normalizeExpression(exprRaw || '');
    if (!expr) return { yValues: null, error: '' };
    if (!window.math) return { yValues: null, error: 'Math library loading...' };
    let compiled;
    try {
      compiled = window.math.compile(expr);
    } catch (err) {
      return { yValues: null, error: `Invalid expression: ${err.message || err}` };
    }
    const xValues = axis === 'gamma' ? gammaValues : nValues;
    const yValues = [];
    let hasFinite = false;
    for (let i = 0; i < xValues.length; i += 1) {
      const x = Number(xValues[i]);
      if (!Number.isFinite(x)) {
        yValues.push(null);
        continue;
      }
      const scope = {};
      Object.entries(patternParams).forEach(([name, value]) => {
        const numeric = Number(value);
        if (Number.isFinite(numeric)) scope[name] = numeric;
      });
      if (axis === 'gamma') {
        scope.x = x;
        scope.gamma = x;
        scope.n = Number(nValues[nIdx]);
      } else {
        scope.x = x;
        scope.n = x;
        scope.gamma = Number(gammaValues[gammaIdx]);
      }
      try {
        const value = compiled.evaluate(scope);
        const numeric = typeof value === 'number' ? value : Number(value);
        if (Number.isFinite(numeric)) {
          yValues.push(numeric);
          hasFinite = true;
        } else {
          yValues.push(null);
        }
      } catch (err) {
        return { yValues: null, error: `Invalid expression: ${err.message || err}` };
      }
    }
    if (!hasFinite) return { yValues, error: 'Expression did not produce finite values.' };
    return { yValues, error: '' };
  }

  function renderTauPlots() {
    if (!window.Plotly) return;
    const gammaPlot = root.querySelector('#tau-plot-gamma');
    const nPlot = root.querySelector('#tau-plot-n');
    if (!gammaPlot || !nPlot) return;

    tauTraceRegistry.gamma.clear();
    tauTraceRegistry.n.clear();

    const gammaPattern = tauPatternGammaInput ? tauPatternGammaInput.value : '';
    const nPattern = tauPatternNInput ? tauPatternNInput.value : '';
    const gammaOverlay = tauPatternOverlay('gamma', gammaPattern);
    const nOverlay = tauPatternOverlay('n', nPattern);
    if (tauPatternGammaError) tauPatternGammaError.textContent = gammaOverlay.error;
    if (tauPatternNError) tauPatternNError.textContent = nOverlay.error;
    if (tauPatternGammaInput) tauPatternGammaInput.classList.toggle('is-error', Boolean(gammaOverlay.error));
    if (tauPatternNInput) tauPatternNInput.classList.toggle('is-error', Boolean(nOverlay.error));

    const currentTau = Array.isArray(tauGrid[gammaIdx]) ? tauGrid[gammaIdx][nIdx] : null;
    const gammaTraces = [
      {
        x: gammaValues,
        y: tauColumn(tauGrid, nIdx),
        mode: 'lines',
        line: { color: '#9aa0a6', width: 2 },
        hovertemplate: 'gamma=%{x:.3f}<br>tau=%{y:.3e}<extra></extra>',
        showlegend: false,
      },
    ];
    if (currentTau !== null && currentTau !== undefined) {
      gammaTraces.push({
        x: [gammaValues[gammaIdx]],
        y: [currentTau],
        mode: 'markers',
        marker: { size: 12 },
        hovertemplate: 'gamma=%{x:.3f}<br>tau=%{y:.3e}<extra></extra>',
        showlegend: false,
      });
    }

    const nTraces = [
      {
        x: nValues,
        y: tauRow(tauGrid, gammaIdx),
        mode: 'lines',
        line: { color: '#9aa0a6', width: 2 },
        hovertemplate: 'n=%{x:.3f}<br>tau=%{y:.3e}<extra></extra>',
        showlegend: false,
      },
    ];
    if (currentTau !== null && currentTau !== undefined) {
      nTraces.push({
        x: [nValues[nIdx]],
        y: [currentTau],
        mode: 'markers',
        marker: { size: 12 },
        hovertemplate: 'n=%{x:.3f}<br>tau=%{y:.3e}<extra></extra>',
        showlegend: false,
      });
    }

    dualRuns.forEach((run) => {
      const grid = run.tau_grid || [];
      const gammaTraceIdx = gammaTraces.length;
      gammaTraces.push({
        x: gammaValues,
        y: tauColumn(grid, nIdx),
        mode: 'lines',
        line: { color: run.color || '#f97316', width: 2 },
        hovertemplate: `gamma=%{x:.3f}<br>${escapeHtml(run.name || run.id || 'Run')}=%{y:.3e}<extra></extra>`,
        showlegend: false,
        visible: runVisibility.get(run.id) !== false ? true : 'legendonly',
      });
      tauTraceRegistry.gamma.set(run.id, gammaTraceIdx);

      const nTraceIdx = nTraces.length;
      nTraces.push({
        x: nValues,
        y: tauRow(grid, gammaIdx),
        mode: 'lines',
        line: { color: run.color || '#f97316', width: 2 },
        hovertemplate: `n=%{x:.3f}<br>${escapeHtml(run.name || run.id || 'Run')}=%{y:.3e}<extra></extra>`,
        showlegend: false,
        visible: runVisibility.get(run.id) !== false ? true : 'legendonly',
      });
      tauTraceRegistry.n.set(run.id, nTraceIdx);
    });

    // Put hypothesis overlays on top of all tau traces.
    if (gammaOverlay.yValues && !gammaOverlay.error) {
      gammaTraces.push({
        x: gammaValues,
        y: gammaOverlay.yValues,
        mode: 'lines',
        line: { color: '#ff8a00', width: 2 },
        hovertemplate: 'gamma=%{x:.3f}<br>pattern=%{y:.3e}<extra></extra>',
        showlegend: false,
      });
    }
    if (nOverlay.yValues && !nOverlay.error) {
      nTraces.push({
        x: nValues,
        y: nOverlay.yValues,
        mode: 'lines',
        line: { color: '#ff8a00', width: 2 },
        hovertemplate: 'n=%{x:.3f}<br>pattern=%{y:.3e}<extra></extra>',
        showlegend: false,
      });
    }

    Plotly.newPlot(
      'tau-plot-gamma',
      gammaTraces,
      { autosize: true, title: { text: 'Tau vs gamma' }, xaxis: { title: 'gamma' }, yaxis: { title: 'tau' }, margin: { t: 40, l: 45, r: 10, b: 35 } },
      { displayModeBar: false, responsive: true },
    );
    Plotly.newPlot(
      'tau-plot-n',
      nTraces,
      { autosize: true, title: { text: 'Tau vs n' }, xaxis: { title: 'n' }, yaxis: { title: 'tau' }, margin: { t: 40, l: 45, r: 10, b: 35 } },
      { displayModeBar: false, responsive: true },
    );
    updateTauLabels();
  }

  function emitCursorIfChanged() {
    const currentKey = `${gammaIdx}:${nIdx}`;
    if (currentKey === lastCursorEventKey) return;
    lastCursorEventKey = currentKey;
    setTriggerValue('cursor', {
      request_id: Date.now(),
      gamma_idx: gammaIdx,
      n_idx: nIdx,
      pattern_gamma: tauPatternGammaInput ? tauPatternGammaInput.value : '',
      pattern_n: tauPatternNInput ? tauPatternNInput.value : '',
    });
  }

  function clearDualPlots() {
    const gammaPlot = root.querySelector('#dual-plot-gamma');
    const nPlot = root.querySelector('#dual-plot-n');
    if (gammaPlot) gammaPlot.innerHTML = '';
    if (nPlot) nPlot.innerHTML = '';
    selectedPlotCards.gamma.clear();
    selectedPlotCards.n.clear();
    overlayState.clear();
    dualTraceRegistry.clear();
    updateOverlayButtonVisibility(false);
    updateRemoveButtons();
  }

  function togglePlotCardSelection(card, setRef) {
    const seriesId = card.getAttribute('data-series-id');
    if (!seriesId) return;
    const select = !setRef.has(seriesId);
    const apply = (targetSet) => {
      if (select) targetSet.add(seriesId);
      else targetSet.delete(seriesId);
    };
    apply(selectedPlotCards.gamma);
    apply(selectedPlotCards.n);
    root.querySelectorAll(`.dual-plot-card[data-series-id="${seriesId}"]`).forEach((el) => {
      el.classList.toggle('is-selected', select);
    });
    updateRemoveButtons();
  }

  function removeSelectedSeries(seriesIds) {
    seriesIds.forEach((seriesId) => {
      if (plotSelectionMap.has(seriesId)) {
        plotSelectionMap.delete(seriesId);
      }
    });
    selectedPlotCards.gamma.clear();
    selectedPlotCards.n.clear();
    updateDeactivatedList();
    updateClearButton();
    clearDualPlots();
    plotSelected();
    updateRemoveButtons();
  }

  function applyZeroFilter() {
    root.querySelectorAll('.dual-button').forEach((btn) => {
      const seriesId = btn.getAttribute('data-series-id');
      const series = seriesData[seriesId];
      const section = btn.getAttribute('data-section');
      const isAllZero = series && ((section === 'gamma' && series.all_zero_gamma) || (section === 'n' && series.all_zero_n));
      btn.classList.toggle('is-all-zero', Boolean(isAllZero));
      if (hideZero && isAllZero) {
        if (plotSelectionMap.has(seriesId)) {
          plotSelectionMap.delete(seriesId);
        }
        btn.classList.add('is-hidden');
      } else {
        btn.classList.remove('is-hidden');
      }
    });
    updateDeactivatedList();
    syncDualButtonState();
    updateClearButton();
  }

  function plotSelected() {
    const gammaGrid = root.querySelector('#dual-plot-gamma');
    const nGrid = root.querySelector('#dual-plot-n');
    if (!gammaGrid || !nGrid) return;
    const seriesIds = Array.from(plotSelectionMap.keys());
    gammaGrid.innerHTML = '';
    nGrid.innerHTML = '';
    selectedPlotCards.gamma.clear();
    selectedPlotCards.n.clear();
    overlayState.clear();
    dualTraceRegistry.clear();
    updateRemoveButtons();

    if (!seriesIds.length) {
      gammaGrid.textContent = 'Select dual values to plot.';
      nGrid.textContent = 'Select dual values to plot.';
      updateOverlayButtonVisibility(false);
      return;
    }

    updateOverlayButtonVisibility(true);
    const grouped = new Map();
    seriesIds.forEach((seriesId) => {
      const series = seriesData[seriesId];
      if (!series) return;
      const constraint = series.constraint || 'Other';
      if (!grouped.has(constraint)) grouped.set(constraint, []);
      grouped.get(constraint).push(seriesId);
    });

    grouped.forEach((ids, constraint) => {
      const constraintLabel = escapeHtml(constraint);
      let gammaConstraintGrid = null;
      let nConstraintGrid = null;

      function ensureGammaSection() {
        if (gammaConstraintGrid) return gammaConstraintGrid;
        const gammaSection = document.createElement('div');
        gammaSection.className = 'dual-plot-constraint';
        gammaSection.innerHTML = `<div class="dual-plot-constraint-title">${constraintLabel}</div><div class="dual-plot-cards" data-constraint="${constraintLabel}"></div>`;
        gammaGrid.appendChild(gammaSection);
        gammaConstraintGrid = gammaSection.querySelector('.dual-plot-cards');
        return gammaConstraintGrid;
      }

      function ensureNSection() {
        if (nConstraintGrid) return nConstraintGrid;
        const nSection = document.createElement('div');
        nSection.className = 'dual-plot-constraint';
        nSection.innerHTML = `<div class="dual-plot-constraint-title">${constraintLabel}</div><div class="dual-plot-cards" data-constraint="${constraintLabel}"></div>`;
        nGrid.appendChild(nSection);
        nConstraintGrid = nSection.querySelector('.dual-plot-cards');
        return nConstraintGrid;
      }

      ids.forEach((seriesId) => {
        const series = seriesData[seriesId];
        if (!series) return;
        const gammaCount = series.gamma_dual.filter((value) => value !== null && Number.isFinite(value)).length;
        const nCount = series.n_dual.filter((value) => value !== null && Number.isFinite(value)).length;
        const hasGammaData = gammaCount > 0;
        const hasNData = nCount > 0;
        if (!hasGammaData && !hasNData) return;

        const safeId = sanitizeId(seriesId);
        const safeKey = `${safeId}-${Math.random().toString(36).slice(2, 8)}`;
        const gammaPlotId = `gamma-${safeKey}`;
        const nPlotId = `n-${safeKey}`;

        if (hasGammaData) {
          const gammaCard = document.createElement('div');
          gammaCard.className = 'dual-plot-card';
          gammaCard.setAttribute('data-series-id', seriesId);
          gammaCard.innerHTML = `<div class="dual-plot-card-title">${formatDualLabel(series.label)}</div><div id="${gammaPlotId}" class="dual-plot-chart"></div><input class="dual-overlay-input" type="text" placeholder="${randomOverlayPlaceholder()}" data-series-id="${escapeHtml(seriesId)}" data-axis="gamma" data-plot-id="${gammaPlotId}">`;
          ensureGammaSection().appendChild(gammaCard);
          gammaCard.addEventListener('click', () => togglePlotCardSelection(gammaCard, selectedPlotCards.gamma));

          const gammaTraces = [];
          gammaTraces.push({
            x: series.gamma_values,
            y: series.gamma_dual,
            mode: gammaCount <= 1 ? 'markers' : 'lines',
            name: 'Baseline',
            line: { color: '#9aa0a6', width: 2 },
            hovertemplate: 'gamma=%{x:.3f}<br>Baseline=%{y:.3e}<extra></extra>',
            showlegend: false,
          });
          const gammaRunMap = new Map();
          dualRuns.forEach((run) => {
            const runSeries = (run.series_data || {})[seriesId];
            if (!runSeries) return;
            const runGammaCount = runSeries.gamma_dual.filter((value) => value !== null && Number.isFinite(value)).length;
            if (!runGammaCount) return;
            const traceIndex = gammaTraces.length;
            gammaTraces.push({
              x: runSeries.gamma_values,
              y: runSeries.gamma_dual,
              mode: runGammaCount <= 1 ? 'markers' : 'lines',
              name: run.name || run.id || 'Run',
              line: { color: run.color || '#f97316', width: 2 },
              hovertemplate: `gamma=%{x:.3f}<br>${escapeHtml(run.name || run.id || 'Run')}=%{y:.3e}<extra></extra>`,
              showlegend: false,
              visible: runVisibility.get(run.id) !== false ? true : 'legendonly',
            });
            gammaRunMap.set(run.id, traceIndex);
          });
          dualTraceRegistry.set(gammaPlotId, gammaRunMap);
          Plotly.newPlot(
            gammaPlotId,
            gammaTraces,
            { autosize: true, xaxis: { title: '', tickfont: { size: 9 } }, yaxis: { title: '', tickfont: { size: 9 } }, margin: { t: 10, l: 30, r: 10, b: 15 } },
            { displayModeBar: false, responsive: true },
          );

          const gammaOverlayInput = gammaCard.querySelector('.dual-overlay-input');
          if (gammaOverlayInput) {
            gammaOverlayInput.addEventListener('focus', () => gammaOverlayInput.setAttribute('placeholder', ''));
            gammaOverlayInput.addEventListener('click', (event) => event.stopPropagation());
            gammaOverlayInput.addEventListener('input', (event) => {
              const key = gammaOverlayInput.getAttribute('data-plot-id');
              if (overlayDebounce.has(key)) clearTimeout(overlayDebounce.get(key));
              overlayDebounce.set(key, setTimeout(() => ensureMath(() => handleOverlayInput(event)), 250));
            });
          }
        }

        if (hasNData) {
          const nCard = document.createElement('div');
          nCard.className = 'dual-plot-card';
          nCard.setAttribute('data-series-id', seriesId);
          nCard.innerHTML = `<div class="dual-plot-card-title">${formatDualLabel(series.label)}</div><div id="${nPlotId}" class="dual-plot-chart"></div><input class="dual-overlay-input" type="text" placeholder="${randomOverlayPlaceholder()}" data-series-id="${escapeHtml(seriesId)}" data-axis="n" data-plot-id="${nPlotId}">`;
          ensureNSection().appendChild(nCard);
          nCard.addEventListener('click', () => togglePlotCardSelection(nCard, selectedPlotCards.n));

          const nTraces = [];
          nTraces.push({
            x: series.n_values,
            y: series.n_dual,
            mode: nCount <= 1 ? 'markers' : 'lines',
            name: 'Baseline',
            line: { color: '#9aa0a6', width: 2 },
            hovertemplate: 'n=%{x:.3f}<br>Baseline=%{y:.3e}<extra></extra>',
            showlegend: false,
          });
          const nRunMap = new Map();
          dualRuns.forEach((run) => {
            const runSeries = (run.series_data || {})[seriesId];
            if (!runSeries) return;
            const runNCount = runSeries.n_dual.filter((value) => value !== null && Number.isFinite(value)).length;
            if (!runNCount) return;
            const traceIndex = nTraces.length;
            nTraces.push({
              x: runSeries.n_values,
              y: runSeries.n_dual,
              mode: runNCount <= 1 ? 'markers' : 'lines',
              name: run.name || run.id || 'Run',
              line: { color: run.color || '#f97316', width: 2 },
              hovertemplate: `n=%{x:.3f}<br>${escapeHtml(run.name || run.id || 'Run')}=%{y:.3e}<extra></extra>`,
              showlegend: false,
              visible: runVisibility.get(run.id) !== false ? true : 'legendonly',
            });
            nRunMap.set(run.id, traceIndex);
          });
          dualTraceRegistry.set(nPlotId, nRunMap);
          Plotly.newPlot(
            nPlotId,
            nTraces,
            { autosize: true, xaxis: { title: '', tickfont: { size: 9 } }, yaxis: { title: '', tickfont: { size: 9 } }, margin: { t: 10, l: 30, r: 10, b: 15 } },
            { displayModeBar: false, responsive: true },
          );

          const nOverlayInput = nCard.querySelector('.dual-overlay-input');
          if (nOverlayInput) {
            nOverlayInput.addEventListener('focus', () => nOverlayInput.setAttribute('placeholder', ''));
            nOverlayInput.addEventListener('click', (event) => event.stopPropagation());
            nOverlayInput.addEventListener('input', (event) => {
              const key = nOverlayInput.getAttribute('data-plot-id');
              if (overlayDebounce.has(key)) clearTimeout(overlayDebounce.get(key));
              overlayDebounce.set(key, setTimeout(() => ensureMath(() => handleOverlayInput(event)), 250));
            });
          }
        }
      });
    });
    if (!gammaGrid.childElementCount) gammaGrid.textContent = 'No gamma-series data for selected dual values.';
    if (!nGrid.childElementCount) nGrid.textContent = 'No n-series data for selected dual values.';
  }

  root.addEventListener('click', (event) => {
    const target = event.target;
    if (!(target instanceof Element)) return;
    const button = target.closest('button');
    if (!button) return;

    if (button.classList.contains('run-toggle')) {
      event.preventDefault();
      const runId = button.getAttribute('data-run-id');
      if (runId) {
        toggleRunVisibility(runId);
      }
      return;
    }
    if (button.classList.contains('run-remove')) {
      event.preventDefault();
      const runId = button.getAttribute('data-run-id');
      if (!runId) return;
      setTriggerValue('remove_run', {
        request_id: Date.now(),
        run_id: runId,
        pattern_gamma: tauPatternGammaInput ? tauPatternGammaInput.value : '',
        pattern_n: tauPatternNInput ? tauPatternNInput.value : '',
      });
      return;
    }

    if (button.classList.contains('dual-button')) {
      event.preventDefault();
      const seriesId = button.getAttribute('data-series-id');
      const label = button.getAttribute('data-label');
      if (!seriesId) return;
      if (actionTab === 'recompute') {
        if (!deactivatedForRecompute.has(seriesId)) {
          deactivatedForRecompute.set(seriesId, label || '');
        }
        syncDualButtonState();
        updateDeactivatedList();
      } else {
        if (plotSelectionMap.has(seriesId)) {
          plotSelectionMap.delete(seriesId);
        } else {
          plotSelectionMap.set(seriesId, label || '');
        }
        syncDualButtonState();
        updateClearButton();
      }
      return;
    }

    if (button.classList.contains('dual-reactivate-button')) {
      event.preventDefault();
      const seriesId = button.getAttribute('data-series-id');
      if (!seriesId) return;
      deactivatedForRecompute.delete(seriesId);
      syncDualButtonState();
      updateDeactivatedList();
      return;
    }
    if (button.id === 'dual-activate-all') {
      event.preventDefault();
      deactivatedForRecompute.clear();
      syncDualButtonState();
      updateDeactivatedList();
      return;
    }
    if (button.id === 'dual-deactivate-all') {
      event.preventDefault();
      const buttons = visibleButtons();
      buttons.forEach((btn) => {
        const seriesId = btn.getAttribute('data-series-id');
        const label = btn.getAttribute('data-label');
        if (!seriesId) return;
        if (!deactivatedForRecompute.has(seriesId)) {
          deactivatedForRecompute.set(seriesId, label || '');
        }
      });
      syncDualButtonState();
      updateDeactivatedList();
      return;
    }

    if (button.id === 'dual-toggle-zero') {
      event.preventDefault();
      hideZero = !hideZero;
      button.classList.toggle('is-active', hideZero);
      button.textContent = hideZero ? 'Show all-zero duals' : 'Hide all-zero duals';
      applyZeroFilter();
      updateClearButton();
      return;
    }

    if (button.id === 'dual-plot') {
      event.preventDefault();
      ensurePlotly(plotSelected);
      return;
    }

    if (button.id === 'dual-overlay') {
      event.preventDefault();
      const wrapperEl = root.querySelector('.dual-wrapper');
      ensureMath(() => {
        root.querySelectorAll('.dual-overlay-input').forEach((input) => input.setAttribute('placeholder', randomOverlayPlaceholder()));
        if (!wrapperEl) return;
        wrapperEl.classList.toggle('dual-show-overlay');
        button.classList.toggle('is-active', wrapperEl.classList.contains('dual-show-overlay'));
      });
      return;
    }

    if (button.id === 'dual-clear') {
      event.preventDefault();
      const buttons = visibleButtons();
      if (!buttons.length) return;
      const visibleSeries = new Set(buttons.map((btn) => btn.getAttribute('data-series-id')));
      const allSelected = Array.from(visibleSeries).every((id) => plotSelectionMap.has(id));
      if (allSelected) {
        visibleSeries.forEach((id) => plotSelectionMap.delete(id));
      } else {
        buttons.forEach((btn) => {
          const seriesId = btn.getAttribute('data-series-id');
          const label = btn.getAttribute('data-label');
          if (seriesId && !plotSelectionMap.has(seriesId)) plotSelectionMap.set(seriesId, label || '');
        });
      }
      syncDualButtonState();
      updateClearButton();
      clearDualPlots();
      return;
    }

    if (button.id === 'dual-remove-gamma' || button.id === 'dual-remove-n') {
      event.preventDefault();
      removeSelectedSeries(new Set([...selectedPlotCards.gamma, ...selectedPlotCards.n]));
      return;
    }

    if (button.id === 'dual-recompute') {
      event.preventDefault();
      if (!deactivatedForRecompute.size) return;
      const allSeriesIds = Object.keys(seriesData);
      const activeSeriesIds = allSeriesIds.filter((seriesId) => !deactivatedForRecompute.has(seriesId));
      setTriggerValue('recompute', {
        request_id: Date.now(),
        selected_series_ids: activeSeriesIds,
        deactivated_series_ids: Array.from(deactivatedForRecompute.keys()),
        deactivated_labels: Array.from(deactivatedForRecompute.values()),
        pattern_gamma: tauPatternGammaInput ? tauPatternGammaInput.value : '',
        pattern_n: tauPatternNInput ? tauPatternNInput.value : '',
      });
    }
  });

  if (tauGammaSlider) {
    tauGammaSlider.addEventListener('input', () => {
      gammaIdx = Number(tauGammaSlider.value);
      ensurePlotly(renderTauPlots);
    });
    tauGammaSlider.addEventListener('change', () => {
      gammaIdx = Number(tauGammaSlider.value);
      emitCursorIfChanged();
    });
  }
  if (tauNSlider) {
    tauNSlider.addEventListener('input', () => {
      nIdx = Number(tauNSlider.value);
      ensurePlotly(renderTauPlots);
    });
    tauNSlider.addEventListener('change', () => {
      nIdx = Number(tauNSlider.value);
      emitCursorIfChanged();
    });
  }
  if (tauPatternGammaInput) {
    tauPatternGammaInput.addEventListener('input', () => {
      ensureMath(() => ensurePlotly(renderTauPlots));
    });
  }
  if (tauPatternNInput) {
    tauPatternNInput.addEventListener('input', () => {
      ensureMath(() => ensurePlotly(renderTauPlots));
    });
  }
  if (metricSelect) {
    metricSelect.addEventListener('change', () => {
      setTriggerValue('metric', {
        request_id: Date.now(),
        metric: metricSelect.value,
      });
    });
  }
  if (tabPlotBtn) {
    tabPlotBtn.addEventListener('click', () => setActionTab('plot'));
  }
  if (tabRecomputeBtn) {
    tabRecomputeBtn.addEventListener('click', () => setActionTab('recompute'));
  }

  if (tauPatternGammaHint) tauPatternGammaHint.textContent = buildPatternHintText('gamma');
  if (tauPatternNHint) tauPatternNHint.textContent = buildPatternHintText('n');

  toggleBtn.classList.toggle('is-active', hideZero);
  toggleBtn.textContent = hideZero ? 'Show all-zero duals' : 'Hide all-zero duals';

  const selectedSeries = (data && data.selected_series_ids) || [];
  selectedSeries.forEach((seriesId) => {
    const series = seriesData[seriesId];
    if (!series) return;
    plotSelectionMap.set(seriesId, series.label);
  });

  applyZeroFilter();
  setActionTab('plot');
  updateClearButton();
  updateRemoveButtons();
  updateDeactivatedList();
  ensureMath(() => ensurePlotly(renderTauPlots));
  if (plotSelectionMap.size) {
    ensurePlotly(plotSelected);
  }
}
