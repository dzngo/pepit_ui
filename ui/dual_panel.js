export default function(component) {
  const { data, parentElement, setTriggerValue } = component;

  const seriesDataByParam = (data && data.series_data_by_param) || ((data && data.tau_payload && data.tau_payload.series_data_by_param) || {});
  const rawDualRuns = (data && data.dual_runs) || [];
  const tauPayload = (data && data.tau_payload) || {};
  const sectionsHtmlByParam = (tauPayload && tauPayload.sections_html_by_param) || {};
  const plotTitlesByParam = (tauPayload && tauPayload.plot_titles_by_param) || {};
  const metricLabels = (data && data.metric_labels) || {};
  const metricKeys = Object.keys(metricLabels);
  const currentMetric = (data && data.metric) || 'non_zero_pct_with_none';

  const plotSelectionMap = new Map();
  const deactivatedForRecompute = new Map();
  const selectedPlotCards = new Map();
  const overlayState = new Map();
  const overlayDebounce = new Map();
  const dualTraceRegistry = new Map();
  const tauTraceRegistry = new Map();
  let hideZero = true;
  let actionTab = 'plot';

  const tauSeriesByParam = (tauPayload && tauPayload.tau_series_by_param) || {};
  const paramOrder = Array.isArray(tauPayload.param_order) ? tauPayload.param_order : [];
  const fallbackParams = Object.keys((tauPayload && tauPayload.param_values_by_name) || {});
  const dualParams = (paramOrder.length ? paramOrder : fallbackParams).filter((name, idx, arr) => arr.indexOf(name) === idx);
  dualParams.forEach((name) => selectedPlotCards.set(name, new Set()));
  const paramValuesByName = (tauPayload && tauPayload.param_values_by_name) || {};
  const cursorIndicesByParamPayload = (tauPayload && tauPayload.cursor_indices_by_param) || {};
  const localCursorIndicesByAxisPayload = (tauPayload && tauPayload.local_cursor_indices_by_axis) || {};
  const patternsByParamPayload = (tauPayload && tauPayload.patterns_by_param) || {};
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

  function dualSectionsHtml() {
    if (!dualParams.length) return '';
    return dualParams.map((param) => {
      const sectionHtml = sectionsHtmlByParam[param] || '';
      return `<div class="dual-panel"><div class="dual-section" data-param-section="${escapeHtml(param)}">${sectionHtml}</div></div>`;
    }).join('');
  }

  function dualPlotsHtml() {
    if (!dualParams.length) return '';
    return dualParams.map((param) => {
      const safeParam = sanitizeId(param);
      const defaultTitle = `Dual value vs ${param}`;
      const title = plotTitlesByParam[param] || defaultTitle;
      return (
        `<div class="dual-panel">` +
          `<div class="dual-plot-section">` +
            `<div class="dual-plot-title">${escapeHtml(title || defaultTitle)}</div>` +
            `<div id="dual-plot-${safeParam}" class="dual-plot-sections" data-param="${escapeHtml(param)}"></div>` +
            `<div class="dual-plot-actions dual-plot-actions-inline">` +
              `<button type="button" class="dual-remove-button" data-param="${escapeHtml(param)}" id="dual-remove-${safeParam}" style="display:none;">Remove selected dual values</button>` +
            `</div>` +
          `</div>` +
        `</div>`
      );
    }).join('');
  }

  function clampIdx(value, maxIdx) {
    return Math.min(Math.max(Number(value || 0), 0), Math.max(maxIdx, 0));
  }

  function currentCursorIndicesByParam() {
    const indices = {};
    dualParams.forEach((name) => {
      const values = Array.isArray(paramValuesByName[name]) ? paramValuesByName[name] : [];
      indices[name] = clampIdx(cursorIndicesByParamState[name] ?? 0, values.length - 1);
    });
    return indices;
  }

  function currentLocalCursorIndicesByAxis() {
    const byAxis = {};
    const cursor = currentCursorIndicesByParam();
    dualParams.forEach((axis) => {
      const axisLocals = localCursorIndicesByAxisState[axis] || {};
      const nextAxis = {};
      dualParams.forEach((name) => {
        if (name === axis) return;
        const values = Array.isArray(paramValuesByName[name]) ? paramValuesByName[name] : [];
        nextAxis[name] = clampIdx(axisLocals[name] ?? cursor[name] ?? 0, values.length - 1);
      });
      byAxis[axis] = nextAxis;
    });
    return byAxis;
  }

  function getLocalIndex(axisParam, paramName) {
    const localByAxis = currentLocalCursorIndicesByAxis();
    const axisLocals = localByAxis[axisParam] || {};
    if (axisLocals[paramName] !== undefined) return axisLocals[paramName];
    const cursor = currentCursorIndicesByParam();
    return cursor[paramName] ?? 0;
  }

  function currentPatternsByParam() {
    const patterns = { ...patternsByParamPayload };
    tauPatternInputsByParam.forEach((inputEl, param) => {
      patterns[param] = inputEl ? String(inputEl.value || '') : String(patterns[param] || '');
    });
    return patterns;
  }

  function extraLocalSlidersHtml(axisParam) {
    const skip = new Set([axisParam]);
    const items = dualParams.filter((name) => !skip.has(name));
    if (!items.length) return '';
    return items.map((name) => {
      const values = Array.isArray(paramValuesByName[name]) ? paramValuesByName[name] : [];
      const maxIdx = Math.max(values.length - 1, 0);
      const axisPayload = localCursorIndicesByAxisPayload[axisParam] || {};
      const defaultIdx = clampIdx(axisPayload[name] ?? cursorIndicesByParamPayload[name] ?? 0, maxIdx);
      return (
        `<div style="display:flex;align-items:center;gap:8px;">` +
          `<strong>Local ${escapeHtml(name)}</strong>` +
          `<input type="range" class="tau-local-slider" data-axis-param="${escapeHtml(axisParam)}" data-param="${escapeHtml(name)}" min="0" max="${maxIdx}" step="1" value="${defaultIdx}" style="flex:1;" />` +
          `<span class="tau-local-value" data-axis-param="${escapeHtml(axisParam)}" data-param="${escapeHtml(name)}"></span>` +
        `</div>`
      );
    }).join('');
  }

  function tauPanelsHtml() {
    if (!dualParams.length) return '';
    return dualParams.map((axisParam) => {
      const safeAxis = sanitizeId(axisParam);
      const axisValues = Array.isArray(paramValuesByName[axisParam]) ? paramValuesByName[axisParam] : [];
      const globalIdx = clampIdx(cursorIndicesByParamPayload[axisParam] ?? 0, axisValues.length - 1);
      const patternValue = String(patternsByParamPayload[axisParam] ?? '');
      return (
        `<div class="dual-panel">` +
          `<div class="dual-plot-section">` +
            `${extraLocalSlidersHtml(axisParam)}` +
            `<div id="tau-plot-${safeAxis}" class="dual-plot-chart" style="min-height:320px;"></div>` +
            `<div class="tau-global-control">` +
              `<div style="display:flex;align-items:center;gap:8px;">` +
                `<strong>Global ${escapeHtml(axisParam)}</strong>` +
                `<input type="range" class="tau-global-slider" data-param="${escapeHtml(axisParam)}" min="0" max="${Math.max(axisValues.length - 1, 0)}" step="1" value="${globalIdx}" style="flex:1;" />` +
                `<span class="tau-global-value" data-param="${escapeHtml(axisParam)}"></span>` +
              `</div>` +
            `</div>` +
            `<input class="dual-overlay-input tau-pattern-input" data-param="${escapeHtml(axisParam)}" type="text" value="${escapeHtml(patternValue)}" placeholder="${randomOverlayPlaceholder()}" style="display:block;">` +
            `<div class="tau-pattern-hint" data-param="${escapeHtml(axisParam)}" style="font-size:12px;color:#666;"></div>` +
            `<div class="tau-pattern-error" data-param="${escapeHtml(axisParam)}" style="font-size:12px;color:#cb0000;min-height:16px;"></div>` +
          `</div>` +
        `</div>`
      );
    }).join('');
  }

  parentElement.innerHTML = `
    <style>${data.css || ''}</style>
    <div class="dual-wrapper">
      <div class="dual-section-title">Worst-case guarantee</div>
      <div class="tau-panels-grid">
        ${tauPanelsHtml()}
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
      ${dualSectionsHtml()}
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
      ${dualPlotsHtml()}
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
  const removeButtonsByParam = new Map(dualParams.map((param) => [param, root.querySelector(`#dual-remove-${sanitizeId(param)}`)]));
  const recomputeBtn = root.querySelector('#dual-recompute');
  const activateAllBtn = root.querySelector('#dual-activate-all');
  const deactivateAllBtn = root.querySelector('#dual-deactivate-all');
  const tauGlobalSlidersByParam = new Map();
  const tauGlobalValuesByParam = new Map();
  const tauLocalSliders = Array.from(root.querySelectorAll('.tau-local-slider'));
  const tauLocalValues = Array.from(root.querySelectorAll('.tau-local-value'));
  const tauPatternInputsByParam = new Map();
  const tauPatternHintsByParam = new Map();
  const tauPatternErrorsByParam = new Map();
  dualParams.forEach((param) => {
    tauGlobalSlidersByParam.set(param, root.querySelector(`.tau-global-slider[data-param="${param}"]`));
    tauGlobalValuesByParam.set(param, root.querySelector(`.tau-global-value[data-param="${param}"]`));
    tauPatternInputsByParam.set(param, root.querySelector(`.tau-pattern-input[data-param="${param}"]`));
    tauPatternHintsByParam.set(param, root.querySelector(`.tau-pattern-hint[data-param="${param}"]`));
    tauPatternErrorsByParam.set(param, root.querySelector(`.tau-pattern-error[data-param="${param}"]`));
  });
  const metricSelect = root.querySelector('#dual-ranking-metric');

  let cursorIndicesByParamState = { ...cursorIndicesByParamPayload };
  let localCursorIndicesByAxisState = {};
  dualParams.forEach((name) => {
    const values = Array.isArray(paramValuesByName[name]) ? paramValuesByName[name] : [];
    cursorIndicesByParamState[name] = clampIdx(cursorIndicesByParamState[name] ?? 0, values.length - 1);
  });
  dualParams.forEach((axis) => {
    const axisPayload = localCursorIndicesByAxisPayload[axis] || {};
    const axisState = {};
    dualParams.forEach((name) => {
      if (name === axis) return;
      const values = Array.isArray(paramValuesByName[name]) ? paramValuesByName[name] : [];
      const fallback = cursorIndicesByParamState[name] ?? 0;
      axisState[name] = clampIdx(axisPayload[name] ?? fallback, values.length - 1);
    });
    localCursorIndicesByAxisState[axis] = axisState;
  });
  tauGlobalSlidersByParam.forEach((slider, param) => {
    if (!slider) return;
    const values = Array.isArray(paramValuesByName[param]) ? paramValuesByName[param] : [];
    const idx = clampIdx(cursorIndicesByParamState[param] ?? 0, values.length - 1);
    slider.value = String(idx);
  });
  tauLocalSliders.forEach((slider) => {
    const axis = slider.getAttribute('data-axis-param');
    const param = slider.getAttribute('data-param');
    if (!axis || !param) return;
    const values = Array.isArray(paramValuesByName[param]) ? paramValuesByName[param] : [];
    const idx = clampIdx(((localCursorIndicesByAxisState[axis] || {})[param]) ?? (cursorIndicesByParamState[param] ?? 0), values.length - 1);
    slider.value = String(idx);
  });
  let lastCursorEventKey = JSON.stringify({
    cursor_by_param: currentCursorIndicesByParam(),
    local_by_axis: currentLocalCursorIndicesByAxis(),
    patterns: currentPatternsByParam(),
  });

  function visibleButtonsForPlot() {
    return Array.from(root.querySelectorAll('.dual-button:not(.is-hidden)'));
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
    const buttons = visibleButtonsForPlot();
    if (!buttons.length) {
      clearBtn.textContent = 'Select all';
      return;
    }
    const visibleSeries = new Set(buttons.map((btn) => btn.getAttribute('data-series-id')));
    const allSelected = Array.from(visibleSeries).every((id) => plotSelectionMap.has(id));
    clearBtn.textContent = allSelected ? 'Deselect all' : 'Select all';
  }

  function updateRemoveButtons() {
    const hasSelection = Array.from(selectedPlotCards.values()).some((setRef) => setRef.size);
    const display = hasSelection ? 'inline-flex' : 'none';
    removeButtonsByParam.forEach((btn) => {
      if (btn) btn.style.display = display;
    });
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
    const generic = seriesDataByParam[seriesId];
    if (!generic || !axis || !plotId) return;
    const rawExpr = normalizeExpression(input.value);
    if (!rawExpr) {
      input.classList.remove('is-error');
      updateOverlayTrace(plotId, [], [], '');
      return;
    }
    let xValues = [];
    if (generic && generic.by_param && generic.by_param[axis]) {
      xValues = generic.by_param[axis].x_values || [];
    }
    const result = buildOverlayYValues(rawExpr, xValues, (x) => {
      const scope = { x };
      Object.entries(patternParams).forEach(([name, value]) => {
        const numeric = Number(value);
        if (Number.isFinite(numeric)) scope[name] = numeric;
      });
      dualParams.forEach((param) => {
        const values = Array.isArray(paramValuesByName[param]) ? paramValuesByName[param] : [];
        const idx = clampIdx(cursorIndicesByParamState[param] ?? 0, values.length - 1);
        scope[param] = values[idx] !== undefined ? Number(values[idx]) : null;
      });
      scope[axis] = Number(x);
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
    dualParams.forEach((axis) => {
      const plotId = `tau-plot-${sanitizeId(axis)}`;
      const plotDiv = root.querySelector(`#${plotId}`);
      if (!plotDiv) return;
      const runMap = tauTraceRegistry.get(axis) || new Map();
      runMap.forEach((traceIndex, runId) => {
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

  function tauSeriesForParam(paramName, fallbackX, fallbackY, fallbackCursorIdx) {
    const payload = tauSeriesByParam[paramName];
    if (!payload || !Array.isArray(payload.x_values) || !Array.isArray(payload.y_values)) {
      return {
        xValues: fallbackX,
        yValues: fallbackY,
        cursorIdx: fallbackCursorIdx,
      };
    }
    const cursorIdx = (paramName === 'gamma' || paramName === 'n')
      ? fallbackCursorIdx
      : (Number.isFinite(Number(payload.cursor_idx)) ? Number(payload.cursor_idx) : fallbackCursorIdx);
    return {
      xValues: payload.x_values,
      yValues: payload.y_values,
      cursorIdx,
    };
  }

  function updateTauLabels() {
    tauGlobalValuesByParam.forEach((valueEl, param) => {
      if (!valueEl) return;
      const values = Array.isArray(paramValuesByName[param]) ? paramValuesByName[param] : [];
      const idx = clampIdx(cursorIndicesByParamState[param] ?? 0, values.length - 1);
      valueEl.textContent = values[idx] !== undefined ? String(values[idx]) : '';
    });
    tauLocalValues.forEach((valueEl) => {
      const axis = valueEl.getAttribute('data-axis-param');
      const param = valueEl.getAttribute('data-param');
      if (!axis || !param) return;
      const values = Array.isArray(paramValuesByName[param]) ? paramValuesByName[param] : [];
      const idx = clampIdx(getLocalIndex(axis, param), values.length - 1);
      valueEl.textContent = values[idx] !== undefined ? String(values[idx]) : '';
    });
  }

  function buildPatternHintText(primaryAxisName) {
    const entries = Object.entries(patternParams).filter(([, value]) => Number.isFinite(Number(value)));
    const paramsHint = entries.length
      ? entries.map(([name, value]) => `${name}=${Number(value).toString()}`).join(', ')
      : 'none';
    const parts = [`Available parameters: ${paramsHint}.`];
    if (patternInvalidParams.length) parts.push(`Ignored (non-scalar): ${patternInvalidParams.join(', ')}.`);
    if (patternConflictParams.length) parts.push(`Conflicts: ${patternConflictParams.join(', ')}.`);
    parts.push(`Variables: x, ${dualParams.join(', ')}. Axis '${primaryAxisName}' uses x.`);
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
    const axisValues = Array.isArray(paramValuesByName[axis]) ? paramValuesByName[axis] : [];
    const xValues = axisValues;
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
      dualParams.forEach((param) => {
        const values = Array.isArray(paramValuesByName[param]) ? paramValuesByName[param] : [];
        const idx = clampIdx(getLocalIndex(axis, param), values.length - 1);
        scope[param] = values[idx] !== undefined ? Number(values[idx]) : null;
      });
      scope.x = x;
      scope[axis] = x;
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
    dualParams.forEach((axisParam) => {
      const plotId = `tau-plot-${sanitizeId(axisParam)}`;
      const plotDiv = root.querySelector(`#${plotId}`);
      if (!plotDiv) return;

      const axisValues = Array.isArray(paramValuesByName[axisParam]) ? paramValuesByName[axisParam] : [];
      const cursorIdx = clampIdx(cursorIndicesByParamState[axisParam] ?? 0, axisValues.length - 1);
      const axisSeries = tauSeriesForParam(axisParam, axisValues, [], cursorIdx);
      const traces = [
        {
          x: axisSeries.xValues,
          y: axisSeries.yValues,
          mode: 'lines',
          line: { color: '#9aa0a6', width: 2 },
          hovertemplate: `${escapeHtml(axisParam)}=%{x:.3f}<br>tau=%{y:.3e}<extra></extra>`,
          showlegend: false,
        },
      ];
      const currentTau = axisSeries.yValues[cursorIdx] ?? null;
      if (currentTau !== null && currentTau !== undefined) {
        traces.push({
          x: [axisSeries.xValues[cursorIdx]],
          y: [currentTau],
          mode: 'markers',
          marker: { size: 12 },
          hovertemplate: `${escapeHtml(axisParam)}=%{x:.3f}<br>tau=%{y:.3e}<extra></extra>`,
          showlegend: false,
        });
      }

      const runMap = new Map();
      dualRuns.forEach((run) => {
        const runSeries = ((run.tau_series_by_param || {})[axisParam] || {});
        if (!Array.isArray(runSeries.y_values)) return;
        const traceIndex = traces.length;
        traces.push({
          x: Array.isArray(runSeries.x_values) ? runSeries.x_values : axisSeries.xValues,
          y: runSeries.y_values,
          mode: 'lines',
          line: { color: run.color || '#f97316', width: 2 },
          hovertemplate: `${escapeHtml(axisParam)}=%{x:.3f}<br>${escapeHtml(run.name || run.id || 'Run')}=%{y:.3e}<extra></extra>`,
          showlegend: false,
          visible: runVisibility.get(run.id) !== false ? true : 'legendonly',
        });
        runMap.set(run.id, traceIndex);
      });
      tauTraceRegistry.set(axisParam, runMap);

      const patternInput = tauPatternInputsByParam.get(axisParam);
      const overlay = tauPatternOverlay(axisParam, patternInput ? patternInput.value : '');
      const errorEl = tauPatternErrorsByParam.get(axisParam);
      if (errorEl) errorEl.textContent = overlay.error;
      if (patternInput) patternInput.classList.toggle('is-error', Boolean(overlay.error));
      if (overlay.yValues && !overlay.error) {
        traces.push({
          x: axisSeries.xValues,
          y: overlay.yValues,
          mode: 'lines',
          line: { color: '#ff8a00', width: 2 },
          hovertemplate: `${escapeHtml(axisParam)}=%{x:.3f}<br>pattern=%{y:.3e}<extra></extra>`,
          showlegend: false,
        });
      }

      const fixedParts = dualParams
        .filter((name) => name !== axisParam)
        .map((name) => {
          const values = Array.isArray(paramValuesByName[name]) ? paramValuesByName[name] : [];
          const idx = clampIdx(getLocalIndex(axisParam, name), values.length - 1);
          return `${name}=${values[idx] !== undefined ? values[idx] : ''}`;
        });
      const title = fixedParts.length ? `Tau vs ${axisParam} (${fixedParts.join(', ')})` : `Tau vs ${axisParam}`;
      Plotly.newPlot(
        plotId,
        traces,
        { autosize: true, title: { text: title }, xaxis: { title: axisParam }, yaxis: { title: 'tau' }, margin: { t: 40, l: 45, r: 10, b: 35 } },
        { displayModeBar: false, responsive: true },
      );
    });
    updateTauLabels();
  }

  function emitCursorIfChanged() {
    const cursorByParam = currentCursorIndicesByParam();
    const localCursorByAxis = currentLocalCursorIndicesByAxis();
    const patternsByParam = currentPatternsByParam();
    const currentKey = JSON.stringify({
      cursor_by_param: cursorByParam,
      local_by_axis: localCursorByAxis,
      patterns: patternsByParam,
    });
    if (currentKey === lastCursorEventKey) return;
    lastCursorEventKey = currentKey;
    setTriggerValue('cursor', {
      request_id: Date.now(),
      cursor_indices_by_param: cursorByParam,
      local_cursor_indices_by_axis: localCursorByAxis,
      patterns_by_param: patternsByParam,
    });
  }

  function clearDualPlots() {
    dualParams.forEach((param) => {
      const grid = root.querySelector(`#dual-plot-${sanitizeId(param)}`);
      if (grid) grid.innerHTML = '';
    });
    selectedPlotCards.forEach((setRef) => setRef.clear());
    overlayState.clear();
    dualTraceRegistry.clear();
    updateOverlayButtonVisibility(false);
    updateRemoveButtons();
  }

  function togglePlotCardSelection(card, paramName) {
    const seriesId = card.getAttribute('data-series-id');
    if (!seriesId) return;
    const targetSet = selectedPlotCards.get(paramName) || selectedPlotCards.values().next().value;
    if (!targetSet) return;
    const select = !targetSet.has(seriesId);
    const apply = (targetSet) => {
      if (select) targetSet.add(seriesId);
      else targetSet.delete(seriesId);
    };
    selectedPlotCards.forEach((setRef) => apply(setRef));
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
    selectedPlotCards.forEach((setRef) => setRef.clear());
    updateDeactivatedList();
    updateClearButton();
    clearDualPlots();
    plotSelected();
    updateRemoveButtons();
  }

  function baseSeriesForParam(seriesId, paramName) {
    const generic = (seriesDataByParam[seriesId] && seriesDataByParam[seriesId].by_param && seriesDataByParam[seriesId].by_param[paramName]) || null;
    if (generic && Array.isArray(generic.x_values) && Array.isArray(generic.y_values)) {
      return {
        x: generic.x_values,
        y: generic.y_values,
        allZero: Boolean(generic.all_zero),
      };
    }
    return null;
  }

  function runSeriesForParam(run, seriesId, paramName) {
    const generic = (run.series_data_by_param || {})[seriesId];
    const genericParam = generic && generic.by_param && generic.by_param[paramName];
    if (genericParam && Array.isArray(genericParam.x_values) && Array.isArray(genericParam.y_values)) {
      return {
        x: genericParam.x_values,
        y: genericParam.y_values,
      };
    }
    return null;
  }

  function applyZeroFilter() {
    root.querySelectorAll('.dual-button').forEach((btn) => {
      const seriesId = btn.getAttribute('data-series-id');
      const section = btn.getAttribute('data-section');
      const payload = baseSeriesForParam(seriesId, section || '');
      const isAllZero = Boolean(payload && payload.allZero);
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
    const plotGridsByParam = new Map(dualParams.map((param) => [param, root.querySelector(`#dual-plot-${sanitizeId(param)}`)]));
    const seriesIds = Array.from(plotSelectionMap.keys());
    plotGridsByParam.forEach((grid) => {
      if (grid) grid.innerHTML = '';
    });
    selectedPlotCards.forEach((setRef) => setRef.clear());
    overlayState.clear();
    dualTraceRegistry.clear();
    updateRemoveButtons();

    if (!seriesIds.length) {
      plotGridsByParam.forEach((grid) => {
        if (grid) grid.textContent = 'Select dual values to plot.';
      });
      updateOverlayButtonVisibility(false);
      return;
    }

    updateOverlayButtonVisibility(true);
    const grouped = new Map();
    seriesIds.forEach((seriesId) => {
      const generic = seriesDataByParam[seriesId];
      if (!generic) return;
      const constraint = generic.constraint || 'Other';
      if (!grouped.has(constraint)) grouped.set(constraint, []);
      grouped.get(constraint).push(seriesId);
    });

    grouped.forEach((ids, constraint) => {
      const constraintLabel = escapeHtml(constraint);
      const constraintGridByParam = new Map();
      dualParams.forEach((param) => {
        const grid = plotGridsByParam.get(param);
        if (!grid) return;
        const section = document.createElement('div');
        section.className = 'dual-plot-constraint';
        section.innerHTML = `<div class="dual-plot-constraint-title">${constraintLabel}</div><div class="dual-plot-cards" data-constraint="${constraintLabel}" data-param="${escapeHtml(param)}"></div>`;
        grid.appendChild(section);
        constraintGridByParam.set(param, section.querySelector('.dual-plot-cards'));
      });

      ids.forEach((seriesId) => {
        const generic = seriesDataByParam[seriesId];
        const label = (generic && generic.label) || seriesId;
        if (!generic) return;

        const safeId = sanitizeId(seriesId);
        const safeKey = `${safeId}-${Math.random().toString(36).slice(2, 8)}`;
        dualParams.forEach((param) => {
          const payload = baseSeriesForParam(seriesId, param);
          if (!payload || !Array.isArray(payload.y)) return;
          const count = payload.y.filter((value) => value !== null && Number.isFinite(value)).length;
          if (!count) return;
          const plotId = `${sanitizeId(param)}-${safeKey}`;
          const card = document.createElement('div');
          card.className = 'dual-plot-card';
          card.setAttribute('data-series-id', seriesId);
          card.innerHTML = `<div class="dual-plot-card-title">${formatDualLabel(label)}</div><div id="${plotId}" class="dual-plot-chart"></div><input class="dual-overlay-input" type="text" placeholder="${randomOverlayPlaceholder()}" data-series-id="${escapeHtml(seriesId)}" data-axis="${escapeHtml(param)}" data-plot-id="${plotId}">`;
          const container = constraintGridByParam.get(param);
          if (!container) return;
          container.appendChild(card);
          card.addEventListener('click', () => togglePlotCardSelection(card, param));

          const traces = [];
          traces.push({
            x: payload.x,
            y: payload.y,
            mode: count <= 1 ? 'markers' : 'lines',
            name: 'Baseline',
            line: { color: '#9aa0a6', width: 2 },
            hovertemplate: `${escapeHtml(param)}=%{x:.3f}<br>Baseline=%{y:.3e}<extra></extra>`,
            showlegend: false,
          });
          const runMap = new Map();
          dualRuns.forEach((run) => {
            const runPayload = runSeriesForParam(run, seriesId, param);
            if (!runPayload || !Array.isArray(runPayload.y)) return;
            const runCount = runPayload.y.filter((value) => value !== null && Number.isFinite(value)).length;
            if (!runCount) return;
            const traceIndex = traces.length;
            traces.push({
              x: runPayload.x,
              y: runPayload.y,
              mode: runCount <= 1 ? 'markers' : 'lines',
              name: run.name || run.id || 'Run',
              line: { color: run.color || '#f97316', width: 2 },
              hovertemplate: `${escapeHtml(param)}=%{x:.3f}<br>${escapeHtml(run.name || run.id || 'Run')}=%{y:.3e}<extra></extra>`,
              showlegend: false,
              visible: runVisibility.get(run.id) !== false ? true : 'legendonly',
            });
            runMap.set(run.id, traceIndex);
          });
          dualTraceRegistry.set(plotId, runMap);
          Plotly.newPlot(
            plotId,
            traces,
            { autosize: true, xaxis: { title: '', tickfont: { size: 9 } }, yaxis: { title: '', tickfont: { size: 9 } }, margin: { t: 10, l: 30, r: 10, b: 15 } },
            { displayModeBar: false, responsive: true },
          );

          const overlayInput = card.querySelector('.dual-overlay-input');
          if (overlayInput) {
            overlayInput.addEventListener('focus', () => overlayInput.setAttribute('placeholder', ''));
            overlayInput.addEventListener('click', (event) => event.stopPropagation());
            overlayInput.addEventListener('input', (event) => {
              const key = overlayInput.getAttribute('data-plot-id');
              if (overlayDebounce.has(key)) clearTimeout(overlayDebounce.get(key));
              overlayDebounce.set(key, setTimeout(() => ensureMath(() => handleOverlayInput(event)), 250));
            });
          }
        });
      });
    });
    dualParams.forEach((param) => {
      const grid = plotGridsByParam.get(param);
      if (grid && !grid.childElementCount) grid.textContent = `No ${param}-series data for selected dual values.`;
    });
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
        patterns_by_param: currentPatternsByParam(),
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
      if (actionTab !== 'recompute') return;
      deactivatedForRecompute.clear();
      syncDualButtonState();
      updateDeactivatedList();
      return;
    }
    if (button.id === 'dual-deactivate-all') {
      event.preventDefault();
      setActionTab('recompute');
      const buttons = Array.from(root.querySelectorAll('.dual-button:not(.is-hidden)'));
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
      root.querySelectorAll('.dual-overlay-input').forEach((input) => input.setAttribute('placeholder', randomOverlayPlaceholder()));
      if (!wrapperEl) return;
      wrapperEl.classList.toggle('dual-show-overlay');
      button.classList.toggle('is-active', wrapperEl.classList.contains('dual-show-overlay'));
      return;
    }

    if (button.id === 'dual-clear') {
      event.preventDefault();
      if (actionTab !== 'plot') return;
      const buttons = visibleButtonsForPlot();
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

    if (button.classList.contains('dual-remove-button')) {
      event.preventDefault();
      const selectedIds = new Set();
      selectedPlotCards.forEach((setRef) => {
        setRef.forEach((id) => selectedIds.add(id));
      });
      removeSelectedSeries(selectedIds);
      return;
    }

    if (button.id === 'dual-recompute') {
      event.preventDefault();
      if (!deactivatedForRecompute.size) return;
      const allSeriesIds = Object.keys(seriesDataByParam);
      const activeSeriesIds = allSeriesIds.filter((seriesId) => !deactivatedForRecompute.has(seriesId));
      setTriggerValue('recompute', {
        request_id: Date.now(),
        selected_series_ids: activeSeriesIds,
        deactivated_series_ids: Array.from(deactivatedForRecompute.keys()),
        deactivated_labels: Array.from(deactivatedForRecompute.values()),
        local_cursor_indices_by_axis: currentLocalCursorIndicesByAxis(),
        patterns_by_param: currentPatternsByParam(),
      });
    }
  });

  tauGlobalSlidersByParam.forEach((slider, param) => {
    if (!slider) return;
    slider.addEventListener('input', () => {
      const idx = Number(slider.value);
      cursorIndicesByParamState[param] = idx;
      ensurePlotly(renderTauPlots);
    });
    slider.addEventListener('change', () => {
      const idx = Number(slider.value);
      cursorIndicesByParamState[param] = idx;
      emitCursorIfChanged();
    });
  });
  tauLocalSliders.forEach((slider) => {
    slider.addEventListener('input', () => {
      const axis = slider.getAttribute('data-axis-param');
      const param = slider.getAttribute('data-param');
      if (!axis || !param) return;
      const idx = Number(slider.value);
      localCursorIndicesByAxisState[axis] = { ...(localCursorIndicesByAxisState[axis] || {}), [param]: idx };
      ensurePlotly(renderTauPlots);
    });
    slider.addEventListener('change', () => {
      const axis = slider.getAttribute('data-axis-param');
      const param = slider.getAttribute('data-param');
      if (!axis || !param) return;
      const idx = Number(slider.value);
      localCursorIndicesByAxisState[axis] = { ...(localCursorIndicesByAxisState[axis] || {}), [param]: idx };
      ensurePlotly(renderTauPlots);
      emitCursorIfChanged();
    });
  });
  tauPatternInputsByParam.forEach((inputEl) => {
    if (!inputEl) return;
    inputEl.addEventListener('input', () => {
      ensureMath(() => ensurePlotly(renderTauPlots));
    });
  });
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

  tauPatternHintsByParam.forEach((hintEl, param) => {
    if (!hintEl) return;
    hintEl.textContent = buildPatternHintText(param);
  });

  toggleBtn.classList.toggle('is-active', hideZero);
  toggleBtn.textContent = hideZero ? 'Show all-zero duals' : 'Hide all-zero duals';

  const selectedSeries = (data && data.selected_series_ids) || [];
  selectedSeries.forEach((seriesId) => {
    const series = seriesDataByParam[seriesId];
    if (!series) return;
    plotSelectionMap.set(seriesId, series.label || seriesId);
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
