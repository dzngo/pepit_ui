export default function(component) {
  const { data, parentElement, setTriggerValue } = component;

  const seriesData = (data && data.series_data) || {};
  const dualRuns = (data && data.dual_runs) || [];
  const selected = new Map();
  const selectedPlotCards = { gamma: new Set(), n: new Set() };
  const overlayState = new Map();
  const overlayDebounce = new Map();
  let hideZero = true;

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

  parentElement.innerHTML = `
    <style>${data.css || ''}</style>
    <div class="dual-wrapper">
      <div class="dual-plot-actions">
        <button type="button" class="dual-toggle-button is-active" id="dual-toggle-zero">Show all-zero duals</button>
      </div>
      <div class="dual-panel"><div class="dual-section">${data.gamma_html || ''}</div></div>
      <div class="dual-panel"><div class="dual-section">${data.n_html || ''}</div></div>
      <div class="dual-plot-actions">
        <button type="button" class="dual-plot-button" id="dual-plot">Plot dual values</button>
        <button type="button" class="dual-overlay-button" id="dual-overlay" style="display:none;">Overlay plot</button>
        <button type="button" class="dual-clear-button" id="dual-clear">Select all</button>
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
      <div class="dual-selected-header"><div class="dual-selected-title">Selected dual values</div></div>
      <div id="dual-selected-list" class="dual-selected-list">None</div>
      <div class="dual-plot-actions dual-plot-actions-inline">
        <button type="button" class="dual-plot-button" id="dual-recompute" disabled>Recompute</button>
      </div>
    </div>
  `;

  const root = parentElement;
  const listEl = root.querySelector('#dual-selected-list');
  const toggleBtn = root.querySelector('#dual-toggle-zero');
  const plotBtn = root.querySelector('#dual-plot');
  const overlayBtn = root.querySelector('#dual-overlay');
  const clearBtn = root.querySelector('#dual-clear');
  const removeGammaBtn = root.querySelector('#dual-remove-gamma');
  const removeNBtn = root.querySelector('#dual-remove-n');
  const recomputeBtn = root.querySelector('#dual-recompute');

  function visibleButtons() {
    return Array.from(root.querySelectorAll('.dual-button:not(.is-hidden)'));
  }

  function setButtonsSelected(seriesId, isSelected) {
    root.querySelectorAll('.dual-button').forEach((btn) => {
      if (btn.getAttribute('data-series-id') !== seriesId) return;
      btn.classList.toggle('is-clicked', isSelected);
    });
  }

  function updateList() {
    if (!listEl) return;
    if (!selected.size) {
      listEl.textContent = 'None';
      if (recomputeBtn) recomputeBtn.disabled = true;
      return;
    }
    listEl.innerHTML = Array.from(selected.values())
      .map((txt) => `<span class="dual-badge">${formatDualLabel(txt)}</span>`)
      .join('');
    if (recomputeBtn) recomputeBtn.disabled = false;
  }

  function updateClearButton() {
    if (!clearBtn) return;
    const buttons = visibleButtons();
    if (!buttons.length) {
      clearBtn.textContent = 'Select all';
      return;
    }
    const visibleSeries = new Set(buttons.map((btn) => btn.getAttribute('data-series-id')));
    const allSelected = Array.from(visibleSeries).every((id) => selected.has(id));
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

  function clearPlots() {
    const gammaPlot = root.querySelector('#dual-plot-gamma');
    const nPlot = root.querySelector('#dual-plot-n');
    if (gammaPlot) gammaPlot.innerHTML = '';
    if (nPlot) nPlot.innerHTML = '';
    selectedPlotCards.gamma.clear();
    selectedPlotCards.n.clear();
    overlayState.clear();
    updateOverlayButtonVisibility(false);
    updateRemoveButtons();
  }

  function normalizeExpression(raw) {
    if (!raw) return '';
    const trimmed = raw.trim();
    if (!trimmed) return '';
    return trimmed.replace(/^y\s*=\s*/i, '');
  }

  function buildOverlayYValues(expr, xValues) {
    let compiled;
    try {
      compiled = window.math.compile(expr);
    } catch (err) {
      return { yValues: xValues.map(() => null), hasError: true };
    }
    let hasValid = false;
    const yValues = xValues.map((x) => {
      if (!Number.isFinite(x)) return null;
      try {
        const value = compiled.evaluate({ x });
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
    const result = buildOverlayYValues(rawExpr, xValues);
    updateOverlayTrace(plotId, xValues, result.yValues, rawExpr);
    input.classList.toggle('is-error', result.hasError);
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
      if (selected.has(seriesId)) {
        selected.delete(seriesId);
        setButtonsSelected(seriesId, false);
      }
    });
    selectedPlotCards.gamma.clear();
    selectedPlotCards.n.clear();
    updateList();
    updateClearButton();
    clearPlots();
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
        if (selected.has(seriesId)) {
          selected.delete(seriesId);
          setButtonsSelected(seriesId, false);
        }
        btn.classList.add('is-hidden');
      } else {
        btn.classList.remove('is-hidden');
      }
    });
    updateList();
    updateClearButton();
  }

  function plotSelected() {
    const gammaGrid = root.querySelector('#dual-plot-gamma');
    const nGrid = root.querySelector('#dual-plot-n');
    if (!gammaGrid || !nGrid) return;
    const seriesIds = Array.from(selected.keys());
    gammaGrid.innerHTML = '';
    nGrid.innerHTML = '';
    selectedPlotCards.gamma.clear();
    selectedPlotCards.n.clear();
    overlayState.clear();
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
      const gammaSection = document.createElement('div');
      gammaSection.className = 'dual-plot-constraint';
      gammaSection.innerHTML = `<div class="dual-plot-constraint-title">${constraintLabel}</div><div class="dual-plot-cards" data-constraint="${constraintLabel}"></div>`;
      gammaGrid.appendChild(gammaSection);

      const nSection = document.createElement('div');
      nSection.className = 'dual-plot-constraint';
      nSection.innerHTML = `<div class="dual-plot-constraint-title">${constraintLabel}</div><div class="dual-plot-cards" data-constraint="${constraintLabel}"></div>`;
      nGrid.appendChild(nSection);

      const gammaConstraintGrid = gammaSection.querySelector('.dual-plot-cards');
      const nConstraintGrid = nSection.querySelector('.dual-plot-cards');

      ids.forEach((seriesId) => {
        const series = seriesData[seriesId];
        if (!series) return;
        const safeId = sanitizeId(seriesId);
        const safeKey = `${safeId}-${Math.random().toString(36).slice(2, 8)}`;

        const gammaCard = document.createElement('div');
        gammaCard.className = 'dual-plot-card';
        gammaCard.setAttribute('data-series-id', seriesId);
        gammaCard.innerHTML = `<div class="dual-plot-card-title">${formatDualLabel(series.label)}</div><div id="gamma-${safeKey}" class="dual-plot-chart"></div><input class="dual-overlay-input" type="text" placeholder="${randomOverlayPlaceholder()}" data-series-id="${escapeHtml(seriesId)}" data-axis="gamma" data-plot-id="gamma-${safeKey}">`;
        gammaConstraintGrid.appendChild(gammaCard);
        gammaCard.addEventListener('click', () => togglePlotCardSelection(gammaCard, selectedPlotCards.gamma));

        const nCard = document.createElement('div');
        nCard.className = 'dual-plot-card';
        nCard.setAttribute('data-series-id', seriesId);
        nCard.innerHTML = `<div class="dual-plot-card-title">${formatDualLabel(series.label)}</div><div id="n-${safeKey}" class="dual-plot-chart"></div><input class="dual-overlay-input" type="text" placeholder="${randomOverlayPlaceholder()}" data-series-id="${escapeHtml(seriesId)}" data-axis="n" data-plot-id="n-${safeKey}">`;
        nConstraintGrid.appendChild(nCard);
        nCard.addEventListener('click', () => togglePlotCardSelection(nCard, selectedPlotCards.n));

        const gammaTraces = [];
        const gammaCount = series.gamma_dual.filter((value) => value !== null && Number.isFinite(value)).length;
        gammaTraces.push({
          x: series.gamma_values,
          y: series.gamma_dual,
          mode: gammaCount <= 1 ? 'markers' : 'lines',
          name: 'Baseline',
          line: { color: '#9aa0a6', width: 2 },
          hovertemplate: 'gamma=%{x:.3f}<br>Baseline=%{y:.3e}<extra></extra>',
          showlegend: false,
        });
        dualRuns.forEach((run) => {
          const runSeriesData = run.series_data || {};
          const runSeries = runSeriesData[seriesId];
          if (!runSeries) return;
          const runGammaCount = runSeries.gamma_dual.filter((value) => value !== null && Number.isFinite(value)).length;
          gammaTraces.push({
            x: runSeries.gamma_values,
            y: runSeries.gamma_dual,
            mode: runGammaCount <= 1 ? 'markers' : 'lines',
            name: run.name || run.id || 'Run',
            line: { color: run.color || '#f97316', width: 2 },
            hovertemplate: `gamma=%{x:.3f}<br>${escapeHtml(run.name || run.id || 'Run')}=%{y:.3e}<extra></extra>`,
            showlegend: false,
          });
        });
        Plotly.newPlot(
          `gamma-${safeKey}`,
          gammaTraces,
          { autosize: true, xaxis: { title: '', tickfont: { size: 9 } }, yaxis: { title: '', tickfont: { size: 9 } }, margin: { t: 10, l: 30, r: 10, b: 15 } },
          { displayModeBar: false, responsive: true },
        );

        const nTraces = [];
        const nCount = series.n_dual.filter((value) => value !== null && Number.isFinite(value)).length;
        nTraces.push({
          x: series.n_values,
          y: series.n_dual,
          mode: nCount <= 1 ? 'markers' : 'lines',
          name: 'Baseline',
          line: { color: '#9aa0a6', width: 2 },
          hovertemplate: 'n=%{x:.3f}<br>Baseline=%{y:.3e}<extra></extra>',
          showlegend: false,
        });
        dualRuns.forEach((run) => {
          const runSeriesData = run.series_data || {};
          const runSeries = runSeriesData[seriesId];
          if (!runSeries) return;
          const runNCount = runSeries.n_dual.filter((value) => value !== null && Number.isFinite(value)).length;
          nTraces.push({
            x: runSeries.n_values,
            y: runSeries.n_dual,
            mode: runNCount <= 1 ? 'markers' : 'lines',
            name: run.name || run.id || 'Run',
            line: { color: run.color || '#f97316', width: 2 },
            hovertemplate: `n=%{x:.3f}<br>${escapeHtml(run.name || run.id || 'Run')}=%{y:.3e}<extra></extra>`,
            showlegend: false,
          });
        });
        Plotly.newPlot(
          `n-${safeKey}`,
          nTraces,
          { autosize: true, xaxis: { title: '', tickfont: { size: 9 } }, yaxis: { title: '', tickfont: { size: 9 } }, margin: { t: 10, l: 30, r: 10, b: 15 } },
          { displayModeBar: false, responsive: true },
        );

        [gammaCard.querySelector('.dual-overlay-input'), nCard.querySelector('.dual-overlay-input')].forEach((input) => {
          if (!input) return;
          input.addEventListener('focus', () => input.setAttribute('placeholder', ''));
          input.addEventListener('click', (event) => event.stopPropagation());
          input.addEventListener('input', (event) => {
            const key = input.getAttribute('data-plot-id');
            if (overlayDebounce.has(key)) clearTimeout(overlayDebounce.get(key));
            overlayDebounce.set(key, setTimeout(() => ensureMath(() => handleOverlayInput(event)), 250));
          });
        });
      });
    });
  }

  root.addEventListener('click', (event) => {
    const target = event.target;
    if (!(target instanceof Element)) return;
    const button = target.closest('button');
    if (!button) return;

    if (button.classList.contains('dual-button')) {
      event.preventDefault();
      const seriesId = button.getAttribute('data-series-id');
      const label = button.getAttribute('data-label');
      if (!seriesId) return;
      if (selected.has(seriesId)) {
        selected.delete(seriesId);
        setButtonsSelected(seriesId, false);
      } else {
        selected.set(seriesId, label || '');
        setButtonsSelected(seriesId, true);
      }
      updateList();
      updateClearButton();
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
      const allSelected = Array.from(visibleSeries).every((id) => selected.has(id));
      if (allSelected) {
        visibleSeries.forEach((id) => selected.delete(id));
        root.querySelectorAll('.dual-button').forEach((btn) => {
          const id = btn.getAttribute('data-series-id');
          if (visibleSeries.has(id)) btn.classList.remove('is-clicked');
        });
      } else {
        buttons.forEach((btn) => {
          const seriesId = btn.getAttribute('data-series-id');
          const label = btn.getAttribute('data-label');
          if (seriesId && !selected.has(seriesId)) selected.set(seriesId, label || '');
        });
        visibleSeries.forEach((id) => setButtonsSelected(id, true));
      }
      updateList();
      updateClearButton();
      clearPlots();
      return;
    }

    if (button.id === 'dual-remove-gamma' || button.id === 'dual-remove-n') {
      event.preventDefault();
      removeSelectedSeries(new Set([...selectedPlotCards.gamma, ...selectedPlotCards.n]));
      return;
    }

    if (button.id === 'dual-recompute') {
      event.preventDefault();
      if (!selected.size) return;
      setTriggerValue('recompute', {
        request_id: Date.now(),
        selected_series_ids: Array.from(selected.keys()),
        selected_labels: Array.from(selected.values()),
      });
    }
  });

  toggleBtn.classList.toggle('is-active', hideZero);
  toggleBtn.textContent = hideZero ? 'Show all-zero duals' : 'Hide all-zero duals';

  const selectedSeries = (data && data.selected_series_ids) || [];
  selectedSeries.forEach((seriesId) => {
    const series = seriesData[seriesId];
    if (!series) return;
    selected.set(seriesId, series.label);
    setButtonsSelected(seriesId, true);
  });

  applyZeroFilter();
  updateClearButton();
  updateRemoveButtons();
  updateList();
}
