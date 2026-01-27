const selected = new Map();
const listEl = document.getElementById('dual-selected-list');
const clearBtn = document.getElementById('dual-clear');
const plotBtn = document.getElementById('dual-plot');
const overlayBtn = document.getElementById('dual-overlay');
const wrapperEl = document.querySelector('.dual-wrapper');
const toggleBtn = document.getElementById('dual-toggle-zero');
const removeGammaBtn = document.getElementById('dual-remove-gamma');
const removeNBtn = document.getElementById('dual-remove-n');
let hideZero = true;
const selectedPlotCards = { gamma: new Set(), n: new Set() };
const overlayState = new Map();
const overlayDebounce = new Map();
const overlayExamples = [
  '1/log(x)',
  'sin(x)',
  '2*x^2 + 3',
  'log(x)',
  'sqrt(x)',
  'exp(-x)',
];

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (c) => {
    return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
  });
}

function formatSubscriptText(text) {
  if (!text) {
    return '';
  }
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
      const subText = value.slice(i + 2, end);
      out += `<sub>${escapeHtml(subText)}</sub>`;
      i = end + 1;
      continue;
    }
    let end = i + 1;
    while (end < value.length && /[a-zA-Z0-9*]/.test(value[end])) {
      end += 1;
    }
    if (end === i + 1) {
      out += `<sub>${escapeHtml(value[i + 1])}</sub>`;
      i += 2;
      continue;
    }
    const subText = value.slice(i + 1, end);
    out += `<sub>${escapeHtml(subText)}</sub>`;
    i = end;
  }
  return out;
}

function formatDualLabel(text) {
  if (!text) {
    return '';
  }
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

function updateList() {
  if (!selected.size) {
    listEl.textContent = 'None';
    return;
  }
  const items = Array.from(selected.values());
  listEl.innerHTML = items.map((txt) => `<span class="dual-badge">${formatDualLabel(txt)}</span>`).join('');
}

function visibleButtons() {
  return Array.from(document.querySelectorAll('.dual-button:not(.is-hidden)'));
}

function updateClearButton() {
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
  removeGammaBtn.style.display = display;
  removeNBtn.style.display = display;
}

function updateOverlayButtonVisibility(show) {
  if (!overlayBtn) {
    return;
  }
  overlayBtn.style.display = show ? 'inline-flex' : 'none';
  if (!show && wrapperEl) {
    wrapperEl.classList.remove('dual-show-overlay');
    overlayBtn.classList.remove('is-active');
  }
}

function randomOverlayPlaceholder() {
  const example = overlayExamples[Math.floor(Math.random() * overlayExamples.length)];
  return `example: ${example}`;
}

function refreshOverlayPlaceholders() {
  document.querySelectorAll('.dual-overlay-input').forEach((input) => {
    input.setAttribute('placeholder', randomOverlayPlaceholder());
  });
}

function togglePlotCardSelection(card, set) {
  const seriesId = card.getAttribute('data-series-id');
  if (!seriesId) {
    return;
  }
  const select = !set.has(seriesId);
  const applyToSet = (targetSet) => {
    if (select) {
      targetSet.add(seriesId);
    } else {
      targetSet.delete(seriesId);
    }
  };
  applyToSet(selectedPlotCards.gamma);
  applyToSet(selectedPlotCards.n);
  document.querySelectorAll(`.dual-plot-card[data-series-id="${seriesId}"]`).forEach((el) => {
    if (select) {
      el.classList.add('is-selected');
    } else {
      el.classList.remove('is-selected');
    }
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
function setButtonsSelected(seriesId, isSelected) {
  const matcher = (btn) => btn.getAttribute('data-series-id') === seriesId;
  document.querySelectorAll('.dual-button').forEach((btn) => {
    if (!matcher(btn)) {
      return;
    }
    if (isSelected) {
      btn.classList.add('is-clicked');
    } else {
      btn.classList.remove('is-clicked');
    }
  });
}

function applyZeroFilter() {
  document.querySelectorAll('.dual-button').forEach((btn) => {
    const seriesId = btn.getAttribute('data-series-id');
    const series = seriesData[seriesId];
    const section = btn.getAttribute('data-section');
    const isAllZero = series && (
      (section === 'gamma' && series.all_zero_gamma) ||
      (section === 'n' && series.all_zero_n)
    );
    if (isAllZero) {
      btn.classList.add('is-all-zero');
    } else {
      btn.classList.remove('is-all-zero');
    }
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

document.querySelectorAll('.dual-button').forEach((btn) => {
  btn.addEventListener('click', () => {
    const seriesId = btn.getAttribute('data-series-id');
    const label = btn.getAttribute('data-label');
    const display = `${label}`;
    if (selected.has(seriesId)) {
      selected.delete(seriesId);
      setButtonsSelected(seriesId, false);
    } else {
      selected.set(seriesId, display);
      setButtonsSelected(seriesId, true);
    }
    updateList();
    updateClearButton();
  });
});

toggleBtn.addEventListener('click', () => {
  hideZero = !hideZero;
  toggleBtn.classList.toggle('is-active', hideZero);
  toggleBtn.textContent = hideZero ? 'Show all-zero duals' : 'Hide all-zero duals';
  applyZeroFilter();
  updateClearButton();
});

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

function sanitizeId(value) {
  return value.replace(/[^a-zA-Z0-9_-]/g, '-');
}

function clearPlots() {
  document.getElementById('dual-plot-gamma').innerHTML = '';
  document.getElementById('dual-plot-n').innerHTML = '';
  selectedPlotCards.gamma.clear();
  selectedPlotCards.n.clear();
  overlayState.clear();
  updateOverlayButtonVisibility(false);
  updateRemoveButtons();
}

function normalizeExpression(raw) {
  if (!raw) {
    return '';
  }
  const trimmed = raw.trim();
  if (!trimmed) {
    return '';
  }
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
    if (!Number.isFinite(x)) {
      return null;
    }
    try {
      const value = compiled.evaluate({ x });
      const numeric = typeof value === 'number' ? value : Number(value);
      if (Number.isFinite(numeric)) {
        hasValid = true;
        return numeric;
      }
      return null;
    } catch (err) {
      return null;
    }
  });
  return { yValues, hasError: !hasValid };
}

function updateOverlayTrace(plotId, xValues, yValues, expr) {
  const plotDiv = document.getElementById(plotId);
  if (!plotDiv || !window.Plotly) {
    return;
  }
  const state = overlayState.get(plotId);
  if (!expr) {
    if (state) {
      Plotly.deleteTraces(plotDiv, [state.traceIndex]);
      overlayState.delete(plotId);
    }
    return;
  }
  const traceName = `Overlay: ${expr}`;
  if (state && plotDiv.data && plotDiv.data[state.traceIndex]) {
    Plotly.restyle(plotDiv, { x: [xValues], y: [yValues], name: [traceName] }, [state.traceIndex]);
    return;
  }
  const traceIndex = plotDiv.data ? plotDiv.data.length : 0;
  const trace = {
    x: xValues,
    y: yValues,
    mode: 'lines',
    name: traceName,
    showlegend: false,
    line: { color: '#ff8a00', width: 2 },
  };
  Plotly.addTraces(plotDiv, trace).then(() => {
    overlayState.set(plotId, { traceIndex });
  });
}

function handleOverlayInput(event) {
  const input = event.target;
  if (!input || !window.math) {
    return;
  }
  const seriesId = input.getAttribute('data-series-id');
  const axis = input.getAttribute('data-axis');
  const plotId = input.getAttribute('data-plot-id');
  const series = seriesData[seriesId];
  if (!series || !axis || !plotId) {
    return;
  }
  const rawExpr = normalizeExpression(input.value);
  if (!rawExpr) {
    input.classList.remove('is-error');
    updateOverlayTrace(plotId, [], [], '');
    return;
  }
  try {
    const xValues = axis === 'gamma' ? series.gamma_values : series.n_values;
    const result = buildOverlayYValues(rawExpr, xValues);
    updateOverlayTrace(plotId, xValues, result.yValues, rawExpr);
    input.classList.toggle('is-error', result.hasError);
  } catch (err) {
    input.classList.add('is-error');
  }
}

function plotSelected() {
  const seriesIds = Array.from(selected.keys());
  const gammaGrid = document.getElementById('dual-plot-gamma');
  const nGrid = document.getElementById('dual-plot-n');
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
    if (!series) {
      return;
    }
    const constraint = series.constraint || 'Other';
    if (!grouped.has(constraint)) {
      grouped.set(constraint, []);
    }
    grouped.get(constraint).push(seriesId);
  });

  grouped.forEach((ids, constraint) => {
    const gammaSection = document.createElement('div');
    gammaSection.className = 'dual-plot-constraint';
    const constraintLabel = escapeHtml(constraint);
    gammaSection.innerHTML = `
      <div class="dual-plot-constraint-title">${constraintLabel}</div>
      <div class="dual-plot-cards" data-constraint="${constraintLabel}"></div>
    `;
    gammaGrid.appendChild(gammaSection);

    const nSection = document.createElement('div');
    nSection.className = 'dual-plot-constraint';
    nSection.innerHTML = `
      <div class="dual-plot-constraint-title">${constraintLabel}</div>
      <div class="dual-plot-cards" data-constraint="${constraintLabel}"></div>
    `;
    nGrid.appendChild(nSection);

    const gammaConstraintGrid = gammaSection.querySelector('.dual-plot-cards');
    const nConstraintGrid = nSection.querySelector('.dual-plot-cards');

    ids.forEach((seriesId) => {
      const series = seriesData[seriesId];
      if (!series) {
        return;
      }
      const safeId = sanitizeId(seriesId);
      const safeKey = `${safeId}-${Math.random().toString(36).slice(2, 8)}`;
      const gammaCard = document.createElement('div');
      gammaCard.className = 'dual-plot-card';
      gammaCard.setAttribute('data-series-id', seriesId);
      gammaCard.innerHTML = `
        <div class="dual-plot-card-title">${formatDualLabel(series.label)}</div>
        <div id="gamma-${safeKey}" class="dual-plot-chart"></div>
        <input class="dual-overlay-input" type="text" placeholder="example: 1/log(x)" data-series-id="${escapeHtml(seriesId)}" data-axis="gamma" data-plot-id="gamma-${safeKey}">
      `;
      gammaConstraintGrid.appendChild(gammaCard);
      gammaCard.addEventListener('click', () => togglePlotCardSelection(gammaCard, selectedPlotCards.gamma));

      const nCard = document.createElement('div');
      nCard.className = 'dual-plot-card';
      nCard.setAttribute('data-series-id', seriesId);
      nCard.innerHTML = `
        <div class="dual-plot-card-title">${formatDualLabel(series.label)}</div>
        <div id="n-${safeKey}" class="dual-plot-chart"></div>
        <input class="dual-overlay-input" type="text" placeholder="example: 1/log(x)" data-series-id="${escapeHtml(seriesId)}" data-axis="n" data-plot-id="n-${safeKey}">
      `;
      nConstraintGrid.appendChild(nCard);
      nCard.addEventListener('click', () => togglePlotCardSelection(nCard, selectedPlotCards.n));

      const gammaCount = series.gamma_dual.filter((value) => value !== null && Number.isFinite(value)).length;
      Plotly.newPlot(`gamma-${safeKey}`, [{
        x: series.gamma_values,
        y: series.gamma_dual,
        mode: gammaCount <= 1 ? 'markers' : 'lines',
        name: series.label,
        showlegend: false,
      }], {
        autosize: true,
        xaxis: { title: '', tickfont: { size: 9 } },
        yaxis: { title: '', tickfont: { size: 9 } },
        margin: { t: 10, l: 30, r: 10, b: 15 },
      }, { displayModeBar: false, responsive: true });

      const nCount = series.n_dual.filter((value) => value !== null && Number.isFinite(value)).length;
      Plotly.newPlot(`n-${safeKey}`, [{
        x: series.n_values,
        y: series.n_dual,
        mode: nCount <= 1 ? 'markers' : 'lines',
        name: series.label,
        showlegend: false,
      }], {
        autosize: true,
        xaxis: { title: '', tickfont: { size: 9 } },
        yaxis: { title: '', tickfont: { size: 9 } },
        margin: { t: 10, l: 30, r: 10, b: 15 },
      }, { displayModeBar: false, responsive: true });

      const gammaInput = gammaCard.querySelector('.dual-overlay-input');
      const nInput = nCard.querySelector('.dual-overlay-input');
      [gammaInput, nInput].forEach((input) => {
        if (!input) {
          return;
        }
        input.addEventListener('focus', () => {
          input.setAttribute('placeholder', '');
        });
        input.addEventListener('click', (event) => {
          event.stopPropagation();
        });
        input.addEventListener('input', (event) => {
          const key = input.getAttribute('data-plot-id');
          if (overlayDebounce.has(key)) {
            clearTimeout(overlayDebounce.get(key));
          }
          overlayDebounce.set(key, setTimeout(() => {
            ensureMath(() => handleOverlayInput(event));
          }, 250));
        });
      });
    });
  });
}

plotBtn.addEventListener('click', () => {
  ensurePlotly(plotSelected);
});

overlayBtn.addEventListener('click', () => {
  ensureMath(() => {
    refreshOverlayPlaceholders();
    if (!wrapperEl) {
      return;
    }
    wrapperEl.classList.toggle('dual-show-overlay');
    overlayBtn.classList.toggle('is-active', wrapperEl.classList.contains('dual-show-overlay'));
  });
});

removeGammaBtn.addEventListener('click', () => {
  const seriesIds = new Set([
    ...selectedPlotCards.gamma,
    ...selectedPlotCards.n,
  ]);
  removeSelectedSeries(seriesIds);
});

removeNBtn.addEventListener('click', () => {
  const seriesIds = new Set([
    ...selectedPlotCards.gamma,
    ...selectedPlotCards.n,
  ]);
  removeSelectedSeries(seriesIds);
});

clearBtn.addEventListener('click', () => {
  const buttons = visibleButtons();
  if (!buttons.length) {
    return;
  }
  const visibleSeries = new Set(buttons.map((btn) => btn.getAttribute('data-series-id')));
  const allSelected = Array.from(visibleSeries).every((id) => selected.has(id));
  if (allSelected) {
    visibleSeries.forEach((id) => selected.delete(id));
    document.querySelectorAll('.dual-button').forEach((btn) => {
      const id = btn.getAttribute('data-series-id');
      if (visibleSeries.has(id)) {
        btn.classList.remove('is-clicked');
      }
    });
  } else {
    buttons.forEach((btn) => {
      const seriesId = btn.getAttribute('data-series-id');
      const label = btn.getAttribute('data-label');
      if (!selected.has(seriesId)) {
        selected.set(seriesId, label);
      }
    });
    visibleSeries.forEach((id) => setButtonsSelected(id, true));
  }
  updateList();
  updateClearButton();
  clearPlots();
});

toggleBtn.classList.toggle('is-active', hideZero);
toggleBtn.textContent = hideZero ? 'Show all-zero duals' : 'Hide all-zero duals';
applyZeroFilter();
updateClearButton();
updateRemoveButtons();
