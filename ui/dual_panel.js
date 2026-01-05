const selected = new Map();
const listEl = document.getElementById('dual-selected-list');
const clearBtn = document.getElementById('dual-clear');
const plotBtn = document.getElementById('dual-plot');
const toggleBtn = document.getElementById('dual-toggle-zero');
let hideZero = true;

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (c) => {
    return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
  });
}

function updateList() {
  if (!selected.size) {
    listEl.textContent = 'None';
    return;
  }
  const items = Array.from(selected.values());
  listEl.innerHTML = items.map((txt) => `<span class="dual-badge">${escapeHtml(txt)}</span>`).join('');
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
    const value = btn.getAttribute('data-value');
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

function sanitizeId(value) {
  return value.replace(/[^a-zA-Z0-9_-]/g, '-');
}

function clearPlots() {
  document.getElementById('dual-plot-gamma').innerHTML = '';
  document.getElementById('dual-plot-n').innerHTML = '';
}

function plotSelected() {
  const seriesIds = Array.from(selected.keys());
  const gammaGrid = document.getElementById('dual-plot-gamma');
  const nGrid = document.getElementById('dual-plot-n');
  gammaGrid.innerHTML = '';
  nGrid.innerHTML = '';
  if (!seriesIds.length) {
    gammaGrid.textContent = 'Select dual values to plot.';
    nGrid.textContent = 'Select dual values to plot.';
    return;
  }
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
      const gammaCard = document.createElement('div');
      gammaCard.className = 'dual-plot-card';
      gammaCard.innerHTML = `
        <div class="dual-plot-card-title">${escapeHtml(series.label)}</div>
        <div id="gamma-${safeId}" class="dual-plot-chart"></div>
      `;
      gammaConstraintGrid.appendChild(gammaCard);

      const nCard = document.createElement('div');
      nCard.className = 'dual-plot-card';
      nCard.innerHTML = `
        <div class="dual-plot-card-title">${escapeHtml(series.label)}</div>
        <div id="n-${safeId}" class="dual-plot-chart"></div>
      `;
      nConstraintGrid.appendChild(nCard);

      const gammaCount = series.gamma_dual.filter((value) => value !== null && Number.isFinite(value)).length;
      Plotly.newPlot(`gamma-${safeId}`, [{
      x: series.gamma_values,
      y: series.gamma_dual,
      mode: gammaCount <= 1 ? 'markers' : 'lines',
      name: series.label,
    }], {
      autosize: true,
      xaxis: { title: '', tickfont: { size: 9 } },
      yaxis: { title: '', tickfont: { size: 9 } },
      margin: { t: 10, l: 30, r: 10, b: 15 },
    }, { displayModeBar: false, responsive: true });

      const nCount = series.n_dual.filter((value) => value !== null && Number.isFinite(value)).length;
      Plotly.newPlot(`n-${safeId}`, [{
      x: series.n_values,
      y: series.n_dual,
      mode: nCount <= 1 ? 'markers' : 'lines',
      name: series.label,
    }], {
      autosize: true,
      xaxis: { title: '', tickfont: { size: 9 } },
      yaxis: { title: '', tickfont: { size: 9 } },
      margin: { t: 10, l: 30, r: 10, b: 15 },
    }, { displayModeBar: false, responsive: true });
    });
  });
}

plotBtn.addEventListener('click', () => {
  ensurePlotly(plotSelected);
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
