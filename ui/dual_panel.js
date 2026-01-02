const selected = new Map();
const listEl = document.getElementById('dual-selected-list');
const clearBtn = document.getElementById('dual-clear');
const plotBtn = document.getElementById('dual-plot');
const toggleBtn = document.getElementById('dual-toggle-zero');
let hideZero = false;

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
      btn.classList.add('is-hidden');
    } else {
      btn.classList.remove('is-hidden');
    }
  });
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
  });
});

toggleBtn.addEventListener('click', () => {
  hideZero = !hideZero;
  toggleBtn.classList.toggle('is-active', hideZero);
  toggleBtn.textContent = hideZero ? 'Show all-zero duals' : 'Hide all-zero duals';
  applyZeroFilter();
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
  seriesIds.forEach((seriesId) => {
    const series = seriesData[seriesId];
    if (!series) {
      return;
    }
    const safeId = sanitizeId(seriesId);
    const gammaCard = document.createElement('div');
    gammaCard.className = 'dual-plot-card';
    gammaCard.innerHTML = `
      <div class="dual-plot-card-title">${series.label}</div>
      <div id="gamma-${safeId}" class="dual-plot-chart"></div>
    `;
    gammaGrid.appendChild(gammaCard);

    const nCard = document.createElement('div');
    nCard.className = 'dual-plot-card';
    nCard.innerHTML = `
      <div class="dual-plot-card-title">${series.label}</div>
      <div id="n-${safeId}" class="dual-plot-chart"></div>
    `;
    nGrid.appendChild(nCard);

    Plotly.newPlot(`gamma-${safeId}`, [{
      x: series.gamma_values,
      y: series.gamma_dual,
      mode: 'lines',
      name: series.label,
    }], {
      xaxis: { title: '', tickfont: { size: 9 } },
      yaxis: { title: '', tickfont: { size: 9 } },
      margin: { t: 10, l: 30, r: 10, b: 26 },
    }, { displayModeBar: false });

    Plotly.newPlot(`n-${safeId}`, [{
      x: series.n_values,
      y: series.n_dual,
      mode: 'lines',
      name: series.label,
    }], {
      xaxis: { title: '', tickfont: { size: 9 } },
      yaxis: { title: '', tickfont: { size: 9 } },
      margin: { t: 10, l: 30, r: 10, b: 26 },
    }, { displayModeBar: false });
  });
}

plotBtn.addEventListener('click', () => {
  ensurePlotly(plotSelected);
});

clearBtn.addEventListener('click', () => {
  selected.clear();
  document.querySelectorAll('.dual-button').forEach((btn) => {
    btn.classList.remove('is-clicked');
  });
  updateList();
  clearPlots();
});

applyZeroFilter();
