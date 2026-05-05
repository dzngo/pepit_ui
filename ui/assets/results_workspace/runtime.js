function renderResultsWorkspaceRuntime(component) {
  const { parentElement, setTriggerValue } = component;
  const ns = (globalThis.__resultsWorkspace = globalThis.__resultsWorkspace || {});
  const payload = ns.parsePayload(component);
  const {
    data,
    seriesDataByParam,
    rawDualRuns,
    sectionsHtmlByParam,
    plotTitlesByParam,
    metricLabels,
    metricKeys,
    currentMetric,
    tauSeriesByParam,
    dualParams,
    paramValuesByName,
    cursorIndicesByParamPayload,
    localCursorIndicesByAxisPayload,
    patternsByParamPayload,
    patternParams,
    patternInvalidParams,
    patternConflictParams,
  } = payload;
  const state = ns.createState(dualParams);
  const {
    plotSelectionMap,
    deactivatedForRecompute,
    selectedPlotCards,
    overlayState,
    overlayDebounce,
    dualTraceRegistry,
    tauTraceRegistry,
  } = state;

  const dualRuns = rawDualRuns.map((run, idx) => ({
    ...run,
    color: run.color || ns.runColorFor(run.id, idx),
  }));
  const runsById = new Map();
  dualRuns.forEach((run) => {
    runsById.set(run.id, { ...run, visible: run.visible !== false });
  });
  const runVisibility = new Map(Array.from(runsById.values()).map((run) => [run.id, run.visible !== false]));

  function randomOverlayPlaceholder() {
    const overlayExamples = (ns.constants && ns.constants.OVERLAY_EXAMPLES) || [];
    const example = overlayExamples[Math.floor(Math.random() * overlayExamples.length)];
    return `example: ${example}`;
  }

  function ensurePlotly(callback) {
    if (window.Plotly) {
      callback();
      return;
    }
    const script = document.createElement("script");
    script.src = "https://cdn.plot.ly/plotly-2.32.0.min.js";
    script.onload = callback;
    document.head.appendChild(script);
  }

  function ensureMath(callback) {
    if (window.math) {
      callback();
      return;
    }
    const script = document.createElement("script");
    script.src = "https://cdn.jsdelivr.net/npm/mathjs@12.4.2/lib/browser/math.js";
    script.onload = callback;
    document.head.appendChild(script);
  }

  function metricOptionsHtml() {
    if (!metricKeys.length) return "";
    return metricKeys
      .map((key) => {
        const selected = key === currentMetric ? " selected" : "";
        const label = metricLabels[key] || key;
        return `<option value="${ns.escapeHtml(key)}"${selected}>${ns.escapeHtml(label)}</option>`;
      })
      .join("");
  }

  const legendCollapsedByDefault = ns.readLegendCollapsed();
  const ctx = {
    root: parentElement,
    setTriggerValue,
    ensureMath,
    ensurePlotly,
    escapeHtml: ns.escapeHtml,
    sanitizeId: ns.sanitizeId,
    formatDualLabel: ns.formatDualLabel,
    randomOverlayPlaceholder,
    dualParams,
    paramValuesByName,
    seriesDataByParam,
    sectionsHtmlByParam,
    plotTitlesByParam,
    tauSeriesByParam,
    patternParams,
    patternInvalidParams,
    patternConflictParams,
    patternsByParamPayload,
    cursorIndicesByParamPayload,
    localCursorIndicesByAxisPayload,
    cursorIndicesByParamState: {},
    localCursorIndicesByAxisState: {},
    dualRuns,
    runVisibility,
    plotSelectionMap,
    deactivatedForRecompute,
    selectedPlotCards,
    overlayState,
    overlayDebounce,
    dualTraceRegistry,
    tauTraceRegistry,
    hideZero: state.hideZero,
    actionTab: state.actionTab,
    tauGlobalSlidersByParam: new Map(),
    tauGlobalValuesByParam: new Map(),
    tauLocalSliders: [],
    tauLocalValues: [],
    tauPatternInputsByParam: new Map(),
    tauPatternHintsByParam: new Map(),
    tauPatternErrorsByParam: new Map(),
  };

  const tauCtrl = ns.tauPlots.createTauController(ctx);
  const dualCtrl = ns.dualPlots.createDualController(ctx, tauCtrl);

  parentElement.innerHTML = `
    <style>${data.css || ""}</style>
    <div class="dual-wrapper">
      <div class="dual-shell${legendCollapsedByDefault ? " is-legend-collapsed" : ""}">
        <aside class="dual-rail">
          <div class="dual-rail-sticky">
            <div class="dual-section-title dual-rail-title">
              <span class="dual-rail-title-text">Recompute runs legend</span>
              <button type="button" class="dual-legend-toggle" id="dual-legend-toggle" aria-expanded="${legendCollapsedByDefault ? "false" : "true"}" aria-label="${legendCollapsedByDefault ? "Expand recompute runs legend" : "Collapse recompute runs legend"}">${legendCollapsedByDefault ? "→" : "←"}</button>
            </div>
            <div class="dual-panel dual-rail-panel">${ns.runRowsHtml(runsById, runVisibility)}</div>
          </div>
        </aside>
        <main class="dual-main">
          <div class="dual-section-title">Worst-case guarantee</div>
          <div class="tau-panels-grid">${tauCtrl.tauPanelsHtml()}</div>
          <div class="dual-section-title">Dual values</div>
          <div class="dual-control-bar">
            <div class="dual-control-row">
              <label class="dual-control-label" for="dual-ranking-metric">Ranking metric</label>
              <select id="dual-ranking-metric" class="dual-ranking-select">${metricOptionsHtml()}</select>
              <button type="button" class="dual-toggle-button is-active" id="dual-toggle-zero">Show all-zero duals</button>
            </div>
          </div>
          ${dualCtrl.dualSectionsHtml()}
          <div class="dual-mode-panel">
            <div class="dual-mode-topline">
              <div class="dual-mode-switch" role="tablist" aria-label="Dual interaction mode">
                <button type="button" class="dual-mode-option is-active" id="tab-plot" role="tab" aria-selected="true" aria-controls="tab-panel-plot">Explore plots</button>
                <button type="button" class="dual-mode-option" id="tab-recompute" role="tab" aria-selected="false" aria-controls="tab-panel-recompute" tabindex="-1">Prepare recompute</button>
              </div>
              <div class="dual-mode-meta">
                <div id="dual-mode-title" class="dual-mode-title">Explore plots</div>
                <div id="dual-mode-description" class="dual-mode-description">Select dual values to inspect their curves.</div>
              </div>
              <div id="dual-mode-count" class="dual-mode-count">0 selected</div>
            </div>
            <div id="tab-panel-plot" class="dual-mode-toolbar" role="tabpanel" aria-labelledby="tab-plot">
              <button type="button" class="dual-plot-button" id="dual-plot" disabled>Plot selected</button>
              <button type="button" class="dual-overlay-button" id="dual-overlay" style="display:none;">Overlay plot</button>
              <button type="button" class="dual-clear-button" id="dual-clear">Select all</button>
            </div>
            <div id="tab-panel-recompute" class="dual-mode-toolbar dual-mode-toolbar-stacked" role="tabpanel" aria-labelledby="tab-recompute" hidden>
              <div class="dual-selected-header"><div class="dual-selected-title">Deactivated dual values</div></div>
              <div id="dual-deactivated-list" class="dual-selected-list">None</div>
              <div class="dual-recompute-divider" aria-hidden="true"></div>
              <div class="dual-plot-actions dual-plot-actions-inline dual-recompute-actions">
                <button type="button" class="dual-plot-button" id="dual-recompute" disabled>Recompute without selected duals</button>
                <button type="button" class="dual-clear-button" id="dual-activate-all">Reactivate all</button>
                <button type="button" class="dual-toggle-button is-active" id="dual-deactivate-all">Deactivate visible duals</button>
              </div>
            </div>
          </div>
          ${dualCtrl.dualPlotsHtml()}
        </main>
      </div>
    </div>
  `;

  ctx.root = parentElement;
  ctx.shellEl = ctx.root.querySelector(".dual-shell");
  ctx.legendToggleBtn = ctx.root.querySelector("#dual-legend-toggle");
  ctx.deactivatedListEl = ctx.root.querySelector("#dual-deactivated-list");
  ctx.toggleBtn = ctx.root.querySelector("#dual-toggle-zero");
  ctx.tabPlotBtn = ctx.root.querySelector("#tab-plot");
  ctx.tabRecomputeBtn = ctx.root.querySelector("#tab-recompute");
  ctx.tabPanelPlot = ctx.root.querySelector("#tab-panel-plot");
  ctx.tabPanelRecompute = ctx.root.querySelector("#tab-panel-recompute");
  ctx.modeTitle = ctx.root.querySelector("#dual-mode-title");
  ctx.modeDescription = ctx.root.querySelector("#dual-mode-description");
  ctx.modeCount = ctx.root.querySelector("#dual-mode-count");
  ctx.plotBtn = ctx.root.querySelector("#dual-plot");
  ctx.overlayBtn = ctx.root.querySelector("#dual-overlay");
  ctx.clearBtn = ctx.root.querySelector("#dual-clear");
  ctx.removeButtonsByParam = new Map(ctx.dualParams.map((param) => [param, ctx.root.querySelector(`#dual-remove-${ctx.sanitizeId(param)}`)]));
  ctx.recomputeBtn = ctx.root.querySelector("#dual-recompute");
  ctx.activateAllBtn = ctx.root.querySelector("#dual-activate-all");
  ctx.deactivateAllBtn = ctx.root.querySelector("#dual-deactivate-all");
  ctx.metricSelect = ctx.root.querySelector("#dual-ranking-metric");

  ctx.dualParams.forEach((param) => {
    ctx.tauGlobalSlidersByParam.set(param, ctx.root.querySelector(`.tau-global-slider[data-param="${param}"]`));
    ctx.tauGlobalValuesByParam.set(param, ctx.root.querySelector(`.tau-global-value[data-param="${param}"]`));
    ctx.tauPatternInputsByParam.set(param, ctx.root.querySelector(`.tau-pattern-input[data-param="${param}"]`));
    ctx.tauPatternHintsByParam.set(param, ctx.root.querySelector(`.tau-pattern-hint[data-param="${param}"]`));
    ctx.tauPatternErrorsByParam.set(param, ctx.root.querySelector(`.tau-pattern-error[data-param="${param}"]`));
  });
  ctx.tauLocalSliders = Array.from(ctx.root.querySelectorAll(".tau-local-slider"));
  ctx.tauLocalValues = Array.from(ctx.root.querySelectorAll(".tau-local-value"));
  tauCtrl.initTauStateAndDom();

  const emitCursorIfChanged = ns.events.createCursorEmitter({
    setTriggerValue,
    getCursor: tauCtrl.currentCursorIndicesByParam,
    getLocalByAxis: tauCtrl.currentLocalCursorIndicesByAxis,
    getPatterns: tauCtrl.currentPatternsByParam,
  });

  ctx.root.addEventListener("click", (event) => {
    const target = event.target;
    if (!(target instanceof Element)) return;
    const button = target.closest("button");
    if (!button) return;

    if (button.classList.contains("run-toggle")) {
      event.preventDefault();
      const runId = button.getAttribute("data-run-id");
      if (runId) dualCtrl.toggleRunVisibility(runId);
      return;
    }
    if (button.classList.contains("run-remove")) {
      event.preventDefault();
      const runId = button.getAttribute("data-run-id");
      if (!runId) return;
      setTriggerValue("remove_run", {
        request_id: Date.now(),
        run_id: runId,
        patterns_by_param: tauCtrl.currentPatternsByParam(),
      });
      return;
    }
    if (button.classList.contains("dual-button")) {
      event.preventDefault();
      const seriesId = button.getAttribute("data-series-id");
      const label = button.getAttribute("data-label");
      if (!seriesId) return;
      if (ctx.actionTab === "recompute") {
        if (ctx.deactivatedForRecompute.has(seriesId)) ctx.deactivatedForRecompute.delete(seriesId);
        else dualCtrl.setDeactivatedSeries(seriesId, label || "", button);
        dualCtrl.syncDualButtonState();
        dualCtrl.updateDeactivatedList();
        ensurePlotly(dualCtrl.plotSelected);
      } else {
        if (ctx.plotSelectionMap.has(seriesId)) ctx.plotSelectionMap.delete(seriesId);
        else ctx.plotSelectionMap.set(seriesId, label || "");
        dualCtrl.syncDualButtonState();
        dualCtrl.updateClearButton();
      }
      return;
    }
    if (button.classList.contains("dual-reactivate-button")) {
      event.preventDefault();
      const seriesId = button.getAttribute("data-series-id");
      if (!seriesId) return;
      ctx.deactivatedForRecompute.delete(seriesId);
      dualCtrl.syncDualButtonState();
      dualCtrl.updateDeactivatedList();
      ensurePlotly(dualCtrl.plotSelected);
      return;
    }
    if (button.id === "dual-activate-all") {
      event.preventDefault();
      if (ctx.actionTab !== "recompute") return;
      ctx.deactivatedForRecompute.clear();
      dualCtrl.syncDualButtonState();
      dualCtrl.updateDeactivatedList();
      ensurePlotly(dualCtrl.plotSelected);
      return;
    }
    if (button.id === "dual-deactivate-all") {
      event.preventDefault();
      dualCtrl.setActionTab("recompute");
      const buttons = Array.from(ctx.root.querySelectorAll(".dual-button:not(.is-hidden)"));
      buttons.forEach((btn) => {
        const seriesId = btn.getAttribute("data-series-id");
        const label = btn.getAttribute("data-label");
        if (!seriesId) return;
        if (!ctx.deactivatedForRecompute.has(seriesId)) dualCtrl.setDeactivatedSeries(seriesId, label || "", btn);
      });
      dualCtrl.syncDualButtonState();
      dualCtrl.updateDeactivatedList();
      ensurePlotly(dualCtrl.plotSelected);
      return;
    }
    if (button.id === "dual-toggle-zero") {
      event.preventDefault();
      ctx.hideZero = !ctx.hideZero;
      button.classList.toggle("is-active", ctx.hideZero);
      button.textContent = ctx.hideZero ? "Show all-zero duals" : "Hide all-zero duals";
      dualCtrl.applyZeroFilter();
      dualCtrl.updateClearButton();
      return;
    }
    if (button.id === "dual-plot") {
      event.preventDefault();
      ensurePlotly(dualCtrl.plotSelected);
      return;
    }
    if (button.id === "dual-overlay") {
      event.preventDefault();
      const wrapperEl = ctx.root.querySelector(".dual-wrapper");
      ctx.root.querySelectorAll(".dual-overlay-input").forEach((input) => input.setAttribute("placeholder", randomOverlayPlaceholder()));
      if (!wrapperEl) return;
      wrapperEl.classList.toggle("dual-show-overlay");
      button.classList.toggle("is-active", wrapperEl.classList.contains("dual-show-overlay"));
      return;
    }
    if (button.id === "dual-clear") {
      event.preventDefault();
      if (ctx.actionTab !== "plot") return;
      const buttons = dualCtrl.visibleButtonsForPlot();
      if (!buttons.length) return;
      const visibleSeries = new Set(buttons.map((btn) => btn.getAttribute("data-series-id")));
      const allSelected = Array.from(visibleSeries).every((id) => ctx.plotSelectionMap.has(id));
      if (allSelected) visibleSeries.forEach((id) => ctx.plotSelectionMap.delete(id));
      else {
        buttons.forEach((btn) => {
          const seriesId = btn.getAttribute("data-series-id");
          const label = btn.getAttribute("data-label");
          if (seriesId && !ctx.plotSelectionMap.has(seriesId)) ctx.plotSelectionMap.set(seriesId, label || "");
        });
      }
      dualCtrl.syncDualButtonState();
      dualCtrl.updateClearButton();
      dualCtrl.clearDualPlots();
      return;
    }
    if (button.classList.contains("dual-remove-button")) {
      event.preventDefault();
      const selectedIds = new Set();
      ctx.selectedPlotCards.forEach((setRef) => setRef.forEach((id) => selectedIds.add(id)));
      if (ctx.actionTab === "recompute") {
        dualCtrl.deactivateSelectedSeries(selectedIds);
      } else {
        dualCtrl.removeSelectedSeries(selectedIds);
      }
      return;
    }
    if (button.id === "dual-recompute") {
      event.preventDefault();
      if (!ctx.deactivatedForRecompute.size) return;
      const allSeriesIds = Object.keys(seriesDataByParam);
      const activeSeriesIds = allSeriesIds.filter((seriesId) => !ctx.deactivatedForRecompute.has(seriesId));
      setTriggerValue("recompute", {
        request_id: Date.now(),
        selected_series_ids: activeSeriesIds,
        deactivated_series_ids: Array.from(ctx.deactivatedForRecompute.keys()),
        deactivated_labels: dualCtrl.deactivatedLabels(),
        local_cursor_indices_by_axis: tauCtrl.currentLocalCursorIndicesByAxis(),
        patterns_by_param: tauCtrl.currentPatternsByParam(),
      });
    }
  });

  tauCtrl.bindTauEvents(ensurePlotly, ensureMath, emitCursorIfChanged);
  if (ctx.metricSelect) {
    ctx.metricSelect.addEventListener("change", () => {
      setTriggerValue("metric", { request_id: Date.now(), metric: ctx.metricSelect.value });
    });
  }
  if (ctx.tabPlotBtn) ctx.tabPlotBtn.addEventListener("click", () => dualCtrl.setActionTab("plot"));
  if (ctx.tabRecomputeBtn) ctx.tabRecomputeBtn.addEventListener("click", () => dualCtrl.setActionTab("recompute"));
  ns.layout.bindLegendToggle({
    legendToggleBtn: ctx.legendToggleBtn,
    shellEl: ctx.shellEl,
    writeLegendCollapsed: ns.writeLegendCollapsed,
    onResized: () => ns.layout.resizeAllPlots(ctx.root),
  });

  ctx.tauPatternHintsByParam.forEach((hintEl, param) => {
    if (!hintEl) return;
    hintEl.textContent = tauCtrl.buildPatternHintText(param);
  });
  ctx.toggleBtn.classList.toggle("is-active", ctx.hideZero);
  ctx.toggleBtn.textContent = ctx.hideZero ? "Show all-zero duals" : "Hide all-zero duals";
  (payload.selectedSeriesIds || []).forEach((seriesId) => {
    const series = seriesDataByParam[seriesId];
    if (!series) return;
    plotSelectionMap.set(seriesId, series.label || seriesId);
  });

  dualCtrl.applyZeroFilter();
  dualCtrl.setActionTab("plot");
  dualCtrl.updateClearButton();
  dualCtrl.updateRemoveButtons();
  dualCtrl.updateDeactivatedList();
  ensureMath(() => ensurePlotly(tauCtrl.renderTauPlots));
  if (plotSelectionMap.size) ensurePlotly(dualCtrl.plotSelected);
}

globalThis.__resultsWorkspace = globalThis.__resultsWorkspace || {};
globalThis.__resultsWorkspace.renderResultsWorkspaceRuntime = renderResultsWorkspaceRuntime;
