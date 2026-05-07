(() => {
  const ns = (globalThis.__resultsWorkspace = globalThis.__resultsWorkspace || {});

  function setScopeValue(scope, name, value) {
    if (!name || !Number.isFinite(Number(value))) return;
    if (name.includes(".")) {
      const parts = name.split(".").filter(Boolean);
      if (!parts.length) return;
      let target = scope;
      parts.slice(0, -1).forEach((part) => {
        if (!target[part] || typeof target[part] !== "object") target[part] = {};
        target = target[part];
      });
      target[parts[parts.length - 1]] = Number(value);
    } else {
      scope[name] = Number(value);
    }
  }

  function createDualController(ctx, tauCtrl) {
    function constraintFromLabel(label) {
      const value = String(label || "");
      const splitIndex = value.indexOf(" | ");
      return splitIndex === -1 ? "Other" : value.slice(0, splitIndex) || "Other";
    }

    function constraintForButton(button) {
      const seriesId = button.getAttribute("data-series-id");
      const generic = seriesId ? ctx.seriesDataByParam[seriesId] : null;
      return (generic && generic.constraint) || constraintFromLabel(button.getAttribute("data-label"));
    }

    function groupedDualSectionHtml(param, sectionHtml) {
      const template = document.createElement("template");
      template.innerHTML = String(sectionHtml || "").trim();
      const sourceSection = template.content.querySelector("section");
      if (!sourceSection) return sectionHtml || "";

      const buttons = Array.from(sourceSection.querySelectorAll(".dual-button"));
      const titleEl = sourceSection.querySelector("h4");
      const titleHtml = titleEl ? titleEl.innerHTML : ctx.escapeHtml(ctx.plotTitlesByParam[param] || `Ranking vs ${param}`);
      const sectionId = sourceSection.getAttribute("id") || "";
      const sectionKey = sourceSection.getAttribute("data-section-key") || param;
      if (!buttons.length) {
        const emptyEl = sourceSection.querySelector(".dual-empty");
        const emptyHtml = emptyEl ? emptyEl.outerHTML : "<p class='dual-empty'>No dual variables detected.</p>";
        return (
          `<section id="${ctx.escapeHtml(sectionId)}" class="dual-section" data-section-key="${ctx.escapeHtml(sectionKey)}">` +
          `<h4>${titleHtml}</h4>${emptyHtml}</section>`
        );
      }

      const grouped = new Map();
      buttons.forEach((button) => {
        const constraint = constraintForButton(button);
        if (!grouped.has(constraint)) grouped.set(constraint, []);
        grouped.get(constraint).push(button.outerHTML);
      });

      const groupsHtml = Array.from(grouped.entries())
        .map(([constraint, groupButtons]) => (
          `<div class="dual-constraint-group">` +
          `<div class="dual-constraint-title">` +
          `<span>${ctx.escapeHtml(constraint)}</span>` +
          `</div>` +
          `<div class="dual-grid">${groupButtons.join("")}</div>` +
          `</div>`
        ))
        .join("");

      return (
        `<section id="${ctx.escapeHtml(sectionId)}" class="dual-section" data-section-key="${ctx.escapeHtml(sectionKey)}">` +
        `<h4>${titleHtml}</h4>` +
        `<div class="dual-constraint-list">${groupsHtml}</div>` +
        `</section>`
      );
    }

    function dualSectionsHtml() {
      if (!ctx.dualParams.length) return "";
      return ctx.dualParams
        .map((param) => {
          const sectionHtml = ctx.sectionsHtmlByParam[param] || "";
          return `<div class="dual-panel" data-param-section="${ctx.escapeHtml(param)}">${groupedDualSectionHtml(param, sectionHtml)}</div>`;
        })
        .join("");
    }

    function dualPlotsHtml() {
      if (!ctx.dualParams.length) return "";
      return ctx.dualParams
        .map((param) => {
          const safeParam = ctx.sanitizeId(param);
          const defaultTitle = `Dual value vs ${param}`;
          const title = ctx.plotTitlesByParam[param] || defaultTitle;
          return (
            `<div class="dual-panel">` +
            `<div class="dual-plot-section">` +
            `<div class="dual-plot-title">${ctx.escapeHtml(title || defaultTitle)}</div>` +
            `<div id="dual-plot-${safeParam}" class="dual-plot-sections" data-param="${ctx.escapeHtml(param)}"></div>` +
            `<div class="dual-plot-actions dual-plot-actions-inline">` +
            `<button type="button" class="dual-remove-button" data-param="${ctx.escapeHtml(param)}" id="dual-remove-${safeParam}" style="display:none;">Remove selected dual values</button>` +
            `</div>` +
            `</div>` +
            `</div>`
          );
        })
        .join("");
    }

    function visibleButtonsForPlot() {
      return Array.from(ctx.root.querySelectorAll(".dual-button:not(.is-hidden)"));
    }

    function visibleSeriesIds() {
      return new Set(visibleButtonsForPlot().map((btn) => btn.getAttribute("data-series-id")).filter(Boolean));
    }

    function selectedPlotCount() {
      const visibleSeries = visibleSeriesIds();
      return Array.from(visibleSeries).filter((id) => ctx.plotSelectionMap.has(id)).length;
    }

    function sourceButtonForSeries(seriesId) {
      return Array.from(ctx.root.querySelectorAll(".dual-button")).find(
        (btn) => btn.getAttribute("data-series-id") === seriesId
      );
    }

    function deactivatedLabel(entry) {
      if (entry && typeof entry === "object") return entry.label || "";
      return String(entry || "");
    }

    function deactivatedLabels() {
      return Array.from(ctx.deactivatedForRecompute.values()).map(deactivatedLabel);
    }

    function deactivatedEntry(seriesId, label, sourceButton) {
      const button = sourceButton || sourceButtonForSeries(seriesId);
      const bg = button && button.style ? button.style.background : "";
      const color = button && button.style ? button.style.color : "";
      return {
        label: label || "",
        bg: bg || "",
        color: color || "",
      };
    }

    function deactivatedBadgeStyle(entry) {
      if (!entry || typeof entry !== "object" || !entry.bg) return "";
      const color = entry.color || "#ffffff";
      return ` style="background:${ctx.escapeHtml(entry.bg)};border-color:${ctx.escapeHtml(entry.bg)};color:${ctx.escapeHtml(color)};"`;
    }

    function setDeactivatedSeries(seriesId, label, sourceButton) {
      ctx.deactivatedForRecompute.set(seriesId, deactivatedEntry(seriesId, label, sourceButton));
    }

    function updateModeSummary() {
      const isRecompute = ctx.actionTab === "recompute";
      const deactivatedCount = ctx.deactivatedForRecompute.size;
      const plotCount = selectedPlotCount();
      if (ctx.modeTitle) ctx.modeTitle.textContent = isRecompute ? "Prepare recompute" : "Explore plots";
      if (ctx.modeDescription) {
        ctx.modeDescription.textContent = isRecompute
          ? "Click dual buttons above to deactivate them, then click them in the list below to reactivate them."
          : "Select dual values to inspect their curves.";
      }
      if (ctx.modeCount) {
        ctx.modeCount.textContent = isRecompute ? `${deactivatedCount} deactivated` : `${plotCount} selected`;
      }
      if (ctx.plotBtn) ctx.plotBtn.disabled = plotCount === 0;
      if (ctx.recomputeBtn) ctx.recomputeBtn.disabled = deactivatedCount === 0;
    }

    function updateDeactivatedList() {
      if (!ctx.deactivatedListEl) return;
      if (!ctx.deactivatedForRecompute.size) {
        ctx.deactivatedListEl.textContent = "None";
        if (ctx.recomputeBtn) {
          ctx.recomputeBtn.disabled = true;
        }
        updateModeSummary();
        return;
      }
      ctx.deactivatedListEl.innerHTML = Array.from(ctx.deactivatedForRecompute.entries())
        .map(([seriesId, entry]) => {
          const label = deactivatedLabel(entry);
          return `<button type="button" class="dual-badge dual-reactivate-button" data-series-id="${ctx.escapeHtml(seriesId)}"${deactivatedBadgeStyle(entry)}>${ctx.formatDualLabel(label)}</button>`;
        })
        .join("");
      if (ctx.recomputeBtn) {
        ctx.recomputeBtn.disabled = false;
      }
      updateModeSummary();
    }

    function syncDualButtonState() {
      ctx.root.querySelectorAll(".dual-button").forEach((btn) => {
        const seriesId = btn.getAttribute("data-series-id");
        const deactivated = ctx.deactivatedForRecompute.has(seriesId);
        const selectedForPlot = ctx.actionTab === "plot" && ctx.plotSelectionMap.has(seriesId);
        btn.classList.toggle("is-clicked", selectedForPlot);
        btn.classList.toggle("is-deactivated", deactivated);
        btn.setAttribute("aria-pressed", selectedForPlot || deactivated ? "true" : "false");
      });
      updateModeSummary();
    }

    function setActionTab(nextTab) {
      ctx.actionTab = nextTab === "recompute" ? "recompute" : "plot";
      if (ctx.tabPlotBtn) {
        const active = ctx.actionTab === "plot";
        ctx.tabPlotBtn.classList.toggle("is-active", active);
        ctx.tabPlotBtn.setAttribute("aria-selected", active ? "true" : "false");
        ctx.tabPlotBtn.tabIndex = active ? 0 : -1;
      }
      if (ctx.tabRecomputeBtn) {
        const active = ctx.actionTab === "recompute";
        ctx.tabRecomputeBtn.classList.toggle("is-active", active);
        ctx.tabRecomputeBtn.setAttribute("aria-selected", active ? "true" : "false");
        ctx.tabRecomputeBtn.tabIndex = active ? 0 : -1;
      }
      if (ctx.tabPanelPlot) ctx.tabPanelPlot.hidden = ctx.actionTab !== "plot";
      if (ctx.tabPanelRecompute) ctx.tabPanelRecompute.hidden = ctx.actionTab !== "recompute";
      syncDualButtonState();
      updateClearButton();
      updateRemoveButtonsLabel();
      updateModeSummary();
      // Keep rendered plots in sync with active tab semantics.
      plotSelected();
    }

    function updateClearButton() {
      if (!ctx.clearBtn) return;
      const buttons = visibleButtonsForPlot();
      if (!buttons.length) {
        ctx.clearBtn.textContent = "Select all";
        ctx.clearBtn.disabled = true;
        updateModeSummary();
        return;
      }
      const visibleSeries = new Set(buttons.map((btn) => btn.getAttribute("data-series-id")));
      const allSelected = Array.from(visibleSeries).every((id) => ctx.plotSelectionMap.has(id));
      ctx.clearBtn.textContent = allSelected ? "Deselect all" : "Select all";
      ctx.clearBtn.disabled = false;
      updateModeSummary();
    }

    function updateRemoveButtons() {
      const hasSelection = Array.from(ctx.selectedPlotCards.values()).some((setRef) => setRef.size);
      const display = hasSelection ? "inline-flex" : "none";
      ctx.removeButtonsByParam.forEach((btn) => {
        if (btn) btn.style.display = display;
      });
    }

    function updateRemoveButtonsLabel() {
      const label = ctx.actionTab === "recompute" ? "Deactivate selected plotted duals" : "Remove selected plotted duals";
      ctx.removeButtonsByParam.forEach((btn) => {
        if (btn) btn.textContent = label;
      });
    }

    function updateOverlayButtonVisibility(show) {
      if (!ctx.overlayBtn) return;
      ctx.overlayBtn.style.display = show ? "inline-flex" : "none";
      const wrapperEl = ctx.root.querySelector(".dual-wrapper");
      if (!show && wrapperEl) {
        wrapperEl.classList.remove("dual-show-overlay");
        ctx.overlayBtn.classList.remove("is-active");
      }
    }

    function normalizeExpression(raw) {
      if (!raw) return "";
      const trimmed = raw.trim();
      if (!trimmed) return "";
      return trimmed.replace(/^y\s*=\s*/i, "");
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
        const scope = typeof scopeBuilder === "function" ? scopeBuilder(x, idx) : { x };
        try {
          const value = compiled.evaluate(scope);
          const numeric = typeof value === "number" ? value : Number(value);
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
      const plotDiv = ctx.root.querySelector(`#${plotId}`);
      if (!plotDiv || !window.Plotly) return;
      const prevState = ctx.overlayState.get(plotId);
      if (!expr) {
        if (prevState) {
          Plotly.deleteTraces(plotDiv, [prevState.traceIndex]);
          ctx.overlayState.delete(plotId);
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
        mode: "lines",
        name: traceName,
        showlegend: false,
        line: { color: "#ff8a00", width: 2, dash: "dash" },
      }).then(() => {
        ctx.overlayState.set(plotId, { traceIndex });
      });
    }

    function handleOverlayInput(event) {
      const input = event.target;
      if (!input || !window.math) return;
      const axis = input.getAttribute("data-axis");
      const plotId = input.getAttribute("data-plot-id");
      const seriesId = input.getAttribute("data-series-id");
      const generic = ctx.seriesDataByParam[seriesId];
      if (!generic || !axis || !plotId) return;
      const rawExpr = normalizeExpression(input.value);
      if (!rawExpr) {
        input.classList.remove("is-error");
        updateOverlayTrace(plotId, [], [], "");
        return;
      }
      let xValues = [];
      if (generic && generic.by_param && generic.by_param[axis]) {
        xValues = generic.by_param[axis].x_values || [];
      }
      const result = buildOverlayYValues(rawExpr, xValues, (x) => {
        const scope = { x };
        Object.entries(ctx.patternParams).forEach(([name, value]) => {
          const numeric = Number(value);
          if (Number.isFinite(numeric)) setScopeValue(scope, name, numeric);
        });
        ctx.dualParams.forEach((param) => {
          const values = Array.isArray(ctx.paramValuesByName[param]) ? ctx.paramValuesByName[param] : [];
          const idx = tauCtrl.clampIdx(ctx.cursorIndicesByParamState[param] ?? 0, values.length - 1);
          setScopeValue(scope, param, values[idx] !== undefined ? Number(values[idx]) : null);
        });
        setScopeValue(scope, axis, Number(x));
        return scope;
      });
      updateOverlayTrace(plotId, xValues, result.yValues, rawExpr);
      input.classList.toggle("is-error", result.hasError);
    }

    function setRunToggleButton(runId) {
      const button = ctx.root.querySelector(`.run-toggle[data-run-id="${runId}"]`);
      if (!button) return;
      const isVisible = ctx.runVisibility.get(runId) !== false;
      button.textContent = isVisible ? "Hide" : "View";
      button.classList.toggle("is-active", isVisible);
      button.classList.toggle("is-neutral", !isVisible);
    }

    function applyRunVisibilityToTau() {
      if (!window.Plotly) return;
      ctx.dualParams.forEach((axis) => {
        const plotId = `tau-plot-${ctx.sanitizeId(axis)}`;
        const plotDiv = ctx.root.querySelector(`#${plotId}`);
        if (!plotDiv) return;
        const runMap = ctx.tauTraceRegistry.get(axis) || new Map();
        runMap.forEach((traceIndex, runId) => {
          const visible = ctx.runVisibility.get(runId) !== false;
          Plotly.restyle(plotDiv, { visible: visible ? true : "legendonly" }, [traceIndex]);
        });
      });
    }

    function applyRunVisibilityToDual() {
      if (!window.Plotly) return;
      ctx.dualTraceRegistry.forEach((runMap, plotId) => {
        const plotDiv = ctx.root.querySelector(`#${plotId}`);
        if (!plotDiv) return;
        runMap.forEach((traceIndex, runId) => {
          const visible = ctx.runVisibility.get(runId) !== false;
          Plotly.restyle(plotDiv, { visible: visible ? true : "legendonly" }, [traceIndex]);
        });
      });
    }

    function toggleRunVisibility(runId) {
      const current = ctx.runVisibility.get(runId) !== false;
      ctx.runVisibility.set(runId, !current);
      setRunToggleButton(runId);
      applyRunVisibilityToTau();
      applyRunVisibilityToDual();
    }

    function clearDualPlots() {
      ctx.dualParams.forEach((param) => {
        const grid = ctx.root.querySelector(`#dual-plot-${ctx.sanitizeId(param)}`);
        if (grid) grid.innerHTML = "";
      });
      ctx.selectedPlotCards.forEach((setRef) => setRef.clear());
      ctx.overlayState.clear();
      ctx.dualTraceRegistry.clear();
      updateOverlayButtonVisibility(false);
      updateRemoveButtons();
    }

    function togglePlotCardSelection(card, paramName) {
      const seriesId = card.getAttribute("data-series-id");
      if (!seriesId) return;
      const targetSet = ctx.selectedPlotCards.get(paramName) || ctx.selectedPlotCards.values().next().value;
      if (!targetSet) return;
      const select = !targetSet.has(seriesId);
      const apply = (setRef) => {
        if (select) setRef.add(seriesId);
        else setRef.delete(seriesId);
      };
      ctx.selectedPlotCards.forEach((setRef) => apply(setRef));
      ctx.root.querySelectorAll(`.dual-plot-card[data-series-id="${seriesId}"]`).forEach((el) => {
        el.classList.toggle("is-selected", select);
      });
      updateRemoveButtons();
    }

    function removeSelectedSeries(seriesIds) {
      seriesIds.forEach((seriesId) => {
        if (ctx.plotSelectionMap.has(seriesId)) ctx.plotSelectionMap.delete(seriesId);
      });
      syncDualButtonState();
      ctx.selectedPlotCards.forEach((setRef) => setRef.clear());
      updateDeactivatedList();
      updateClearButton();
      clearDualPlots();
      plotSelected();
      updateRemoveButtons();
    }

    function deactivateSelectedSeries(seriesIds) {
      if (!seriesIds || !seriesIds.size) return;
      seriesIds.forEach((seriesId) => {
        if (ctx.deactivatedForRecompute.has(seriesId)) return;
        const generic = ctx.seriesDataByParam[seriesId];
        const label = (generic && generic.label) || seriesId;
        setDeactivatedSeries(seriesId, label);
      });
      ctx.selectedPlotCards.forEach((setRef) => setRef.clear());
      updateDeactivatedList();
      syncDualButtonState();
      updateRemoveButtons();
      plotSelected();
    }

    function baseSeriesForParam(seriesId, paramName) {
      const generic = (ctx.seriesDataByParam[seriesId] && ctx.seriesDataByParam[seriesId].by_param && ctx.seriesDataByParam[seriesId].by_param[paramName]) || null;
      if (generic && Array.isArray(generic.x_values) && Array.isArray(generic.y_values)) {
        return { x: generic.x_values, y: generic.y_values, allZero: Boolean(generic.all_zero) };
      }
      return null;
    }

    function runSeriesForParam(run, seriesId, paramName) {
      const generic = (run.series_data_by_param || {})[seriesId];
      const genericParam = generic && generic.by_param && generic.by_param[paramName];
      if (genericParam && Array.isArray(genericParam.x_values) && Array.isArray(genericParam.y_values)) {
        return { x: genericParam.x_values, y: genericParam.y_values };
      }
      return null;
    }

    function applyZeroFilter() {
      ctx.root.querySelectorAll(".dual-button").forEach((btn) => {
        const seriesId = btn.getAttribute("data-series-id");
        const section = btn.getAttribute("data-section");
        const payload = baseSeriesForParam(seriesId, section || "");
        const isAllZero = Boolean(payload && payload.allZero);
        btn.classList.toggle("is-all-zero", Boolean(isAllZero));
        if (ctx.hideZero && isAllZero) {
          if (ctx.plotSelectionMap.has(seriesId)) ctx.plotSelectionMap.delete(seriesId);
          btn.classList.add("is-hidden");
        } else {
          btn.classList.remove("is-hidden");
        }
      });
      ctx.root.querySelectorAll(".dual-constraint-group").forEach((group) => {
        const buttons = Array.from(group.querySelectorAll(".dual-button"));
        const hasVisibleButton = buttons.some((btn) => !btn.classList.contains("is-hidden"));
        group.classList.toggle("is-empty", !hasVisibleButton);
      });
      updateDeactivatedList();
      syncDualButtonState();
      updateClearButton();
    }

    function plotSelected() {
      const plotGridsByParam = new Map(ctx.dualParams.map((param) => [param, ctx.root.querySelector(`#dual-plot-${ctx.sanitizeId(param)}`)]));
      const seriesIds = Array.from(ctx.plotSelectionMap.keys()).filter((seriesId) => {
        if (ctx.actionTab !== "recompute") return true;
        return !ctx.deactivatedForRecompute.has(seriesId);
      });
      plotGridsByParam.forEach((grid) => {
        if (grid) grid.innerHTML = "";
      });
      ctx.selectedPlotCards.forEach((setRef) => setRef.clear());
      ctx.overlayState.clear();
      ctx.dualTraceRegistry.clear();
      updateRemoveButtons();

      if (!seriesIds.length) {
        plotGridsByParam.forEach((grid) => {
          if (grid) grid.textContent = "Select dual values to plot.";
        });
        updateOverlayButtonVisibility(false);
        return;
      }

      updateOverlayButtonVisibility(true);
      const grouped = new Map();
      seriesIds.forEach((seriesId) => {
        const generic = ctx.seriesDataByParam[seriesId];
        if (!generic) return;
        const constraint = generic.constraint || "Other";
        if (!grouped.has(constraint)) grouped.set(constraint, []);
        grouped.get(constraint).push(seriesId);
      });

      grouped.forEach((ids, constraint) => {
        const constraintLabel = ctx.escapeHtml(constraint);
        const constraintGridByParam = new Map();
        ctx.dualParams.forEach((param) => {
          const grid = plotGridsByParam.get(param);
          if (!grid) return;
          const section = document.createElement("div");
          section.className = "dual-plot-constraint";
          section.innerHTML = `<div class="dual-plot-constraint-title">${constraintLabel}</div><div class="dual-plot-cards" data-constraint="${constraintLabel}" data-param="${ctx.escapeHtml(param)}"></div>`;
          grid.appendChild(section);
          constraintGridByParam.set(param, section.querySelector(".dual-plot-cards"));
        });

        ids.forEach((seriesId) => {
          const generic = ctx.seriesDataByParam[seriesId];
          const label = (generic && generic.label) || seriesId;
          if (!generic) return;
          const safeId = ctx.sanitizeId(seriesId);
          const safeKey = `${safeId}-${Math.random().toString(36).slice(2, 8)}`;
          ctx.dualParams.forEach((param) => {
            const payload = baseSeriesForParam(seriesId, param);
            if (!payload || !Array.isArray(payload.y)) return;
            const count = payload.y.filter((value) => value !== null && Number.isFinite(value)).length;
            if (!count) return;
            const plotId = `${ctx.sanitizeId(param)}-${safeKey}`;
            const card = document.createElement("div");
            card.className = "dual-plot-card";
            card.setAttribute("data-series-id", seriesId);
            card.innerHTML = `<div class="dual-plot-card-title">${ctx.formatDualLabel(label)}</div><div id="${plotId}" class="dual-plot-chart"></div><input class="dual-overlay-input" type="text" placeholder="${ctx.randomOverlayPlaceholder()}" data-series-id="${ctx.escapeHtml(seriesId)}" data-axis="${ctx.escapeHtml(param)}" data-plot-id="${plotId}">`;
            const container = constraintGridByParam.get(param);
            if (!container) return;
            container.appendChild(card);
            card.addEventListener("click", () => togglePlotCardSelection(card, param));
            const traces = [
              {
                x: payload.x,
                y: payload.y,
                mode: count <= 1 ? "markers" : "lines",
                name: "Baseline",
                line: { color: "#9aa0a6", width: 2 },
                hovertemplate: `${ctx.escapeHtml(param)}=%{x:.3f}<br>Baseline=%{y:.3e}<extra></extra>`,
                showlegend: false,
              },
            ];
            const runMap = new Map();
            ctx.dualRuns.forEach((run) => {
              const runPayload = runSeriesForParam(run, seriesId, param);
              if (!runPayload || !Array.isArray(runPayload.y)) return;
              const runCount = runPayload.y.filter((value) => value !== null && Number.isFinite(value)).length;
              if (!runCount) return;
              const traceIndex = traces.length;
              traces.push({
                x: runPayload.x,
                y: runPayload.y,
                mode: runCount <= 1 ? "markers" : "lines",
                name: run.name || run.id || "Run",
                line: { color: run.color || "#f97316", width: 2 },
                hovertemplate: `${ctx.escapeHtml(param)}=%{x:.3f}<br>${ctx.escapeHtml(run.name || run.id || "Run")}=%{y:.3e}<extra></extra>`,
                showlegend: false,
                visible: ctx.runVisibility.get(run.id) !== false ? true : "legendonly",
              });
              runMap.set(run.id, traceIndex);
            });
            ctx.dualTraceRegistry.set(plotId, runMap);
            Plotly.newPlot(plotId, traces, { autosize: true, xaxis: { title: "", tickfont: { size: 9 } }, yaxis: { title: "", tickfont: { size: 9 } }, margin: { t: 10, l: 30, r: 10, b: 15 } }, { displayModeBar: false, responsive: true });
            const overlayInput = card.querySelector(".dual-overlay-input");
            if (overlayInput) {
              overlayInput.addEventListener("focus", () => overlayInput.setAttribute("placeholder", ""));
              overlayInput.addEventListener("click", (event) => event.stopPropagation());
              overlayInput.addEventListener("input", (event) => {
                const key = overlayInput.getAttribute("data-plot-id");
                if (ctx.overlayDebounce.has(key)) clearTimeout(ctx.overlayDebounce.get(key));
                ctx.overlayDebounce.set(key, setTimeout(() => ctx.ensureMath(() => handleOverlayInput(event)), 250));
              });
            }
          });
        });
      });
      ctx.dualParams.forEach((param) => {
        const grid = plotGridsByParam.get(param);
        if (grid && !grid.childElementCount) grid.textContent = `No ${param}-series data for selected dual values.`;
      });
    }

    return {
      dualSectionsHtml,
      dualPlotsHtml,
      visibleButtonsForPlot,
      updateDeactivatedList,
      syncDualButtonState,
      setActionTab,
      updateClearButton,
      updateRemoveButtons,
      updateOverlayButtonVisibility,
      setRunToggleButton,
      applyRunVisibilityToTau,
      applyRunVisibilityToDual,
      toggleRunVisibility,
      clearDualPlots,
      removeSelectedSeries,
      deactivateSelectedSeries,
      applyZeroFilter,
      plotSelected,
      deactivatedLabels,
      setDeactivatedSeries,
    };
  }

  ns.dualPlots = { createDualController };
})();
