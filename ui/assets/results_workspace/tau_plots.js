(() => {
  const ns = (globalThis.__resultsWorkspace = globalThis.__resultsWorkspace || {});

  function clampIdx(value, maxIdx) {
    return Math.min(Math.max(Number(value || 0), 0), Math.max(maxIdx, 0));
  }

  function normalizeExpression(raw) {
    if (!raw) return "";
    const trimmed = raw.trim();
    if (!trimmed) return "";
    return trimmed.replace(/^y\s*=\s*/i, "");
  }

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

  function createTauController(ctx) {
    function currentCursorIndicesByParam() {
      const indices = {};
      ctx.dualParams.forEach((name) => {
        const values = Array.isArray(ctx.paramValuesByName[name]) ? ctx.paramValuesByName[name] : [];
        indices[name] = clampIdx(ctx.cursorIndicesByParamState[name] ?? 0, values.length - 1);
      });
      return indices;
    }

    function currentLocalCursorIndicesByAxis() {
      const byAxis = {};
      const cursor = currentCursorIndicesByParam();
      ctx.dualParams.forEach((axis) => {
        const axisLocals = ctx.localCursorIndicesByAxisState[axis] || {};
        const nextAxis = {};
        ctx.dualParams.forEach((name) => {
          if (name === axis) return;
          const values = Array.isArray(ctx.paramValuesByName[name]) ? ctx.paramValuesByName[name] : [];
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
      const patterns = { ...ctx.patternsByParamPayload };
      ctx.tauPatternInputsByParam.forEach((inputEl, param) => {
        patterns[param] = inputEl ? String(inputEl.value || "") : String(patterns[param] || "");
      });
      return patterns;
    }

    function extraLocalSlidersHtml(axisParam) {
      const skip = new Set([axisParam]);
      const items = ctx.dualParams.filter((name) => !skip.has(name));
      if (!items.length) return "";
      return items
        .map((name) => {
          const values = Array.isArray(ctx.paramValuesByName[name]) ? ctx.paramValuesByName[name] : [];
          const maxIdx = Math.max(values.length - 1, 0);
          const axisPayload = ctx.localCursorIndicesByAxisPayload[axisParam] || {};
          const defaultIdx = clampIdx(axisPayload[name] ?? ctx.cursorIndicesByParamPayload[name] ?? 0, maxIdx);
          return (
            `<div style="display:flex;align-items:center;gap:8px;">` +
            `<strong>Local ${ctx.escapeHtml(name)}</strong>` +
            `<input type="range" class="tau-local-slider" data-axis-param="${ctx.escapeHtml(axisParam)}" data-param="${ctx.escapeHtml(name)}" min="0" max="${maxIdx}" step="1" value="${defaultIdx}" style="flex:1;" />` +
            `<span class="tau-local-value" data-axis-param="${ctx.escapeHtml(axisParam)}" data-param="${ctx.escapeHtml(name)}"></span>` +
            `</div>`
          );
        })
        .join("");
    }

    function tauPanelsHtml() {
      if (!ctx.dualParams.length) return "";
      return ctx.dualParams
        .map((axisParam) => {
          const safeAxis = ctx.sanitizeId(axisParam);
          const axisValues = Array.isArray(ctx.paramValuesByName[axisParam]) ? ctx.paramValuesByName[axisParam] : [];
          const globalIdx = clampIdx(ctx.cursorIndicesByParamPayload[axisParam] ?? 0, axisValues.length - 1);
          const patternValue = String(ctx.patternsByParamPayload[axisParam] ?? "");
          return (
            `<div class="dual-panel">` +
            `<div class="dual-plot-section">` +
            `${extraLocalSlidersHtml(axisParam)}` +
            `<div id="tau-plot-${safeAxis}" class="dual-plot-chart tau-plot-chart"></div>` +
            `<div class="tau-global-control">` +
            `<div style="display:flex;align-items:center;gap:8px;">` +
            `<strong>Global ${ctx.escapeHtml(axisParam)}</strong>` +
            `<input type="range" class="tau-global-slider" data-param="${ctx.escapeHtml(axisParam)}" min="0" max="${Math.max(axisValues.length - 1, 0)}" step="1" value="${globalIdx}" style="flex:1;" />` +
            `<span class="tau-global-value" data-param="${ctx.escapeHtml(axisParam)}"></span>` +
            `</div>` +
            `</div>` +
            `<input class="dual-prediction-input tau-pattern-input" data-param="${ctx.escapeHtml(axisParam)}" type="text" value="${ctx.escapeHtml(patternValue)}" placeholder="${ctx.randomPredictionPlaceholder()}">` +
            `<div class="tau-pattern-hint" data-param="${ctx.escapeHtml(axisParam)}" style="font-size:12px;color:#666;"></div>` +
            `<div class="tau-pattern-error" data-param="${ctx.escapeHtml(axisParam)}"></div>` +
            `</div>` +
            `</div>`
          );
        })
        .join("");
    }

    function tauSeriesForParam(paramName, fallbackX, fallbackY, fallbackCursorIdx) {
      const payload = ctx.tauSeriesByParam[paramName];
      if (!payload || !Array.isArray(payload.x_values) || !Array.isArray(payload.y_values)) {
        return { xValues: fallbackX, yValues: fallbackY, cursorIdx: fallbackCursorIdx };
      }
      const cursorIdx =
        paramName === "gamma" || paramName === "n"
          ? fallbackCursorIdx
          : Number.isFinite(Number(payload.cursor_idx))
            ? Number(payload.cursor_idx)
            : fallbackCursorIdx;
      return { xValues: payload.x_values, yValues: payload.y_values, cursorIdx };
    }

    function updateTauLabels() {
      ctx.tauGlobalValuesByParam.forEach((valueEl, param) => {
        if (!valueEl) return;
        const values = Array.isArray(ctx.paramValuesByName[param]) ? ctx.paramValuesByName[param] : [];
        const idx = clampIdx(ctx.cursorIndicesByParamState[param] ?? 0, values.length - 1);
        valueEl.textContent = values[idx] !== undefined ? String(values[idx]) : "";
      });
      ctx.tauLocalValues.forEach((valueEl) => {
        const axis = valueEl.getAttribute("data-axis-param");
        const param = valueEl.getAttribute("data-param");
        if (!axis || !param) return;
        const values = Array.isArray(ctx.paramValuesByName[param]) ? ctx.paramValuesByName[param] : [];
        const idx = clampIdx(getLocalIndex(axis, param), values.length - 1);
        valueEl.textContent = values[idx] !== undefined ? String(values[idx]) : "";
      });
    }

    function buildPatternHintText(primaryAxisName) {
      const entries = Object.entries(ctx.patternParams).filter(([, value]) => Number.isFinite(Number(value)));
      const parts = [];
      if (entries.length) {
        const paramsHint = entries.map(([name, value]) => `${name}=${Number(value).toString()}`).join(", ");
        parts.push(`Available parameters: ${paramsHint}.`);
      }
      if (ctx.patternInvalidParams.length) parts.push(`Ignored (non-scalar): ${ctx.patternInvalidParams.join(", ")}.`);
      if (ctx.patternConflictParams.length) parts.push(`Conflicts: ${ctx.patternConflictParams.join(", ")}.`);
      parts.push(`Variables: x, ${ctx.dualParams.join(", ")}.`);
      return parts.join(" ");
    }

    function tauPatternPrediction(axis, exprRaw) {
      const expr = normalizeExpression(exprRaw || "");
      if (!expr) return { yValues: null, error: "" };
      if (!window.math) return { yValues: null, error: "Math library loading..." };
      let compiled;
      try {
        compiled = window.math.compile(expr);
      } catch (err) {
        return { yValues: null, error: `Invalid expression: ${err.message || err}` };
      }
      const axisValues = Array.isArray(ctx.paramValuesByName[axis]) ? ctx.paramValuesByName[axis] : [];
      const yValues = [];
      let hasFinite = false;
      for (let i = 0; i < axisValues.length; i += 1) {
        const x = Number(axisValues[i]);
        if (!Number.isFinite(x)) {
          yValues.push(null);
          continue;
        }
        const scope = {};
        Object.entries(ctx.patternParams).forEach(([name, value]) => {
          const numeric = Number(value);
          if (Number.isFinite(numeric)) setScopeValue(scope, name, numeric);
        });
        ctx.dualParams.forEach((param) => {
          const values = Array.isArray(ctx.paramValuesByName[param]) ? ctx.paramValuesByName[param] : [];
          const idx = clampIdx(getLocalIndex(axis, param), values.length - 1);
          const resolved = values[idx] !== undefined ? Number(values[idx]) : null;
          setScopeValue(scope, param, resolved);
        });
        scope.x = x;
        setScopeValue(scope, axis, x);
        try {
          const value = compiled.evaluate(scope);
          const numeric = typeof value === "number" ? value : Number(value);
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
      if (!hasFinite) return { yValues, error: "Expression did not produce finite values." };
      return { yValues, error: "" };
    }

    function renderTauPlots() {
      if (!window.Plotly) return;
      ctx.dualParams.forEach((axisParam) => {
        const plotId = `tau-plot-${ctx.sanitizeId(axisParam)}`;
        const plotDiv = ctx.root.querySelector(`#${plotId}`);
        if (!plotDiv) return;
        const axisValues = Array.isArray(ctx.paramValuesByName[axisParam]) ? ctx.paramValuesByName[axisParam] : [];
        const cursorIdx = clampIdx(ctx.cursorIndicesByParamState[axisParam] ?? 0, axisValues.length - 1);
        const axisSeries = tauSeriesForParam(axisParam, axisValues, [], cursorIdx);
        const traces = [
          {
            x: axisSeries.xValues,
            y: axisSeries.yValues,
            mode: "lines",
            line: { color: "#9aa0a6", width: 2 },
            hovertemplate: `${ctx.escapeHtml(axisParam)}=%{x:.3f}<br>tau=%{y:.3e}<extra></extra>`,
            showlegend: false,
          },
        ];
        const currentTau = axisSeries.yValues[cursorIdx] ?? null;
        if (currentTau !== null && currentTau !== undefined) {
          traces.push({
            x: [axisSeries.xValues[cursorIdx]],
            y: [currentTau],
            mode: "markers",
            marker: { size: 12 },
            hovertemplate: `${ctx.escapeHtml(axisParam)}=%{x:.3f}<br>tau=%{y:.3e}<extra></extra>`,
            showlegend: false,
          });
        }

        const runMap = new Map();
        ctx.dualRuns.forEach((run) => {
          const runSeries = ((run.tau_series_by_param || {})[axisParam] || {});
          if (!Array.isArray(runSeries.y_values)) return;
          const traceIndex = traces.length;
          traces.push({
            x: Array.isArray(runSeries.x_values) ? runSeries.x_values : axisSeries.xValues,
            y: runSeries.y_values,
            mode: "lines",
            line: { color: run.color || "#f97316", width: 2 },
            hovertemplate: `${ctx.escapeHtml(axisParam)}=%{x:.3f}<br>${ctx.escapeHtml(run.name || run.id || "Run")}=%{y:.3e}<extra></extra>`,
            showlegend: false,
            visible: ctx.runVisibility.get(run.id) !== false ? true : "legendonly",
          });
          runMap.set(run.id, traceIndex);
        });
        ctx.tauTraceRegistry.set(axisParam, runMap);

        const patternInput = ctx.tauPatternInputsByParam.get(axisParam);
        const prediction = tauPatternPrediction(axisParam, patternInput ? patternInput.value : "");
        const errorEl = ctx.tauPatternErrorsByParam.get(axisParam);
        if (errorEl) errorEl.textContent = prediction.error;
        if (patternInput) patternInput.classList.toggle("is-error", Boolean(prediction.error));
        ctx.tauPredictionTraceRegistry.delete(axisParam);
        if (prediction.yValues && !prediction.error) {
          const traceIndex = traces.length;
          traces.push({
            x: axisSeries.xValues,
            y: prediction.yValues,
            mode: "lines",
            line: { color: "#ff8a00", width: 2, dash: "dash" },
            hovertemplate: `${ctx.escapeHtml(axisParam)}=%{x:.3f}<br>prediction=%{y:.3e}<extra></extra>`,
            showlegend: false,
            visible: ctx.tauPredictionVisible ? true : "legendonly",
          });
          ctx.tauPredictionTraceRegistry.set(axisParam, traceIndex);
        }

        const fixedParts = ctx.dualParams
          .filter((name) => name !== axisParam)
          .map((name) => {
            const values = Array.isArray(ctx.paramValuesByName[name]) ? ctx.paramValuesByName[name] : [];
            const idx = clampIdx(getLocalIndex(axisParam, name), values.length - 1);
            return `${name}=${values[idx] !== undefined ? values[idx] : ""}`;
          });
        const title = fixedParts.length ? `Tau vs ${axisParam} (${fixedParts.join(", ")})` : `Tau vs ${axisParam}`;
        Plotly.newPlot(
          plotId,
          traces,
          {
            autosize: true,
            title: { text: title },
            xaxis: { title: axisParam },
            yaxis: { title: "tau" },
            height: 320,
            margin: { t: 40, l: 45, r: 10, b: 35 },
          },
          { displayModeBar: false, responsive: true }
        );
      });
      updateTauLabels();
    }

    function setTauPredictionTraceVisibility(isVisible) {
      if (!window.Plotly) return;
      ctx.tauPredictionTraceRegistry.forEach((traceIndex, axisParam) => {
        const plotId = `tau-plot-${ctx.sanitizeId(axisParam)}`;
        const plotDiv = ctx.root.querySelector(`#${plotId}`);
        if (!plotDiv || !plotDiv.data || !plotDiv.data[traceIndex]) {
          ctx.tauPredictionTraceRegistry.delete(axisParam);
          return;
        }
        Plotly.restyle(plotDiv, { visible: isVisible ? true : "legendonly" }, [traceIndex]);
      });
    }

    function initTauStateAndDom() {
      ctx.cursorIndicesByParamState = { ...ctx.cursorIndicesByParamPayload };
      ctx.localCursorIndicesByAxisState = {};
      ctx.dualParams.forEach((name) => {
        const values = Array.isArray(ctx.paramValuesByName[name]) ? ctx.paramValuesByName[name] : [];
        ctx.cursorIndicesByParamState[name] = clampIdx(ctx.cursorIndicesByParamState[name] ?? 0, values.length - 1);
      });
      ctx.dualParams.forEach((axis) => {
        const axisPayload = ctx.localCursorIndicesByAxisPayload[axis] || {};
        const axisState = {};
        ctx.dualParams.forEach((name) => {
          if (name === axis) return;
          const values = Array.isArray(ctx.paramValuesByName[name]) ? ctx.paramValuesByName[name] : [];
          const fallback = ctx.cursorIndicesByParamState[name] ?? 0;
          axisState[name] = clampIdx(axisPayload[name] ?? fallback, values.length - 1);
        });
        ctx.localCursorIndicesByAxisState[axis] = axisState;
      });
      ctx.tauGlobalSlidersByParam.forEach((slider, param) => {
        if (!slider) return;
        const values = Array.isArray(ctx.paramValuesByName[param]) ? ctx.paramValuesByName[param] : [];
        const idx = clampIdx(ctx.cursorIndicesByParamState[param] ?? 0, values.length - 1);
        slider.value = String(idx);
      });
      ctx.tauLocalSliders.forEach((slider) => {
        const axis = slider.getAttribute("data-axis-param");
        const param = slider.getAttribute("data-param");
        if (!axis || !param) return;
        const values = Array.isArray(ctx.paramValuesByName[param]) ? ctx.paramValuesByName[param] : [];
        const idx = clampIdx(
          ((ctx.localCursorIndicesByAxisState[axis] || {})[param]) ?? (ctx.cursorIndicesByParamState[param] ?? 0),
          values.length - 1
        );
        slider.value = String(idx);
      });
    }

    function bindTauEvents(ensurePlotly, ensureMath, emitCursorIfChanged) {
      ctx.tauGlobalSlidersByParam.forEach((slider, param) => {
        if (!slider) return;
        slider.addEventListener("input", () => {
          const idx = Number(slider.value);
          ctx.cursorIndicesByParamState[param] = idx;
          ensurePlotly(renderTauPlots);
        });
        slider.addEventListener("change", () => {
          const idx = Number(slider.value);
          ctx.cursorIndicesByParamState[param] = idx;
          emitCursorIfChanged();
        });
      });
      ctx.tauLocalSliders.forEach((slider) => {
        slider.addEventListener("input", () => {
          const axis = slider.getAttribute("data-axis-param");
          const param = slider.getAttribute("data-param");
          if (!axis || !param) return;
          const idx = Number(slider.value);
          ctx.localCursorIndicesByAxisState[axis] = { ...(ctx.localCursorIndicesByAxisState[axis] || {}), [param]: idx };
          ensurePlotly(renderTauPlots);
        });
        slider.addEventListener("change", () => {
          const axis = slider.getAttribute("data-axis-param");
          const param = slider.getAttribute("data-param");
          if (!axis || !param) return;
          const idx = Number(slider.value);
          ctx.localCursorIndicesByAxisState[axis] = { ...(ctx.localCursorIndicesByAxisState[axis] || {}), [param]: idx };
          ensurePlotly(renderTauPlots);
          emitCursorIfChanged();
        });
      });
      ctx.tauPatternInputsByParam.forEach((inputEl) => {
        if (!inputEl) return;
        inputEl.addEventListener("input", () => {
          ensureMath(() => ensurePlotly(renderTauPlots));
        });
      });
    }

    return {
      clampIdx,
      currentCursorIndicesByParam,
      currentLocalCursorIndicesByAxis,
      getLocalIndex,
      currentPatternsByParam,
      tauPanelsHtml,
      initTauStateAndDom,
      bindTauEvents,
      buildPatternHintText,
      renderTauPlots,
      setTauPredictionTraceVisibility,
    };
  }

  ns.tauPlots = {
    createTauController,
    clampIdx,
  };
})();
