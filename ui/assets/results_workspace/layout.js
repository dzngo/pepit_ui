(() => {
  const ns = (globalThis.__resultsWorkspace = globalThis.__resultsWorkspace || {});

  ns.sanitizeId = function sanitizeId(value) {
    return String(value).replace(/[^a-zA-Z0-9_-]/g, "-");
  };

  ns.layout = {
    resizeAllPlots(root) {
      if (!window.Plotly || !window.Plotly.Plots || typeof window.Plotly.Plots.resize !== "function") return;
      root.querySelectorAll(".dual-plot-chart").forEach((plotDiv) => {
        if (!plotDiv || !plotDiv.data) return;
        window.Plotly.Plots.resize(plotDiv);
      });
    },
    bindLegendToggle({ legendToggleBtn, shellEl, writeLegendCollapsed, onResized }) {
      if (!legendToggleBtn || !shellEl) return;
      legendToggleBtn.addEventListener("click", () => {
        const collapsed = !shellEl.classList.contains("is-legend-collapsed");
        shellEl.classList.toggle("is-legend-collapsed", collapsed);
        legendToggleBtn.textContent = collapsed ? "→" : "←";
        legendToggleBtn.setAttribute("aria-expanded", collapsed ? "false" : "true");
        legendToggleBtn.setAttribute(
          "aria-label",
          collapsed ? "Expand recompute runs legend" : "Collapse recompute runs legend"
        );
        writeLegendCollapsed(collapsed);
        requestAnimationFrame(() => requestAnimationFrame(onResized));
      });
    },
  };
})();
