(() => {
  const ns = (globalThis.__resultsWorkspace = globalThis.__resultsWorkspace || {});

  ns.constants = {
    RUN_COLORS: ['#0ea5e9', '#14b8a6', '#e11d48', '#f4ec0b', '#22c55e', '#ef4444', '#3b82f6', '#84cc16', '#a855f7'],
    CURVE_PREDICTION_EXAMPLES: ['1/log(x)', 'sin(x)', '2*x^2 + 3', 'log(x)', 'sqrt(x)', 'exp(-x)'],
    LEGEND_COLLAPSED_KEY: 'dual_legend_collapsed',
    DEFAULT_METRIC: 'non_zero_pct_with_none',
  };

  ns.stableHash = function stableHash(text) {
    const value = String(text || '');
    let hash = 0;
    for (let i = 0; i < value.length; i += 1) {
      hash = (hash * 31 + value.charCodeAt(i)) >>> 0;
    }
    return hash;
  };

  ns.runColorFor = function runColorFor(runId, fallbackIndex = 0) {
    const colors = ns.constants.RUN_COLORS;
    if (!colors.length) return '#0ea5e9';
    const hash = ns.stableHash(runId || fallbackIndex);
    return colors[hash % colors.length];
  };
})();
