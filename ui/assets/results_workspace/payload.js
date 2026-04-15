(() => {
  const ns = (globalThis.__resultsWorkspace = globalThis.__resultsWorkspace || {});

  /**
   * @typedef {Object} TauPayload
   * @property {Object<string, Object<string, Object>>} tau_series_by_param
   * @property {string[]} param_order
   * @property {Object<string, number[]>} param_values_by_name
   * @property {Object<string, number>} cursor_indices_by_param
   * @property {Object<string, Object<string, number>>} local_cursor_indices_by_axis
   * @property {Object<string, string>} patterns_by_param
   * @property {Object<string, string>} sections_html_by_param
   * @property {Object<string, string>} plot_titles_by_param
   * @property {Object<string, Object<string, Object>>} series_data_by_param
   * @property {Object<string, number>} pattern_params
   * @property {string[]} pattern_invalid_params
   * @property {string[]} pattern_conflict_params
   */

  /**
   * @typedef {Object} DualRun
   * @property {string} id
   * @property {string} [name]
   * @property {boolean} [visible]
   * @property {string[]} [selected_labels]
   * @property {Object<string, Object<string, Object>>} [tau_series_by_param]
   * @property {Object<string, Object>} [series_data_by_param]
   */

  /**
   * @typedef {Object} ResultsWorkspacePayload
   * @property {TauPayload} tau_payload
   * @property {Object<string, Object<string, Object>>} series_data_by_param
   * @property {DualRun[]} dual_runs
   * @property {string[]} selected_series_ids
   * @property {string} metric
   * @property {Object<string, string>} metric_labels
   * @property {string} [css]
   */

  ns.parsePayload = function parsePayload(component) {
    const data = (component && component.data) || {};
    const tauPayload = data.tau_payload || {};
    const metricLabels = data.metric_labels || {};
    const metricKeys = Object.keys(metricLabels);
    const defaultMetric = (ns.constants && ns.constants.DEFAULT_METRIC) || "non_zero_pct_with_none";
    const currentMetric = metricKeys.includes(data.metric) ? data.metric : defaultMetric;

    const seriesDataByParam = data.series_data_by_param || tauPayload.series_data_by_param || {};
    const rawDualRuns = Array.isArray(data.dual_runs) ? data.dual_runs : [];
    const paramOrder = Array.isArray(tauPayload.param_order) ? tauPayload.param_order : [];
    const fallbackParams = Object.keys(tauPayload.param_values_by_name || {});
    const dualParams = (paramOrder.length ? paramOrder : fallbackParams).filter(
      (name, idx, arr) => arr.indexOf(name) === idx
    );

    return {
      data,
      tauPayload,
      metricLabels,
      metricKeys,
      currentMetric,
      seriesDataByParam,
      rawDualRuns,
      dualParams,
      paramValuesByName: tauPayload.param_values_by_name || {},
      cursorIndicesByParamPayload: tauPayload.cursor_indices_by_param || {},
      localCursorIndicesByAxisPayload: tauPayload.local_cursor_indices_by_axis || {},
      patternsByParamPayload: tauPayload.patterns_by_param || {},
      patternParams: tauPayload.pattern_params || {},
      patternInvalidParams: Array.isArray(tauPayload.pattern_invalid_params) ? tauPayload.pattern_invalid_params : [],
      patternConflictParams: Array.isArray(tauPayload.pattern_conflict_params)
        ? tauPayload.pattern_conflict_params
        : [],
      sectionsHtmlByParam: tauPayload.sections_html_by_param || {},
      plotTitlesByParam: tauPayload.plot_titles_by_param || {},
      tauSeriesByParam: tauPayload.tau_series_by_param || {},
      selectedSeriesIds: Array.isArray(data.selected_series_ids) ? data.selected_series_ids : [],
      css: data.css || "",
    };
  };
})();
