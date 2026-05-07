(() => {
  const ns = (globalThis.__resultsWorkspace = globalThis.__resultsWorkspace || {});

  ns.createState = function createState(dualParams) {
    const selectedPlotCards = new Map();
    dualParams.forEach((name) => selectedPlotCards.set(name, new Set()));
    return {
      plotSelectionMap: new Map(),
      deactivatedForRecompute: new Map(),
      selectedPlotCards,
      predictionState: new Map(),
      predictionDebounce: new Map(),
      dualTraceRegistry: new Map(),
      tauTraceRegistry: new Map(),
      tauPredictionTraceRegistry: new Map(),
      tauPredictionVisible: false,
      hideZero: true,
      actionTab: "plot",
    };
  };
})();
