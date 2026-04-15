(() => {
  const ns = (globalThis.__resultsWorkspace = globalThis.__resultsWorkspace || {});

  ns.createState = function createState(dualParams) {
    const selectedPlotCards = new Map();
    dualParams.forEach((name) => selectedPlotCards.set(name, new Set()));
    return {
      plotSelectionMap: new Map(),
      deactivatedForRecompute: new Map(),
      selectedPlotCards,
      overlayState: new Map(),
      overlayDebounce: new Map(),
      dualTraceRegistry: new Map(),
      tauTraceRegistry: new Map(),
      hideZero: true,
      actionTab: "plot",
    };
  };
})();
