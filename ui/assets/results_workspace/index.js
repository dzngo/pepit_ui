function renderResultsWorkspace(component) {
  const ns = (globalThis.__resultsWorkspace = globalThis.__resultsWorkspace || {});
  const runtime = ns.renderResultsWorkspaceRuntime;
  if (typeof runtime !== "function") {
    throw new Error("results_workspace runtime is not loaded.");
  }
  return runtime(component);
}

globalThis.__resultsWorkspace = globalThis.__resultsWorkspace || {};
globalThis.__resultsWorkspace.entry = renderResultsWorkspace;

export default renderResultsWorkspace;
