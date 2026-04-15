(() => {
  const ns = (globalThis.__resultsWorkspace = globalThis.__resultsWorkspace || {});

  /**
   * Event envelope emitted to Streamlit component:
   * - { cursor: {...} }
   * - { metric: {...} }
   * - { remove_run: {...} }
   * - { recompute: {...} }
   */
  ns.emitComponentEvent = function emitComponentEvent(setTriggerValue, payload) {
    setTriggerValue(payload);
  };

  ns.events = {
    createCursorEmitter({ setTriggerValue, getCursor, getLocalByAxis, getPatterns }) {
      let lastCursorEventKey = JSON.stringify({
        cursor_by_param: getCursor(),
        local_by_axis: getLocalByAxis(),
        patterns: getPatterns(),
      });
      return function emitCursorIfChanged() {
        const cursorByParam = getCursor();
        const localCursorByAxis = getLocalByAxis();
        const patternsByParam = getPatterns();
        const currentKey = JSON.stringify({
          cursor_by_param: cursorByParam,
          local_by_axis: localCursorByAxis,
          patterns: patternsByParam,
        });
        if (currentKey === lastCursorEventKey) return;
        lastCursorEventKey = currentKey;
        setTriggerValue("cursor", {
          request_id: Date.now(),
          cursor_indices_by_param: cursorByParam,
          local_cursor_indices_by_axis: localCursorByAxis,
          patterns_by_param: patternsByParam,
        });
      };
    },
  };
})();
