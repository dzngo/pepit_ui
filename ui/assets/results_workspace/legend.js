(() => {
  const ns = (globalThis.__resultsWorkspace = globalThis.__resultsWorkspace || {});

  ns.escapeHtml = function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  };

  ns.formatSubscriptText = function formatSubscriptText(text) {
    if (!text) return "";
    const value = String(text);
    let out = "";
    let i = 0;
    while (i < value.length) {
      const ch = value[i];
      if (ch !== "_") {
        out += ns.escapeHtml(ch);
        i += 1;
        continue;
      }
      if (i + 1 >= value.length) {
        out += ns.escapeHtml(ch);
        i += 1;
        continue;
      }
      if (value[i + 1] === "{") {
        const end = value.indexOf("}", i + 2);
        if (end === -1) {
          out += ns.escapeHtml(value.slice(i));
          break;
        }
        out += `<sub>${ns.escapeHtml(value.slice(i + 2, end))}</sub>`;
        i = end + 1;
        continue;
      }
      let end = i + 1;
      while (end < value.length && /[a-zA-Z0-9*]/.test(value[end])) end += 1;
      if (end === i + 1) {
        out += `<sub>${ns.escapeHtml(value[i + 1])}</sub>`;
        i += 2;
        continue;
      }
      out += `<sub>${ns.escapeHtml(value.slice(i + 1, end))}</sub>`;
      i = end;
    }
    return out;
  };

  ns.formatDualLabel = function formatDualLabel(text) {
    if (!text) return "";
    const value = String(text);
    const separator = " | ";
    const splitIndex = value.indexOf(separator);
    if (splitIndex !== -1) {
      const left = value.slice(0, splitIndex);
      const right = value.slice(splitIndex + separator.length);
      return `${ns.escapeHtml(left)}${separator}${ns.formatSubscriptText(right)}`;
    }
    return ns.formatSubscriptText(value);
  };

  ns.readLegendCollapsed = function readLegendCollapsed() {
    const key = (ns.constants && ns.constants.LEGEND_COLLAPSED_KEY) || "dual_legend_collapsed";
    try {
      return window.sessionStorage.getItem(key) === "1";
    } catch (err) {
      return false;
    }
  };

  ns.writeLegendCollapsed = function writeLegendCollapsed(collapsed) {
    const key = (ns.constants && ns.constants.LEGEND_COLLAPSED_KEY) || "dual_legend_collapsed";
    try {
      window.sessionStorage.setItem(key, collapsed ? "1" : "0");
    } catch (err) {
      return;
    }
  };

  ns.runRowsHtml = function runRowsHtml(runsById, runVisibility) {
    function groupedDeactivatedDualsHtml(labels) {
      const groups = new Map();
      labels.forEach((rawLabel) => {
        const label = String(rawLabel || "").trim();
        if (!label) return;
        const separator = " | ";
        const splitIndex = label.indexOf(separator);
        if (splitIndex < 0) {
          const bucket = groups.get(label) || [];
          groups.set(label, bucket);
          return;
        }
        const constraint = label.slice(0, splitIndex).trim() || "unknown_constraint";
        const dualValue = label.slice(splitIndex + separator.length).trim();
        const bucket = groups.get(constraint) || [];
        if (dualValue && !bucket.includes(dualValue)) {
          bucket.push(dualValue);
        }
        groups.set(constraint, bucket);
      });

      if (!groups.size) {
        return '<span class="dual-badge">No dual values selected.</span>';
      }

      return Array.from(groups.entries())
        .map(([constraint, dualValues]) => {
          if (!dualValues.length) {
            return `<div class="dual-group-row"><strong>${ns.escapeHtml(constraint)}</strong></div>`;
          }
          const valuesHtml = dualValues
            .map((value) => `<span class="dual-badge">${ns.formatSubscriptText(value)}</span>`)
            .join("");
          return (
            `<div class="dual-group-row">` +
            `<strong>${ns.escapeHtml(constraint)}:</strong>` +
            `<span class="dual-group-tags">${valuesHtml}</span>` +
            `</div>`
          );
        })
        .join("");
    }

    if (!runsById.size) {
      return '<div class="dual-selected-list">No recompute runs yet.</div>';
    }
    const rows = [];
    rows.push(
      '<div class="run-row" style="margin-bottom:10px;">' +
        '<div style="display:flex;align-items:center;gap:8px;">' +
          '<span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:#9aa0a6;"></span>' +
          "<strong>Baseline</strong>" +
        "</div>" +
        '<div class="dual-selected-list" style="margin-top:8px;">All duals</div>' +
      "</div>"
    );
    Array.from(runsById.values()).forEach((run) => {
      const isVisible = runVisibility.get(run.id) !== false;
      const buttonText = isVisible ? "Hide" : "View";
      const buttonStateClass = isVisible ? "is-active" : "is-neutral";
      const labels = Array.isArray(run.selected_labels) ? run.selected_labels : [];
      const groupedDuals = groupedDeactivatedDualsHtml(labels);
      rows.push(
        `<div class="run-row" data-run-row="${ns.escapeHtml(run.id)}">` +
          `<div style="display:flex;justify-content:space-between;align-items:center;gap:12px;">` +
            `<div><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:${ns.escapeHtml(run.color || "#f97316")};margin-right:8px;"></span><strong>${ns.escapeHtml(run.name || run.id)}</strong></div>` +
            '<div style="display:flex;align-items:center;gap:8px;">' +
              `<button type="button" class="dual-toggle-button run-toggle ${buttonStateClass}" data-run-id="${ns.escapeHtml(run.id)}">${buttonText}</button>` +
              `<button type="button" class="dual-remove-button run-remove" data-run-id="${ns.escapeHtml(run.id)}">Remove</button>` +
            "</div>" +
          "</div>" +
          '<details style="margin-top:8px;">' +
            `<summary style="cursor:pointer;font-weight:600;">View deactivated duals (${labels.length})</summary>` +
            `<div class="dual-selected-list" style="margin-top:8px;">${groupedDuals}</div>` +
          "</details>" +
        "</div>"
      );
    });
    return rows.join("");
  };
})();
