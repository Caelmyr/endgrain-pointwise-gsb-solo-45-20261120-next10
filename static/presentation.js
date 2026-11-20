const number = new Intl.NumberFormat("en", {maximumFractionDigits: 2});
const detail = new Intl.NumberFormat("en", {maximumFractionDigits: 3});
const time = new Intl.DateTimeFormat("en", {month: "short", day: "numeric", hour: "2-digit", minute: "2-digit"});

export function el(tag, options = {}, ...children) {
  const node = document.createElement(tag);
  Object.entries(options).forEach(([key, value]) => {
    if (key === "dataset") Object.assign(node.dataset, value);
    else if (key === "text") node.textContent = value;
    else if (key === "onclick") node.addEventListener("click", value);
    else if (key in node) node[key] = value;
    else node.setAttribute(key, value);
  });
  node.append(...children);
  return node;
}

export function displayValue(value, mode, unit = "px", metrics = {}) {
  if (value === null || value === undefined) return "—";
  if (mode === "bool") return value ? "Yes" : "No";
  if (mode === "%") return number.format(value * 100) + "%";
  if (mode === "area") return number.format(value) + " " + unit + "²";
  if (mode === "linear") return number.format(value) + " " + unit;
  if (mode === "density") return number.format(value) + " / " + metrics.density_basis;
  if (mode === "°") return number.format(value) + "°";
  if (mode === "px") return number.format(value) + " px";
  return detail.format(value);
}

export function shortTime(value) {
  return time.format(new Date(value));
}

export function notices(entries) {
  return el("ul", {className: "analysis-notices"}, ...entries.map(text => el("li", {text})));
}

export function button(text, onclick, className = "button outline") {
  return el("button", {type: "button", text, className, onclick});
}

export function link(text, href, className = "button outline") {
  return el("a", {text, href, className});
}

export function createToolInput(spec, current) {
  const fields = {"data-setting": spec.key, name: spec.key};
  const defaults = current ?? spec.default;
  if (spec.values.length) {
    const select = el("select", fields, ...spec.values.map(value => el("option", {value, text: value.replaceAll("_", " ")})));
    select.value = defaults;
    return el("label", {text: spec.title}, select);
  }
  if (typeof spec.default === "boolean") {
    const input = el("input", {...fields, type: "checkbox", checked: defaults});
    return el("label", {className: "checkbox full"}, input, document.createTextNode(spec.title));
  }
  return el("label", {text: spec.title}, el("input", {...fields,
    type: "number", value: defaults, step: "any", inputMode: "decimal", required: true,
  }));
}

function lines(ctx, xs, ys, xf, yf, color) {
  ctx.strokeStyle = color;
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  let fresh = true;
  xs.forEach((x, index) => {
    if (ys[index] === null) {
      fresh = true;
    } else {
      if (fresh) ctx.moveTo(xf(x), yf(ys[index]));
      else ctx.lineTo(xf(x), yf(ys[index]));
      fresh = false;
    }
  });
  ctx.stroke();
}

export function plot(specification) {
  const graph = el("canvas", {width: 600, height: 184, className: "plot",
    role: "img", "aria-label": specification.x_label + " versus " + specification.y_label,
  });
  const ctx = graph.getContext("2d");
  const left = 42, right = 580, top = 12, bottom = 145;
  let xmin = 0, xmax = 1, ymin = 0, ymax = 1;
  if (specification.kind === "histogram") {
    xmax = 31;
    ymax = Math.max(1, ...Object.values(specification.channels).flat());
  } else if (specification.kind === "profile" && specification.x.length) {
    const valid = specification.y.filter(y => y !== null);
    xmin = Math.min(...specification.x);
    xmax = Math.max(...specification.x);
    ymin = valid.length ? Math.min(...valid) - 0.01 : 0;
    ymax = valid.length ? Math.max(...valid) + 0.01 : 1;
  } else if (specification.kind === "bars" && specification.rows.length) {
    xmin = specification.rows[0].x;
    xmax = specification.rows.at(-1).x;
    ymax = Math.max(1e-5, ...specification.rows.map(row => row.value));
  }
  const xf = value => left + (value - xmin) * (right - left) / Math.max(xmax - xmin, 0.000001);
  const yf = value => bottom - (value - ymin) * (bottom - top) / Math.max(ymax - ymin, 0.000001);
  ctx.strokeStyle = "#d7daca";
  ctx.lineWidth = 1;
  ctx.font = "12px system-ui";
  ctx.fillStyle = "#676c5e";
  for (let i = 0; i < 4; i++) {
    const value = ymin + (ymax - ymin) * i / 3;
    const y = yf(value);
    ctx.beginPath();
    ctx.moveTo(left, y);
    ctx.lineTo(right, y);
    ctx.stroke();
    ctx.fillText(detail.format(value), 2, y + 4, 37);
  }
  ctx.textAlign = "center";
  ctx.fillText(specification.x_label, (left + right) / 2, 174, 425);
  ctx.textAlign = "left";
  ctx.fillText(detail.format(xmin), left, 160);
  ctx.textAlign = "right";
  ctx.fillText(detail.format(xmax), right, 160);
  if (specification.kind === "profile") {
    ctx.strokeStyle = "#b98048";
    ctx.lineWidth = 1;
    (specification.markers || []).forEach(mark => {
      const x = xf(mark);
      ctx.beginPath();
      ctx.moveTo(x, top);
      ctx.lineTo(x, bottom);
      ctx.stroke();
    });
    lines(ctx, specification.x, specification.y, xf, yf, "#465438");
  } else if (specification.kind === "bars") {
    const width = Math.max(2, Math.min(35, (right - left) / Math.max(specification.rows.length, 1) * 0.78));
    ctx.fillStyle = "#728565";
    specification.rows.forEach(row => ctx.fillRect(xf(row.x) - width / 2, yf(row.value), width, bottom - yf(row.value)));
  } else {
    Object.entries(specification.channels).forEach(([channel, values]) => {
      const colors = {r: "#b2684d", g: "#698155", b: "#5d759b"};
      lines(ctx, values.map((_, ix) => ix), values, xf, yf, colors[channel]);
    });
  }
  return graph;
}

export function metricCards(tool, result) {
  const points = (tool?.metrics && Object.entries(tool.metrics)) || [];
  return el("div", {className: "metrics"}, ...points.map(([key, [label, mode]]) => el("div", {className: "metric"},
    el("span", {text: label}),
    el("strong", {text: displayValue(result.metrics[key], mode, result.unit, result.metrics)}),
  )));
}

export function objectTable(run, onSegment) {
  const measures = run.result;
  let rows = [];
  let headers = [];
  if (run.kind === "rings") {
    headers = ["Crossing", "Position", "Previous interval", "Origin"];
    rows = (measures.crossings || []).map((entry, ix) => [
      ix + 1, displayValue(entry.distance, "linear", measures.unit),
      displayValue(entry.gap_from_previous, "linear", measures.unit), entry.origin,
    ]);
  } else if (run.kind === "cracks") {
    headers = ["Trace", "Length", "Width", "Junctions", "Edge", "Segments"];
    rows = (measures.objects || []).map(entry => [
      "T" + entry.id, displayValue(entry.length, "linear", measures.unit),
      displayValue(entry.median_width, "linear", measures.unit), entry.junctions,
      entry.touches_support_edge ? "Yes" : "No", entry.segments ?? "—",
    ]);
    if (!rows.length) return null;
    return el("details", {className: "manual", open: true},
      el("summary", {text: "Show whole-trace readings · " + rows.length}),
      table(headers, rows),
      ...crackReview(measures, onSegment),
    );
  } else if (run.kind === "pores") {
    headers = ["Pore", "Area", "Diameter", "Roundness"];
    rows = (measures.objects || []).map(entry => [
      entry.id, displayValue(entry.area, "area", measures.unit),
      displayValue(entry.diameter, "linear", measures.unit), displayValue(entry.roundness),
    ]);
  } else if (run.kind === "colors") {
    headers = ["Swatch", "Qualified pixels", "Coverage"];
    rows = (measures.objects || []).map(entry => [entry.label, entry.pixels, displayValue(entry.coverage, "%")]);
  }
  if (!rows.length) return null;
  return el("details", {className: "manual"}, el("summary", {text: "Show readings · " + rows.length}),
    table(headers, rows),
  );
}

function crackReview(measures, onSegment) {
  const segments = measures.segments || [];
  const forks = measures.forks || [];
  if (!segments.length && !forks.length) return [];
  const unit = measures.unit;
  const segmentRows = segments.map(entry => [
    entry.label, "T" + entry.trace, entry.from, entry.to,
    displayValue(entry.length, "linear", unit),
    entry.share === null || entry.share === undefined ? "—" : displayValue(entry.share, "%"),
    displayValue(entry.median_width, "linear", unit),
    entry.boundary === "region" ? "Region edge" : entry.boundary === "source" ? "Source gap" : "",
  ]);
  const forkRows = forks.map(entry => [
    entry.label, "T" + entry.trace, displayValue(entry.held_length, "linear", unit),
    entry.segments.length ? entry.segments.join(", ") : "—",
    entry.main_segments.length ? entry.main_segments.join(" – ") : "—",
    Object.entries(entry.side_branches || {}).map(([label, angle]) => label + " (" + angle.toFixed(0) + "°)").join(", ") || "—",
  ]);
  const total = measures.metrics.segment_length + measures.metrics.fork_length;
  const reconciliation = el("p",
    {className: "crack-reconcile",
     text: "Segments " + displayValue(measures.metrics.segment_length, "linear", unit)
       + " + held at forks " + displayValue(measures.metrics.fork_length, "linear", unit)
       + " = " + displayValue(total, "linear", unit)
       + " (whole trace " + displayValue(measures.metrics.total_length, "linear", unit) + ")"},
  );
  const segmentTable = table(["Segment", "Trace", "From", "To", "Length", "Share", "Width", "Stops at"], segmentRows);
  if (onSegment) {
    segmentTable.querySelectorAll("tbody tr").forEach((row, index) => {
      row.classList.add("pickable");
      row.tabIndex = 0;
      const choose = () => onSegment(segments[index]);
      row.addEventListener("click", choose);
      row.addEventListener("keydown", event => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          choose();
        }
      });
    });
  }
  return [
    reconciliation,
    el("details", {className: "manual", open: true},
      el("summary", {text: "Segment review · " + segmentRows.length}),
      segmentTable,
      el("p", {className: "subtle", text: "Select a row or click a segment on the Segments view to highlight it on the image."}),
    ),
    ...(forkRows.length ? [el("details", {className: "manual"},
      el("summary", {text: "Fork clusters · " + forkRows.length}),
      table(["Fork", "Trace", "Held length", "Adjoining segments", "Main line", "Side branches"], forkRows),
      el("p", {className: "subtle", text: "Length held at a fork has no unambiguous branch, so it is counted once at the fork and never added into a segment. Main line shows the two adjoining segments that continue most nearly straight through the fork; other adjoining segments are side branches with their deviation from the main line."}),
    )] : []),
  ];
}

export function table(headers, rows) {
  const heading = el("thead", {},
    el("tr", {}, ...headers.map(value => el("th", {text: value, scope: "col"}))),
  );
  const body = el("tbody", {}, ...rows.map(row => el("tr", {}, ...row.map(value => el("td", {text: value ?? "—"})))));
  return el("div", {className: "table-wrap"}, el("table", {}, heading, body));
}
