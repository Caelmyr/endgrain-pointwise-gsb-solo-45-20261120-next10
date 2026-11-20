import {button, displayValue, el, notices, shortTime} from "./presentation.js";

const pick = selector => document.querySelector(selector);
const GROUP_ORDER = ["single", "multiple", "none"];
const COLORS = {
  exact: "#70ba79",
  segments: "#e2bd67",
  fragment: "#de9356",
  ambiguous: "#87a0cb",
  truncated: "#cea66a",
  unmatched: "#c76f94",
  paired: "#96a89b",
};
const GROUP_TITLES = {
  single: "Single pair to re-check",
  multiple: "More than one defensible relation",
  none: "No evidence at this position",
};
const TYPE_TITLES = {
  exact: "Compatible single pair",
  segments: "Several short traces could combine",
  fragment: "One trace is much shorter",
  truncated: "One trace may be cut by the photograph",
  ambiguous: "Several candidates are compatible",
  unmatched: "No compatible trace recorded",
  paired: "Compatible with the first face",
};
const ORIGIN_NAME = {left: "A", right: "B"};
const COVERAGE_LABELS = {
  observed: "read here",
  no_trace: "no trace there",
  outside_region: "not read here",
  unsupported: "not photographed",
  partial: "partly unseen",
};

const local = {
  faces: [],
  a: {inspectionId: null, runId: null, anchors: null, records: [], image: null, plane: null, region: null, selected: null},
  b: {inspectionId: null, runId: null, anchors: null, records: [], image: null, plane: null, region: null, selected: null},
  runs: {a: [], b: []},
  marking: null,
  saved: [],
  current: null,
  generation: 0,
};

function tone(kind) {
  return COLORS[kind] || COLORS.unmatched;
}

function groupTone(key) {
  return key === "none" ? COLORS.unmatched : key === "single" ? COLORS.exact : COLORS.segments;
}

function secondFace(app) {
  const first = local.a.inspectionId || app.state.current?.id;
  return local.faces.find(record => record.id !== first)?.id || first;
}

function describeAnchor(anchors) {
  if (!anchors || !anchors.corner) return "Not marked";
  const [cx, cy] = anchors.corner;
  if (!anchors.edge) return `Corner ${cx.toFixed(0)}, ${cy.toFixed(0)} · now click a point along the same edge`;
  const [ex, ey] = anchors.edge;
  const span = Math.hypot(ex - cx, ey - cy);
  return `Corner ${cx.toFixed(0)}, ${cy.toFixed(0)} · edge ${ex.toFixed(0)}, ${ey.toFixed(0)} · ${span.toFixed(0)} px`;
}

function faceLink(face) {
  return "/api/inspections/" + face.inspectionId + "/working.png?view=clean";
}

function readingOptions(runs, selected) {
  return runs.map(run => el("option", {
    value: run.id,
    text: "Rev " + run.revision + " · " + shortTime(run.created_at) + " · " + (run.result?.metrics?.traces ?? 0) + " traces",
    selected: run.id === selected,
  }));
}

async function loadFaces(app) {
  const data = await app.api("/api/inspections?limit=100");
  local.faces = data.items;
  for (const key of ["a", "b"]) {
    const select = pick("#twoends-" + key + "-face");
    select.replaceChildren(...local.faces.map(record => el("option", {value: record.id, text: record.title})));
    const preferred = local[key].inspectionId || (key === "b" ? secondFace(app) : app.state.current?.id);
    const chosen = local.faces.some(record => record.id === preferred) ? preferred : local.faces[0]?.id;
    select.value = chosen;
    await selectFace(app, key, chosen);
  }
}

async function selectFace(app, key, inspectionId) {
  const side = local[key];
  if (!inspectionId) return;
  const changed = side.inspectionId !== inspectionId;
  if (changed) side.anchors = null;
  side.inspectionId = inspectionId;
  const [runs, record] = await Promise.all([
    app.api("/api/inspections/" + inspectionId + "/runs"),
    app.api("/api/inspections/" + inspectionId).catch(() => null),
  ]);
  if (record) {
    const index = local.faces.findIndex(item => item.id === inspectionId);
    if (index >= 0) local.faces[index] = record;
    if (key === "a") local.aRevision = record.revision;
  }
  const cracks = runs.items.filter(run => run.kind === "cracks");
  local.runs[key] = cracks;
  const select = pick("#twoends-" + key + "-run");
  const keep = cracks.some(run => run.id === side.runId) ? side.runId : cracks[0]?.id;
  select.replaceChildren(...readingOptions(cracks, keep));
  if (!keep) select.replaceChildren(el("option", {value: "", text: "No saved Crack traces reading"}));
  side.runId = keep || null;
  await selectReading(app, key, side.runId);
}

async function selectReading(app, key, runId) {
  const side = local[key];
  side.runId = runId || null;
  side.image = null;
  side.plane = null;
  side.region = null;
  side.records = [];
  if (runId) {
    const inspection = local.faces.find(record => record.id === side.inspectionId);
    const run = local.runs[key].find(entry => entry.id === runId);
    side.plane = run?.snapshot?.plane || inspection?.plane || null;
    side.region = run?.snapshot?.state?.region || null;
    side.records = (run?.result?.objects || []).map(entry => ({id: entry.id, bbox: entry.bbox}));
    side.image = run?.links?.preview || faceLink(side);
  }
  renderSide(key);
}

function fitCanvas(image, canvas) {
  const width = image.naturalWidth;
  const height = image.naturalHeight;
  if (!width || !height) return;
  canvas.width = width;
  canvas.height = height;
  canvas.style.aspectRatio = width + " / " + height;
}

function drawSide(key) {
  const side = local[key];
  const image = pick("#twoends-" + key + "-image");
  const canvas = pick("#twoends-" + key + "-canvas");
  if (!image.naturalWidth || !side.plane) return;
  fitCanvas(image, canvas);
  const context = canvas.getContext("2d");
  context.clearRect(0, 0, canvas.width, canvas.height);
  const scale = canvas.width / image.naturalWidth;
  const factor = Math.max(1, scale);
  if (side.anchors?.corner) {
    const {corner, edge} = side.anchors;
    context.strokeStyle = "#f7d889";
    context.lineWidth = 2.4 * factor;
    context.setLineDash([]);
    context.beginPath();
    context.moveTo(corner[0] + 0.5, corner[1] + 0.5);
    if (edge) context.lineTo(edge[0] + 0.5, edge[1] + 0.5);
    context.stroke();
    context.font = "600 " + 16 * factor + "px system-ui";
    for (const [point, label] of edge ? [[corner, "C"], [edge, "E"]] : [[corner, "C"]]) {
      context.beginPath();
      context.arc(point[0] + 0.5, point[1] + 0.5, 7 * factor, 0, Math.PI * 2);
      context.stroke();
      context.fillStyle = "#f7d889";
      context.fillText(label, point[0] + 11 * factor, point[1] + 5 * factor);
    }
  }
  const rows = local.current?.analysis?.rows || [];
  for (const row of rows) {
    if (row.origin !== (key === "a" ? "left" : "right")) continue;
    const selected = side.selected === row.trace_id;
    context.strokeStyle = tone(row.type);
    context.lineWidth = (selected ? 4.5 : 2.2) * factor;
    if (row.origin === "left") {
      const record = side.records.find(entry => entry.id === row.trace_id);
      if (!record) continue;
      const [x0, y0, x1, y1] = record.bbox;
      context.strokeRect(x0, y0, Math.max(1, x1 - x0), Math.max(1, y1 - y0));
    } else {
      const point = row.position;
      context.beginPath();
      context.arc(point[0] + 0.5, point[1] + 0.5, (selected ? 13 : 8) * factor, 0, Math.PI * 2);
      context.stroke();
    }
    context.font = "700 " + 13 * factor + "px system-ui";
    context.fillStyle = tone(row.type);
    const label = (key === "a" ? "A" : "B") + row.trace_id;
    const [lx, ly] = row.origin === "left" ? [row.position[0] + 6, row.position[1] - 8] : [row.position[0] + 10, row.position[1] - 9];
    context.fillText(label, lx, ly);
  }
}

function renderSide(key) {
  const side = local[key];
  const image = pick("#twoends-" + key + "-image");
  const wrapper = image.closest(".twoends-image");
  if (side.image) {
    if (image.getAttribute("src") !== side.image) image.setAttribute("src", side.image);
  } else {
    image.removeAttribute("src");
  }
  pick("#twoends-" + key + "-caption").textContent =
    (key === "a" ? "Face A" : "Face B") + " · " + (local.faces.find(record => record.id === side.inspectionId)?.title || "not chosen");
  const run = local.runs[key].find(entry => entry.id === side.runId);
  pick("#twoends-" + key + "-note").textContent = run
    ? "reading " + run.id.slice(0, 8) + " · " + (run.result?.metrics?.traces ?? 0) + " traces · " + (side.plane?.unit || "px")
    : "no Crack traces reading chosen";
  wrapper.classList.toggle("marking", local.marking === key);
  pick("#twoends-" + key + "-anchors").textContent = describeAnchor(side.anchors);
  pick("#twoends-" + key + "-mark").textContent = local.marking === key
    ? "Cancel marking"
    : "Mark corner and edge on " + key.toUpperCase();
  drawSide(key);
}

function clearSelection() {
  local.a.selected = null;
  local.b.selected = null;
}

function selectRow(row) {
  if (!row) {
    clearSelection();
  } else if (row.origin === "left") {
    local.a.selected = row.trace_id;
    local.b.selected = row.candidates.length ? row.candidates[0].trace_id : null;
  } else {
    local.b.selected = row.trace_id;
    local.a.selected = row.candidates.length ? row.candidates[0].trace_id : null;
  }
  renderOutput();
}

function renderOutput() {
  const output = pick("#twoends-output");
  const current = local.current;
  if (!current) {
    output.replaceChildren(el("p", {className: "result-empty", text: "Mark the shared corner and edge on both faces, choose the two readings and save the review to see the grouping."}));
    return;
  }
  const analysis = current.analysis;
  const groupRows = key => analysis.rows.filter(row => row.group === key);
  const summary = el("div", {className: "metrics"},
    ...GROUP_ORDER.map(key => el("div", {className: "metric"},
      el("span", {text: GROUP_TITLES[key]}),
      el("strong", {text: String(groupRows(key).length)}),
    )),
    el("div", {className: "metric"},
      el("span", {text: "First-face relations"}),
      el("strong", {text: String(analysis.relations)}),
    ),
  );
  const meta = el("p", {className: "subtle twoends-hint", text:
    "A " + current.left.inspection_title + " · reading " + current.left.run_id.slice(0, 8) + " · revision " + current.left.run_revision
    + "  /  B " + current.right.inspection_title + " · reading " + current.right.run_id.slice(0, 8) + " · revision " + current.right.run_revision
    + ". " + (current.relationship === "mirror" ? "Second face treated as mirrored." : "Second face treated as the same viewpoint."),
  });
  const parts = [meta, summary, notices(analysis.notices)];
  for (const key of GROUP_ORDER) {
    const rows = groupRows(key);
    const heading = el("h4", {},
      el("span", {className: "swatch", style: `background:${groupTone(key)}`}),
      GROUP_TITLES[key],
      el("span", {className: "subtle", text: rows.length + (rows.length === 1 ? " row" : " rows")}),
    );
    parts.push(el("div", {className: "twoends-group"}, heading,
      rows.length ? groupTable(rows) : el("p", {text: "None."})));
  }
  parts.push(el("p", {className: "subtle twoends-hint", text:
    "Selecting a row marks that trace on both photographs. Compatibility describes the two photographed faces only; it does not show that a check runs through the stock."
  }));
  output.replaceChildren(...parts);
  renderSide("a");
  renderSide("b");
}

function groupTable(rows) {
  const header = el("thead", {}, el("tr", {},
    ...["Trace", "Relation", "Other-face evidence", "Length", "Reading"].map(text => el("th", {text, scope: "col"})),
  ));
  const unit = local.a.plane?.unit || "px";
  const body = el("tbody", {}, ...rows.map(row => {
    const selected = (row.origin === "left" && local.a.selected === row.trace_id)
      || (row.origin === "right" && local.b.selected === row.trace_id);
    const candidate = row.candidates.length
      ? row.candidates.map(match => ORIGIN_NAME[row.origin === "left" ? "right" : "left"] + match.trace_id).join(", ")
      : (row.coverage.status === "unsupported" || row.coverage.status === "outside_region"
        ? "Not visible"
        : "None recorded");
    const coverage = el("span", {
      className: "coverage " + row.coverage.status,
      text: COVERAGE_LABELS[row.coverage.status] || row.coverage.status,
    });
    return el("tr", {
      tabIndex: 0,
      className: selected ? "selected" : "",
      onclick: () => { selectRow(row); },
      onkeydown: event => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          selectRow(row);
        }
      },
    },
      el("td", {text: ORIGIN_NAME[row.origin] + row.trace_id}),
      el("td", {text: TYPE_TITLES[row.type]}),
      el("td", {}, candidate, " ", coverage),
      el("td", {text: displayValue(row.length, "linear", unit)}),
      el("td", {text: row.explanation}),
    );
  }));
  return el("div", {className: "table-wrap"}, el("table", {className: "twoends-table"}, header, body));
}

async function loadSaved(app) {
  const id = local.a.inspectionId || app.state.current?.id;
  if (!id) return;
  const data = await app.api("/api/inspections/" + id + "/pairings");
  local.saved = data.items;
  const target = pick("#twoends-saved");
  if (!data.items.length) {
    target.replaceChildren(el("p", {className: "result-empty", text: "No two-end reviews saved on this face record yet."}));
    return;
  }
  target.replaceChildren(...data.items.map(item => el("div", {className: "issued-row"},
    el("div", {}, el("strong", {text: item.title}), el("small", {text: "Revision " + item.revision + " · " + (item.reviewer || "unassigned") + " · " + shortTime(item.created_at)})),
    button("Open", () => openSaved(app, item.id), "text-link"),
  )));
}

async function openSaved(app, reviewId) {
  const id = local.a.inspectionId || app.state.current?.id;
  const record = await app.api("/api/inspections/" + id + "/pairings/" + reviewId);
  local.a.inspectionId = record.left.inspection_id;
  local.b.inspectionId = record.right.inspection_id;
  local.a.runId = record.left.run_id;
  local.b.runId = record.right.run_id;
  local.a.anchors = record.anchors.left;
  local.b.anchors = record.anchors.right;
  await loadFaces(app);
  local.current = record;
  clearSelection();
  const form = pick("#twoends-form");
  form.elements.title.value = record.title;
  form.elements.reviewer.value = record.reviewer || "";
  form.elements.note.value = record.note || "";
  form.elements.relationship.value = record.relationship;
  const sheet = pick("#twoends-sheet-link");
  sheet.hidden = false;
  sheet.href = record.links.sheet;
  renderOutput();
}

function handleMarkClick(app, key) {
  const canvas = pick("#twoends-" + key + "-canvas");
  const side = local[key];
  const mark = event => {
    if (local.marking !== key || !side.plane) return;
    const image = pick("#twoends-" + key + "-image");
    const box = canvas.getBoundingClientRect();
    if (!box.width || !image.naturalWidth) return;
    const x = Math.min(image.naturalWidth - 1, Math.max(0, (event.clientX - box.left) * image.naturalWidth / box.width - 0.5));
    const y = Math.min(image.naturalHeight - 1, Math.max(0, (event.clientY - box.top) * image.naturalHeight / box.height - 0.5));
    const value = [Number(x.toFixed(2)), Number(y.toFixed(2))];
    const prior = side.anchors || {corner: null, edge: null};
    side.anchors = prior.corner && !prior.edge
      ? {corner: prior.corner, edge: value}
      : {corner: value, edge: null};
    if (side.anchors.edge) {
      local.marking = null;
      notifyMark("Marks saved on face " + key.toUpperCase());
    }
    renderSide(key);
  };
  canvas.addEventListener("click", mark);
}

function notifyMark(text) {
  const toast = pick("#toast");
  toast.textContent = text;
  toast.hidden = false;
  toast.classList.remove("error");
  clearTimeout(notifyMark.timer);
  notifyMark.timer = setTimeout(() => { toast.hidden = true; }, 2400);
}

function reviewPayload() {
  const form = pick("#twoends-form");
  if (!local.a.anchors || !local.b.anchors || !local.a.anchors.corner || !local.a.anchors.edge || !local.b.anchors.corner || !local.b.anchors.edge) {
    throw new Error("Mark the shared corner and edge on both faces.");
  }
  if (!local.a.runId || !local.b.runId) throw new Error("Choose a saved Crack traces reading on each face.");
  return {
    revision: local.aRevision || 1,
    title: form.elements.title.value,
    reviewer: form.elements.reviewer.value,
    note: form.elements.note.value,
    relationship: form.elements.relationship.value,
    left_run_id: local.a.runId,
    right_run_id: local.b.runId,
    right_inspection_id: local.b.inspectionId,
    anchors: {left: local.a.anchors, right: local.b.anchors},
  };
}


export async function enterTwoEnds(app) {
  const epoch = ++local.generation;
  pick("#twoends-subtitle").textContent = app.state.current
    ? "Compare saved Crack traces readings from the two ends of " + app.state.current.title + "."
    : "Compare two saved Crack traces readings.";
  await loadFaces(app);
  if (epoch !== local.generation) return;
  await loadSaved(app);
}

export function wireTwoEnds(app) {
  const {api, bindForm, notify} = app;
  pick("#twoends-a-face").addEventListener("change", event => {
    local.current = null;
    selectFace(app, "a", event.target.value).catch(error => notify(error.message, true));
  });
  pick("#twoends-b-face").addEventListener("change", event => {
    local.current = null;
    selectFace(app, "b", event.target.value).catch(error => notify(error.message, true));
  });
  for (const key of ["a", "b"]) {
    pick("#twoends-" + key + "-run").addEventListener("change", event => {
      local.current = null;
      selectReading(app, key, event.target.value).catch(error => notify(error.message, true));
      const sheet = pick("#twoends-sheet-link");
      sheet.hidden = true;
    });
    pick("#twoends-" + key + "-mark").addEventListener("click", () => {
      local.marking = local.marking === key ? null : key;
      if (local.marking) notifyMark("Click the shared corner, then a point along the same edge.");
      renderSide("a");
      renderSide("b");
    });
    handleMarkClick(app, key);
  }
  for (const key of ["a", "b"]) {
    pick("#twoends-" + key + "-image").addEventListener("load", () => renderSide(key));
  }
  bindForm("#twoends-form", async () => {
    const payload = reviewPayload();
    const created = await api("/api/inspections/" + local.a.inspectionId + "/pairings", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload),
    });
    local.current = created;
    const sheet = pick("#twoends-sheet-link");
    sheet.hidden = false;
    sheet.href = created.links.sheet;
    renderOutput();
    await loadSaved(app);
    notify("Two-end review saved");
  });
}
