import {button, el, link} from "./presentation.js";

const pick = selector => document.querySelector(selector);
const INK = {planned: "#7ebe89", shot: "#68b7ca", void: "#d87661"};

function faceDraft(state) {
  const draft = state.sampling ?? (state.sampling = {});
  draft.faces ??= {A: null, B: null};
  draft.minimums ??= {A: 3, B: 3};
  draft.plan ??= null;
  return draft;
}

function metres(value, unit) {
  return (Math.round(value * 10) / 10).toFixed(1) + " " + unit;
}

function facePolygon(state, id) {
  const draft = faceDraft(state).faces[id];
  const saved = state.sampling?.plan?.plan.config.faces.find(face => face.id === id);
  if (draft && (!saved || JSON.stringify(draft) !== JSON.stringify(saved.polygon))) return draft;
  if (saved) return saved.polygon;
  return draft && draft.length >= 3 ? draft : null;
}

export function renderSamplingPanel(app) {
  const state = app.state;
  const draft = faceDraft(state);
  const plane = state.current?.plane;
  const chip = pick("#sampling-chip");
  if (draft.plan) {
    const counts = draft.plan.summary.counts;
    chip.textContent = counts.planned + " planned · " + counts.shot + " captured"
      + (draft.plan.summary.shortfall_total ? " · short" : "");
  } else {
    chip.textContent = plane?.unit === "mm" ? "" : "Needs scale";
  }
  const faces = pick("#sampling-faces");
  const savedConfig = draft.plan?.plan.config;
  const polygonStates = ["A", "B"].map(id => {
    const polygon = facePolygon(state, id);
    if (!polygon) return el("p", {}, el("strong", {text: "Face " + id + ": "}), document.createTextNode("not drawn yet."));
    const saved = savedConfig?.faces.find(face => face.id === id);
    const stale = saved && JSON.stringify(saved.polygon) !== JSON.stringify(polygon);
    const xs = polygon.map(point => point[0]);
    const ys = polygon.map(point => point[1]);
    const width = (Math.max(...xs) - Math.min(...xs)) * (plane.mm_per_x ?? 1);
    const height = (Math.max(...ys) - Math.min(...ys)) * (plane.mm_per_y ?? 1);
    return el("p", {}, el("strong", {text: "Face " + id + ": "}), document.createTextNode(
      polygon.length + " points · about " + metres(width, plane.unit) + " × " + metres(height, plane.unit)
      + (stale ? " · redrawn, place stations again" : ""),
    ));
  });
  faces.replaceChildren(...polygonStates);
  if (draft.plan && !draft.synced) {
    for (const face of draft.plan.plan.config.faces) draft.minimums[face.id] = face.minimum;
    const form = pick("#sampling-form");
    form.elements.frame_width_mm.value = draft.plan.plan.config.frame_width_mm;
    form.elements.frame_height_mm.value = draft.plan.plan.config.frame_height_mm;
    form.elements.budget.value = draft.plan.summary.budget;
    form.elements.spacing_mm.value = draft.plan.plan.config.spacing_mm;
    draft.synced = true;
  }
  const minimums = pick("#sampling-minimums");
  minimums.replaceChildren(...["A", "B"].map(id => {
    const input = el("input", {
      type: "number", min: "0", max: "96", step: "1", value: draft.minimums[id],
      oninput: event => { draft.minimums[id] = Number(event.target.value); },
    });
    input.name = "minimum_" + id;
    return el("label", {text: "Face " + id + " minimum"}, input);
  }));
  pick("#sampling-plan-button").disabled = plane?.unit !== "mm" || !facePolygon(state, "A") || !facePolygon(state, "B");
  renderSamplingState(app);
}

function renderSamplingState(app) {
  const state = app.state;
  const draft = faceDraft(state);
  const output = pick("#sampling-state");
  const links = pick("#sampling-links");
  pick("#sampling-plan-open").hidden = !draft.plan;
  pick("#sampling-reset").hidden = !draft.plan;
  if (!draft.plan) {
    output.replaceChildren(el("p", {className: "meta", text:
      "Draw both representative faces, set the field size in mm and the maximum number of frames. "
      + "Stations land only on photographed, clear pixels and never overlap.",
    }));
    links.replaceChildren();
    return;
  }
  const summary = draft.plan.summary;
  const rows = summary.faces.map(face => {
    const stateText = face.shortfall
      ? "short " + face.shortfall + (face.limit === "fit" ? " · no room" : " · budget")
      : "met";
    return el("p", {className: face.shortfall ? "shortfall" : ""},
      el("strong", {text: face.label + ": "}),
      document.createTextNode(face.covered + " / " + face.minimum + " · " + stateText),
    );
  });
  const parts = [
    el("p", {className: "meta", text:
      summary.active + " of " + summary.budget + " frames placed · "
      + summary.counts.shot + " captured · " + summary.counts.void + " void · "
      + summary.counts.planned + " still to shoot",
    }),
    ...rows,
  ];
  if (draft.plan.notices.length) {
    parts.push(el("ul", {}, ...draft.plan.notices.map(text => el("li", {text}))));
  }
  output.replaceChildren(...parts);
  links.replaceChildren(
    link("Marked PNG ↓", draft.plan.links.image + "&download=1"),
    link("Walk list CSV ↓", draft.plan.links.csv),
  );
}

export function startFace(app, id) {
  app.startDrawing(
    "sample-face",
    "Trace the whole representative face for " + id + ", then choose Use points.",
    3, 64, "working", {face: id},
  );
}

export function acceptFace(app, points, id) {
  const draft = faceDraft(app.state);
  draft.faces[id] = points;
  draft.synced = true;
  renderSamplingPanel(app);
  app.notify("Face " + id + " outlined; place stations again to re-plan");
}

export function clearFaces(app) {
  const draft = faceDraft(app.state);
  draft.faces = {A: null, B: null};
  renderSamplingPanel(app);
}

function samplingBody(app) {
  const form = pick("#sampling-form");
  const fields = form.elements;
  const draft = faceDraft(app.state);
  return {
    frame_width_mm: fields.frame_width_mm.valueAsNumber,
    frame_height_mm: fields.frame_height_mm.valueAsNumber,
    budget: Number(fields.budget.value),
    spacing_mm: fields.spacing_mm.valueAsNumber,
    // Keep captured and void marks. The server refuses a redraw that would drop
    // captured frames and blocks any plan whose measuring revision has moved.
    reset: false,
    faces: ["A", "B"].map(id => ({
      id, label: "Face " + id, minimum: draft.minimums[id] || 0, polygon: facePolygon(app.state, id),
    })),
  };
}

export async function planStations(app) {
  const state = app.state;
  const draft = faceDraft(state);
  const result = await app.api("/api/inspections/" + state.current.id + "/sampling",
    {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(samplingBody(app))});
  draft.plan = result;
  renderSamplingPanel(app);
  app.renderImage();
  return result;
}

async function markStation(app, station, status, reason = "") {
  const revision = app.state.current.revision;
  const result = await app.api(
    "/api/inspections/" + app.state.current.id + "/sampling/station",
    {method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({revision, station, status, reason})},
  );
  faceDraft(app.state).plan = result;
  renderSamplingPanel(app);
  app.renderImage();
  if (pick("#sampling-dialog").open) renderSamplingPlan(app);
  return result;
}

function act(app, station, status, message) {
  return () => markStation(app, station.id, status)
    .then(() => app.notify(message))
    .catch(error => app.notify(error.message, true));
}

function stationActions(app, station) {
  if (station.status === "planned") {
    return el("div", {className: "station-actions"},
      button("Captured", act(app, station, "shot", station.id + " captured"), "button outline"),
      button("Void", () => openVoid(app, station), "button ghost"),
    );
  }
  if (station.status === "shot") {
    return el("div", {className: "station-actions"},
      button("Undo", act(app, station, "planned", station.id + " reopened"), "button ghost"),
      button("Void", () => openVoid(app, station), "button ghost"),
    );
  }
  return el("div", {className: "station-actions"},
    button("Restore", act(app, station, "planned", station.id + " restored"), "button ghost"),
  );
}

async function resetPlan(app) {
  try {
    const result = await app.api(
      "/api/inspections/" + app.state.current.id + "/sampling",
      {method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({...samplingBody(app), reset: true})},
    );
    faceDraft(app.state).plan = result;
    renderSamplingPanel(app);
    app.renderImage();
    app.notify("Plan reset; captured and void marks cleared");
  } catch (error) {
    app.notify(error.message, true);
  }
}

function openVoid(app, station) {
  app.state.sampling.voidTarget = station.id;
  const form = pick("#void-form");
  form.reset();
  pick("#void-station").textContent = station.id + " on Face " + station.face + " · centre "
    + metres(station.center_mm[0], "mm") + " / " + metres(station.center_mm[1], "mm");
  app.showDialog("#void-dialog");
}

export function renderSamplingPlan(app) {
  const state = app.state;
  const draft = faceDraft(state);
  const body = pick("#sampling-plan-body");
  if (!draft.plan) {
    body.replaceChildren(el("p", {className: "result-empty", text: "Place stations first."}));
    return;
  }
  const plan = draft.plan;
  const summary = plan.summary;
  const frame = plan.plan.frame_px;
  const image = el("img", {
    className: "station-map", src: plan.links.image + "&t=" + Date.now(),
    alt: "Numbered station frames over the whole face photograph",
  });
  const legend = el("div", {className: "station-legend"},
    el("span", {}, el("i", {style: "border-color:" + INK.planned + ";background:transparent"}), document.createTextNode("planned")),
    el("span", {}, el("i", {style: "border-color:" + INK.shot + ";background:" + INK.shot + "33"}), document.createTextNode("captured")),
    el("span", {}, el("i", {style: "border-color:" + INK.void + ";background:transparent"}), document.createTextNode("void")),
  );
  const fields = state.current.plane;
  const heading = el("p", {className: "subtle", text:
    "Each field " + metres(fields.mm_per_x * frame.width, "mm") + " × "
    + metres(fields.mm_per_y * frame.height, "mm") + " · gap "
    + metres(plan.plan.config.spacing_mm, "mm") + " · revision " + plan.revision,
  });
  const faces = el("div", {className: "sampling-state"}, ...summary.faces.map(face => el("p", {
    className: face.shortfall ? "shortfall" : "",
  }, el("strong", {text: face.label + ": "}),
  document.createTextNode(
    face.covered + " / " + face.minimum + " represented"
    + (face.shortfall ? " · short " + face.shortfall : "")
    + (face.extra ? " · " + face.extra + " spare frame(s)" : ""),
  ))));
  const live = plan.plan.stations.slice().sort((a, b) => {
    const order = value => value.walk_order ?? 9999;
    return order(a) - order(b) || a.id.localeCompare(b.id);
  });
  const rows = live.filter(station => station.status !== "void").map(station => el("div", {className: "station-row"},
    el("span", {className: "station-id", text: (station.walk_order ? station.walk_order + ". " : "") + station.id}),
    el("span", {className: "station-state " + station.status, text: station.status}),
    el("span", {className: "station-where", text:
      "c " + metres(station.center_mm[0], "mm") + " / " + metres(station.center_mm[1], "mm")
      + (station.replaces ? " · replaces " + station.replaces : "")
      + (station.substitutes_shortfall.length ? " · not " + station.substitutes_shortfall.join("/") : ""),
    }),
    stationActions(app, station),
  ));
  const voids = plan.plan.stations.filter(station => station.status === "void");
  const replacements = plan.plan.replacement_of ?? {};
  const voidRows = voids.length ? [el("p", {className: "small-label", text: "VOIDED POSITIONS"}),
    ...voids.map(item => el("div", {className: "station-row"},
      el("span", {className: "station-id", text: item.id}),
      el("span", {className: "station-state void", text: "void"}),
      el("span", {className: "station-where", text:
        (item.void_reason || "—") + (replacements[item.id] ? " · replaced by " + replacements[item.id] : " · no replacement fits"),
      }),
      stationActions(app, item),
    )),
  ] : [];
  body.replaceChildren(
    image, legend, heading, faces,
    ...(plan.notices.length ? [el("ul", {}, ...plan.notices.map(text => el("li", {text})))] : []),
    el("div", {className: "station-list"}, ...rows, ...voidRows),
    el("div", {className: "panel-actions"}, link("Marked PNG ↓", plan.links.image + "&download=1"),
      link("Walk list CSV ↓", plan.links.csv),
      el("span", {className: "subtle", text: "Marks save immediately; captured frames never move."}),
    ),
  );
}

export function wireSampling(app) {
  const {state} = app;
  pick("#sampling-reset").addEventListener("click", () => resetPlan(app));
  pick("#sampling-plan-open").addEventListener("click", () => app.showSamplingPlan());
  pick("#sample-face-A").addEventListener("click", () => startFace(app, "A"));
  pick("#sample-face-B").addEventListener("click", () => startFace(app, "B"));
  pick("#sample-clear").addEventListener("click", () => clearFaces(app));
  app.bindForm("#sampling-form", async () => {
    const result = await planStations(app);
    if (result) app.notify("Stations placed");
  });
  app.bindForm("#void-form", async form => {
    const station = state.sampling?.voidTarget;
    if (!station) throw new Error("Choose a station to void.");
    const result = await markStation(app, station, "void", form.elements.reason.value);
    form.closest("dialog").close();
    app.notify(result.notices[0] || station + " voided");
  });
}
