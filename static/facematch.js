import {button, el, link, notices, shortTime} from "./presentation.js";

const pick = selector => document.querySelector(selector);

const VERDICT_TEXT = {
  same_face: "Likely the same end face",
  different_face: "Likely different faces",
  inconclusive: "Evidence insufficient",
};

const VERDICT_CLASS = {
  same_face: "verdict-same",
  different_face: "verdict-different",
  inconclusive: "verdict-inconclusive",
};

const DISPOSITION_TEXT = {
  undecided: "Not decided",
  confirmed: "Confirmed as same face",
  rejected: "Rejected",
};

let generation = 0;

function metricRow(label, value) {
  return el("div", {className: "fm-metric"}, el("span", {text: label}), el("strong", {text: value}));
}

function verdictBadge(verdict) {
  return el("span", {className: "fm-verdict " + VERDICT_CLASS[verdict], text: VERDICT_TEXT[verdict]});
}

function candidateCheckboxes(app, candidates) {
  const state = app.state;
  const box = pick("#facematch-candidates");
  if (!candidates.length) {
    box.replaceChildren(el("p", {
      className: "result-empty",
      text: state.current.lot
        ? "No other saved records share this lot. Assign the same lot ID to the records you want to compare."
        : "This re-photo has no lot ID. Set a lot ID on the candidate records first.",
    }));
    return;
  }
  box.replaceChildren(...candidates.map((record, index) => el("label", {className: "checkbox facematch-option"},
    el("input", {type: "checkbox", name: "candidate", value: record.id, checked: index === 0}),
    el("span", {},
      el("strong", {text: record.title}),
      el("small", {text: "Lot " + record.lot + " · updated " + shortTime(record.updated_at)}),
    ),
  )));
}

function resultCard(app, item) {
  const disposition = el("span", {
    className: "fm-disposition",
    text: DISPOSITION_TEXT[item.disposition] || item.disposition,
  });

  async function decide(value) {
    try {
      const updated = await app.api(
        "/api/face-matches/" + item.id + "/decision",
        app.json("POST", {disposition: value, note: ""}),
      );
      app.notify(value === "confirmed" ? "Match confirmed — both source records are kept separately." : "Candidate rejected.");
      await enterFaceMatch(app, true);
      return updated;
    } catch (error) {
      app.notify(error.message, true);
    }
  }

  const confirmButton = button("Confirm same face", () => decide("confirmed"), "button strong");
  const rejectButton = button("Reject candidate", () => decide("rejected"), "button outline");
  if (item.disposition === "confirmed") confirmButton.disabled = true;
  if (item.disposition === "rejected") rejectButton.disabled = true;

  const transform = item.transform;
  const evidence = item.evidence;
  const card = el("article", {className: "fm-card " + VERDICT_CLASS[item.verdict]},
    el("header", {},
      el("div", {}, verdictBadge(item.verdict), disposition),
      el("strong", {className: "fm-candidate-title", text: "Candidate: " + (item.candidate_title || "record")}),
    ),
    el("div", {className: "fm-metrics"},
      metricRow("Texture correlation r", Number(transform.ncc).toFixed(2)),
      metricRow("Overlap", Math.round(transform.overlap_fraction * 100) + "%"),
      metricRow("Rotation", Math.round(transform.angle_deg) + "°"),
      metricRow("Scale", Number(transform.scale).toFixed(2)),
      metricRow("Matching pores", String(evidence.pore_matches.length)),
      metricRow("Matching texture tiles", String(evidence.texture_tiles.length)),
    ),
    el("img", {className: "fm-sheet", src: item.links.sheet,
      alt: "Side-by-side evidence showing matching pores and texture for " + item.candidate.title}),
    notices(item.reasons),
    el("div", {className: "panel-actions fm-decide"}, confirmButton, rejectButton),
  );
  return card;
}

function historyCard(app, item) {
  const isQuery = item.query_inspection_id === app.state.current.id;
  const otherTitle = isQuery ? item.candidate_title : item.query_title;
  return el("div", {className: "fm-history-row"},
    el("div", {},
      el("strong", {text: VERDICT_TEXT[item.verdict]}),
      el("small", {
        text: "vs " + (otherTitle || "another record") + " · " + DISPOSITION_TEXT[item.disposition]
          + " · " + shortTime(item.updated_at),
      }),
    ),
    link("Evidence sheet ↓", item.links.sheet, "button outline"),
  );
}

export async function enterFaceMatch(app, keepHistory = false) {
  const epoch = ++generation;
  const state = app.state;
  pick("#facematch-subtitle").textContent =
    state.current.title + " · lot " + (state.current.lot || "—") + " · find which old record shows this same end face";
  try {
    const data = await app.api("/api/inspections/" + state.current.id + "/face-match-candidates");
    if (epoch !== generation) return;
    candidateCheckboxes(app, data.items);
    pick("#facematch-form button").disabled = !data.items.length;
    const historyData = await app.api("/api/inspections/" + state.current.id + "/face-matches");
    if (epoch !== generation) return;
    const history = pick("#facematch-history");
    history.replaceChildren(...(historyData.items.length
      ? historyData.items.map(item => historyCard(app, item))
      : [el("p", {className: "notes-empty", text: "No face checks recorded yet."})]));
    if (!keepHistory) pick("#facematch-results").replaceChildren();
  } catch (error) {
    app.notify(error.message, true);
  }
}

export function wireFaceMatch(app) {
  const {api, bindForm, json, notify} = app;
  bindForm("#facematch-form", async form => {
    const selected = [...form.querySelectorAll('input[name="candidate"]:checked')].map(input => input.value);
    if (!selected.length) throw new Error("Select at least one same-lot record to screen.");
    const results = pick("#facematch-results");
    results.replaceChildren(el("p", {className: "notes-empty", text: "Screening " + selected.length + " record(s)… this compares rotation, crop and brightness changes and can take a little while."}));
    const data = await api("/api/face-matches/screen", json("POST", {
      query_inspection_id: app.state.current.id,
      candidate_inspection_ids: selected,
    }));
    results.replaceChildren(...data.items.map(item => resultCard(app, item)));
    notify("Screening complete. Review the evidence before confirming.");
    await enterFaceMatch(app, true);
  });
}
