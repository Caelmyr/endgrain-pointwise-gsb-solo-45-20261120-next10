"""Application service for end-face re-photo matching.

Only the screening record and the two source identifiers are stored here.
Photographs, inspections and their measurement readings stay independent;
confirming or rejecting a candidate never merges or rewrites either record.
"""

from typing import Any

from . import facematch
from .database import Database
from .errors import Missing, WorkbenchError
from .validation import choice, document, fields, list_of, string

MAX_CANDIDATE_INPUTS = facematch.MAX_CANDIDATES


def _inspection(db: Database, inspection_id: Any, label: str) -> dict:
    inspection_id = string(inspection_id, label, 40, required=True)
    try:
        return db.inspection(inspection_id)
    except Missing:
        raise WorkbenchError(f"Choose an existing inspection as the {label}.", label) from None


def candidate_options(db: Database, inspection_id: str) -> list[dict]:
    inspection = _inspection(db, inspection_id, "inspection")
    with db.connection() as conn:
        rows = conn.execute(
            """SELECT id, title, lot, material, surface, capture_id, updated_at
               FROM inspections WHERE lot = ? AND id <> ?
               ORDER BY updated_at DESC, id LIMIT ?""",
            (inspection["lot"], inspection_id, MAX_CANDIDATE_INPUTS),
        ).fetchall()
    return [dict(row) for row in rows]


def _store(db: Database, document: dict) -> dict:
    payload = {
        "query_inspection_id": document["query"]["inspection_id"],
        "candidate_inspection_id": document["candidate"]["inspection_id"],
        "query_capture_id": document["query"]["capture_id"],
        "candidate_capture_id": document["candidate"]["capture_id"],
        "verdict": document["verdict"],
        "evidence": document["evidence"],
        "result": {"transform": document["transform"], "reasons": document["reasons"]},
    }
    return db.upsert_face_match(payload)


def screen(db: Database, raw: Any) -> list[dict]:
    body = document(raw, "face match request")
    fields(body, {"query_inspection_id", "candidate_inspection_ids"}, "face match request")
    query = _inspection(db, body.get("query_inspection_id"), "query inspection")
    requested = list_of(
        body.get("candidate_inspection_ids"), "candidate inspections", 1, MAX_CANDIDATE_INPUTS
    )
    candidate_ids = []
    for value in requested:
        candidate_id = string(value, "candidate inspection", 40, required=True)
        if candidate_id == query["id"]:
            raise WorkbenchError("A face cannot be screened against itself.", "candidate_inspection_ids")
        if candidate_id not in candidate_ids:
            candidate_ids.append(candidate_id)
    candidates = []
    for candidate_id in candidate_ids:
        record = _inspection(db, candidate_id, "candidate inspection")
        if record["lot"] != query["lot"]:
            raise WorkbenchError(
                "Choose candidate photographs from the same lot as the re-photo.",
                "candidate_inspection_ids",
            )
        candidates.append(record)
    documents = facematch.rank_candidates(db, query, candidates)
    stored = [_store(db, item) for item in documents]
    titles = _titles_for(db, stored)
    return [describe(item, titles) for item in stored]


def describe(record: dict, titles: dict | None = None) -> dict:
    result = record["result"]
    titles = titles or {}
    return {
        "id": record["id"],
        "query_inspection_id": record["query_inspection_id"],
        "candidate_inspection_id": record["candidate_inspection_id"],
        "query_title": titles.get(record["query_inspection_id"], ""),
        "candidate_title": titles.get(record["candidate_inspection_id"], ""),
        "query_capture_id": record["query_capture_id"],
        "candidate_capture_id": record["candidate_capture_id"],
        "verdict": record["verdict"],
        "disposition": record["disposition"],
        "transform": result["transform"],
        "evidence": record["evidence"],
        "reasons": result["reasons"],
        "created_at": record["created_at"],
        "updated_at": record["updated_at"],
        "links": {
            "sheet": "/api/face-matches/" + record["id"] + "/sheet.png",
        },
    }


def _titles_for(db: Database, records: list[dict]) -> dict:
    ids = {record["query_inspection_id"] for record in records}
    ids.update(record["candidate_inspection_id"] for record in records)
    titles = {}
    with db.connection() as conn:
        for inspection_id in ids:
            row = conn.execute("SELECT title FROM inspections WHERE id = ?", (inspection_id,)).fetchone()
            if row is not None:
                titles[inspection_id] = row["title"]
    return titles


def history(db: Database, inspection_id: str) -> list[dict]:
    _inspection(db, inspection_id, "inspection")
    records = db.face_matches(inspection_id)
    titles = _titles_for(db, records)
    return [describe(item, titles) for item in records]


def decide(db: Database, match_id: str, raw: Any) -> dict:
    body = document(raw, "face match decision")
    fields(body, {"disposition", "note"}, "face match decision")
    disposition = choice(body.get("disposition"), "decision", facematch.DISPOSITIONS)
    note = string(body.get("note"), "decision note", 600, multiline=True)
    try:
        record = db.face_match(match_id)
    except Missing:
        raise WorkbenchError("Choose a recorded face match to decide on.", "match") from None
    updated = db.set_face_match_disposition(record["id"], disposition, note)
    return describe(updated, _titles_for(db, [updated]))


def sheet_document(db: Database, match_id: str) -> dict:
    try:
        record = db.face_match(match_id)
    except Missing:
        raise WorkbenchError("Choose a recorded face match to display.", "match") from None
    return {
        "query": {
            "inspection_id": record["query_inspection_id"],
            "capture_id": record["query_capture_id"],
        },
        "candidate": {
            "inspection_id": record["candidate_inspection_id"],
            "capture_id": record["candidate_capture_id"],
        },
        "transform": record["result"]["transform"],
        "evidence": record["evidence"],
        "verdict": record["verdict"],
        "reasons": record["result"]["reasons"],
    }
