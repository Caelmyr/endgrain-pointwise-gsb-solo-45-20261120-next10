"""End-to-end checks for the face-match HTTP service and persistence.

Run with: python3 -I tests/test_facematch_service.py

Builds two same-lot inspections (one a rotated/cropped/re-exposed copy of the
other) plus a different-lot inspection, then exercises the candidate list,
screening, decision, history and evidence-sheet routes through the Flask test
client against a real temporary SQLite database.
"""

import sys
import tempfile

sys.path.insert(0, "/opt/pointwise-env/site-packages")
sys.path.insert(0, "/workspace")

import numpy as np
from PIL import Image

from endgrain.captures import capture_payload
from endgrain.database import Database
from endgrain.facematch import VERDICTS
from endgrain.imaging import encode_png, read_image
from endgrain.web import create_app
import tests.test_facematch as synth


def make_inspection(db, rgb, support, *, title, lot, identifier):
    rgba = np.dstack([rgb, (support * 255).astype(np.uint8)])
    image = read_image(encode_png(rgba))
    capture = capture_payload(image, title, f"{identifier}.png", synthetic=True)
    values = {
        "title": title, "material": "White oak", "lot": lot, "surface": "sanded",
        "state": {"calibration": {"mode": "pixels"}, "region": None, "lighting": None},
    }
    return db.import_inspection(capture, values)


def main():
    with tempfile.TemporaryDirectory() as folder:
        db = Database(__import__("pathlib").Path(folder))
        app = create_app(data_folder=folder)
        client = app.test_client()

        base_rgb, base_support, _ = synth.synthetic_face(seed=23)
        rephoto_rgb, rephoto_support = synth.rephotograph(base_rgb, angle=20.0, brightness=1.12, crop=0.12)
        other_rgb, other_support, _ = synth.synthetic_face(seed=91)

        old = make_inspection(db, base_rgb, base_support, title="Strip 7 old", lot="L45", identifier="old")
        new = make_inspection(db, rephoto_rgb, rephoto_support, title="Strip 7 rephoto", lot="L45", identifier="new")
        foreign = make_inspection(db, other_rgb, other_support, title="Other lot", lot="ZZ", identifier="foreign")

        # Candidate list is restricted to the same lot and excludes the query.
        resp = client.get(f"/api/inspections/{new['id']}/face-match-candidates")
        assert resp.status_code == 200, resp.data
        candidates = resp.get_json()["items"]
        candidate_ids = {item["id"] for item in candidates}
        assert old["id"] in candidate_ids
        assert new["id"] not in candidate_ids
        assert foreign["id"] not in candidate_ids
        print("candidate list ok:", [item["title"] for item in candidates])

        # Cross-lot screening must be rejected.
        resp = client.post("/api/face-matches/screen", json={
            "query_inspection_id": new["id"],
            "candidate_inspection_ids": [foreign["id"]],
        })
        assert resp.status_code == 400, resp.data
        print("cross-lot rejection ok:", resp.get_json()["error"])

        # Same-lot screening returns a stored match with the two source ids.
        resp = client.post("/api/face-matches/screen", json={
            "query_inspection_id": new["id"],
            "candidate_inspection_ids": [old["id"]],
        })
        assert resp.status_code == 201, resp.data
        items = resp.get_json()["items"]
        assert len(items) == 1
        match = items[0]
        assert match["query_inspection_id"] == new["id"]
        assert match["candidate_inspection_id"] == old["id"]
        assert match["verdict"] in VERDICTS
        assert match["disposition"] == "undecided"
        print("screening ok: verdict", match["verdict"], "ncc", match["transform"]["ncc"])

        # The evidence sheet renders as a PNG.
        resp = client.get(match["links"]["sheet"])
        assert resp.status_code == 200 and resp.data[:8] == b"\x89PNG\r\n\x1a\n"
        print("evidence sheet ok:", len(resp.data), "bytes")

        # Operator confirms; the machine verdict must not auto-confirm anything.
        resp = client.post(f"/api/face-matches/{match['id']}/decision", json={
            "disposition": "confirmed", "note": "Pore layout matches.",
        })
        assert resp.status_code == 200, resp.data
        decided = resp.get_json()
        assert decided["disposition"] == "confirmed"
        print("decision ok:", decided["disposition"])

        # History exposes the recorded check.
        resp = client.get(f"/api/inspections/{new['id']}/face-matches")
        history = resp.get_json()["items"]
        assert any(item["id"] == match["id"] and item["disposition"] == "confirmed" for item in history)
        print("history ok:", len(history), "record(s)")

        # Re-screening keeps the human decision while the two captures are fixed.
        resp = client.post("/api/face-matches/screen", json={
            "query_inspection_id": new["id"],
            "candidate_inspection_ids": [old["id"]],
        })
        again = resp.get_json()["items"][0]
        assert again["id"] == match["id"] and again["disposition"] == "confirmed"
        print("decision retained on re-screen ok")

        # Source inspections and their readings remain independent records.
        assert db.inspection(new["id"])["capture_id"] == new["capture_id"]
        assert db.inspection(old["id"])["capture_id"] == old["capture_id"]
        assert new["capture_id"] != old["capture_id"]
        print("source records remain independent ok")

        print("\nservice integration checks passed")


if __name__ == "__main__":
    main()
