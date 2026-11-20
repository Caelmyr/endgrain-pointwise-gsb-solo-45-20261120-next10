import io
import zipfile

from endgrain.web import create_app


def client(tmp_path):
    app = create_app(str(tmp_path / "data"))
    return app.test_client()


def crack_run(app, inspection_id, revision=1, params=None):
    response = app.post(
        f"/api/inspections/{inspection_id}/runs",
        json={"kind": "cracks", "revision": revision, "params": params or {}},
    )
    assert response.status_code == 201
    return response.get_json()


def test_crack_run_saves_and_reconciles_its_views(tmp_path):
    app = client(tmp_path)
    app.post("/api/demo")
    inspection = app.get("/api/inspections").get_json()["items"][0]
    run = crack_run(app, inspection["id"])
    metrics = run["result"]["metrics"]
    assert abs(metrics["segment_length"] + metrics["fork_length"] - metrics["total_length"]) < 1e-8
    assert run["result"]["segments"]
    assert set(run["links"]["views"]) == {"segments", "review"}
    for url in run["links"]["views"].values():
        assert app.get(url).status_code == 200


def test_segment_overlay_is_available_after_reload(tmp_path):
    app = client(tmp_path)
    app.post("/api/demo")
    inspection = app.get("/api/inspections").get_json()["items"][0]
    run = crack_run(app, inspection["id"])
    stored = app.get(run["links"]["data"]).get_json()
    assert stored["links"]["views"] == run["links"]["views"]
    assert app.get(stored["links"]["views"]["segments"]).status_code == 200


def test_report_bundle_carries_the_segment_overlays(tmp_path):
    app = client(tmp_path)
    app.post("/api/demo")
    inspection = app.get("/api/inspections").get_json()["items"][0]
    run = crack_run(app, inspection["id"])
    run_id = run["links"]["data"].rsplit("/", 1)[-1]
    report = app.post(
        f"/api/inspections/{inspection['id']}/reports",
        json={
            "revision": 1, "title": "Crack review", "reviewer": "QA", "disposition": "recorded",
            "summary": "", "run_ids": [run_id],
        },
    ).get_json()
    archive = app.get(report["links"]["zip"])
    assert archive.status_code == 200
    names = zipfile.ZipFile(io.BytesIO(archive.data)).namelist()
    assert "measurements/01-cracks.png" in names
    assert "measurements/01-cracks-segments.png" in names
    assert "measurements/01-cracks-review.png" in names
    check = app.post(
        "/api/evidence/check",
        data={"file": (io.BytesIO(archive.data), "evidence.zip")},
        content_type="multipart/form-data",
    ).get_json()
    assert check["verified"] is True


def test_csv_exposes_trace_segment_fork_and_total_rows(tmp_path):
    app = client(tmp_path)
    app.post("/api/demo")
    inspection = app.get("/api/inspections").get_json()["items"][0]
    run = crack_run(app, inspection["id"])
    lines = app.get(run["links"]["csv"]).data.decode("utf-8-sig").splitlines()
    header = lines[0].split(",")
    assert "row_type" in header and "from" in header and "to" in header
    kind_index = header.index("row_type")
    kinds = set()
    for line in lines[1:]:
        if not line:
            continue
        cells = line.split(",")
        kinds.add(cells[kind_index])
    assert {"trace", "segment", "total"} <= kinds
    if run["result"]["forks"]:
        assert "fork" in kinds
    assert any(line.startswith("segments total") for line in lines[1:])

