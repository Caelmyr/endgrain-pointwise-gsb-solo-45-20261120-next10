"""Plan spaced micrograph stations inside two hand-drawn representative faces."""

from dataclasses import dataclass
from datetime import datetime, timezone
from math import floor
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

from .database import Database
from .errors import Conflict, Missing, WorkbenchError
from .geometry import Raster, signed_area, simple_polygon, pixel_point
from .imaging import encode_png
from .quality import measure_quality
from .reports import render_csv
from .validation import choice, document, fields, flag, integer, list_of, number, string
from .workspaces import describe_plane, frame_for

FACE_IDS = ("A", "B")
STATUSES = ("planned", "shot", "void")
COVERAGE_FLOOR = 0.996
GLARE_CEILING = 0.012
CLIP_CEILING = 0.030
SHADOW_CEILING = 0.020
SOFT_CEILING = 0.300
MAX_STATIONS = 96

PLANNED_INK = (126, 190, 137)
SHOT_INK = (104, 183, 202)
VOID_INK = (216, 118, 97)
FACE_INK = ((236, 205, 139), (143, 202, 209))


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass(frozen=True)
class Screening:
    missing: np.ndarray
    glare: np.ndarray
    clipped: np.ndarray
    shadow: np.ndarray
    soft: np.ndarray

    @property
    def bad(self) -> np.ndarray:
        return self.missing | self.glare | self.clipped | self.shadow | self.soft


@dataclass(frozen=True)
class CandidateGrid:
    cx: np.ndarray
    cy: np.ndarray
    quality: np.ndarray
    edge: np.ndarray
    valid: np.ndarray


@dataclass(frozen=True)
class Placement:
    center: tuple[int, int]
    score: float


def screen(raster: Raster) -> Screening:
    reading = measure_quality(raster, 10, 0.60)
    glare = reading.glare
    return Screening(
        missing=~raster.support,
        glare=glare,
        clipped=reading.clip_highlight & ~glare,
        shadow=reading.clip_shadow,
        soft=reading.soft,
    )


def face_mask(polygon: list[list[float]], width: int, height: int) -> np.ndarray:
    canvas = Image.new("1", (width, height))
    ImageDraw.Draw(canvas).polygon([tuple(point) for point in polygon], fill=1)
    return np.asarray(canvas, dtype=bool)


def integral(mask: np.ndarray) -> np.ndarray:
    prepared = np.zeros((mask.shape[0] + 1, mask.shape[1] + 1), dtype=np.float64)
    prepared[1:, 1:] = np.cumsum(np.cumsum(mask.astype(np.float64), axis=0), axis=1)
    return prepared


def window_sum(table: np.ndarray, x0: np.ndarray, y0: np.ndarray, width: int, height: int) -> np.ndarray:
    return table[y0 + height, x0 + width] - table[y0, x0 + width] - table[y0 + height, x0] + table[y0, x0]


def candidate_grid(
    mask: np.ndarray, spatial: Screening, width_px: int, height_px: int, step: int,
) -> CandidateGrid | None:
    rows, cols = np.nonzero(mask)
    if not len(rows):
        return None
    left, right = int(cols.min()), int(cols.max()) + 1
    top, bottom = int(rows.min()), int(rows.max()) + 1
    half_x, half_y = width_px // 2, height_px // 2
    xs = np.arange(left + half_x, right - (width_px - half_x) + 1, step, dtype=np.int64)
    ys = np.arange(top + half_y, bottom - (height_px - half_y) + 1, step, dtype=np.int64)
    if not len(xs) or not len(ys):
        return None
    cx, cy = np.meshgrid(xs, ys)
    x0, y0 = cx - half_x, cy - half_y
    keep = (x0 >= 0) & (y0 >= 0) & (x0 + width_px <= mask.shape[1]) & (y0 + height_px <= mask.shape[0])
    table = integral(mask)
    total = float(width_px * height_px)
    inside = window_sum(table, np.clip(x0, 0, mask.shape[1]), np.clip(y0, 0, mask.shape[0]), width_px, height_px)
    fields = [integral(item) for item in (spatial.missing, spatial.glare, spatial.clipped, spatial.shadow, spatial.soft)]
    fractions = [
        window_sum(entry, np.clip(x0, 0, mask.shape[1]), np.clip(y0, 0, mask.shape[0]), width_px, height_px) / total
        for entry in fields
    ]
    missing, glare, clipped, shadow, soft = fractions
    valid = (
        keep
        & (inside >= total - 0.5)
        & (missing <= 1 - COVERAGE_FLOOR)
        & (glare <= GLARE_CEILING)
        & (clipped <= CLIP_CEILING)
        & (shadow <= SHADOW_CEILING)
        & (soft <= SOFT_CEILING)
    )
    quality = np.clip(
        1.0 - 2.5 * glare - 1.5 * clipped - 1.5 * shadow - 0.9 * soft - 2.0 * missing, 0.0, 1.0
    )
    usable = mask & ~spatial.bad
    distance = ndi.distance_transform_edt(usable) if usable.any() else np.zeros(mask.shape)
    edge = distance[np.clip(y0, 0, mask.shape[0] - 1), np.clip(x0, 0, mask.shape[1] - 1)]
    return CandidateGrid(cx, cy, np.where(valid, quality, -1.0), edge, valid)


def rect_size(station: dict) -> tuple[int, int]:
    frame = station["frame"]
    return int(frame[2] - frame[0]), int(frame[3] - frame[1])


def station_center(station: dict) -> tuple[float, float]:
    x0, y0, x1, y1 = station["frame"]
    return (x0 + x1) / 2, (y0 + y1) / 2


def choose_candidate(
    grid: CandidateGrid, stations: list[dict], width_px: int, height_px: int,
    gap: float, near: tuple[float, float] | None = None, radius: float = 0.0,
) -> Placement | None:
    allowed = grid.valid.copy()
    distances = []
    for station in stations:
        other_w, other_h = rect_size(station)
        ox, oy = station_center(station)
        dx = np.abs(grid.cx - ox)
        dy = np.abs(grid.cy - oy)
        separated = (dx >= (width_px + other_w) / 2 + gap) | (dy >= (height_px + other_h) / 2 + gap)
        allowed &= separated
        if station["status"] != "void":
            distances.append(np.maximum(
                dx / ((width_px + other_w) / 2 + gap), dy / ((height_px + other_h) / 2 + gap),
            ))
    if not allowed.any():
        return None
    quality = np.where(allowed, grid.quality, 0.0)
    if not np.any(quality > 1e-6):
        return None
    if near is not None:
        away = np.hypot(grid.cx - near[0], grid.cy - near[1])
        value = 0.5 * quality + 0.5 * np.clip(1 - away / max(radius, 1.0), 0, 1)
    elif distances:
        spacing = np.minimum(np.min(distances, axis=0), 2.0)
        value = 0.55 * spacing / 2 + 0.45 * quality
    else:
        reach = np.minimum(grid.edge / max(width_px, height_px), 2.0)
        value = 0.55 * reach / 2 + 0.45 * quality
    flat = int(np.argmax(value))
    return Placement((int(grid.cx.flat[flat]), int(grid.cy.flat[flat])), float(value.flat[flat]))


def lattice_slots(
    grid: CandidateGrid, blockers: list[dict], width_px: int, height_px: int, gap: float,
) -> list[Placement]:
    """Pack separated fields on a lattice so a clear face is actually filled.

    Greedy nearest-first placement can strand room. The candidate grid is a regular
    lattice already, so keeping every n-th row and column, shifted by a few offsets
    and taking the densest clear set, fills the face's true capacity.
    """
    columns, rows = grid.cx[0], grid.cy[:, 0]
    step_x = float(columns[1] - columns[0]) if len(columns) > 1 else 1.0
    step_y = float(rows[1] - rows[0]) if len(rows) > 1 else 1.0
    jump_x = max(1, int(np.ceil((width_px + gap) / step_x - 1e-6)))
    jump_y = max(1, int(np.ceil((height_px + gap) / step_y - 1e-6)))
    blocked = np.zeros(grid.valid.shape, dtype=bool)
    for station in blockers:
        other_w, other_h = rect_size(station)
        ox, oy = station_center(station)
        blocked |= ~(
            (np.abs(grid.cx - ox) >= (width_px + other_w) / 2 + gap)
            | (np.abs(grid.cy - oy) >= (height_px + other_h) / 2 + gap)
        )
    free = grid.valid & ~blocked
    best: tuple[int, list[tuple[int, int]]] | None = None
    for offset_x in range(jump_x):
        for offset_y in range(jump_y):
            points = [
                (row, col)
                for row in range(offset_y, grid.valid.shape[0], jump_y)
                for col in range(offset_x, grid.valid.shape[1], jump_x)
                if free[row, col]
            ]
            if best is None or len(points) > len(best[1]):
                best = (offset_x * 1000 + offset_y, points)
            if len(points) >= MAX_STATIONS:
                break
        if best and len(best[1]) >= MAX_STATIONS:
            break
    if best is None or not best[1]:
        return []
    picked = sorted(
        best[1],
        key=lambda pair: (
            -float(grid.quality[pair]), 
            -float(grid.edge[pair]),
        ),
    )
    return [
        Placement(
            (int(grid.cx[row, col]), int(grid.cy[row, col])),
            float(grid.quality[row, col]),
        )
        for row, col in picked
    ]


def split_budget(minimums: list[int], available: int) -> list[int]:
    wanted = sum(minimums)
    if wanted <= available:
        return list(minimums)
    if available <= 0:
        return [0] * len(minimums)
    shares = [value * available / wanted for value in minimums]
    counts = [floor(value) for value in shares]
    order = sorted(range(len(shares)), key=lambda index: (shares[index] - counts[index], minimums[index]), reverse=True)
    for index in order[: available - sum(counts)]:
        counts[index] += 1
    return counts


def face_spec(raw: Any, plane, position: int) -> dict:
    body = document(raw, f"face {position}")
    fields(body, {"id", "label", "minimum", "polygon"}, f"face {position}")
    identifier = choice(body.get("id"), "face identifier", FACE_IDS)
    label = string(body.get("label") or f"Face {identifier}", "face label", 42)
    polygon = [
        pixel_point(point, f"face {identifier} point", plane.width, plane.height)
        for point in list_of(body.get("polygon"), f"face {identifier} outline", 3, 64)
    ]
    if not simple_polygon(np.asarray(polygon)) or abs(signed_area(polygon)) < 36:
        raise WorkbenchError(f"Draw face {identifier} as a simple outline with real area.", "faces")
    return {
        "id": identifier,
        "label": label,
        "minimum": integer(body.get("minimum", 0), f"face {identifier} minimum stations", 0, MAX_STATIONS),
        "polygon": polygon,
    }


def normalize_config(raw: Any, plane) -> dict:
    body = document(raw, "sampling plan")
    fields(body, {"frame_width_mm", "frame_height_mm", "budget", "spacing_mm", "faces", "reset"}, "sampling plan")
    faces = [face_spec(item, plane, index) for index, item in enumerate(list_of(body.get("faces"), "faces", 2, 2), 1)]
    if faces[0]["id"] == faces[1]["id"]:
        raise WorkbenchError("Name the two faces differently.", "faces")
    config = {
        "frame_width_mm": number(body.get("frame_width_mm"), "field width in mm", 0.01, 500),
        "frame_height_mm": number(body.get("frame_height_mm"), "field height in mm", 0.01, 500),
        "budget": integer(body.get("budget"), "maximum frames", 1, MAX_STATIONS),
        "spacing_mm": number(body.get("spacing_mm", 0), "minimum gap in mm", 0, 100),
        "faces": faces,
        "reset": flag(body.get("reset", False), "reset"),
    }
    return config


def new_station(
    face: str, counters: dict, center: tuple[int, int], width_px: int, height_px: int,
    purpose: str, quota: bool, **extra: Any,
) -> dict:
    counters[face] += 1
    x0 = int(center[0]) - width_px // 2
    y0 = int(center[1]) - height_px // 2
    return {
        "id": f"{face}-{counters[face]:02d}",
        "face": face,
        "status": "planned",
        "purpose": purpose,
        "quota": quota,
        "frame": [x0, y0, x0 + width_px, y0 + height_px],
        "replaces": extra.get("replaces"),
        "void_reason": "",
        "note": extra.get("note", ""),
        "created_at": stamp(),
        "marked_at": None,
    }


def counters_from(stations: list[dict]) -> dict:
    counters = {face: 0 for face in FACE_IDS}
    for station in stations:
        face, _, sequence = station["id"].partition("-")
        if face in counters and sequence.isdigit():
            counters[face] = max(counters[face], int(sequence))
    return counters


def outline_changed(old: list, new: list) -> bool:
    if len(old) != len(new):
        return True
    return any(
        abs(float(a[0]) - float(b[0])) > 0.5 or abs(float(a[1]) - float(b[1])) > 0.5
        for a, b in zip(old, new, strict=True)
    )


def previous_stations(plan: dict | None, revision: int, reset: bool, config: dict) -> list[dict]:
    if not plan or reset:
        return []
    if plan.get("revision") != revision:
        raise Conflict(
            "The measuring setup changed after this station plan was made. Reset the plan to place stations on the current scale.",
            reset_required=True,
        )
    stored = plan.get("config", {})
    same = all(
        stored.get(key) == config[key]
        for key in ("frame_width_mm", "frame_height_mm", "budget", "spacing_mm")
    ) and not any(
        outline_changed(old.get("polygon", []), new["polygon"])
        for old, new in zip(stored.get("faces", []), config["faces"], strict=True)
    )
    return [
        dict(item) for item in plan.get("stations", [])
        if item["status"] in {"shot", "void"} or (same and item["status"] == "planned")
    ]


def place_stations(
    raster: Raster, spatial: Screening, config: dict, kept: list[dict],
) -> tuple[list[dict], dict, dict]:
    plane = raster.plane
    width_px = max(2, int(round(config["frame_width_mm"] / plane.mm_x)))
    height_px = max(2, int(round(config["frame_height_mm"] / plane.mm_y)))
    gap = config["spacing_mm"] / plane.mm_x
    step = max(2, int(round(min(width_px, height_px) / 7)))
    stations = [dict(item) for item in kept]
    counters = counters_from(stations)
    active = sum(1 for item in stations if item["status"] in {"planned", "shot"})
    diagnostics: dict[str, dict] = {}
    grids: dict[str, CandidateGrid | None] = {}
    for face in config["faces"]:
        mask = face_mask(face["polygon"], plane.width, plane.height)
        grids[face["id"]] = candidate_grid(mask, spatial, width_px, height_px, step)
        area = max(1, int(mask.sum()))
        diagnostics[face["id"]] = {
            "area_px": area,
            "outside_image": float((mask & spatial.missing).sum() / area),
            "glare": float((mask & spatial.glare).sum() / area),
            "clipped": float((mask & spatial.clipped).sum() / area),
            "shadow": float((mask & spatial.shadow).sum() / area),
            "low_detail": float((mask & spatial.soft).sum() / area),
            "unusable_fraction": float((mask & spatial.bad).sum() / area),
            "placed": 0,
            "capacity": capacity_of(grids[face["id"]], width_px, height_px, gap),
        }
    # A position that was voided no longer represents its face, so its quota
    # stays unfilled until a replacement candidate is actually captured.
    remaining = {
        face["id"]: max(0, face["minimum"] - sum(
            1 for item in stations
            if item["face"] == face["id"] and item["status"] in {"planned", "shot"} and item["quota"]
        ))
        for face in config["faces"]
    }
    # Cap each share at what the face can actually hold, so the budget is not
    # reserved for fields that cannot be placed there.
    caps = [
        min(remaining[face["id"]], diagnostics[face["id"]]["capacity"])
        for face in config["faces"]
    ]
    targets = split_budget(caps, config["budget"] - active)
    maximum = {face["id"]: count for face, count in zip(config["faces"], targets, strict=True)}
    placed = {face["id"]: 0 for face in config["faces"]}
    all_placed: list[dict] = []

    def refresh(face_id: str) -> list[Placement]:
        return lattice_slots(grids[face_id], stations, width_px, height_px, gap) if grids[face_id] is not None else []

    slots = {face["id"]: refresh(face["id"]) for face in config["faces"]}

    def take(face_id: str, purpose: str, quota: bool) -> bool:
        if not slots[face_id]:
            return False
        station = new_station(
            face_id, counters, slots[face_id][0].center, width_px, height_px,
            purpose, quota, note="" if purpose == "quota" else "Extra coverage; not a replacement for a short face.",
        )
        stations.append(station)
        all_placed.append(station)
        placed[face_id] += 1
        slots[face_id] = refresh(face_id)
        return True

    for _ in range(max(maximum.values(), default=0)):
        for face in config["faces"]:
            face_id = face["id"]
            if placed[face_id] < maximum[face_id]:
                take(face_id, "quota", True)
    while active + len(all_placed) < config["budget"]:
        progressed = False
        for face in config["faces"]:
            if active + len(all_placed) >= config["budget"]:
                break
            if take(face["id"], "supplement", False):
                progressed = True
        if not progressed:
            break
    for face in config["faces"]:
        diagnostics[face["id"]]["placed"] = placed[face["id"]]
    return stations, counters, diagnostics


def capacity_of(grid: CandidateGrid | None, width_px: int, height_px: int, gap: float) -> int:
    """Count the separated clear fields this face holds with unlimited frames."""
    return len(lattice_slots(grid, [], width_px, height_px, gap)) if grid is not None else 0


def plan(db: Database, inspection_id: str, raw: Any) -> dict:
    current = db.inspection(inspection_id)
    plane = describe_plane(current)
    if not plane.physical:
        raise WorkbenchError("Calibrate this face with a ruler or known rectangle before planning micrograph stations.", "calibration")
    config = normalize_config(raw, plane)
    reset = config.pop("reset")
    previous = db.station_plan(inspection_id)
    redrawn = previous is not None and any(
        outline_changed(old.get("polygon", []), new["polygon"])
        for old, new in zip(previous.get("config", {}).get("faces", []), config["faces"], strict=False)
    )
    if redrawn and any(item["status"] == "shot" for item in previous["stations"]) and not reset:
        raise Conflict(
            "A face outline changed after frames were captured. Reset the plan to re-place stations, or keep the outline and the captured frames.",
            reset_required=True,
        )
    if redrawn:
        reset = True
    kept = previous_stations(previous, current["revision"], reset, config)
    frame = frame_for(db, current)
    spatial = screen(frame.raster)
    stations, counters, diagnostics = place_stations(frame.raster, spatial, config, kept)
    now = stamp()
    payload = {
        "version": 1,
        "revision": current["revision"],
        "created_at": now if not previous or reset else previous.get("created_at", now),
        "updated_at": now,
        "config": config,
        "frame_px": {
            "width": max(2, int(round(config["frame_width_mm"] / plane.mm_x))),
            "height": max(2, int(round(config["frame_height_mm"] / plane.mm_y))),
        },
        "counters": counters,
        "stations": stations,
        "diagnostics": diagnostics,
    }
    db.save_station_plan(
        inspection_id, current["revision"], payload, "sampling_planned",
        {"stations": len(stations), "reset": reset},
    )
    return describe(payload, current)


VOID_FIELDS = ("id", "face", "frame", "void_reason", "marked_at", "replaces", "quota")


def replacement_links(stations: list[dict]) -> dict[str, str]:
    """Map each captured station back to the voided position it stands in for."""
    return {
        item["replaces"]: item["id"] for item in stations
        if item.get("replaces") and item["status"] in {"planned", "shot"}
    }


def void_history(payload: dict) -> list[dict]:
    """Keep every withdrawn position, even after a re-plan drops it from the walk list."""
    return [
        {key: item.get(key) for key in VOID_FIELDS}
        for item in payload["stations"] if item["status"] == "void"
    ]


def replacement_for(
    payload: dict, voided: dict, raster: Raster, spatial: Screening,
) -> tuple[dict | None, list[str]]:
    notices: list[str] = []
    config = payload["config"]
    plane = raster.plane
    face = next(item for item in config["faces"] if item["id"] == voided["face"])
    width_px = max(2, int(round(config["frame_width_mm"] / plane.mm_x)))
    height_px = max(2, int(round(config["frame_height_mm"] / plane.mm_y)))
    gap = config["spacing_mm"] / plane.mm_x
    active = sum(
        1 for item in payload["stations"]
        if item["status"] in {"planned", "shot"} and item["id"] != voided["id"]
    )
    if active >= config["budget"]:
        notices.append("The frame budget is full; drop a planned station before adding a replacement.")
        return None, notices
    grid = candidate_grid(
        face_mask(face["polygon"], plane.width, plane.height), spatial,
        width_px, height_px, max(2, int(round(min(width_px, height_px) / 7))),
    )
    if grid is None:
        notices.append(f"{voided['id']} is void and no clear field fits inside {face['label']}.")
        return None, notices
    # The voided frame stays a blocker so the candidate is a genuinely different
    # field rather than a copy of the position that could not be photographed.
    blockers = payload["stations"]
    nearby = station_center(voided)
    for factor in (2.2, 3.5, 6.0):
        chosen = choose_candidate(
            grid, blockers, width_px, height_px, gap,
            near=nearby, radius=factor * max(width_px, height_px),
        )
        if chosen is not None:
            break
    if chosen is None:
        notices.append(
            f"{voided['id']} is void and no separate, non-overlapping replacement fits inside {face['label']}."
        )
        return None, notices
    if np.hypot(chosen.center[0] - nearby[0], chosen.center[1] - nearby[1]) > 3.5 * max(width_px, height_px):
        notices.append(f"The replacement for {voided['id']} sits well away from the voided position.")
    station = new_station(
        face["id"], payload["counters"], chosen.center, width_px, height_px,
        "replacement", bool(voided["quota"]), replaces=voided["id"],
        note=f"Replacement for voided {voided['id']}.",
    )
    payload["stations"].append(station)
    return station, notices


def mark(db: Database, inspection_id: str, raw: Any) -> dict:
    body = document(raw, "station mark")
    fields(body, {"revision", "station", "status", "reason"}, "station mark")
    current = db.inspection(inspection_id)
    revision = integer(body.get("revision"), "revision", 1, 2**31 - 1)
    db.validate_revision(current, revision)
    payload = db.station_plan(inspection_id)
    if payload is None:
        raise Missing("Station plan")
    if payload.get("revision") != revision:
        raise Conflict(
            "The measuring setup changed after this station plan was made. Reset the plan to place stations on the current scale.",
            reset_required=True,
        )
    wanted = choice(body.get("status"), "station status", STATUSES)
    target = next((item for item in payload["stations"] if item["id"] == body.get("station")), None)
    if target is None:
        raise Missing("Station")
    if target["status"] == wanted:
        return describe(payload, current)
    if target["status"] == "void" and wanted == "shot":
        raise WorkbenchError("Restore this voided position before marking it captured.", "status")
    notices: list[str] = []
    if wanted == "void":
        reason = string(body.get("reason"), "void reason", 240, required=True)
        target["status"] = "void"
        target["void_reason"] = reason
        target["marked_at"] = stamp()
        frame = frame_for(db, current)
        spatial = screen(frame.raster)
        replacement, alerts = replacement_for(payload, target, frame.raster, spatial)
        notices.extend(alerts)
        if replacement:
            notices.append(f"{replacement['id']} is the replacement candidate for voided {target['id']}.")
    elif wanted == "shot":
        target["status"] = "shot"
        target["marked_at"] = stamp()
        linked = [
            item for item in payload["stations"]
            if item.get("replaces") == target["id"] and item["status"] == "planned"
        ]
        for item in linked:
            payload["stations"].remove(item)
        if linked:
            notices.append(f"Dropped {len(linked)} unused replacement(s) for {target['id']}.")
    else:
        target["status"] = "planned"
        target["marked_at"] = stamp()
        target["void_reason"] = ""
        linked = [
            item for item in payload["stations"]
            if item.get("replaces") == target["id"] and item["status"] == "planned"
        ]
        for item in linked:
            payload["stations"].remove(item)
        if linked:
            notices.append(f"Removed the unused replacement for {target['id']}.")
        target["replaces"] = None
    payload["updated_at"] = stamp()
    action = {"shot": "station_captured", "void": "station_voided", "planned": "station_restored"}[wanted]
    db.save_station_plan(inspection_id, revision, payload, action, {"station": target["id"], "status": wanted})
    described = describe(payload, current)
    described["notices"] = notices + described["notices"]
    return described


def load(db: Database, inspection_id: str) -> dict:
    current = db.inspection(inspection_id)
    payload = db.station_plan(inspection_id)
    if payload is None:
        return {"revision": current["revision"], "plan": None, "summary": None, "notices": [], "links": plan_links(inspection_id, current["revision"])}
    return describe(payload, current)


def plan_links(inspection_id: str, revision: int) -> dict:
    root = "/api/inspections/" + inspection_id + "/sampling"
    return {"image": root + "/plan.png?r=" + str(revision), "csv": root + "/plan.csv?r=" + str(revision)}


REASON_LABELS = (
    ("outside_image", "outside the photographed support"),
    ("glare", "glare"),
    ("clipped", "clipped highlights"),
    ("shadow", "deep shadow"),
    ("low_detail", "low local detail"),
)


def dominant_reason(reasons: dict) -> str:
    key, value = max(reasons.items(), key=lambda pair: pair[1], default=("", 0.0))
    labels = dict(REASON_LABELS)
    if not key or value < 0.02:
        return ""
    return f"; the largest obstacle is {labels[key]} ({value:.0%} of the face)"


def face_counts(payload: dict) -> tuple[list[dict], list[str]]:
    config = payload["config"]
    stations = payload["stations"]
    diagnostics = payload.get("diagnostics", {})
    summary = []
    for face in config["faces"]:
        own = [item for item in stations if item["face"] == face["id"]]
        quota = [item for item in own if item["quota"] and item["status"] in {"planned", "shot"}]
        extra = [item for item in own if not item["quota"] and item["status"] in {"planned", "shot"}]
        covered = len(quota)
        fits = diagnostics.get(face["id"], {}).get("placed")
        summary.append({
            "id": face["id"],
            "label": face["label"],
            "minimum": face["minimum"],
            "covered": covered,
            "planned": sum(1 for item in own if item["status"] == "planned"),
            "shot": sum(1 for item in own if item["status"] == "shot"),
            "void": sum(1 for item in own if item["status"] == "void"),
            "shortfall": max(0, face["minimum"] - covered),
            "extra": len(extra),
            "fits": fits,
            "capacity": diagnostics.get(face["id"], {}).get("capacity"),
            "limit": "budget",
            "unusable_fraction": diagnostics.get(face["id"], {}).get("unusable_fraction"),
            "reasons": {
                key: diagnostics.get(face["id"], {}).get(key, 0.0)
                for key in ("outside_image", "glare", "clipped", "shadow", "low_detail")
            },
        })
    notices = []
    for face in summary:
        if not face["shortfall"]:
            continue
        if face["capacity"] is not None and face["capacity"] < face["minimum"]:
            face["limit"] = "fit"
            reason = (
                f"only {face['capacity']} non-overlapping field(s) fit on clear, photographed pixels"
                + dominant_reason(face["reasons"])
            )
        else:
            face["limit"] = "budget"
            reason = "the frame budget runs out before this minimum is met"
        notices.append(f"{face['label']} is short {face['shortfall']} station(s): {reason}.")
        for other in summary:
            if other["id"] == face["id"] or not other["extra"]:
                continue
            notices.append(
                f"The {other['extra']} extra frame(s) on {other['label']} are spare coverage only; they do not represent {face['label']}."
            )
    return summary, notices


def describe(payload: dict, current: dict) -> dict:
    plane = describe_plane(current)
    config = payload["config"]
    face_summary, notices = face_counts(payload)
    short = {item["id"]: item["shortfall"] for item in face_summary}
    stations = []
    for station in payload["stations"]:
        x0, y0, x1, y1 = station["frame"]
        center_x, center_y = (x0 + x1) / 2, (y0 + y1) / 2
        mapped = plane.map_source(np.asarray([center_x, center_y]))
        source = mapped.tolist() if np.isfinite(mapped).all() else None
        other_short = [face for face, missing in short.items() if missing and face != station["face"]]
        stations.append({
            **station,
            "width_px": int(x1 - x0),
            "height_px": int(y1 - y0),
            "width_mm": float((x1 - x0) * (plane.mm_x or 0)),
            "height_mm": float((y1 - y0) * (plane.mm_y or 0)),
            "center": [center_x, center_y],
            "center_mm": [float((center_x + 0.5) * (plane.mm_x or 0)), float((center_y + 0.5) * (plane.mm_y or 0))],
            "top_left_mm": [float(x0 * (plane.mm_x or 0)), float(y0 * (plane.mm_y or 0))],
            "source": source,
            "substitutes_shortfall": other_short if (not station["quota"] and station["status"] != "void") else [],
        })
    order = {face["id"]: index for index, face in enumerate(config["faces"])}
    walk = sorted(
        (item for item in stations if item["status"] != "void"),
        key=lambda item: (order.get(item["face"], 9), item["center"][1], item["center"][0]),
    )
    for index, station in enumerate(walk, 1):
        station["walk_order"] = index

    counts = {status: sum(1 for item in stations if item["status"] == status) for status in STATUSES}
    active = counts["planned"] + counts["shot"]
    summary = {
        "budget": config["budget"],
        "active": active,
        "counts": counts,
        "faces": face_summary,
        "shortfall_total": sum(item["shortfall"] for item in face_summary),
    }
    if payload.get("revision") != current["revision"]:
        notices.insert(0, "This plan belongs to an earlier measuring setup. Reset and re-plan before walking it.")
    return {
        "revision": current["revision"],
        "plan": {
            "revision": payload["revision"],
            "created_at": payload["created_at"],
            "updated_at": payload["updated_at"],
            "config": config,
            "frame_px": payload["frame_px"],
            "stations": stations,
            "void_history": void_history(payload),
            "replacement_of": replacement_links(payload["stations"]),
        },
        "summary": summary,
        "notices": notices,
        "links": plan_links(current["id"], current["revision"]),
    }


def font(size: int) -> ImageFont.ImageFont:
    return ImageFont.load_default(size=size)


def dashed(pen: ImageDraw.ImageDraw, box: tuple[float, float, float, float], color, width: int, dash: float = 9) -> None:
    x0, y0, x1, y1 = box
    for start, end in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
        length = max(abs(end[0] - start[0]), abs(end[1] - start[1]))
        if not length:
            continue
        for step in np.arange(0, length, dash * 2):
            ratio = min(1.0, step / length)
            ratio_end = min(1.0, (step + dash) / length)
            pen.line(
                (
                    (start[0] + (end[0] - start[0]) * ratio, start[1] + (end[1] - start[1]) * ratio),
                    (start[0] + (end[0] - start[0]) * ratio_end, start[1] + (end[1] - start[1]) * ratio_end),
                ),
                fill=color, width=width,
            )


def chip(pen: ImageDraw.ImageDraw, position: tuple[float, float], text: str, fill, ink, size: int) -> None:
    handle = font(size)
    width = handle.getlength(text) + size * 0.7
    x, y = position
    pen.rectangle((x, y, x + width, y + size * 1.35), fill=fill)
    pen.text((x + size * 0.32, y + size * 0.12), text, font=handle, fill=ink)


def box_gap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Separation between two axis-aligned boxes; negative when they overlap."""
    return max(b[0] - a[2], a[0] - b[2], b[1] - a[3], a[1] - b[3])


def face_label_spot(
    polygon: list[list[float]], text: str, size: int, scale: float,
    bounds: tuple[int, int], obstacles: list[tuple[float, float, float, float]],
) -> tuple[float, float]:
    """Find the clearest label anchor outside a face outline.

    Every vertex is tried, pushed outwards from the face centre, and scored by
    distance from station frames, the legend, the scale bar and the other face label.
    """
    middle = (
        sum(point[0] for point in polygon) / len(polygon),
        sum(point[1] for point in polygon) / len(polygon),
    )
    label_width = font(size).getlength(text) + size * 0.7
    label_height = size * 1.35
    width, height = bounds
    best: tuple[float, float, float] | None = None
    for point in polygon:
        dx, dy = point[0] - middle[0], point[1] - middle[1]
        span = float(np.hypot(dx, dy)) or 1.0
        for reach in (size * 1.15, size * 3.2):
            x = point[0] + dx / span * reach + (-label_width if dx < 0 else 0)
            y = point[1] + dy / span * reach - (label_height if dy < 0 else 0)
            x = min(max(x, 6 * scale), width - label_width - 6 * scale)
            y = min(max(y, 6 * scale), height - label_height - 6 * scale)
            box = (x, y, x + label_width, y + label_height)
            clearance = min((box_gap(box, other) for other in obstacles), default=float("inf"))
            score = min(clearance, size * 2.0) - 0.2 * reach
            if best is None or score > best[0]:
                best = (score, x, y)
    _, x, y = best or (0.0, polygon[0][0], polygon[0][1])
    return x, y


def summary_height(lines: int, size: int, scale: float) -> float:
    return lines * size * 1.5 + 10 * scale


def plan_png(db: Database, inspection_id: str) -> bytes:
    current = db.inspection(inspection_id)
    payload = db.station_plan(inspection_id)
    if payload is None:
        raise Missing("Station plan")
    frame = frame_for(db, current)
    raster = frame.raster
    plane = raster.plane
    described = describe(payload, current)
    spatial = screen(raster)
    tint = raster.rgb.astype(np.float32)
    overlay = [
        (spatial.missing, (128, 92, 86), 0.55),
        (spatial.glare, (246, 205, 92), 0.45),
        (spatial.clipped, (219, 124, 93), 0.35),
        (spatial.soft, (104, 142, 186), 0.28),
    ]
    for mask, color, strength in overlay:
        if mask.any():
            tint[mask] = (1 - strength) * tint[mask] + strength * np.asarray(color)
    base = Image.fromarray(np.rint(np.clip(tint, 0, 255)).astype(np.uint8))
    pen = ImageDraw.Draw(base)
    scale = max(1.0, base.width / 900)
    label_size = int(round(18 * scale))
    config = described["plan"]["config"]
    summary = described["summary"]
    lines = [
        f"Micrograph stations · revision {described['revision']}",
        f"Budget {summary['budget']} · planned {summary['counts']['planned']} · captured {summary['counts']['shot']} · void {summary['counts']['void']}",
    ]
    for face in summary["faces"]:
        detail = f"{face['label']}: {face['covered']}/{face['minimum']} represented"
        if face["shortfall"]:
            detail += f" · short {face['shortfall']}"
        if face["extra"]:
            detail += f" · extra {face['extra']}"
        lines.append(detail)
    box_width = max(font(label_size).getlength(line) for line in lines) + 22 * scale
    legend_top = base.height - summary_height(len(lines), label_size, scale) - 10 * scale
    obstacles = [tuple(station["frame"]) for station in described["plan"]["stations"]]
    obstacles += [
        (10 * scale, legend_top, 10 * scale + box_width, base.height),
        (10 * scale, legend_top - 30 * scale, 10 * scale + box_width, legend_top),
    ]
    for index, face in enumerate(config["faces"]):
        ink = FACE_INK[index % len(FACE_INK)]
        points = [tuple(point) for point in face["polygon"]]
        pen.line(points + points[:1], fill=ink, width=max(2, int(3 * scale)))
        spot = face_label_spot(
            face["polygon"], face["label"], label_size, scale, (base.width, base.height), obstacles,
        )
        chip(pen, spot, face["label"], (32, 40, 32), ink, label_size)
        obstacles.append(
            (spot[0], spot[1], spot[0] + font(label_size).getlength(face["label"]) + label_size * 0.7,
             spot[1] + label_size * 1.35)
        )
    for station in described["plan"]["stations"]:
        x0, y0, x1, y1 = station["frame"]
        box = (x0, y0, x1, y1)
        if station["status"] == "void":
            dashed(pen, box, VOID_INK, max(2, int(3 * scale)), 10 * scale)
            pen.line((x0, y0, x1, y1), fill=VOID_INK, width=max(2, int(2 * scale)))
            pen.line((x1, y0, x0, y1), fill=VOID_INK, width=max(2, int(2 * scale)))
            ink = VOID_INK
        else:
            ink = SHOT_INK if station["status"] == "shot" else PLANNED_INK
            pen.rectangle(box, outline=ink, width=max(2, int(3 * scale)))
            if station["status"] == "shot":
                pen.rectangle((x0 + 3, y0 + 3, x1 - 3, y1 - 3), outline=ink, width=max(1, int(scale)))
        text = station["id"] if station["status"] != "void" else station["id"] + " void"
        chip(pen, (x0 + 2 * scale, y0 + 2 * scale), text, (26, 32, 26), ink, max(12, int(label_size * 0.82)))
    handle = font(label_size)
    box_height = summary_height(len(lines), label_size, scale)
    left = 10 * scale
    top = legend_top
    pen.rectangle((left, top, left + box_width, top + box_height), fill=(26, 31, 26))
    for index, line in enumerate(lines):
        pen.text(
            (left + 8 * scale, top + 6 * scale + index * label_size * 1.5), line,
            font=handle, fill=(243, 235, 205),
        )
    if plane.unit == "mm":
        small = font(max(12, int(label_size * 0.8)))
        span_mm = max(1, round(plane.width * plane.mm_x * 0.15))
        length = span_mm / plane.mm_x
        line_top = top - 10 * scale
        pen.rectangle(
            (left, line_top - 18 * scale, left + length + 26 * scale + small.getlength(f"{span_mm:g} mm"), line_top + 6 * scale),
            fill=(26, 31, 26),
        )
        pen.line((left + 9 * scale, line_top, left + 9 * scale + length, line_top), fill=(243, 235, 205), width=max(2, int(2 * scale)))
        pen.text(
            (left + 14 * scale + length, line_top - 13 * scale), f"{span_mm:g} mm", font=small, fill=(243, 235, 205),
        )
    return encode_png(np.asarray(base))


def plan_csv(db: Database, inspection_id: str) -> bytes:
    current = db.inspection(inspection_id)
    payload = db.station_plan(inspection_id)
    if payload is None:
        raise Missing("Station plan")
    described = describe(payload, current)
    rows = []
    for station in sorted(
        described["plan"]["stations"],
        key=lambda item: (item.get("walk_order") is None, item.get("walk_order") or 0, item["id"]),
    ):
        rows.append([
            station.get("walk_order", ""),
            station["id"],
            station["face"],
            station["status"],
            "yes" if station["quota"] else "extra",
            round(station["width_mm"], 3),
            round(station["height_mm"], 3),
            round(station["center_mm"][0], 2),
            round(station["center_mm"][1], 2),
            round(station["top_left_mm"][0], 2),
            round(station["top_left_mm"][1], 2),
            None if station["source"] is None else round(station["source"][0], 1),
            None if station["source"] is None else round(station["source"][1], 1),
            ", ".join(station["substitutes_shortfall"]) or "",
            station["replaces"] or "",
            station["void_reason"] or station["note"] or "",
        ])
    return render_csv(
        [
            "walk_order", "station", "face", "status", "quota", "width_mm", "height_mm",
            "center_x_mm", "center_y_mm", "top_left_x_mm", "top_left_y_mm",
            "source_x_px", "source_y_px", "does_not_represent", "replaces", "note",
        ],
        rows,
    )
