"""Compare crack traces seen on the two end faces of one piece of stock.

The two photographs may be turned or mirrored relative to each other, so the
review is built on two marks the operator makes on each face: the same physical
corner and one point further along the same machined edge. Every trace is then
expressed as a fraction of that marked edge, giving both faces a shared,
scale-free frame.

A review reports whether the evidence at the two photographed faces is
compatible. It does not establish that a check runs continuously through the
stock, and an unphotographed position is reported as unseen rather than as an
absence of cracks.
"""

from dataclasses import dataclass
from html import escape
from io import BytesIO
import json
from math import acos, cos, degrees, hypot, isfinite, radians, sin
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from .captures import decoded
from .database import Database, encode, identifier, timestamp
from .errors import Missing, WorkbenchError
from .geometry import plane_for, rasterize
from .imaging import encode_png
from .lighting import field_for
from .reports import render_csv
from .validation import document, fields, list_of, string

KIND = "cracks"
RELATIONSHIPS = ("mirror", "same")

# Gates are fractions of the marked common edge, so they do not depend on the
# camera scale of either face.
ACROSS_GATE = 0.10
ALONG_GATE = 0.18
ANGLE_GATE = 55.0
EXACT_ACROSS = 0.05
EXACT_ANGLE = 22.0
EXACT_LENGTH = (0.5, 2.0)
CHAIN_ANGLE = 28.0
CHAIN_GAP = 0.25

COLORS = {
    "exact": (112, 186, 121),
    "segments": (226, 189, 103),
    "fragment": (222, 147, 86),
    "ambiguous": (135, 160, 203),
    "truncated": (206, 166, 106),
    "unmatched": (199, 111, 148),
    "paired": (150, 168, 155),
}
GROUP_OF = {
    "exact": "single",
    "segments": "multiple",
    "fragment": "multiple",
    "ambiguous": "multiple",
    "truncated": "multiple",
    "unmatched": "none",
    "paired": "paired",
}
GROUP_TITLES = {
    "single": "Single pair to re-check",
    "multiple": "More than one defensible relation",
    "none": "No evidence at this position",
    "paired": "Second-face trace already described",
}
TYPE_TITLES = {
    "exact": "Compatible single pair",
    "segments": "Several short traces could combine",
    "fragment": "One trace is much shorter",
    "ambiguous": "Several candidates are compatible",
    "unmatched": "No compatible trace recorded",
    "truncated": "One trace may be cut by the photograph",
    "paired": "Compatible with the first face",
}
COVERAGE_NOTES = {
    "observed": "The second face covers this position inside the selected reading region.",
    "no_trace": "The second face covers this position and its saved crack reading records no trace there.",
    "outside_region": "This position falls outside the second reading's selected measuring region, so that face was not read here.",
    "unsupported": "This position falls on second-face pixels the photograph does not cover, so there is no evidence either way.",
    "partial": "Part of this position touches unsupported pixels or the edge of the selected region; treat a blank result here as unconfirmed.",
}


def rounded(value: float | None) -> float | None:
    return None if value is None else round(float(value), 4)


@dataclass(frozen=True)
class Face:
    key: str
    inspection: dict
    run: dict
    anchors: dict
    plane: dict
    corner: np.ndarray
    along: np.ndarray
    across: np.ndarray
    span: float

    @property
    def scale(self) -> tuple[float, float]:
        return (self.plane["mm_per_x"] or 1.0, self.plane["mm_per_y"] or 1.0)

    def physical(self, point) -> np.ndarray:
        sx, sy = self.scale
        return np.asarray([float(point[0]) * sx, float(point[1]) * sy], dtype=np.float64)

    def relative(self, point) -> tuple[float, float]:
        delta = self.physical(point) - self.corner
        return float(delta @ self.along) / self.span, float(delta @ self.across) / self.span

    def pixel(self, t: float, n: float) -> list[float]:
        sx, sy = self.scale
        location = self.corner + self.along * (t * self.span) + self.across * (n * self.span)
        return [float(location[0] / sx), float(location[1] / sy)]

    def direction(self, angle_deg: float) -> np.ndarray:
        angle = radians(angle_deg)
        physical = np.array([cos(angle), sin(angle)], dtype=np.float64)
        return np.array([physical @ self.along, physical @ self.across], dtype=np.float64)


def face_for(key: str, inspection: dict, run: dict, anchors: dict, relationship: str) -> Face:
    plane = run["snapshot"]["plane"]
    sx, sy = plane["mm_per_x"] or 1.0, plane["mm_per_y"] or 1.0
    corner = np.array([anchors["corner"][0] * sx, anchors["corner"][1] * sy], dtype=np.float64)
    edge = np.array([anchors["edge"][0] * sx, anchors["edge"][1] * sy], dtype=np.float64)
    span = float(np.linalg.norm(edge - corner))
    if span < 1e-9:
        raise WorkbenchError(f"Place the {key} corner and edge point apart on the image.", f"{key}_anchors")
    along = (edge - corner) / span
    across = np.array([along[1], -along[0]], dtype=np.float64)
    if relationship == "mirror" and key == "right":
        across = -across
    return Face(key, inspection, run, anchors, plane, corner, along, across, span)


def anchor_pair(raw: Any, key: str, plane: dict) -> dict:
    values = document(raw, f"{key} anchors")
    fields(values, {"corner", "edge"}, f"{key} anchors")
    result = {}
    for name in ("corner", "edge"):
        point = list_of(values.get(name), f"{key} {name} point", 2, 2)
        coordinates = []
        for axis, coordinate in zip(("x", "y"), point):
            if isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)):
                raise WorkbenchError(f"Mark the {key} {name} on the image.", f"{key}_anchors")
            limit = plane["width"] if axis == "x" else plane["height"]
            if not isfinite(float(coordinate)) or not -1 <= float(coordinate) <= limit + 1:
                raise WorkbenchError(f"Mark the {key} {name} inside the measuring image.", f"{key}_anchors")
            coordinates.append(round(float(coordinate), 2))
        result[name] = coordinates
    first, second = np.asarray(result["corner"]), np.asarray(result["edge"])
    if float(np.linalg.norm(first - second)) < 8:
        raise WorkbenchError(f"Place the {key} edge point at least eight pixels from the corner.", f"{key}_anchors")
    return result


def trace_records(face: Face) -> list[dict]:
    """Place one saved crack reading's traces in the face's relative frame."""
    records = []
    for item in face.run["result"].get("objects") or []:
        centre = face.relative([item["x"], item["y"]])
        x0, y0, x1, y1 = item["bbox"]
        corners = [face.relative(point) for point in ((x0, y0), (x1, y0), (x0, y1), (x1, y1))]
        minima = np.min(corners, axis=0)
        maxima = np.max(corners, axis=0)
        tilt = item.get("orientation_deg")
        axis = face.direction(float(tilt)) if tilt is not None else np.array([0.0, 1.0])
        if axis[0] < 0 or (abs(axis[0]) < 1e-9 and axis[1] < 0):
            axis = -axis
        length = float(item.get("length") or 0.0)
        fraction = length / face.span if face.span else 0.0
        reach = 0.5 * fraction
        # Ends are taken along the fitted run direction, not from the box
        # corners, so a mirrored second face still gives one near and one far end.
        # The skeleton length under-reports how far the stroke itself reaches,
        # so the ends use the larger of the run length and the box extent.
        half_along = max(reach, 0.5 * abs(float(maxima[0] - minima[0]) * axis[0]), 0.5 * abs(float(maxima[1] - minima[1]) * axis[1]))
        near = (centre[0] - axis[0] * half_along, centre[1] - axis[1] * half_along)
        far = (centre[0] + axis[0] * half_along, centre[1] + axis[1] * half_along)
        records.append({
            "id": int(item["id"]),
            "length": length,
            "fraction": fraction,
            "reach": reach,
            "width": float(item.get("median_width") or 0.0),
            "edge": bool(item.get("touches_support_edge")),
            "centre": (float(centre[0]), float(centre[1])),
            "axis": axis,
            "ends": (near, far),
            "span": (float(minima[0]), float(maxima[0]), float(minima[1]), float(maxima[1])),
        })
    records.sort(key=lambda entry: entry["id"])
    return records


@dataclass(frozen=True)
class FaceImage:
    rgb: np.ndarray
    missing: np.ndarray
    outside: np.ndarray


def face_image(db: Database, inspection: dict, run: dict) -> FaceImage:
    """Rebuild the measuring raster that a saved reading was computed on."""
    snapshot = run["snapshot"]
    state = snapshot.get("state") or {}
    try:
        pixels, available, capture = decoded(db, inspection["capture_id"])
        metadata = capture["metadata"]
        plane = plane_for(metadata["width"], metadata["height"], snapshot["plane"]["calibration"])
        base = pixels
        if state.get("lighting"):
            correction, _ = field_for(db, pixels, available, state["lighting"])
            base = correction.rgb
        raster = rasterize(base, available, plane, state.get("region"))
        return FaceImage(raster.rgb, ~raster.supported, ~raster.inspected)
    except (WorkbenchError, KeyError, ValueError, TypeError):
        try:
            raw = db.run_attachment(inspection["id"], run["id"], "preview")
            with Image.open(BytesIO(raw)) as image:
                rgb = np.asarray(image.convert("RGB"))
            plane = snapshot["plane"]
            if rgb.shape[:2] != (plane["height"], plane["width"]):
                with Image.open(BytesIO(raw)) as image:
                    rgb = np.asarray(image.convert("RGB").resize((plane["width"], plane["height"])))
            empty = np.zeros(rgb.shape[:2], dtype=bool)
            return FaceImage(rgb.copy(), empty, empty)
        except (WorkbenchError, OSError, KeyError):
            raise WorkbenchError("A saved reading's measuring image could not be rebuilt.", "file") from None


def edge_state(image: FaceImage, face: Face, t: float, n: float) -> str:
    """Classify one measuring-plane position as seen, unphotographed or unread."""
    x, y = face.pixel(t, n)
    height, width = image.missing.shape[:2]
    cx, cy = int(round(x)), int(round(y))
    if not (0 <= cx < width and 0 <= cy < height):
        return "unsupported"
    if image.missing[cy, cx]:
        return "unsupported"
    if image.outside[cy, cx]:
        return "outside_region"
    return "seen"


def cut_ends(image: FaceImage, face: Face, record: dict, beyond: float = 0.012) -> tuple[bool, bool]:
    """True when an end of this trace runs into an unseen boundary.

    A trace that stops where the photograph stops, or where the selected
    region stops, may continue beyond it. Taking such an end for the real end
    of a check would force a false pair, so each end is probed just beyond the
    recorded run as well as on it.
    """
    (t0, n0), (t1, n1) = record["ends"]
    axis = record["axis"]
    probes = []
    for t, n, sign in ((t0, n0, -1.0), (t1, n1, 1.0)):
        states = [
            edge_state(image, face, t + axis[0] * sign * offset, n + axis[1] * sign * offset)
            for offset in (0.0, beyond)
        ]
        probes.append(any(state != "seen" for state in states))
    return bool(probes[0]), bool(probes[1])


def angle_between(first: np.ndarray, second: np.ndarray) -> float:
    return degrees(float(acos(min(1.0, abs(float(first @ second))))))


def lateral_of(delta: np.ndarray, axis: np.ndarray) -> float:
    return abs(float(delta[0] * -axis[1] + delta[1] * axis[0]))


def longitudinal_of(delta: np.ndarray, axis: np.ndarray) -> float:
    return float(delta @ axis)


def directed_matches(primary: list[dict], other: list[dict]) -> dict[int, list[dict]]:
    """Candidate traces on the other face, judged in each trace's own run frame.

    A short piece of one check may sit well along the parent's run, so the gate
    on the along-run offset grows with the two recorded lengths. The lateral
    offset, measured across the run, stays tight.
    """
    result = {}
    for left in primary:
        candidates = []
        for right in other:
            delta = np.array(
                [right["centre"][0] - left["centre"][0], right["centre"][1] - left["centre"][1]],
                dtype=np.float64,
            )
            axis = left["axis"]
            lateral = lateral_of(delta, axis)
            longitudinal = abs(longitudinal_of(delta, axis))
            angle = angle_between(left["axis"], right["axis"])
            reach = left["reach"] + right["reach"] + ALONG_GATE
            if lateral > ACROSS_GATE or angle > ANGLE_GATE or longitudinal > reach:
                continue
            signed = longitudinal_of(delta, axis)
            low, high = signed - right["reach"], signed + right["reach"]
            overlap = min(high, left["reach"]) - max(low, -left["reach"])
            gap = max(0.0, max(low, -left["reach"]) - min(high, left["reach"]))
            candidates.append({
                "trace_id": right["id"],
                "position": [rounded(right["centre"][0]), rounded(right["centre"][1])],
                "length": right["length"],
                "length_fraction": rounded(right["fraction"]),
                "distance": rounded(hypot(lateral, longitudinal)),
                "along_run": rounded(longitudinal),
                "across_run": rounded(lateral),
                "angle": rounded(angle),
                "overlap": rounded(overlap),
                "gap": rounded(gap),
                "score": rounded(lateral / ACROSS_GATE + max(0.0, -overlap) / max(reach, 1e-9) + angle / ANGLE_GATE),
                "_record": right,
            })
        candidates.sort(key=lambda entry: (entry["score"], entry["trace_id"]))
        result[left["id"]] = candidates
    return result


def chained(segments: list[dict], parent: dict) -> bool:
    """True when the candidates line up end to end as pieces of one run."""
    if len(segments) < 2:
        return False
    axis = parent["axis"]
    ordered = []
    for entry in segments:
        record = entry["_record"]
        delta = np.array([record["centre"][0] - parent["centre"][0], record["centre"][1] - parent["centre"][1]])
        centre = longitudinal_of(delta, axis)
        ordered.append((centre - record["reach"], centre + record["reach"], record, entry["across_run"]))
    ordered.sort(key=lambda item: item[0])
    for first, second in zip(ordered, ordered[1:]):
        separation = second[0] - first[1]
        if separation > CHAIN_GAP or separation < -0.02:
            return False
        if abs(first[3] - second[3]) > EXACT_ACROSS * 2:
            return False
        if angle_between(first[2]["axis"], second[2]["axis"]) > CHAIN_ANGLE:
            return False
    return True


def settled(entry: dict) -> bool:
    """Same line on the face: close across the run and pointing the same way."""
    return entry["across_run"] <= EXACT_ACROSS and entry["angle"] <= EXACT_ANGLE


def classify(matches: list[dict], record: dict) -> str:
    if not matches:
        return "unmatched"
    if len(matches) > 1:
        return "segments" if chained(matches, record) else "ambiguous"
    only = matches[0]
    if not settled(only):
        return "ambiguous"
    if only["overlap"] <= 0 and only["gap"] > CHAIN_GAP:
        return "ambiguous"
    other = only["_record"]["fraction"]
    if record["fraction"] <= 0 or other <= 0:
        return "ambiguous"
    ratio = max(record["fraction"], other) / min(record["fraction"], other)
    return "exact" if EXACT_LENGTH[0] <= ratio <= EXACT_LENGTH[1] else "fragment"


def edge_note(record: dict) -> str:
    if not record.get("edge"):
        return ""
    return " This trace also touches the edge of the photographed support, so it may continue beyond it."


def explanation(kind: str, coverage: str, record: dict | None = None) -> str:
    if kind == "unmatched":
        return COVERAGE_NOTES.get(coverage, "There is no compatible evidence at this position.") + (
            edge_note(record) if record else ""
        )
    if kind == "exact":
        return "One trace on the second face sits at the same position, runs the same way and is of similar length." + (
            edge_note(record) if record else ""
        )
    if kind == "segments":
        return "Two or more short second-face traces line up end to end here; they could be pieces of one check, but the gaps are not evidence of continuity." + (
            edge_note(record) if record else ""
        )
    if kind == "fragment":
        return "A single compatible trace exists but its recorded length differs sharply. Read it as a possible fragment, not a confirmed match." + (
            edge_note(record) if record else ""
        )
    if kind == "truncated":
        return "A compatible trace ends where the photograph or the selected region ends, so its recorded length may be cut short. Treat the pair as possible, not established."
    return "Compatible traces exist but not tightly enough, or more than one unrelated trace is nearby. The relation is possible, not singular."


def coverage_state(image: FaceImage, face: Face, point) -> tuple[str, list[float]]:
    x, y = face.pixel(point[0], point[1])
    height, width = image.missing.shape[:2]
    if not (-0.5 <= x <= width - 0.5 and -0.5 <= y <= height - 0.5):
        return "unsupported", [x, y]
    cx, cy = int(round(x)), int(round(y))
    support = inspected = total = 0
    for offset_y in range(-4, 5, 2):
        for offset_x in range(-4, 5, 2):
            px, py = cx + offset_x, cy + offset_y
            if 0 <= px < width and 0 <= py < height:
                total += 1
                support += int(not image.missing[py, px])
                inspected += int(not image.outside[py, px])
    if not total:
        return "unsupported", [x, y]
    if support / total <= 0.15:
        return "unsupported", [x, y]
    if inspected / total <= 0.15:
        return "outside_region", [x, y]
    if support < total or inspected < total:
        return "partial", [x, y]
    return "observed", [x, y]


def trace_row(origin: str, record: dict, kind: str, state: str, location, matches: list[dict]) -> dict:
    target = matches[0]["position"] if matches else record["centre"]
    row = {
        "trace_id": record["id"],
        "origin": origin,
        "type": kind,
        "group": GROUP_OF[kind],
        "position": [rounded(record["centre"][0]), rounded(record["centre"][1])],
        "length": record["length"],
        "length_fraction": rounded(record["fraction"]),
        "width": record["width"],
        "touches_support_edge": record["edge"],
        "match_position": [rounded(target[0]), rounded(target[1])],
        "second_face_seen": state != "unsupported",
        "coverage": {
            "status": state,
            "point": [rounded(location[0]), rounded(location[1])] if location else [],
        },
        "candidates": [
            {name: value for name, value in match.items() if name != "_record"} for match in matches
        ],
        "explanation": explanation(kind, state, record),
    }
    if kind == "segments":
        row["combined_length"] = rounded(sum(match["_record"]["fraction"] for match in matches))
    return row


def result_for(rows: list[dict], relationship: str, left: list[dict], right: list[dict]) -> dict:
    order = {"exact": 0, "fragment": 1, "segments": 2, "truncated": 3, "ambiguous": 4, "unmatched": 5, "paired": 6}
    rows.sort(key=lambda entry: (order[entry["type"]], entry["position"][0], entry["position"][1]))
    counts = {
        name: sum(1 for row in rows if row["type"] == name and row["origin"] == "left")
        for name in TYPE_TITLES if name != "paired"
    }
    counts["relations"] = sum(1 for row in rows if row["origin"] == "left")
    groups = [
        {
            "key": name,
            "title": GROUP_TITLES[name],
            "rows": [
                {"origin": row["origin"], "trace_id": row["trace_id"]}
                for row in rows if row["group"] == name
            ],
        }
        for name in GROUP_TITLES
    ]
    notices = [
        "Positions are fractions of the marked common edge. Matching uses position, direction and recorded length only.",
        "A compatible pair shows that both faces carry similar evidence. It does not show that a check runs continuously through the stock.",
        "The second face is treated as the reverse side of the first: positions are mirrored across the marked edge."
        if relationship == "mirror"
        else "The second face is treated as seen from the same viewpoint; no mirroring is applied.",
    ]
    if not left:
        notices.append("The first saved reading records no traces.")
    if not right:
        notices.append("The second saved reading records no traces; unphotographed positions cannot be matched.")
    return {
        "relationship": relationship,
        "counts": counts,
        "groups": groups,
        "rows": rows,
        "relations": counts["relations"],
        "notices": notices,
    }


def unseen(state: str) -> str:
    """A covered position with no compatible trace is 'no trace', never a match."""
    return "no_trace" if state == "observed" else state


def analyze(
    left: Face, right: Face, relationship: str, left_image: FaceImage, right_image: FaceImage,
) -> dict:
    left_records = trace_records(left)
    right_records = trace_records(right)
    forward = directed_matches(left_records, right_records)
    rows = []
    left_cuts = {record["id"]: cut_ends(left_image, left, record) for record in left_records}
    right_cuts = {record["id"]: cut_ends(right_image, right, record) for record in right_records}
    for record in left_records:
        matches = forward[record["id"]]
        kind = classify(matches, record)
        mine = any(bool(value) for value in left_cuts[record["id"]])
        theirs = any(any(bool(value) for value in right_cuts[match["trace_id"]]) for match in matches)
        if matches and bool(mine or theirs):
            kind = "truncated"
        target = matches[0]["position"] if matches else record["centre"]
        state, location = coverage_state(right_image, right, target)
        rows.append(trace_row("left", record, kind, unseen(state) if not matches else state, location, matches))
    paired = {}
    for matches in forward.values():
        for match in matches:
            paired.setdefault(match["trace_id"], []).append(match)
    partner = {
        match["trace_id"]: record["id"]
        for record in left_records for match in forward[record["id"]]
    }
    for record in right_records:
        if record["id"] in paired:
            row = trace_row("right", record, "paired", "observed", None, paired[record["id"]])
            row["explanation"] = "Compatible with first-face trace " + str(partner.get(record["id"])) + "; that row describes the relation."
            row["paired_with"] = partner.get(record["id"])
            rows.append(row)
            continue
        state, location = coverage_state(left_image, left, record["centre"])
        rows.append(trace_row("right", record, "unmatched", unseen(state), location, []))
    return result_for(rows, relationship, left_records, right_records)


PLATE_SIZE = (666, 520)
PLATE_TOP = 52


@dataclass(frozen=True)
class Plate:
    canvas: Image.Image
    offset: tuple[float, float]
    scale: float

    def position(self, point) -> tuple[float, float]:
        return (
            self.offset[0] + (float(point[0]) + 0.5) * self.scale,
            self.offset[1] + (float(point[1]) + 0.5) * self.scale,
        )


def toned(image: FaceImage) -> Image.Image:
    rgb = image.rgb.astype(np.float32)
    outside = image.outside & ~image.missing
    if image.missing.any():
        rgb[image.missing] = rgb[image.missing] * 0.20 + np.array([78, 46, 42]) * 0.80
    if outside.any():
        rgb[outside] = rgb[outside] * 0.55 + np.array([52, 56, 50]) * 0.45
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))


def tag(pen: ImageDraw.ImageDraw, position, text: str, tone) -> None:
    x, y = position
    width = pen.textlength(text)
    pen.rectangle((x - 3, y - 2, x + width + 5, y + 15), fill=(36, 42, 36))
    pen.text((x, y), text, fill=tone)


def make_plate(image: FaceImage, face: Face, records: list[dict], rows: list[dict], header: str) -> Plate:
    base = toned(image)
    fitted = ImageOps.contain(base, (PLATE_SIZE[0] - 16, PLATE_SIZE[1] - PLATE_TOP - 12), Image.Resampling.LANCZOS)
    scale = fitted.width / base.width
    offset = (
        (PLATE_SIZE[0] - fitted.width) / 2,
        PLATE_TOP + (PLATE_SIZE[1] - PLATE_TOP - fitted.height) / 2,
    )
    canvas = Image.new("RGB", PLATE_SIZE, (36, 41, 37))
    canvas.paste(fitted, (int(round(offset[0])), int(round(offset[1]))))
    display = Plate(canvas, offset, scale)
    pen = ImageDraw.Draw(canvas)
    lookup = {record["id"]: record for record in records}
    for row in rows:
        record = lookup.get(row["trace_id"])
        if record is None:
            continue
        tone = COLORS[row["type"]]
        label = ("A" if row["origin"] == "left" else "B") + str(row["trace_id"])
        if row["origin"] == "left":
            x0, x1, y0, y1 = record["span"]
            first = display.position(face.pixel(x0, y0))
            second = display.position(face.pixel(x1, y1))
            box = (
                min(first[0], second[0]), min(first[1], second[1]),
                max(first[0], second[0]), max(first[1], second[1]),
            )
            # The trace itself is only a few pixels wide; an outline a little
            # larger than the box keeps the mark visible without hiding it.
            pad = 7
            box = (box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad)
            pen.rectangle(box, outline=(26, 30, 26), width=5)
            pen.rectangle(box, outline=tone, width=2)
            spot = (box[2] + 6, box[1] - 2)
        else:
            centre = display.position(face.pixel(record["centre"][0], record["centre"][1]))
            pen.ellipse((centre[0] - 7, centre[1] - 7, centre[0] + 7, centre[1] + 7), outline=tone, width=2)
            spot = (centre[0] + 10, centre[1] - 7)
        tag(pen, spot, label, tone)
    corner = display.position(face.anchors["corner"])
    along = display.position(face.anchors["edge"])
    guide = (247, 216, 137)
    pen.line([corner, along], fill=guide, width=2)
    for point, name in ((corner, "C"), (along, "E")):
        pen.ellipse((point[0] - 4, point[1] - 4, point[0] + 4, point[1] + 4), outline=guide, width=2)
        tag(pen, (point[0] + 6, point[1] + 3), name, guide)
    tag(pen, (10, 9), header, (240, 233, 205))
    if image.missing.any():
        tag(pen, (10, 29), "Red-brown: not photographed   /   grey: outside the selected region", (206, 197, 170))
    return display


def sheet(left: Plate, right: Plate, record: dict) -> bytes:
    header = 44
    canvas = Image.new("RGB", (PLATE_SIZE[0] * 2 + 24, PLATE_SIZE[1] + header), (33, 38, 34))
    canvas.paste(left.canvas, (8, header))
    canvas.paste(right.canvas, (PLATE_SIZE[0] + 16, header))
    pen = ImageDraw.Draw(canvas)
    relation = "second face mirrored" if record["relationship"] == "mirror" else "second face not mirrored"
    pen.text((12, 10), "Endgrain Lab · two-end crack correspondence", fill=(244, 236, 200))
    pen.text(
        (12, 26),
        "positions are fractions of the marked common edge · " + relation + " · compatibility at the two faces only",
        fill=(176, 186, 162),
    )
    return encode_png(np.asarray(canvas))


@dataclass(frozen=True)
class Review:
    record: dict
    analysis: dict
    left: Face
    right: Face

    @property
    def inspection(self) -> dict:
        return self.record["left_inspection"]


def face_from_snapshot(key: str, record: dict) -> Face:
    return face_for(
        key,
        record[key + "_inspection"],
        record[key + "_run"],
        record["anchors"][key],
        record["relationship"],
    )


def review_for(record: dict) -> Review:
    left = face_from_snapshot("left", record)
    right = face_from_snapshot("right", record)
    return Review(record, record["analysis"], left, right)


def review_plates(db: Database, record: dict) -> tuple[Plate, Plate]:
    value = review_for(record)
    left_image = face_image(db, value.left.inspection, value.left.run)
    right_image = face_image(db, value.right.inspection, value.right.run)
    rows = value.analysis["rows"]
    left = make_plate(
        left_image, value.left, trace_records(value.left),
        [row for row in rows if row["origin"] == "left"],
        "A · " + record["left"]["inspection_title"],
    )
    right = make_plate(
        right_image, value.right, trace_records(value.right),
        [row for row in rows if row["origin"] == "right"],
        "B · " + record["right"]["inspection_title"],
    )
    return left, right


def reading_links(side: dict) -> dict:
    root = "/api/inspections/" + side["inspection_id"] + "/runs/" + side["run_id"]
    return {"preview": root + "/preview.png", "table": root + "/table.csv", "data": root}


def describe_snapshot(record: dict) -> dict:
    inspection_id = record["left"]["inspection_id"]
    root = "/api/inspections/" + inspection_id + "/pairings/" + record["id"]
    return {
        "id": record["id"],
        "inspection_id": inspection_id,
        "title": record["title"],
        "reviewer": record["reviewer"],
        "note": record["note"],
        "revision": record["revision"],
        "left": {**record["left"], "links": reading_links(record["left"])},
        "right": {**record["right"], "links": reading_links(record["right"])},
        "anchors": record["anchors"],
        "relationship": record["relationship"],
        "analysis": record["analysis"],
        "created_at": record["created_at"],
        "links": {
            "sheet": root + "/sheet.png",
            "csv": root + "/review.csv",
            "print": root + "/print",
            "data": root,
        },
    }


def describe(db: Database, inspection_id: str, record: dict) -> dict:
    return describe_snapshot(record)


def validate(db: Database, inspection_id: str, raw: Any) -> dict:
    body = document(raw, "two-end review")
    fields(body, {
        "revision", "title", "reviewer", "note", "relationship",
        "left_run_id", "right_run_id", "right_inspection_id", "anchors",
    }, "two-end review")
    left = db.inspection(inspection_id)
    revision = body.get("revision")
    if isinstance(revision, bool) or not isinstance(revision, int) or revision != left["revision"]:
        raise WorkbenchError(
            "The first face changed. Reload it before saving a two-end review.", "revision"
        )
    left_run = db.run(inspection_id, string(body.get("left_run_id"), "first reading", 40, required=True))
    if left_run["kind"] != KIND:
        raise WorkbenchError("Choose a saved Crack traces reading for the first face.", "left_run_id")
    right_id = string(body.get("right_inspection_id"), "second face", 40, required=True)
    right_inspection = db.inspection(right_id)
    right_run = db.run(right_id, string(body.get("right_run_id"), "second reading", 40, required=True))
    if right_run["kind"] != KIND:
        raise WorkbenchError("Choose a saved Crack traces reading for the second face.", "right_run_id")
    relationship = body.get("relationship", "mirror")
    if relationship not in RELATIONSHIPS:
        raise WorkbenchError("Say whether the second face is mirrored.", "relationship")
    anchors = document(body.get("anchors"), "anchors")
    fields(anchors, {"left", "right"}, "anchors")
    left_anchors = anchor_pair(anchors.get("left"), "left", left_run["snapshot"]["plane"])
    right_anchors = anchor_pair(anchors.get("right"), "right", right_run["snapshot"]["plane"])
    return {
        "title": string(body.get("title"), "review title", 120, required=True),
        "reviewer": string(body.get("reviewer"), "reviewer", 80),
        "note": string(body.get("note"), "review note", 2000, multiline=True),
        "revision": left["revision"],
        "relationship": relationship,
        "anchors": {"left": left_anchors, "right": right_anchors},
        "left": side_binding(db, inspection_id, left_run),
        "right": side_binding(db, right_id, right_run),
        "right_inspection": right_inspection,
    }


def side_binding(db: Database, inspection_id: str, run: dict) -> dict:
    inspection = db.inspection(inspection_id)
    return {
        "inspection_id": inspection_id,
        "inspection_title": inspection["title"],
        "run_id": run["id"],
        "run_revision": run["revision"],
        "run_created_at": run["created_at"],
        "plane": run["snapshot"]["plane"],
        "trace_count": len(run["result"].get("objects") or []),
        "summary": run["result"].get("metrics") or {},
    }


def snapshot(db: Database, inspection_id: str, values: dict) -> dict:
    """Freeze the two readings and their anchors; later edits cannot change it."""
    left_inspection = db.inspection(inspection_id)
    return {
        "id": identifier(),
        "title": values["title"],
        "reviewer": values["reviewer"],
        "note": values["note"],
        "revision": left_inspection["revision"],
        "relationship": values["relationship"],
        "anchors": values["anchors"],
        "left": values["left"],
        "right": values["right"],
        "left_inspection": stored_inspection(left_inspection),
        "right_inspection": stored_inspection(values["right_inspection"]),
        "left_run": frozen_run(db, inspection_id, values["left"]["run_id"]),
        "right_run": frozen_run(db, values["right"]["inspection_id"], values["right"]["run_id"]),
        "created_at": timestamp(),
    }


def stored_inspection(inspection: dict) -> dict:
    metadata = {name: inspection[name] for name in ("id", "title", "material", "lot", "surface", "capture_id")}
    metadata["capture_metadata"] = {
        name: inspection["capture_metadata"][name] for name in ("width", "height", "synthetic")
    }
    return metadata


def frozen_run(db: Database, inspection_id: str, run_id: str) -> dict:
    run = db.run(inspection_id, run_id)
    return {
        "id": run["id"],
        "revision": run["revision"],
        "kind": run["kind"],
        "created_at": run["created_at"],
        "params": run["params"],
        "snapshot": run["snapshot"],
        "result": run["result"],
    }


def save(db: Database, inspection_id: str, raw: Any) -> dict:
    values = validate(db, inspection_id, raw)
    stored = snapshot(db, inspection_id, values)
    left = face_from_snapshot("left", stored)
    right = face_from_snapshot("right", stored)
    stored["analysis"] = analyze(
        left, right, stored["relationship"],
        face_image(db, left.inspection, left.run),
        face_image(db, right.inspection, right.run),
    )
    with db.transaction() as conn:
        conn.execute(
            """INSERT INTO pairings (id,inspection_id,title,reviewer,revision,review,created_at)
               VALUES (?,?,?,?,?,?,?)""",
            (
                stored["id"], inspection_id, stored["title"], stored["reviewer"],
                stored["revision"], encode(stored), stored["created_at"],
            ),
        )
        db.event(conn, inspection_id, stored["revision"], "two_end_review_saved", {
            "review_id": stored["id"],
            "left_run": stored["left"]["run_id"],
            "right_run": stored["right"]["run_id"],
        })
    return describe_snapshot(stored)


def read(db: Database, inspection_id: str, review_id: str) -> dict:
    with db.connection() as conn:
        row = conn.execute(
            "SELECT review FROM pairings WHERE inspection_id = ? AND id = ?",
            (inspection_id, review_id),
        ).fetchone()
    if row is None:
        raise Missing("Two-end review")
    return json.loads(row["review"])


def listing(db: Database, inspection_id: str) -> list[dict]:
    with db.connection() as conn:
        rows = conn.execute(
            """SELECT id,title,reviewer,revision,created_at FROM pairings
               WHERE inspection_id = ? ORDER BY created_at DESC, id LIMIT 60""",
            (inspection_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def export_csv(db: Database, inspection_id: str, record: dict) -> bytes:
    analysis = review_for(record).analysis
    columns = [
        "group", "relation", "origin", "trace", "along_edge_fraction", "across_edge_fraction",
        "length", "width", "candidate_trace", "candidate_along_edge", "candidate_across_edge",
        "candidate_angle_deg", "candidate_length", "second_face_coverage", "reading",
    ]
    rows = []
    for row in analysis["rows"]:
        base = [
            GROUP_TITLES[row["group"]], TYPE_TITLES[row["type"]], row["origin"], row["trace_id"],
            row["position"][0], row["position"][1], row["length"], row["width"],
        ]
        if not row["candidates"]:
            rows.append(base + [None, None, None, None, None, row["coverage"]["status"], row["explanation"]])
            continue
        for match in row["candidates"]:
            rows.append(base + [
                match["trace_id"], match["position"][0], match["position"][1],
                match["angle"], match["length"], row["coverage"]["status"], row["explanation"],
            ])
    return render_csv(columns, rows)


PRINT_STYLES = """
:root { color:#242820; background:#eae7dc; font:15px/1.55 system-ui,sans-serif; }
body { max-width:900px; margin:30px auto; padding:42px; background:#fdfcf5; }
h1,h2 { font-family:Georgia,serif; font-weight:500; }
h1 { font-size:2.4rem; margin:.2em 0 .3em; }
h2 { font-size:1.5rem; margin-top:1.4em; }
.brand { text-transform:uppercase; letter-spacing:.13em; font-weight:700; color:#59604b; font-size:12px; }
table { width:100%; border-collapse:collapse; font-size:13px; }
th,td { text-align:left; padding:.4em .5em; border-bottom:1px solid #cccaaf; vertical-align:top; }
thead tr { background:#ecebdd; }
figure { margin:14px 0; }
figure img { max-width:100%; }
.notice { color:#5b6350; font-size:13px; }
"""


def render_print(db: Database, inspection_id: str, record: dict) -> str:
    analysis = review_for(record).analysis
    summary = "".join(
        "<tr><th>" + escape(group["title"]) + "</th><td>"
        + str(sum(1 for row in analysis["rows"] if row["group"] == group["key"] and row["origin"] == "left"))
        + "</td></tr>"
        for group in analysis["groups"] if group["key"] != "paired"
    )
    detail = []
    for row in analysis["rows"]:
        targets = ", ".join(
            ("B" if row["origin"] == "left" else "A") + str(match["trace_id"]) for match in row["candidates"]
        ) or "None recorded"
        detail.append(
            "<tr><td>" + escape(GROUP_TITLES[row["group"]]) + "</td><td>" + escape(TYPE_TITLES[row["type"]])
            + "</td><td>" + escape(("A" if row["origin"] == "left" else "B") + str(row["trace_id"]))
            + "</td><td>" + escape(targets) + "</td><td>" + escape(row["explanation"]) + "</td></tr>"
        )
    notices = "".join("<li>" + escape(text) + "</li>" for text in analysis["notices"])
    relation = "Second face mirrored" if record["relationship"] == "mirror" else "Second face not mirrored"
    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        "<title>" + escape(record["title"]) + "</title><style>" + PRINT_STYLES + "</style></head><body>"
        "<div class=\"brand\">Endgrain Lab · Two-end correspondence check</div>"
        "<h1>" + escape(record["title"]) + "</h1>"
        "<p>" + escape(record["reviewer"] or "Unassigned reviewer") + " &nbsp;·&nbsp; "
        + escape(record["created_at"][:19].replace("T", " ")) + "</p>"
        "<p class=\"notice\">" + escape(relation)
        + ". Positions are fractions of the marked common edge. This review reports compatible evidence at the two"
        " photographed faces only; it does not establish a continuous check through the stock.</p>"
        "<figure><img src=\"sheet.png\" alt=\"The two measuring frames with marked traces\"></figure>"
        "<h2>Grouping</h2><table><tbody>" + summary + "</tbody></table>"
        "<h2>Rows</h2><table><thead><tr><th>Group</th><th>Relation</th><th>Trace</th>"
        "<th>Compatible with</th><th>Reading</th></tr></thead><tbody>" + "".join(detail) + "</tbody></table>"
        "<h2>Notices</h2><ul>" + notices + "</ul>"
        "<p class=\"notice\">First reading: " + escape(record["left"]["inspection_title"]) + " / "
        + escape(record["left"]["run_id"]) + "<br>Second reading: "
        + escape(record["right"]["inspection_title"]) + " / " + escape(record["right"]["run_id"]) + "</p>"
        "</body></html>"
    )
