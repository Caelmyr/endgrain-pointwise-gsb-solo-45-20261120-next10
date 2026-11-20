from dataclasses import dataclass
from math import ceil
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from .errors import WorkbenchError
from .footprint import isolated, solid_labels
from .geometry import Plane, Raster, ordered_quad, pixel_point, region_mask
from .imaging import safe_ratio, symmetric_overlay
from .validation import document, fields, integer, list_of, number


@dataclass(frozen=True)
class EdgeSample:
    name: str
    station: float
    position: list[float]
    boundary: list[float] | None
    depth: float | None
    status: str


@dataclass(frozen=True)
class LossReading:
    stock: np.ndarray
    expected: np.ndarray
    missing: np.ndarray
    unknown: np.ndarray
    samples: list[EdgeSample]
    corners: np.ndarray
    horizon: float


SIDES = ("top", "right", "bottom", "left")
FLAGS = {"intact", "receded", "over_limit", "unknown", "no_edge_detected"}


def normalize_params(raw: Any, plane: Plane) -> dict:
    body = document(raw, "edge loss settings")
    legal = {"corners", "depth", "station_spacing", "threshold", "minimum_luma", "warmth", "bridge"}
    fields(body, legal, "edge loss")
    w, h = plane.width, plane.height
    defaults = [[w * x, h * y] for x, y in ((0.115, 0.17), (0.91, 0.15), (0.925, 0.835), (0.14, 0.83))]
    polygon = [
        pixel_point(value, f"expected corner {index}", w, h)
        for index, value in enumerate(list_of(body.get("corners", defaults), "expected corners", 4, 4), 1)
    ]
    if not ordered_quad(polygon):
        raise WorkbenchError("Place the expected stock corners clockwise from upper left.", "corners")
    limit = number(body.get("depth", 28), "largest probe depth in pixels", 4, 110)
    return {
        "corners": polygon,
        "depth": limit,
        "station_spacing": integer(body.get("station_spacing", 10), "station spacing in pixels", 3, 100),
        "threshold": number(body.get("threshold", 4), "flagged recession in pixels", 0.4, limit),
        "minimum_luma": number(body.get("minimum_luma", 0.24), "minimum brightness", 0.03, 0.94),
        "warmth": number(body.get("warmth", 0.10), "minimum red minus blue", -0.3, 0.65),
        "bridge": number(body.get("bridge", 2), "edge smoothing radius", 0, 12),
    }


def signed_side(edge_name: str, start: np.ndarray, end: np.ndarray, span: int) -> tuple[float, np.ndarray, np.ndarray]:
    d = end - start
    length = float(np.linalg.norm(d))
    normal = np.array([-d[1], d[0]]) / length
    samples = max(1, ceil(length / span))
    fractions = np.linspace(0, 1, samples + 2)[1:-1]
    return length, start + fractions[:, None] * d, normal


def mask_probe(values: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    coords = np.stack([coordinates[..., 1], coordinates[..., 0]])
    return ndi.map_coordinates(
        values.astype(np.float32), coords, order=1, mode="constant", cval=0, prefilter=False,
    )


def probes(
    raster: Raster, wood: np.ndarray, start: np.ndarray, end: np.ndarray,
    name: str, settings: dict,
) -> list[EdgeSample]:
    length, stations, normal = signed_side(name, start, end, settings["station_spacing"])
    steps = max(2, ceil(settings["depth"] * 2))
    depths = np.linspace(0, settings["depth"], steps + 1)
    expanded = stations[:, None, :] + depths[None, :, None] * normal
    support_fraction = mask_probe(raster.support, expanded)
    present_fraction = mask_probe(wood, expanded)
    good = support_fraction > 0.99
    material = (present_fraction >= 0.55) & good
    plane = raster.plane
    ux, uy = normal
    scale = np.hypot(ux * (plane.mm_x or 1), uy * (plane.mm_y or 1))
    side_scale = plane.distance(start, end) / max(1e-9, length)
    fixed = []
    for index, position in enumerate(stations):
        valid_material = np.flatnonzero(material[index])
        unavailable = np.flatnonzero(~good[index])
        first_support_gap = int(unavailable[0]) if unavailable.size else len(depths)
        station = side_scale * float(np.linalg.norm(position - start))
        bottom = None
        depth = None
        if valid_material.size and valid_material[0] < first_support_gap:
            x = int(valid_material[0])
            hit = float(depths[x])
            if x > 0 and present_fraction[index, x] > present_fraction[index, x - 1]:
                middle = (0.55 - present_fraction[index, x - 1]) / (present_fraction[index, x] - present_fraction[index, x - 1])
                hit = float(depths[x - 1] + middle * (depths[x] - depths[x - 1]))
            depth = hit * scale
            bottom = (position + hit * normal).tolist()
            if hit >= settings["threshold"]:
                status = "over_limit"
            elif hit >= 1:
                status = "receded"
            else:
                status = "intact"
        else:
            status = "unknown" if first_support_gap < len(depths) else "no_edge_detected"
        fixed.append(EdgeSample(name, station, position.tolist(), bottom, depth, status))
    return fixed


def measure_loss(raster: Raster, settings: dict) -> LossReading:
    corners = np.asarray(settings["corners"])
    expected = region_mask(raster.plane, settings["corners"])
    source_labels = solid_labels(raster, settings)
    wood = isolated(source_labels, None)
    missing = expected & raster.support & ~wood
    unknown = expected & ~raster.support
    reports = []
    for index, label in enumerate(SIDES):
        begin = corners[index]
        finish = corners[(index + 1) % len(corners)]
        reports.extend(probes(raster, wood, begin, finish, label, settings))
    return LossReading(wood, expected, missing, unknown, reports, corners, settings["depth"])


def overlay(raster: Raster, reading: LossReading) -> np.ndarray:
    colors = {
        "intact": (139, 221, 175), "receded": (228, 199, 123), "over_limit": (250, 131, 89),
        "no_edge_detected": (243, 129, 137), "unknown": (121, 160, 204),
    }
    base = symmetric_overlay(
        raster.rgb,
        [(reading.missing, colors["over_limit"], 0.55), (reading.unknown, colors["unknown"], 0.62)],
        support=raster.support | reading.expected,
    )
    canvas = Image.fromarray(base)
    pen = ImageDraw.Draw(canvas)
    view = [tuple(point) for point in reading.corners]
    pen.line(view + view[:1], fill=(225, 217, 155), width=2)
    for reading_station in reading.samples:
        x, y = reading_station.position
        tint = colors[reading_station.status]
        pen.ellipse((x - 2, y - 2, x + 2, y + 2), fill=tint)
        if reading_station.boundary and reading_station.status in {"over_limit", "receded"}:
            pen.line([(x, y), tuple(reading_station.boundary)], fill=tint, width=2)
    return np.asarray(canvas)


LOSS_COLUMNS = [
    "side", "station", "expected_x_px", "expected_y_px", "observed_x_px", "observed_y_px", "recession", "classification",
]


def measure(raster: Raster, params: dict) -> tuple[dict, np.ndarray, list[list]]:
    data = measure_loss(raster, params)
    counts = {name: 0 for name in FLAGS}
    rows = []
    for point in data.samples:
        counts[point.status] += 1
        rows.append([
            point.name, point.station, *point.position,
            *(point.boundary or [None, None]), point.depth, point.status,
        ])
    valid = [item.depth for item in data.samples if item.depth is not None]
    unknown = int(np.count_nonzero(data.unknown))
    absent = int(np.count_nonzero(data.missing))
    expected_px = int(np.count_nonzero(data.expected))
    notices = ["Place the expected stock outline on the measuring plane. Colored ticks compare its four sides to the detected face."]
    if counts["unknown"]:
        notices.append("Blue stations encountered unsupported source pixels before a measurable edge. Their recession is unknown.")
    if counts["no_edge_detected"]:
        notices.append("Some probes never reached a wood edge within the requested inspection depth.")
    reading = {
        "metrics": {
            "expected_area": expected_px * raster.plane.pixel_area,
            "visible_loss": absent * raster.plane.pixel_area,
            "visible_loss_fraction": safe_ratio(absent, expected_px - unknown),
            "unknown_fraction": safe_ratio(unknown, expected_px),
            "flagged_stations": counts["over_limit"],
            "deepest_recession": float(max(valid)) if valid else None,
        },
        "notices": notices,
        "unit": raster.plane.unit,
        "stations": counts,
    }
    return reading, overlay(raster, data), rows
