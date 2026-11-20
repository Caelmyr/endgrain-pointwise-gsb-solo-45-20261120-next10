from dataclasses import dataclass
from math import pi, sqrt
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from .geometry import Plane, Raster, mask_bounds
from .imaging import luminance, masked_smooth, safe_ratio, symmetric_overlay
from .validation import document, fields, flag, integer, number

CONNECTIVITY = np.ones((3, 3), dtype=bool)


@dataclass(frozen=True)
class Components:
    observed: np.ndarray
    retained: np.ndarray
    boundaries: np.ndarray
    excluded: np.ndarray
    rows: list[dict]
    reasons: dict[str, int]


def normalize_params(raw: Any, plane: Plane) -> dict:
    body = document(raw, "pore settings")
    fields(body, {"contrast", "min_area", "max_area", "compactness", "reject_edge"}, "pore")
    max_area = integer(body.get("max_area", 130), "largest pore area in square pixels", 5, 6000)
    return {
        "contrast": number(body.get("contrast", 0.095), "pore contrast", 0.01, 0.6),
        "min_area": integer(body.get("min_area", 5), "smallest pore area in square pixels", 2, max_area),
        "max_area": max_area,
        "compactness": number(body.get("compactness", 0.36), "minimum roundness", 0.05, 0.95),
        "reject_edge": flag(body.get("reject_edge", True), "exclude partial edge pores"),
    }


def perimeter(mask: np.ndarray, dx: float, dy: float) -> float:
    padded = np.pad(mask, 1, constant_values=False)
    h = np.count_nonzero(padded[:, 1:] != padded[:, :-1])
    v = np.count_nonzero(padded[1:, :] != padded[:-1, :])
    return float(h * dy + v * dx)


def inertia(positions: tuple[np.ndarray, np.ndarray], plane: Plane) -> tuple[float, float]:
    ys, xs = positions
    dx, dy = plane.mm_x or 1, plane.mm_y or 1
    coords = np.stack([xs * dx, ys * dy], axis=1)
    vectors = coords - coords.mean(axis=0)
    moment = vectors.T @ vectors / max(1, len(xs))
    eigenvalues, axes = np.linalg.eigh(moment)
    direction = float(np.degrees(np.arctan2(axes[1, 1], axes[0, 1])) % 180)
    eccentricity = sqrt(max(0.0, 1 - eigenvalues[0] / max(eigenvalues[1], 1e-12)))
    return direction, eccentricity


def dark_seeds(raster: Raster, settings: dict) -> tuple[np.ndarray, np.ndarray]:
    light = luminance(raster.rgb)
    support = raster.support
    radius = np.clip(sqrt(settings["max_area"] / pi), 3, 26)
    background, available = masked_smooth(light, support, radius)
    difference = background - light
    strongest = (difference >= settings["contrast"]) & support & (available >= 0.55)
    soft = (difference >= 0.68 * settings["contrast"]) & support & (available >= 0.55)
    # Two steps of growth let the body occupy a dim rim without treating every dark band as a seed.
    candidate = ndi.binary_dilation(strongest, structure=CONNECTIVITY, iterations=2, mask=soft)
    return candidate, difference


def pore_geometry(component: np.ndarray, plane: Plane, start: tuple[int, int]) -> dict:
    ys, xs = np.nonzero(component)
    pixel_count = len(xs)
    area = pixel_count * plane.pixel_area
    shape_area = perimeter(component, plane.mm_x or 1, plane.mm_y or 1)
    roundness = min(1.0, 4 * pi * area / max(shape_area ** 2, 1e-9))
    orientation, eccentricity = inertia((ys, xs), plane)
    bounds = mask_bounds(component)
    x, y = start
    return {
        "x": float(np.mean(xs) + x),
        "y": float(np.mean(ys) + y),
        "pixels": pixel_count,
        "area": area,
        "perimeter": shape_area,
        "diameter": sqrt(4 * area / pi),
        "roundness": roundness,
        "eccentricity": eccentricity,
        "orientation": orientation,
        "bbox": [bounds[0] + x, bounds[1] + y, bounds[2] + x, bounds[3] + y],
    }


def detect_components(raster: Raster, settings: dict) -> Components:
    observed, strength = dark_seeds(raster, settings)
    labels, count = ndi.label(observed, structure=CONNECTIVITY)
    accepted = np.zeros(labels.shape, dtype=bool)
    outline = np.zeros(labels.shape, dtype=bool)
    excluded = np.zeros(labels.shape, dtype=bool)
    pixels = np.bincount(labels.ravel(), minlength=count + 1)
    objects = ndi.find_objects(labels)
    valid = raster.support
    boundary = valid & ~ndi.binary_erosion(valid, structure=CONNECTIVITY, border_value=0)
    rows = []
    reasons = {"too_small": 0, "too_big": 0, "shape": 0, "edge": 0}
    for label, bounds in enumerate(objects, 1):
        if bounds is None:
            continue
        count = pixels[label]
        if count < settings["min_area"]:
            reasons["too_small"] += 1
            continue
        body = labels[bounds] == label
        if count > settings["max_area"]:
            reasons["too_big"] += 1
            continue
        shape = pore_geometry(body, raster.plane, (bounds[1].start, bounds[0].start))
        if shape["roundness"] < settings["compactness"]:
            reasons["shape"] += 1
            continue
        clipped = bool(np.any(body & boundary[bounds]))
        if clipped and settings["reject_edge"]:
            excluded[bounds] |= body
            reasons["edge"] += 1
            continue
        closed_rim = body & ~ndi.binary_erosion(body, structure=CONNECTIVITY, border_value=0)
        accepted[bounds] |= body
        outline[bounds] |= closed_rim
        rows.append({
            "id": len(rows) + 1,
            **shape,
            "edge": clipped,
            "contrast": float(np.mean(strength[bounds][body])),
        })
    return Components(observed, accepted, outline, excluded, rows, reasons)


def overlay(raster: Raster, records: Components) -> np.ndarray:
    base = symmetric_overlay(
        raster.rgb,
        [
            (records.retained, (68, 151, 142), 0.6),
            (records.boundaries, (106, 224, 193), 0.86),
            (records.excluded, (251, 158, 93), 0.7),
        ],
        support=raster.support,
    )
    canvas = Image.fromarray(base)
    pen = ImageDraw.Draw(canvas)
    selected = sorted(records.rows, key=lambda item: item["area"], reverse=True)[:22]
    for record in selected:
        x, y = record["x"], record["y"]
        pen.text((x + 3, y - 9), str(record["id"]), fill=(221, 239, 212), stroke_width=2, stroke_fill=(29, 55, 55))
    return np.asarray(canvas)


def diameter_chart(rows: list[dict], unit: str) -> dict:
    sizes = np.array([item["diameter"] for item in rows])
    if not len(sizes):
        return {"kind": "bars", "rows": [], "x_label": f"Pore diameter ({unit})", "y_label": "Count"}
    start = float(sizes.min())
    stop = float(sizes.max())
    if stop <= start:
        stop = start * 1.01
    counts, edges = np.histogram(sizes, bins=10, range=(start, stop))
    return {
        "kind": "bars",
        "rows": [
            {"x": float((left + right) / 2), "value": int(count)}
            for left, right, count in zip(edges[:-1], edges[1:], counts, strict=True)
        ],
        "x_label": f"Pore equivalent diameter ({unit})",
        "y_label": "Count",
    }


PORE_COLUMNS = [
    "id", "x", "y", "area", "diameter", "perimeter", "roundness",
    "eccentricity", "orientation", "edge", "contrast",
]


def measure(raster: Raster, settings: dict) -> tuple[dict, np.ndarray, list[list]]:
    records = detect_components(raster, settings)
    supported = int(raster.support.sum())
    area = supported * raster.plane.pixel_area
    size = [item["diameter"] for item in records.rows]
    count = len(records.rows)
    area_factor = 100 if raster.plane.physical else 10000
    result = {
        "metrics": {
            "pores": count,
            "survey_area": area,
            "pore_area_fraction": safe_ratio(int(records.retained.sum()), supported),
            "density_per_basis": safe_ratio(count * area_factor, area),
            "density_basis": "100 mm²" if raster.plane.physical else "10,000 px²",
            "median_diameter": float(np.median(size)) if count else None,
            "excluded_edge_candidates": records.reasons["edge"],
        },
        "notices": [
            "Dark round components are candidates for vessels. Compare the overlays before treating them as a species diagnostic.",
            "Pixels outside the inspection region or with unavailable source data do not contribute to surveyed area.",
        ],
        "rejected": records.reasons,
        "objects": records.rows[:100],
        "chart": diameter_chart(records.rows, raster.plane.unit),
        "unit": raster.plane.unit,
    }
    return result, overlay(raster, records), [[item[field] for field in PORE_COLUMNS] for item in records.rows]
