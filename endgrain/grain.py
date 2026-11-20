from dataclasses import dataclass
from math import atan2, cos, degrees, hypot, pi, radians, sin
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from .geometry import Plane, Raster
from .imaging import luminance, masked_smooth, safe_ratio, symmetric_overlay
from .validation import document, fields, integer, number


@dataclass(frozen=True)
class OrientationField:
    angle: np.ndarray
    coherence: np.ndarray
    energy: np.ndarray
    support: np.ndarray
    eligible: np.ndarray
    classified: np.ndarray


@dataclass(frozen=True)
class TileDirection:
    x: int
    y: int
    width: int
    height: int
    supported: int
    classified: int
    angle: float | None
    concentration: float | None
    contrast: float | None

    def json(self) -> dict:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "supported": self.supported,
            "classified": self.classified,
            "angle": self.angle,
            "concentration": self.concentration,
            "contrast": self.contrast,
        }


def normalize_params(raw: Any, plane: Plane) -> dict:
    body = document(raw, "grain settings")
    fields(body, {"window", "grid", "minimum_coherence", "minimum_energy", "nominal_axis", "tolerance"}, "grain")
    return {
        "window": number(body.get("window", 4), "tensor integration radius", 1.0, 32.0),
        "grid": integer(body.get("grid", 36), "orientation tile width", 8, 200),
        "minimum_coherence": number(body.get("minimum_coherence", 0.38), "minimum directional coherence", 0.02, 0.99),
        "minimum_energy": number(body.get("minimum_energy", 0.012), "minimum contrast energy", 0.001, 0.75),
        "nominal_axis": number(body.get("nominal_axis", 0), "comparison axis in degrees", 0, 179.9999),
        "tolerance": number(body.get("tolerance", 20), "comparison tolerance in degrees", 2, 90),
    }


def gradients(raster: Raster) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray]:
    gray = luminance(raster.rgb)
    smoothed, usable = masked_smooth(gray, raster.support, 0.85)
    y_sampling = 1.0
    x_sampling = 1.0
    if raster.plane.physical:
        # Use physical axis scale relative to the average step. This keeps
        # controls comparable between photographs acquired at different zooms.
        scale = (raster.plane.mm_y + raster.plane.mm_x) / 2
        y_sampling = raster.plane.mm_y / scale
        x_sampling = raster.plane.mm_x / scale
    y, x = np.gradient(smoothed, y_sampling, x_sampling)
    valid = ndi.binary_erosion(raster.support & (usable >= 0.92), iterations=1, border_value=0)
    return (x, y), valid


def tensor(raster: Raster, params: dict) -> OrientationField:
    (dx, dy), gradient_valid = gradients(raster)
    radius = params["window"]
    xx, amount = masked_smooth(dx * dx, gradient_valid, radius)
    yy, _ = masked_smooth(dy * dy, gradient_valid, radius)
    xy, _ = masked_smooth(dx * dy, gradient_valid, radius)
    total = np.maximum(xx + yy, 0)
    anisotropy = np.sqrt(np.maximum((xx - yy) ** 2 + 4 * xy * xy, 0))
    coherence = np.zeros(total.shape, dtype=np.float32)
    np.divide(anisotropy, total, out=coherence, where=total > 1e-8)
    angle = (0.5 * np.arctan2(2 * xy, xx - yy) + pi / 2) % pi
    energy = np.sqrt(total)
    usable = raster.support & (amount >= 0.66)
    eligible = usable & (energy >= params["minimum_energy"])
    classified = eligible & (coherence >= params["minimum_coherence"])
    return OrientationField(angle, coherence, energy, usable, eligible, classified)


def axial_average(theta: np.ndarray, weight: np.ndarray | None = None) -> tuple[float | None, float | None]:
    if not theta.size:
        return None, None
    if weight is None:
        weight = np.ones_like(theta)
    mass = float(weight.sum())
    if mass <= 1e-9:
        return None, None
    vector = float(np.sum(np.cos(theta * 2) * weight)), float(np.sum(np.sin(theta * 2) * weight))
    concentration = min(1.0, hypot(*vector) / mass)
    if concentration <= 1e-6:
        return None, concentration
    return degrees((atan2(vector[1], vector[0]) / 2) % pi), concentration


def axial_distance(theta: np.ndarray, axis: float) -> np.ndarray:
    return np.abs((theta - axis + pi / 2) % pi - pi / 2)


def summarize_tiles(raster: Raster, field: OrientationField, size: int) -> list[TileDirection]:
    h, w = field.angle.shape
    results = []
    for y in range(0, h, size):
        ys = slice(y, min(y + size, h))
        for x in range(0, w, size):
            xs = slice(x, min(x + size, w))
            domain = np.s_[ys, xs]
            total = int(np.count_nonzero(raster.support[domain]))
            if total == 0:
                continue
            eligible = field.classified[domain]
            count = int(eligible.sum())
            angle, concentration = axial_average(field.angle[domain][eligible], field.coherence[domain][eligible])
            results.append(TileDirection(
                x=x,
                y=y,
                width=xs.stop - x,
                height=ys.stop - y,
                supported=total,
                classified=count,
                angle=angle,
                concentration=concentration,
                contrast=float(np.mean(field.energy[domain][eligible])) if count else None,
            ))
    return results


def color_direction(angle: float) -> tuple[int, int, int]:
    return (
        int(157 + 72 * cos(angle * 2)),
        int(188 + 56 * cos(angle * 2 - 2 * pi / 3)),
        int(172 + 65 * cos(angle * 2 + 2 * pi / 3)),
    )


def paint_orientation(raster: Raster, field: OrientationField, tiles: list[TileDirection], settings: dict) -> np.ndarray:
    composite = symmetric_overlay(
        raster.rgb,
        [(raster.support & ~field.classified, (65, 69, 71), 0.55)],
        support=raster.support,
    )
    preview = Image.fromarray(composite)
    painter = ImageDraw.Draw(preview)
    dx, dy = raster.plane.mm_x or 1, raster.plane.mm_y or 1
    line_size = max(1, min(4, settings["grid"] // 18))
    for note in tiles:
        if note.angle is None or note.classified / note.supported < 0.20:
            continue
        theta = radians(note.angle)
        rx, ry = cos(theta) / dx, sin(theta) / dy
        scale = min(note.width, note.height) * 0.38 / max(hypot(rx, ry), 1e-6)
        scale *= max(0.26, note.concentration)
        ox, oy = note.x + note.width / 2, note.y + note.height / 2
        endpoints = [(ox - rx * scale, oy - ry * scale), (ox + rx * scale, oy + ry * scale)]
        painter.line(endpoints, fill=(34, 41, 42), width=line_size + 2)
        painter.line(endpoints, fill=color_direction(theta), width=line_size)
    return np.asarray(preview)


def angle_distribution(angles: np.ndarray) -> dict:
    values, bins = np.histogram(np.degrees(angles), bins=18, range=(0, 180))
    return {
        "kind": "bars",
        "rows": [
            {"x": float((start + stop) / 2), "value": int(value)}
            for start, stop, value in zip(bins[:-1], bins[1:], values, strict=True)
        ],
        "x_label": "Orientation (degrees, undirected)",
        "y_label": "Classified pixels",
    }


GRAIN_COLUMNS = [
    "x", "y", "width", "height", "supported", "classified", "angle", "concentration", "contrast",
]


def measure(raster: Raster, params: dict) -> tuple[dict, np.ndarray, list[list]]:
    raw = tensor(raster, params)
    tiles = summarize_tiles(raster, raw, params["grid"])
    angles = raw.angle[raw.classified]
    coherence = raw.coherence[raw.classified]
    average, resultant = axial_average(angles, coherence)
    in_tolerance = axial_distance(angles, radians(params["nominal_axis"])) <= radians(params["tolerance"])
    counted = int(len(angles))
    total = int(raster.support.sum())
    messages = ["An orientation is axial: angles 0° and 180° are the same. Gray areas have insufficient support, contrast or coherence."]
    if resultant is not None and resultant < 0.35:
        messages.append("The field contains competing orientations. Use the tiles and distribution instead of the mean alone.")
    if counted < 0.35 * total:
        messages.append("Only a minority of the measuring region supplies a reliable directional signal.")
    output = {
        "metrics": {
            "principal_angle": average,
            "angle_concentration": resultant,
            "classified_fraction": safe_ratio(counted, total),
            "within_axis_tolerance": safe_ratio(int(in_tolerance.sum()), counted),
            "median_coherence": float(np.median(coherence)) if counted else None,
            "low_energy_pixels": int(np.count_nonzero(raw.support & ~raw.eligible)),
        },
        "notices": messages,
        "unit": "°",
        "chart": angle_distribution(angles),
    }
    rows = [[item.json()[name] for name in GRAIN_COLUMNS] for item in tiles]
    return output, paint_orientation(raster, raw, tiles, params), rows
