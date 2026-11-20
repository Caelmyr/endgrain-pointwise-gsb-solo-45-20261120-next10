from dataclasses import dataclass
from math import hypot, log10, sqrt
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .database import Database
from .geometry import Plane, Raster
from .imaging import encode_png, safe_ratio, symmetric_overlay
from .workspaces import frame_for


@dataclass(frozen=True)
class CalibrationReadout:
    plane: Plane
    extents: tuple[float, float]
    grid: float
    anchors: list[dict]
    edges: list[dict]
    source_sampling: dict
    coverage: float | None

    def payload(self) -> dict:
        return {
            "plane": self.plane.json(),
            "width": self.extents[0],
            "height": self.extents[1],
            "unit": self.plane.unit,
            "grid_spacing": self.grid,
            "anchors": self.anchors,
            "edge_readings": self.edges,
            "source_sampling": self.source_sampling,
            "coverage": self.coverage,
        }


def positive_tick(goal: float) -> float:
    if goal <= 0:
        return 1.0
    magnitude = 10 ** np.floor(log10(goal))
    coded = goal / magnitude
    best = min([1, 2, 2.5, 5, 10], key=lambda value: abs(value - coded))
    return float(best * magnitude)


def physical_extents(plane: Plane) -> tuple[float, float]:
    return (plane.width * (plane.mm_x or 1), plane.height * (plane.mm_y or 1))


def reference_corners(plane: Plane) -> np.ndarray:
    w, h = plane.width - 0.5, plane.height - 0.5
    return np.array([[-0.5, -0.5], [w, -0.5], [w, h], [-0.5, h]])
    

def projectable_anchor(matrix_points: np.ndarray, mapped: np.ndarray, plane: Plane) -> list[dict]:
    rows = []
    for index, (local, original) in enumerate(zip(matrix_points, mapped, strict=True), 1):
        rows.append({
            "index": index,
            "x": float(local[0]),
            "y": float(local[1]),
            "source_x": float(original[0]),
            "source_y": float(original[1]),
            "ground_x": float((local[0] + 0.5) * (plane.mm_x or 1)),
            "ground_y": float((local[1] + 0.5) * (plane.mm_y or 1)),
        })
    return rows


def stretch_ratios(plane: Plane) -> dict:
    w, h = plane.width, plane.height
    a, b = np.meshgrid(np.linspace(0.12, 0.87, 5) * (w - 2), np.linspace(0.12, 0.87, 5) * (h - 2))
    origin = np.stack([a, b], axis=-1)
    j_begin = plane.map_source(origin)
    jx = plane.map_source(origin + [1, 0]) - j_begin
    jy = plane.map_source(origin + [0, 1]) - j_begin
    dx = np.linalg.norm(jx, axis=-1)
    dy = np.linalg.norm(jy, axis=-1)
    area = abs(jx[..., 0] * jy[..., 1] - jx[..., 1] * jy[..., 0])
    return {
        "source_px_per_measuring_x": [float(dx.min()), float(dx.max())],
        "source_px_per_measuring_y": [float(dy.min()), float(dy.max())],
        "source_area_stretch": [float(area.min()), float(area.max())],
        "median_source_px_per_measuring_side": sqrt(float(np.median(area))),
    }


def readings(raster: Raster) -> CalibrationReadout:
    plane = raster.plane
    corners = reference_corners(plane)
    mapped = plane.map_source(corners)
    rect_mode = plane.calibration["mode"]
    if rect_mode == "ruler":
        known = np.asarray([plane.calibration["a"], plane.calibration["b"]])
        anchors = projectable_anchor(plane.map_plane(known), known, plane)
    else:
        anchors = projectable_anchor(corners, mapped, plane)
    sides = []
    for index, side in enumerate(("top", "right", "bottom", "left")):
        n = (index + 1) % 4
        p, q = mapped[n] - mapped[index]
        sides.append({
            "edge": side,
            "source_span": hypot(p, q),
            "measuring_span": plane.distance(corners[index], corners[n]),
        })
    width, height = physical_extents(plane)
    spacing = positive_tick(min(width, height) / 7)
    return CalibrationReadout(
        plane, (width, height), spacing, anchors, sides, stretch_ratios(plane),
        safe_ratio(int(raster.supported.sum()), int(raster.supported.size)),
    )


def grid_markings(canvas: Image.Image, data: CalibrationReadout) -> None:
    pen = ImageDraw.Draw(canvas, "RGBA")
    height, width = data.plane.height, data.plane.width
    x_step = data.plane.mm_x or 1
    y_step = data.plane.mm_y or 1
    tint = (227, 218, 159, 175)
    for position in np.arange(data.grid, data.extents[0], data.grid):
        x = position / x_step - 0.5
        pen.line((x, 0, x, height - 1), fill=tint, width=1)
        if x > 6 and x < width - 40:
            pen.text((x + 3, 5), f"{position:g}", fill=(247, 232, 190, 255), stroke_width=1, stroke_fill=(49, 50, 39, 245))
    for position in np.arange(data.grid, data.extents[1], data.grid):
        y = position / y_step - 0.5
        pen.line((0, y, width - 1, y), fill=tint, width=1)
        if y > 8 and y < height - 16:
            pen.text((5, y - 13), f"{position:g}", fill=(247, 232, 190, 255), stroke_width=1, stroke_fill=(49, 50, 39, 245))


def stage_overlay(raster: Raster, data: CalibrationReadout) -> bytes:
    covered = raster.supported
    region = raster.inspected
    rgba = symmetric_overlay(
        raster.rgb,
        [
            (~covered, (135, 87, 77), 0.60),
            (covered & ~region, (66, 75, 66), 0.54),
        ],
    )
    canvas = Image.fromarray(rgba)
    grid_markings(canvas, data)
    pen = ImageDraw.Draw(canvas)
    placement = (7, max(1, data.plane.height - 24))
    header = f"{data.grid:g} {data.plane.unit} grid / supported {100 * (data.coverage or 0):.1f}%"
    length = pen.textlength(header)
    pen.rectangle((placement[0] - 3, placement[1] - 2, placement[0] + length + 5, placement[1] + 14), fill=(42, 46, 35))
    pen.text(placement, header, fill=(247, 234, 184))
    return encode_png(np.asarray(canvas))


def prepare(db: Database, inspection_id: str) -> tuple[Raster, CalibrationReadout, int]:
    info = db.inspection(inspection_id)
    current = frame_for(db, info)
    raster = current.raster
    return raster, readings(raster), info["revision"]
