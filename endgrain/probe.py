from dataclasses import dataclass
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .database import Database
from .errors import WorkbenchError
from .geometry import Raster, pixel_point
from .imaging import luminance, safe_ratio, symmetric_overlay
from .reports import render_csv
from .validation import document, expect_revision, fields, integer
from .workspaces import Frame, describe_plane, frame_for


@dataclass(frozen=True)
class Probe:
    raster: Raster
    source: np.ndarray
    point: list[float]
    bounds: tuple[int, int, int, int]
    marker: tuple[int, int]
    crop: np.ndarray
    supported: np.ndarray
    plane_available: np.ndarray
    scale: int

    @property
    def unit(self) -> str:
        return self.raster.plane.unit


def configure(db: Database, inspection_id: str, raw: Any) -> tuple[Frame, dict]:
    request = document(raw, "pixel probe")
    fields(request, {"revision", "point", "radius", "scale"}, "pixel probe")
    expected = expect_revision(request)
    entry = db.inspection(inspection_id)
    db.validate_revision(entry, expected)
    plane = describe_plane(entry)
    params = {
        "point": pixel_point(request.get("point"), "probe position", plane.width, plane.height),
        "radius": integer(request.get("radius", 12), "probe radius", 2, 48),
        "scale": integer(request.get("scale", 8), "display magnification", 2, 12),
    }
    return frame_for(db, entry), params


def load(db: Database, inspection_id: str, raw: Any) -> Probe:
    frame, options = configure(db, inspection_id, raw)
    raster = frame.raster
    point = options["point"]
    ix = int(np.clip(round(point[0]), 0, raster.plane.width - 1))
    iy = int(np.clip(round(point[1]), 0, raster.plane.height - 1))
    radius = options["radius"]
    x0, y0 = max(0, ix - radius), max(0, iy - radius)
    x1, y1 = min(raster.plane.width, ix + radius + 1), min(raster.plane.height, iy + radius + 1)
    bounds = x0, y0, x1, y1
    tiles = np.s_[y0:y1, x0:x1]
    return Probe(
        raster, frame.source, point, bounds, (ix - x0, iy - y0),
        raster.rgb[tiles], raster.support[tiles], raster.supported[tiles], options["scale"],
    )


def pixel_coordinates(probe: Probe) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x0, y0, x1, y1 = probe.bounds
    ys, xs = np.mgrid[y0:y1, x0:x1]
    base = np.stack([xs, ys], axis=-1)
    coordinate = probe.raster.plane.map_source(base)
    return xs, ys, coordinate


def display_health(probe: Probe) -> dict[str, Any]:
    values = probe.crop[probe.supported]
    count = len(values)
    px = probe.crop.shape[0] * probe.crop.shape[1]
    detail = luminance(probe.crop)[probe.supported]
    percentiles = np.percentile(detail, [5, 50, 95]).tolist() if count else [None] * 3
    medians = np.median(values, axis=0).tolist() if count else [None] * 3
    bottom = {
        "all_pixels": px,
        "source_supported": int(np.count_nonzero(probe.plane_available)),
        "region_supported": count,
        "coverage": safe_ratio(count, px),
        "median_rgb": medians,
        "luma_percentiles": percentiles,
    }
    return bottom


def profile(probe: Probe, axis: int) -> dict:
    x0, y0, x1, y1 = probe.bounds
    u, v = probe.marker
    values = np.arange(x0, x1) if axis == 1 else np.arange(y0, y1)
    along = (values - values[u if axis == 1 else v]).astype(np.float64)
    step = (probe.raster.plane.mm_x if axis == 1 else probe.raster.plane.mm_y) or 1
    line = np.s_[v, :] if axis == 1 else np.s_[:, u]
    present = probe.supported[line]
    intensity = luminance(probe.crop)[line]
    return {
        "kind": "profile",
        "x": (along * step).tolist(),
        "y": [float(value) if visible else None for value, visible in zip(intensity, present, strict=True)],
        "markers": [0],
        "x_label": ("Horizontal" if axis == 1 else "Vertical") + " offset (" + probe.unit + ")",
        "y_label": "Display luminance",
    }


def describe(probe: Probe) -> dict:
    x0, y0, _, _ = probe.bounds
    u, v = probe.marker
    center = [u + x0, v + y0]
    mapped = probe.raster.plane.map_source(np.asarray(center))
    included = bool(probe.supported[v, u])
    return {
        "center": center,
        "source_point": mapped.tolist() if np.isfinite(mapped).all() else None,
        "source_supported": bool(probe.plane_available[v, u]),
        "region_supported": included,
        "rgb": probe.crop[v, u].tolist() if included else None,
        "bounds": probe.bounds,
        "statistics": display_health(probe),
        "horizontal": profile(probe, 1),
        "vertical": profile(probe, 0),
        "unit": probe.unit,
    }


def grid_on(canvas: Image.Image, size: int, label_size: int) -> None:
    pen = ImageDraw.Draw(canvas, "RGBA")
    if size < 8:
        return
    for x in range(size, canvas.width, size):
        pen.line((x, label_size, x, canvas.height), fill=(220, 218, 188, 43))
    for y in range(label_size + size, canvas.height, size):
        pen.line((0, y, canvas.width, y), fill=(220, 218, 188, 43))


def gauge_on(
    canvas: Image.Image, *, center: tuple[int, int], scale: int,
    header: int, point: list[float], units: str,
) -> None:
    px, py = center
    x, y = px * scale, header + py * scale
    pen = ImageDraw.Draw(canvas)
    gold = (248, 221, 149)
    pen.rectangle((x, y, x + scale - 1, y + scale - 1), outline=gold, width=1)
    pen.line((0, y + scale // 2, max(0, x - 1), y + scale // 2), fill=(186, 147, 94), width=1)
    pen.line((x + scale, y + scale // 2, canvas.width, y + scale // 2), fill=(186, 147, 94), width=1)
    display = f"({point[0]:.1f}, {point[1]:.1f}) px / {scale}x / {units}"
    pen.text((6, 6), display, fill=(238, 230, 208))


def preview(probe: Probe) -> bytes:
    colors = symmetric_overlay(
        probe.crop,
        [
            (probe.plane_available & ~probe.supported, (61, 81, 92), 0.72),
            (~probe.plane_available, (115, 83, 75), 0.92),
        ],
    )
    crop = Image.fromarray(colors).resize(
        (colors.shape[1] * probe.scale, colors.shape[0] * probe.scale),
        Image.Resampling.NEAREST,
    )
    top = 23
    canvas = Image.new("RGB", (max(170, crop.width), top + crop.height), (39, 45, 40))
    canvas.paste(crop, (0, top))
    grid_on(canvas, probe.scale, top)
    gauge_on(canvas, center=probe.marker, scale=probe.scale, header=top, point=probe.point, units=probe.unit)
    stream = BytesIO()
    canvas.save(stream, format="PNG")
    return stream.getvalue()


def csv_projection(probe: Probe) -> bytes:
    xs, ys, source_points = pixel_coordinates(probe)
    lightness = luminance(probe.crop)
    rows = []
    for iy, ix in np.ndindex(probe.supported.shape):
        available = bool(probe.plane_available[iy, ix])
        inspected = bool(probe.supported[iy, ix])
        projected = source_points[iy, ix]
        coordinates = projected.tolist() if np.isfinite(projected).all() else [None, None]
        pixel = probe.crop[iy, ix].tolist() if inspected else [None, None, None]
        light = float(lightness[iy, ix]) if inspected else None
        rows.append([int(xs[iy, ix]), int(ys[iy, ix]), *coordinates, available, inspected, *pixel, light])
    return render_csv(
        ["measuring_x", "measuring_y", "source_x", "source_y", "source_available", "included", "r", "g", "b", "display_luma"],
        rows,
    )
