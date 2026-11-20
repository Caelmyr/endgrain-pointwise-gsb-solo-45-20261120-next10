from dataclasses import dataclass
from math import atan2, degrees, pi, sqrt
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from .geometry import Plane, Raster, mask_bounds, pixel_point
from .imaging import luminance, safe_ratio, symmetric_overlay
from .validation import document, fields, number


SQUARE = np.ones((3, 3), dtype=bool)


@dataclass(frozen=True)
class Footprint:
    selected: np.ndarray
    holes: np.ndarray
    rim: np.ndarray
    outer_box: tuple[int, int, int, int] | None
    center: tuple[float, float] | None
    major: float | None
    minor: float | None
    angle: float | None
    samples: list[list]


def normalize_params(raw: Any, plane: Plane) -> dict:
    body = document(raw, "footprint settings")
    fields(body, {"minimum_luma", "warmth", "bridge", "minimum_hole_area", "seed"}, "footprint")
    seed = body.get("seed")
    return {
        "minimum_luma": number(body.get("minimum_luma", 0.24), "minimum brightness", 0.025, 0.95),
        "warmth": number(body.get("warmth", 0.10), "minimum red minus blue", -0.3, 0.65),
        "bridge": number(body.get("bridge", 2), "edge smoothing radius", 0, 12),
        "minimum_hole_area": number(body.get("minimum_hole_area", 110), "smallest reported hole in square pixels", 1, 50000),
        "seed": pixel_point(seed, "footprint selection point", plane.width, plane.height) if seed is not None else None,
    }


def solid_labels(raster: Raster, params: dict) -> np.ndarray:
    image = raster.rgb.astype(np.float32) / 255
    warm = image[..., 0] - image[..., 2]
    initial = raster.support & (luminance(image) > params["minimum_luma"]) & (warm > params["warmth"])
    radius = int(round(params["bridge"]))
    if radius:
        structure = ndi.iterate_structure(ndi.generate_binary_structure(2, 1), radius)
        padded = np.pad(initial, radius, mode="edge")
        closed = ndi.binary_closing(padded, structure=structure)[radius:-radius, radius:-radius]
        initial = closed & raster.support
    return ndi.label(initial, structure=SQUARE)[0]


def isolated(labels: np.ndarray, seed: list[float] | None) -> np.ndarray:
    sizes = np.bincount(labels.ravel())
    if len(sizes) == 1:
        return np.zeros(labels.shape, dtype=bool)
    if seed is not None:
        x = int(np.clip(round(seed[0]), 0, labels.shape[1] - 1))
        y = int(np.clip(round(seed[1]), 0, labels.shape[0] - 1))
        match = int(labels[y, x])
        return labels == match if match else np.zeros(labels.shape, dtype=bool)
    sizes[0] = 0
    return labels == sizes.argmax()


def internal_holes(
    solid: np.ndarray, supported: np.ndarray, area_limit: float,
) -> tuple[np.ndarray, np.ndarray]:
    all_holes = ndi.binary_fill_holes(solid) & ~solid
    hole_labels, count = ndi.label(all_holes)
    sizes = np.bincount(hole_labels.ravel(), minlength=count + 1)
    bad = set(np.unique(hole_labels[~supported]))
    minor = sizes < area_limit
    minor[list(bad)] = False
    minor[0] = False
    filled = minor[hole_labels]
    return solid | filled, all_holes & ~filled


def physical_shape(mask: np.ndarray, plane: Plane) -> tuple:
    ys, xs = np.nonzero(mask)
    if len(xs) < 3:
        return None, None, None, None
    x, y = float(xs.mean()), float(ys.mean())
    dx, dy = plane.mm_x or 1, plane.mm_y or 1
    du, dv = (xs - x) * dx, (ys - y) * dy
    covariance = np.cov([du, dv], bias=True)
    eigs, axes = np.linalg.eigh(covariance)
    turn = degrees(atan2(float(axes[1, -1]), float(axes[0, -1]))) % 180
    return (x, y), 4 * sqrt(max(0.0, eigs[1])), 4 * sqrt(max(0.0, eigs[0])), turn


def border_length(mask: np.ndarray, plane: Plane) -> float:
    padded = np.pad(mask, 1)
    vertical_edges = np.count_nonzero(padded[:, 1:] != padded[:, :-1])
    horizontal_edges = np.count_nonzero(padded[1:, :] != padded[:-1, :])
    return float(vertical_edges * (plane.mm_y or 1) + horizontal_edges * (plane.mm_x or 1))


def strip_profile(mask: np.ndarray, plane: Plane, outer: tuple[int, int, int, int] | None) -> list[list]:
    if not outer:
        return []
    x0, y0, x1, y1 = outer
    increment = max(1, (y1 - y0) // 100)
    out = []
    for row in range(y0, y1, increment):
        picks = np.flatnonzero(mask[row, x0:x1])
        if picks.size:
            span = int(picks[-1] - picks[0] + 1)
            start, end = int(picks[0] + x0), int(picks[-1] + x0)
        else:
            span, start, end = 0, None, None
        out.append([
            row, start, end, span, int(picks.size),
            span * (plane.mm_x or 1), int(picks.size) * (plane.mm_x or 1),
        ])
    return out


def geometry_reading(raster: Raster, params: dict) -> Footprint:
    labels = solid_labels(raster, params)
    raw = isolated(labels, params["seed"])
    selected, holes = internal_holes(raw, raster.support, params["minimum_hole_area"])
    rim = selected & ~ndi.binary_erosion(selected, structure=SQUARE, border_value=0)
    bounds = mask_bounds(selected)
    center, major, minor, angle = physical_shape(selected, raster.plane)
    return Footprint(
        selected, holes, rim, bounds, center, major, minor, angle,
        strip_profile(selected, raster.plane, bounds),
    )


def overlay(raster: Raster, geometry: Footprint) -> np.ndarray:
    base = symmetric_overlay(
        raster.rgb,
        [
            (geometry.selected, (168, 204, 157), 0.14),
            (geometry.rim, (190, 238, 162), 0.97),
            (geometry.holes, (242, 152, 109), 0.54),
        ],
        support=raster.support,
    )
    if not geometry.outer_box:
        return base
    canvas = Image.fromarray(base)
    pen = ImageDraw.Draw(canvas)
    x0, y0, x1, y1 = geometry.outer_box
    pen.rectangle((x0, y0, x1 - 1, y1 - 1), outline=(244, 207, 138), width=2)
    if geometry.center:
        x, y = geometry.center
        half = max(4, min(raster.plane.width, raster.plane.height) // 90)
        pen.line((x - half, y, x + half, y), fill=(231, 231, 196), width=2)
        pen.line((x, y - half, x, y + half), fill=(231, 231, 196), width=2)
    return np.asarray(canvas)


FOOTPRINT_COLUMNS = ["row_px", "start_x_px", "end_x_px", "span_px", "occupied_px", "span", "occupied"]


def measure(raster: Raster, params: dict) -> tuple[dict, np.ndarray, list[list]]:
    data = geometry_reading(raster, params)
    support_edge = raster.support & ~ndi.binary_erosion(raster.support, structure=SQUARE, border_value=0)
    cut = bool(np.any(data.selected & support_edge))
    area_pixels = int(data.selected.sum())
    mass = area_pixels * raster.plane.pixel_area
    extent = raster.plane
    bounds = data.outer_box or (0, 0, 0, 0)
    notices = ["The footprint uses brightness and color contrast against the backing. Inspect its contour before using any area measurements."]
    if cut:
        notices.append("The footprint touches the measuring support and its full dimensions may be cut off.")
    if not area_pixels:
        notices.append("No connected wood-colored silhouette passed the selected thresholds.")
    area_with_voids = mass + int(data.holes.sum()) * extent.pixel_area
    return ({
        "metrics": {
            "wood_area": mass,
            "wood_fraction": safe_ratio(area_pixels, int(raster.support.sum())),
            "outer_extent_x": (bounds[2] - bounds[0]) * (extent.mm_x or 1),
            "outer_extent_y": (bounds[3] - bounds[1]) * (extent.mm_y or 1),
            "trace_perimeter": border_length(data.selected, extent),
            "equivalent_diameter": sqrt(4 * area_with_voids / pi),
            "ellipse_major": data.major,
            "ellipse_minor": data.minor,
            "ellipse_axis": data.angle,
            "boundary_truncated": cut,
        },
        "notices": notices,
        "unit": extent.unit,
        "bounds": data.outer_box,
        "center": data.center,
    }, overlay(raster, data), data.samples)
