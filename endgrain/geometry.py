from dataclasses import dataclass
from math import ceil, hypot, sqrt
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from .errors import WorkbenchError
from .imaging import WORK_LIMIT
from .validation import document, fields, list_of, number, choice


@dataclass(frozen=True)
class Plane:
    source_width: int
    source_height: int
    width: int
    height: int
    to_source: np.ndarray
    to_plane: np.ndarray
    mm_x: float | None
    mm_y: float | None
    calibration: dict[str, Any]

    @property
    def physical(self) -> bool:
        return self.mm_x is not None

    @property
    def unit(self) -> str:
        return "mm" if self.physical else "px"

    @property
    def pixel_area(self) -> float:
        return (self.mm_x or 1) * (self.mm_y or 1)

    def json(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "source_width": self.source_width,
            "source_height": self.source_height,
            "mm_per_x": self.mm_x,
            "mm_per_y": self.mm_y,
            "unit": self.unit,
            "calibration": self.calibration,
        }

    def distance(self, a: np.ndarray, b: np.ndarray) -> float:
        return hypot((float(a[0]) - float(b[0])) * (self.mm_x or 1), (float(a[1]) - float(b[1])) * (self.mm_y or 1))

    def map_source(self, points: np.ndarray) -> np.ndarray:
        return project(self.to_source, points)

    def map_plane(self, points: np.ndarray) -> np.ndarray:
        return project(self.to_plane, points)


@dataclass(frozen=True)
class Raster:
    rgb: np.ndarray
    supported: np.ndarray
    inspected: np.ndarray
    plane: Plane

    @property
    def support(self) -> np.ndarray:
        return self.supported & self.inspected

    def bounds(self) -> tuple[int, int, int, int] | None:
        return mask_bounds(self.support)


def pixel_point(raw: Any, label: str, width: int, height: int) -> list[float]:
    values = list_of(raw, label, 2, 2)
    x = number(values[0], f"{label} x", -0.5, width - 0.5)
    y = number(values[1], f"{label} y", -0.5, height - 0.5)
    return [x, y]


def normalize_calibration(raw: Any, width: int, height: int) -> dict[str, Any]:
    if raw is None:
        return {"mode": "pixels"}
    req = document(raw, "calibration")
    mode = choice(req.get("mode"), "calibration mode", ("pixels", "ruler", "quadrilateral"))
    if mode == "pixels":
        fields(req, {"mode"}, "calibration")
        return {"mode": mode}
    if mode == "ruler":
        fields(req, {"mode", "a", "b", "length_mm"}, "calibration")
        a = pixel_point(req.get("a"), "ruler start", width, height)
        b = pixel_point(req.get("b"), "ruler end", width, height)
        length = number(req.get("length_mm"), "ruler length in mm", 0.05, 12000)
        if np.linalg.norm(np.asarray(a) - b) < 6:
            raise WorkbenchError("Place ruler endpoints at least six source pixels apart.", "calibration")
        return {"mode": mode, "a": a, "b": b, "length_mm": length}
    fields(req, {"mode", "corners", "width_mm", "height_mm"}, "calibration")
    vertices = [
        pixel_point(point, f"corner {index}", width, height)
        for index, point in enumerate(list_of(req.get("corners"), "corners", 4, 4), 1)
    ]
    if not ordered_quad(vertices):
        raise WorkbenchError("Place four distinct convex corners clockwise from the upper left.", "corners")
    physical_width = number(req.get("width_mm"), "rectangle width in mm", 0.05, 12000)
    physical_height = number(req.get("height_mm"), "rectangle height in mm", 0.05, 12000)
    aspect = physical_width / physical_height
    if not 0.05 <= aspect <= 20:
        raise WorkbenchError("The calibration rectangle aspect ratio must be between 1:20 and 20:1.", "calibration")
    return {
        "mode": mode,
        "corners": vertices,
        "width_mm": physical_width,
        "height_mm": physical_height,
    }


def signed_area(points: list | np.ndarray) -> float:
    vertices = np.asarray(points, dtype=np.float64)
    return 0.5 * float(np.sum(vertices[:, 0] * np.roll(vertices[:, 1], -1) - vertices[:, 1] * np.roll(vertices[:, 0], -1)))


def ordered_quad(vertices: list[list[float]]) -> bool:
    points = np.asarray(vertices)
    sides = np.roll(points, -1, axis=0) - points
    next_sides = np.roll(sides, -1, axis=0)
    turns = sides[:, 0] * next_sides[:, 1] - sides[:, 1] * next_sides[:, 0]
    return bool(
        np.all(turns > 1)
        and np.all(np.linalg.norm(sides, axis=1) >= 8)
        and signed_area(points) >= 96
    )


def edge_cross(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def on_segment(a: np.ndarray, p: np.ndarray, b: np.ndarray) -> bool:
    return bool(
        min(a[0], b[0]) - 1e-7 <= p[0] <= max(a[0], b[0]) + 1e-7
        and min(a[1], b[1]) - 1e-7 <= p[1] <= max(a[1], b[1]) + 1e-7
    )


def edges_intersect(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    one, two = edge_cross(a, b, c), edge_cross(a, b, d)
    three, four = edge_cross(c, d, a), edge_cross(c, d, b)
    if one * two < 0 and three * four < 0:
        return True
    return (
        (abs(one) < 1e-7 and on_segment(a, c, b))
        or (abs(two) < 1e-7 and on_segment(a, d, b))
        or (abs(three) < 1e-7 and on_segment(c, a, d))
        or (abs(four) < 1e-7 and on_segment(c, b, d))
    )


def simple_polygon(points: np.ndarray) -> bool:
    count = len(points)
    for index, left in enumerate(points):
        right = points[(index + 1) % count]
        if np.linalg.norm(right - left) < 1:
            return False
        for peer in range(index + 2, count):
            if index == 0 and peer == count - 1:
                continue
            if edges_intersect(left, right, points[peer], points[(peer + 1) % count]):
                return False
    return True


def normalize_region(raw: Any, plane: Plane) -> list[list[float]] | None:
    if raw is None:
        return None
    vertices = [
        pixel_point(point, f"region vertex {index}", plane.width, plane.height)
        for index, point in enumerate(list_of(raw, "region", 3, 48), 1)
    ]
    if not simple_polygon(np.asarray(vertices)) or abs(signed_area(vertices)) < 36:
        raise WorkbenchError("Trace a simple region with at least 36 square pixels of area.", "region")
    return vertices


def bounded_dimensions(width: float, height: float) -> tuple[int, int]:
    scale = min(1.0, WORK_LIMIT / max(width, height))
    return max(2, int(round(width * scale))), max(2, int(round(height * scale)))


def _coordinate_normalization(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = points.mean(axis=0)
    rms = sqrt(float(np.mean(np.sum((points - center) ** 2, axis=1))))
    if rms < 1e-9:
        raise WorkbenchError("The calibration corners are degenerate.", "corners")
    scale = sqrt(2) / rms
    matrix = np.array([[scale, 0, -scale * center[0]], [0, scale, -scale * center[1]], [0, 0, 1]])
    return project(matrix, points), matrix


def homography(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_n, source_scale = _coordinate_normalization(np.asarray(source, dtype=np.float64))
    target_n, target_scale = _coordinate_normalization(np.asarray(target, dtype=np.float64))
    rows = []
    for (x, y), (u, v) in zip(source_n, target_n, strict=True):
        rows.append([-x, -y, -1, 0, 0, 0, x * u, y * u, u])
        rows.append([0, 0, 0, -x, -y, -1, x * v, y * v, v])
    _, singular_values, vectors = np.linalg.svd(np.asarray(rows), full_matrices=True)
    if singular_values[-1] < singular_values[0] * 1e-7:
        raise WorkbenchError("The calibration rectangle is numerically unstable.", "corners")
    normal = vectors[-1].reshape((3, 3))
    transform = np.linalg.solve(target_scale, normal) @ source_scale
    scale = float(np.max(np.abs(transform)))
    if scale < 1e-12 or not np.isfinite(transform).all():
        raise WorkbenchError("The calibration plane could not be formed.", "corners")
    return transform / scale


def project(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    numerators = points @ matrix[:2, :2].T + matrix[:2, 2]
    divisor = points @ matrix[2, :2] + matrix[2, 2]
    projected = np.full_like(numerators, np.nan)
    np.divide(numerators, divisor[..., None], out=projected, where=(np.abs(divisor) > 1e-10)[..., None])
    return projected


def plane_for(width: int, height: int, calibration: dict) -> Plane:
    mode = calibration["mode"]
    if mode in {"pixels", "ruler"}:
        if mode == "pixels":
            scale_native = None
        else:
            span = np.linalg.norm(np.asarray(calibration["a"]) - calibration["b"])
            scale_native = calibration["length_mm"] / float(span)
        view_width, view_height = bounded_dimensions(width, height)
        sx, sy = width / view_width, height / view_height
        matrix = np.array([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]])
        return Plane(
            width,
            height,
            view_width,
            view_height,
            matrix,
            np.linalg.inv(matrix),
            sx * scale_native if scale_native is not None else None,
            sy * scale_native if scale_native is not None else None,
            calibration,
        )
    corners = np.asarray(calibration["corners"])
    sides = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
    mm_width, mm_height = calibration["width_mm"], calibration["height_mm"]
    px_area = abs(signed_area(corners))
    px_per_mm = min(sqrt(px_area / (mm_width * mm_height)), max(sides) * 1.4 / max(mm_width, mm_height))
    view_width, view_height = bounded_dimensions(mm_width * px_per_mm, mm_height * px_per_mm)
    view_width, view_height = max(12, view_width), max(12, view_height)
    outer = np.array(
        [[-0.5, -0.5], [view_width - 0.5, -0.5], [view_width - 0.5, view_height - 0.5], [-0.5, view_height - 0.5]]
    )
    matrix = homography(outer, corners)
    return Plane(
        width,
        height,
        view_width,
        view_height,
        matrix,
        np.linalg.inv(matrix),
        mm_width / view_width,
        mm_height / view_height,
        calibration,
    )


def region_mask(plane: Plane, region: list[list[float]] | None) -> np.ndarray:
    if not region:
        return np.ones((plane.height, plane.width), dtype=bool)
    canvas = Image.new("1", (plane.width, plane.height))
    ImageDraw.Draw(canvas).polygon([tuple(point) for point in region], fill=1)
    return np.asarray(canvas, dtype=bool)


def mask_bounds(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if not len(rows):
        return None
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def _sample_rows(
    source: np.ndarray, avail: np.ndarray, plane: Plane, block: slice,
) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = np.mgrid[block.start:block.stop, 0:plane.width]
    coords = plane.map_source(np.stack([cols, rows], axis=-1))
    sx, sy = coords[..., 0], coords[..., 1]
    inside = (
        np.isfinite(sx)
        & np.isfinite(sy)
        & (sx >= -0.00001)
        & (sx <= plane.source_width - 1 + 0.00001)
        & (sy >= -0.00001)
        & (sy <= plane.source_height - 1 + 0.00001)
    )
    mapped = np.stack([np.clip(np.where(inside, sy, 0), 0, plane.source_height - 1),
                       np.clip(np.where(inside, sx, 0), 0, plane.source_width - 1)])
    weights = ndi.map_coordinates(avail, mapped, order=1, mode="nearest", prefilter=False)
    supported = inside & (weights >= 0.99999)
    colors = np.stack(
        [ndi.map_coordinates(source[..., index], mapped, order=1, mode="nearest", prefilter=False) for index in range(3)],
        axis=-1,
    )
    colors[~inside] = 0
    return colors, supported


def rasterize(
    pixels: np.ndarray, available: np.ndarray, plane: Plane, region: list[list[float]] | None,
) -> Raster:
    if plane.calibration["mode"] != "quadrilateral" and (plane.width, plane.height) == (plane.source_width, plane.source_height):
        return Raster(pixels, available, region_mask(plane, region), plane)
    rgb = np.empty((plane.height, plane.width, 3), dtype=np.uint8)
    supported = np.zeros((plane.height, plane.width), dtype=bool)
    support_float = available.astype(np.float32)
    for start in range(0, plane.height, 120):
        rows = slice(start, min(start + 120, plane.height))
        rgb[rows], supported[rows] = _sample_rows(pixels, support_float, plane, rows)
    return Raster(rgb, supported, region_mask(plane, region), plane)


def sample_line(
    array: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    *,
    max_stride: float = 1.0,
    order: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    count = max(2, ceil(np.linalg.norm(b - a) / max_stride) + 1)
    points = np.linspace(a, b, count)
    values = ndi.map_coordinates(array, points[:, ::-1].T, order=order, mode="constant", prefilter=False, cval=0)
    return points, values
