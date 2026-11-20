from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from .geometry import Plane, Raster
from .imaging import luminance, masked_smooth, safe_ratio, symmetric_overlay
from .validation import document, fields, integer, number

SQUARE = np.ones((3, 3), dtype=bool)
CROSS = ndi.generate_binary_structure(2, 1)
NEIGHBORS = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]


@dataclass(frozen=True)
class CrackRegion:
    mask: np.ndarray
    skeleton: np.ndarray
    strength: np.ndarray
    labels: np.ndarray
    objects: list[dict]
    endpoints: np.ndarray
    junctions: np.ndarray


def normalize_params(raw: Any, plane: Plane) -> dict:
    body = document(raw, "crack settings")
    fields(body, {"contrast", "scale", "minimum_length", "bridge"}, "crack")
    return {
        "contrast": number(body.get("contrast", 0.14), "crack contrast", 0.012, 0.8),
        "scale": number(body.get("scale", 6), "background radius", 2, 44),
        "minimum_length": number(body.get("minimum_length", 13), "minimum trace length", 3, max(13, min(600, max(plane.width, plane.height)))),
        "bridge": integer(body.get("bridge", 1), "gap closing radius", 0, 4),
    }


def shifted_neighbors(binary: np.ndarray) -> list[np.ndarray]:
    padded = np.pad(binary, 1)
    height, width = binary.shape
    return [
        padded[1 + dy:1 + dy + height, 1 + dx:1 + dx + width]
        for dy, dx in NEIGHBORS
    ]


def thin(binary: np.ndarray, maximum: int = 130) -> np.ndarray:
    current = binary.copy()
    for _ in range(maximum):
        changing = False
        for phase in (0, 1):
            around = shifted_neighbors(current)
            count = np.sum(around, axis=0, dtype=np.uint8)
            transitions = np.zeros(binary.shape, dtype=np.uint8)
            for left, right in zip(around, around[1:] + around[:1], strict=True):
                transitions += ~left & right
            n, e, s, w = around[0], around[2], around[4], around[6]
            if phase == 0:
                triplets = ~(n & e & s) & ~(e & s & w)
            else:
                triplets = ~(n & e & w) & ~(n & s & w)
            drop = current & (count >= 2) & (count <= 6) & (transitions == 1) & triplets
            if drop.any():
                current[drop] = False
                changing = True
        if not changing:
            break
    return current


def trace_masks(raster: Raster, params: dict) -> tuple[np.ndarray, np.ndarray]:
    image = luminance(raster.rgb)
    allowed = raster.support
    background, confidence = masked_smooth(image, allowed, params["scale"])
    strength = np.maximum(background - image, 0)
    strong = (strength >= params["contrast"]) & allowed & (confidence >= 0.55)
    if params["bridge"]:
        gap = params["bridge"]
        closed = ndi.binary_closing(strong, structure=ndi.iterate_structure(CROSS, gap), border_value=0)
        strong |= closed & allowed
    return strong, strength


def degrees(skeleton: np.ndarray) -> np.ndarray:
    near = shifted_neighbors(skeleton)
    cardinal = near[0::2]
    count = np.sum(cardinal, axis=0, dtype=np.uint8)
    for index in range(4):
        diagonal = near[index * 2 + 1]
        count += diagonal & ~cardinal[index] & ~cardinal[(index + 1) % 4]
    return count * skeleton


def edge_lengths(skeleton: np.ndarray, dx: float, dy: float) -> np.ndarray:
    near = shifted_neighbors(skeleton)
    right, below = near[2], near[4]
    southeast, southwest = near[3], near[5]
    diagonal = np.hypot(dx, dy)
    corners = southeast & ~right & ~below
    corners_other = southwest & ~near[6] & ~below
    total = right * dx + below * dy + (corners.astype(np.uint8) + corners_other) * diagonal
    return np.where(skeleton, total, 0)


def outside_edge(support: np.ndarray) -> np.ndarray:
    return support & ~ndi.binary_erosion(support, structure=SQUARE, border_value=0)


def diameter_map(component: np.ndarray, plane: Plane) -> np.ndarray:
    padded = np.pad(component, 1)
    step = (plane.mm_y or 1, plane.mm_x or 1)
    radius = ndi.distance_transform_edt(padded, sampling=step)[1:-1, 1:-1]
    return radius * 2


def object_shape(x_indices: np.ndarray, y_indices: np.ndarray, plane: Plane) -> tuple[float, float]:
    if len(x_indices) < 3:
        return 0.0, 1.0
    coords = np.stack([x_indices * (plane.mm_x or 1), y_indices * (plane.mm_y or 1)], axis=1)
    cov = np.cov(coords, rowvar=False, bias=True)
    vals, vectors = np.linalg.eigh(cov)
    angle = float(np.degrees(np.arctan2(vectors[1, -1], vectors[0, -1])) % 180)
    elongation = float(np.sqrt(max(vals[-1], 0.01) / max(vals[0], 0.01)))
    return angle, elongation


def prune_labels(raw: np.ndarray, minimum_px: int) -> np.ndarray:
    labels, count = ndi.label(raw, structure=SQUARE)
    numerators = np.bincount(labels.ravel(), minlength=count + 1)
    keep = numerators >= minimum_px
    keep[0] = False
    return keep[labels]


def analyze_regions(raster: Raster, params: dict) -> CrackRegion:
    proposed, strength = trace_masks(raster, params)
    minimum = params["minimum_length"]
    proposed = prune_labels(proposed, max(4, int(minimum)))
    labels, _ = ndi.label(proposed, structure=SQUARE)
    ids = ndi.find_objects(labels)
    accepted = np.zeros(proposed.shape, dtype=bool)
    skeleton = accepted.copy()
    endpoints = accepted.copy()
    junctions = accepted.copy()
    support_edge = outside_edge(raster.support)
    objects = []
    for label, excerpt in enumerate(ids, 1):
        if excerpt is None:
            continue
        piece = labels[excerpt] == label
        individual = thin(piece)
        length_px = float(edge_lengths(individual, 1, 1).sum())
        if length_px < minimum:
            continue
        # Compact dark objects are reported by pore measurement, not elongated checks.
        ys, xs = np.nonzero(individual)
        angle, elongation = object_shape(xs, ys, raster.plane)
        widths = diameter_map(piece, raster.plane)[individual]
        local_d = degrees(individual)
        local_end = individual & (local_d == 1)
        local_junction = individual & (local_d >= 3)
        if elongation < 1.38 and not local_junction.any():
            continue
        y0, x0 = excerpt[0].start, excerpt[1].start
        accepted[excerpt] |= piece
        skeleton[excerpt] |= individual
        endpoints[excerpt] |= local_end
        junctions[excerpt] |= local_junction
        length = float(edge_lengths(individual, raster.plane.mm_x or 1, raster.plane.mm_y or 1).sum())
        edge = bool(np.any(piece & support_edge[excerpt]))
        objects.append({
            "id": len(objects) + 1,
            "x": float(np.mean(xs) + x0),
            "y": float(np.mean(ys) + y0),
            "bbox": [x0, y0, excerpt[1].stop, excerpt[0].stop],
            "area": int(piece.sum()) * raster.plane.pixel_area,
            "length": length,
            "median_width": float(np.median(widths)),
            "width_p90": float(np.percentile(widths, 90)),
            "orientation_deg": angle,
            "endpoints": int(local_end.sum()),
            "junctions": int(ndi.label(local_junction, structure=SQUARE)[1]),
            "touches_support_edge": edge,
            "contrast": float(np.mean(strength[excerpt][piece])),
        })
    return CrackRegion(accepted, skeleton, strength, labels, objects, endpoints, junctions)


def draw_preview(raster: Raster, regions: CrackRegion) -> np.ndarray:
    composite = symmetric_overlay(
        raster.rgb,
        [
            (regions.mask, (209, 79, 89), 0.60),
            (regions.skeleton, (252, 211, 112), 0.97),
        ],
        support=raster.support,
    )
    output = Image.fromarray(composite)
    pen = ImageDraw.Draw(output)
    radius = max(2, max(raster.plane.width, raster.plane.height) // 310)
    for points, color in ((regions.endpoints, (133, 202, 222)), (regions.junctions, (251, 154, 130))):
        labels, count = ndi.label(points, structure=SQUARE)
        if count:
            for y, x in ndi.center_of_mass(points, labels, np.arange(1, count + 1)):
                pen.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=1)
    for item in sorted(regions.objects, key=lambda candidate: candidate["length"], reverse=True)[:28]:
        x, y = item["x"], item["y"]
        pen.text((x + 5, y + 4), str(item["id"]), fill=(250, 237, 198), stroke_width=2, stroke_fill=(63, 39, 28))
    return np.asarray(output)


CRACK_COLUMNS = [
    "id", "x", "y", "length", "area", "median_width", "width_p90",
    "orientation_deg", "endpoints", "junctions", "touches_support_edge", "contrast",
]


def measure(raster: Raster, params: dict) -> tuple[dict, np.ndarray, list[list]]:
    regions = analyze_regions(raster, params)
    all_regions = regions.objects
    disconnected = Counter("edge" if obj["touches_support_edge"] else "internal" for obj in all_regions)
    notices = ["Edge-touching traces may continue beyond the inspected support. Widths approximate a binary dark region."]
    if not all_regions:
        notices.append("No elongated dark traces passed the current contrast and length settings.")
    result = {
        "metrics": {
            "traces": len(all_regions),
            "total_length": sum(item["length"] for item in all_regions),
            "longest_trace": max((item["length"] for item in all_regions), default=None),
            "edge_traces": disconnected["edge"],
            "internal_traces": disconnected["internal"],
            "trace_area_fraction": safe_ratio(int(regions.mask.sum()), int(raster.support.sum())),
        },
        "notices": notices,
        "objects": all_regions[:100],
        "unit": raster.plane.unit,
    }
    return result, draw_preview(raster, regions), [[item[name] for name in CRACK_COLUMNS] for item in all_regions]
