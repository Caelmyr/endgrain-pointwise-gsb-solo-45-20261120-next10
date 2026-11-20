from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .geometry import Plane, Raster, pixel_point
from .imaging import safe_ratio, srgb_to_linear, symmetric_overlay
from .validation import document, fields, list_of, number, string
from .errors import WorkbenchError

SRGB_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
], dtype=np.float32)
D65 = np.array([0.95047, 1, 1.08883], dtype=np.float32)
COLORS = [(84, 197, 165), (247, 156, 89), (126, 154, 219), (207, 129, 178)]


@dataclass(frozen=True)
class Swatch:
    label: str
    point: list[float]
    rgb: list[int]
    center_lab: np.ndarray
    samples: int
    spread: float


def normalize_params(raw: Any, plane: Plane) -> dict:
    body = document(raw, "color-zone settings")
    fields(body, {"swatches", "radius", "limit", "minimum_margin"}, "color zones")
    defaults = [
        {"label": "Tone A", "point": [plane.width * 0.39, plane.height * 0.47]},
        {"label": "Tone B", "point": [plane.width * 0.68, plane.height * 0.57]},
    ]
    normalized = []
    for index, item in enumerate(list_of(body.get("swatches", defaults), "color swatches", 2, 4), 1):
        item = document(item, "color swatch")
        fields(item, {"label", "point"}, "color swatch")
        label = string(item.get("label"), f"swatch {index} label", 34, required=True)
        point = pixel_point(item.get("point"), f"swatch {index} center", plane.width, plane.height)
        if label in {entry["label"] for entry in normalized}:
            raise WorkbenchError("Give each color swatch a different label.", "swatches")
        normalized.append({"label": label, "point": point})
    return {
        "swatches": normalized,
        "radius": number(body.get("radius", 14), "swatch sampling radius", 3, 60),
        "limit": number(body.get("limit", 15), "largest color difference", 0.5, 45),
        "minimum_margin": number(body.get("minimum_margin", 2.0), "closest-zone margin", 0, 12),
    }


def to_lab(rgb: np.ndarray) -> np.ndarray:
    xyz = srgb_to_linear(rgb) @ SRGB_XYZ.T
    xyz /= D65
    epsilon = (6.0 / 29) ** 3
    linear = (29.0 / 6) ** 2 / 3
    xyz = np.where(xyz > epsilon, np.cbrt(xyz), linear * xyz + 4.0 / 29)
    x, y, z = (xyz[..., index] for index in range(3))
    return np.stack([116 * y - 16, 500 * (x - y), 200 * (y - z)], axis=-1)


def swatch_for(raster: Raster, image: np.ndarray, spec: dict, radius: float) -> Swatch:
    x, y = spec["point"]
    x0, y0 = max(0, int(x - radius)), max(0, int(y - radius))
    x1, y1 = min(raster.plane.width, int(x + radius) + 1), min(raster.plane.height, int(y + radius) + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    block = np.s_[y0:y1, x0:x1]
    disc = (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2
    inside = raster.support[block] & disc
    used = int(inside.sum())
    if used < max(6, 0.55 * disc.sum()):
        raise WorkbenchError(f"Swatch {spec['label']} has too little supported area.", "swatches")
    values = image[block][inside]
    location = np.median(values, axis=0)
    differences = np.linalg.norm(values - location, axis=1)
    return Swatch(
        label=spec["label"], point=spec["point"],
        rgb=np.rint(np.median(raster.rgb[block][inside], axis=0)).astype(int).tolist(),
        center_lab=location,
        samples=used,
        spread=float(np.median(differences)),
    )


def choices_by_color(image: np.ndarray, support: np.ndarray, swatches: list[Swatch], params: dict) -> tuple:
    labels = np.zeros(support.shape, dtype=np.uint8)
    margin = np.zeros(support.shape, dtype=np.float32)
    distance = np.zeros(support.shape, dtype=np.float32)
    missing = np.zeros(support.shape, dtype=bool)
    for start in range(0, height := support.shape[0], 160):
        region = np.s_[start:min(height, start + 160), :]
        pix = image[region]
        delta = np.stack([np.linalg.norm(pix - swatch.center_lab, axis=-1) for swatch in swatches], axis=0)
        winner = np.argmin(delta, axis=0)
        best, next_best = np.partition(delta, 1, axis=0)[:2]
        buffer = next_best - best
        known = support[region] & (best <= params["limit"])
        acceptable = known & (buffer >= params["minimum_margin"])
        labels[region] = np.where(acceptable, winner + 1, 0)
        margin[region] = np.where(known, buffer, 0)
        distance[region] = np.where(known, best, 0)
        missing[region] = known & ~acceptable
    return labels, margin, distance, missing


def overlay(raster: Raster, classes: np.ndarray, swatches: list[Swatch], radius: float, ambiguous: np.ndarray) -> np.ndarray:
    masks = [(classes == index + 1, COLORS[index], 0.56) for index in range(len(swatches))]
    masks.append((ambiguous, (213, 204, 165), 0.25))
    image = Image.fromarray(symmetric_overlay(raster.rgb, masks, support=raster.support))
    pen = ImageDraw.Draw(image)
    for index, swatch in enumerate(swatches):
        x, y = swatch.point
        pen.ellipse((x - radius, y - radius, x + radius, y + radius), outline=COLORS[index], width=3)
        pen.text(
            (x + radius + 4, y - 5), swatch.label,
            fill=(252, 245, 220), stroke_width=2, stroke_fill=(38, 36, 36),
        )
    return np.asarray(image)


COLORS_COLUMNS = [
    "zone", "qualified_px", "area", "region_fraction", "median_delta_e76",
    "median_color_margin", "swatch_pixels", "swatch_spread", "sample_l", "sample_a", "sample_b",
]


def measure(raster: Raster, params: dict) -> tuple[dict, np.ndarray, list[list]]:
    image = to_lab(raster.rgb)
    swatches = [swatch_for(raster, image, spec, params["radius"]) for spec in params["swatches"]]
    classes, margin, distance, ambiguous = choices_by_color(image, raster.support, swatches, params)
    total = int(raster.support.sum())
    rows, zones = [], []
    for index, swatch in enumerate(swatches, 1):
        mask = classes == index
        count = int(mask.sum())
        fraction = safe_ratio(count, total)
        rows.append([
            swatch.label, count, count * raster.plane.pixel_area, fraction,
            float(np.median(distance[mask])) if count else None,
            float(np.median(margin[mask])) if count else None,
            swatch.samples, swatch.spread, *swatch.center_lab.tolist(),
        ])
        zones.append({"label": swatch.label, "pixels": count, "coverage": fraction, "swatch_rgb": swatch.rgb})
    unknown = int(np.count_nonzero(raster.support & (classes == 0)))
    weak = int(ambiguous.sum())
    notices = ["Zone measurements use display-referred CIE L*a*b* differences from the selected swatches under the current lighting adjustment."]
    if unknown:
        notices.append("Pixels far from every swatch or close to competing swatches are unclassified.")
    result = {
        "metrics": {
            "qualified_fraction": safe_ratio(total - unknown, total),
            "ambiguous_pixels": weak,
            "distant_pixels": unknown - weak,
            "swatch_count": len(swatches),
        },
        "notices": notices,
        "unit": raster.plane.unit,
        "objects": zones,
    }
    return result, overlay(raster, classes, swatches, params["radius"], ambiguous), rows
