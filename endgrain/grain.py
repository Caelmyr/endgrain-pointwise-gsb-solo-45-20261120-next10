from dataclasses import dataclass
from math import atan2, cos, degrees, hypot, pi, radians, sin
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

from .geometry import Plane, Raster
from .imaging import luminance, masked_smooth, safe_ratio, symmetric_overlay
from .validation import document, fields, integer, number

GROUP_COLORS = ((224, 158, 74), (86, 166, 186))
GROUP_HEX = ("#e09e4a", "#56a6ba")
GROUP_LABELS = ("A", "B")
AMBIGUOUS_COLOR = (142, 145, 134)
GROUP_BINS = 180
SEED_SMOOTHING = 3.0
VALLEY_RATIO = 0.80


@dataclass(frozen=True)
class OrientationField:
    angle: np.ndarray
    coherence: np.ndarray
    energy: np.ndarray
    support: np.ndarray
    eligible: np.ndarray
    classified: np.ndarray


@dataclass(frozen=True)
class DirectionGroup:
    index: int
    label: str
    angle: float
    share: float
    pixels: int
    concentration: float
    color: tuple[int, int, int]
    hex: str

    def json(self) -> dict:
        return {
            "index": self.index,
            "label": self.label,
            "angle": self.angle,
            "share": self.share,
            "pixels": self.pixels,
            "concentration": self.concentration,
            "color": self.hex,
        }


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
    group: str


def normalize_params(raw: Any, plane: Plane) -> dict:
    body = document(raw, "grain settings")
    fields(
        body,
        {
            "window", "grid", "minimum_coherence", "minimum_energy", "nominal_axis", "tolerance",
            "group_separation", "assignment_band", "ambiguity_margin", "minimum_group_share",
        },
        "grain",
    )
    return {
        "window": number(body.get("window", 4), "tensor integration radius", 1.0, 32.0),
        "grid": integer(body.get("grid", 36), "orientation tile width", 8, 200),
        "minimum_coherence": number(body.get("minimum_coherence", 0.38), "minimum directional coherence", 0.02, 0.99),
        "minimum_energy": number(body.get("minimum_energy", 0.012), "minimum contrast energy", 0.001, 0.75),
        "nominal_axis": number(body.get("nominal_axis", 0), "comparison axis in degrees", 0, 179.9999),
        "tolerance": number(body.get("tolerance", 20), "comparison tolerance in degrees", 2, 90),
        "group_separation": number(body.get("group_separation", 16), "direction group separation in degrees", 5, 90),
        "assignment_band": number(body.get("assignment_band", 24), "group assignment band in degrees", 3, 90),
        "ambiguity_margin": number(body.get("ambiguity_margin", 8), "group ambiguity margin in degrees", 1, 45),
        "minimum_group_share": number(body.get("minimum_group_share", 0.004), "smallest direction group share", 0.0005, 0.4),
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


def _valley(smooth: np.ndarray, first: int, second: int, ratio: float = VALLEY_RATIO) -> bool:
    """True when two circular histogram peaks are separated by a real trough."""
    size = len(smooth)
    forward = np.arange(first, first + size + 1) % size
    stop = int(np.flatnonzero(forward == second)[0]) + 1
    forward = forward[:stop]
    backward = np.arange(second, second + size + 1) % size
    stop = int(np.flatnonzero(backward == first)[0]) + 1
    backward = backward[:stop]
    arc = forward if len(forward) <= len(backward) else backward
    between = smooth[arc[1:-1]]
    if not between.size:
        return True
    return bool(float(between.min()) <= ratio * min(float(smooth[first]), float(smooth[second])))


def _circular_gap(first: int, second: int, bins: int = GROUP_BINS) -> int:
    offset = abs(int(first) - int(second))
    return min(offset, bins - offset)


def _seed_centers(angles: np.ndarray, weights: np.ndarray, separation: float) -> list[float]:
    histogram, _ = np.histogram(angles, bins=GROUP_BINS, range=(0, pi), weights=weights)
    smooth = ndi.gaussian_filter1d(histogram.astype(np.float64), SEED_SMOOTHING, mode="wrap")
    if not np.isfinite(smooth).all() or smooth.max() <= 0:
        return []
    peaks = np.flatnonzero((smooth > np.roll(smooth, 1)) & (smooth >= np.roll(smooth, -1)))
    if not peaks.size:
        return []
    ordered = peaks[np.argsort(smooth[peaks])[::-1]]
    minimum_bins = max(2, int(round(degrees(separation))))
    # A second direction is only credible as a separate lobe: far enough from
    # the leading peak and separated from it by a real trough. A shoulder on
    # one broad distribution is not a second grain direction.
    chosen = [int(ordered[0])]
    for candidate in ordered[1:]:
        candidate = int(candidate)
        if _circular_gap(candidate, chosen[0]) < minimum_bins:
            continue
        if _valley(smooth, chosen[0], candidate):
            chosen.append(candidate)
            break
    centers = []
    window = max(radians(5), separation / 2)
    for peak in chosen:
        seed = radians((peak + 0.5) * 180.0 / GROUP_BINS)
        near = axial_distance(angles, seed) <= window
        if np.count_nonzero(near) >= 16:
            refined, _ = axial_average(angles[near], weights[near])
            if refined is not None:
                seed = radians(refined)
        centers.append(seed)
    return centers


def _assign(
    angles: np.ndarray, coherence: np.ndarray, centers: list[float], params: dict,
) -> list[np.ndarray]:
    band = radians(params["assignment_band"])
    margin = radians(params["ambiguity_margin"])
    floor = max(params["minimum_coherence"], 0.42)
    if len(centers) == 1:
        distance = axial_distance(angles, centers[0])
        return [(distance <= band) & (coherence >= floor)]
    separation = float(axial_distance(np.asarray([centers[0]]), centers[1])[0])
    limit = min(band, max(radians(2), separation / 2 - margin / 2))
    first = axial_distance(angles, centers[0])
    second = axial_distance(angles, centers[1])
    near = np.minimum(first, second) <= limit
    clear = near & (np.abs(first - second) >= margin) & (coherence >= floor)
    return [clear & (first <= second), clear & (second < first)]


def _merge_centers(centers: list[float], separation: float) -> list[float]:
    if len(centers) < 2:
        return centers
    gap = float(axial_distance(np.asarray([centers[0]]), centers[1])[0])
    if gap < separation * 0.75:
        return [centers[0]]
    return centers


def _constrained_average(
    angles: np.ndarray, weights: np.ndarray, start: float, limit: float,
) -> float:
    near = axial_distance(angles, start) <= limit
    if np.count_nonzero(near) < 16:
        return start
    refined, _ = axial_average(angles[near], weights[near])
    return radians(refined) if refined is not None else start


def _local_agreement(
    field: OrientationField, mask: np.ndarray, sigma: float = 2.0,
) -> np.ndarray:
    """Fraction of nearby classified pixels that point the same way.

    Where two textures overlap, the tensor orientation jumps between the two
    directions pixel by pixel. A single direction is locally consistent, so
    this separates mixed areas from genuinely supported evidence.
    """
    unit = np.exp(2j * field.angle.astype(np.float64))
    real = ndi.gaussian_filter(np.where(mask, unit.real, 0).astype(np.float64), sigma, mode="constant")
    imaginary = ndi.gaussian_filter(np.where(mask, unit.imag, 0).astype(np.float64), sigma, mode="constant")
    number = ndi.gaussian_filter(mask.astype(np.float64), sigma, mode="constant")
    total = np.hypot(real, imaginary) / 2
    agreement = np.zeros(mask.shape, dtype=np.float64)
    np.divide(total, number, out=agreement, where=number > 1e-6)
    return agreement


def _evidence_support(
    field: OrientationField, agreement: np.ndarray, floor: float,
) -> np.ndarray:
    return (field.coherence >= floor) & (agreement >= max(0.35, floor * 0.8))


def direction_groups(
    field: OrientationField, params: dict,
) -> tuple[list[DirectionGroup], list[np.ndarray], np.ndarray, dict]:
    shape = field.angle.shape
    classified = field.classified
    measurable = int(field.eligible.sum())
    classified_count = int(classified.sum())
    assigned = np.zeros(shape, dtype=bool)
    ambiguous = np.zeros(shape, dtype=bool)
    if classified_count < 48 or measurable == 0:
        ambiguous |= classified
        return [], [], ambiguous, {
            "measurable": measurable, "classified": classified_count,
            "assigned": 0, "ambiguous": classified_count,
        }
    flat = np.flatnonzero(classified)
    angles = field.angle.ravel()[flat].astype(np.float64)
    weights = np.square(field.coherence.ravel()[flat].astype(np.float64))
    coherence = field.coherence.ravel()[flat]
    separation = radians(params["group_separation"])
    centers = _merge_centers(_seed_centers(angles, weights, separation), separation)
    if not centers:
        ambiguous |= classified
        return [], [], ambiguous, {
            "measurable": measurable, "classified": classified_count,
            "assigned": 0, "ambiguous": classified_count,
        }
    floor = max(params["minimum_coherence"], 0.42)
    agreement = _local_agreement(field, classified)
    support = _evidence_support(field, agreement, floor).ravel()[flat]

    def classify(seeds: list[float]) -> list[np.ndarray]:
        # Clear evidence only: near a direction, closer to it than the other,
        # coherent, and locally consistent. Everything else stays unresolved.
        raw = _assign(angles, coherence, seeds, params)
        return [mask & support for mask in raw]

    def weak_direction(masks: list[np.ndarray]) -> bool:
        shares = [float(mask.sum()) / measurable for mask in masks]
        return min(shares) < params["minimum_group_share"] or min(int(mask.sum()) for mask in masks) < 32

    masks = classify(centers)
    # A second direction that is only a handful of pixels is not evidence.
    if len(centers) == 2 and weak_direction(masks):
        centers = [centers[int(np.argmax([float(mask.sum()) for mask in masks]))]]
        masks = classify(centers)
    # Refine each center only on pixels already near it, so a faint group is
    # never pulled toward the dominant lobe.
    for _ in range(2):
        refined = [
            _constrained_average(angles, weights, center, max(radians(4), separation / 2))
            for center in centers
        ]
        merged = _merge_centers(refined, separation)
        if len(merged) != len(centers):
            centers = merged
            masks = classify(centers)
            break
        centers = refined
        masks = classify(centers)
        if len(centers) == 1:
            break
    groups = []
    full_masks = []
    for mask in masks:
        count = int(mask.sum())
        if count < 32 or count / measurable < params["minimum_group_share"]:
            continue
        angle, concentration = axial_average(angles[mask], weights[mask])
        if angle is None or (concentration or 0.0) < 0.45:
            continue
        full = np.zeros(shape, dtype=bool)
        full.ravel()[flat[mask]] = True
        groups.append(DirectionGroup(
            index=len(groups),
            label=GROUP_LABELS[len(groups)],
            angle=float(angle),
            share=count / measurable,
            pixels=count,
            concentration=float(concentration or 0.0),
            color=GROUP_COLORS[len(groups)],
            hex=GROUP_HEX[len(groups)],
        ))
        full_masks.append(full)
    for mask in full_masks:
        assigned |= mask
    # Every classified pixel that did not become clear group evidence is
    # unresolved. Ambiguous or mixed pixels must never inflate a group share.
    ambiguous.ravel()[flat] = True
    ambiguous &= ~assigned
    return groups, full_masks, ambiguous, {
        "measurable": measurable,
        "classified": classified_count,
        "assigned": int(np.count_nonzero(assigned)),
        "ambiguous": int(np.count_nonzero(ambiguous)),
    }


def summarize_tiles(
    raster: Raster, field: OrientationField, size: int,
    groups: list[DirectionGroup] | None = None, masks: list[np.ndarray] | None = None,
) -> list[TileDirection]:
    h, w = field.angle.shape
    groups = groups or []
    masks = masks or []
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
            group = ""
            chosen = eligible
            if count and groups and masks:
                counts = [int(np.count_nonzero(mask[domain])) for mask in masks]
                assigned = sum(counts)
                if assigned:
                    best = int(np.argmax(counts))
                    strongest = counts[best]
                    if strongest >= 8 and strongest >= 0.55 * assigned and strongest >= 0.30 * count:
                        chosen = masks[best][domain]
                        group = groups[best].label
                    elif sum(1 for value in counts if value) > 1:
                        group = "mixed"
            angle, concentration = axial_average(
                field.angle[domain][chosen], field.coherence[domain][chosen],
            )
            results.append(TileDirection(
                x=x,
                y=y,
                width=xs.stop - x,
                height=ys.stop - y,
                supported=total,
                classified=count,
                angle=angle,
                concentration=concentration,
                contrast=float(np.mean(field.energy[domain][chosen])) if np.any(chosen) else None,
                group=group,
            ))
    return results


def color_direction(angle: float) -> tuple[int, int, int]:
    return (
        int(157 + 72 * cos(angle * 2)),
        int(188 + 56 * cos(angle * 2 - 2 * pi / 3)),
        int(172 + 65 * cos(angle * 2 + 2 * pi / 3)),
    )


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def draw_legend(
    painter: ImageDraw.ImageDraw, lines: list[tuple[str, tuple[int, int, int] | None]],
    size: int,
) -> None:
    font = _font(size)
    widths = []
    for text, _ in lines:
        box = painter.textbbox((0, 0), text, font=font)
        widths.append(box[2] - box[0])
    width = max(widths, default=0) + 22
    height = 10 + len(lines) * (size + 5)
    painter.rectangle((8, 8, 8 + width, 8 + height), fill=(24, 31, 24, 218))
    for index, (text, color) in enumerate(lines):
        y = 13 + index * (size + 5)
        if color is not None:
            painter.rectangle((14, y + 1, 14 + size - 2, y + size - 3), fill=(*color, 255))
            painter.text((18 + size, y), text, font=font, fill=(247, 240, 211))
        else:
            painter.text((14, y), text, font=font, fill=(218, 225, 204))


def _tile_glyph(
    painter: ImageDraw.ImageDraw, note: TileDirection, raster: Raster, settings: dict,
    color: tuple[int, int, int], line_size: int,
) -> None:
    theta = radians(note.angle)
    dx, dy = raster.plane.mm_x or 1, raster.plane.mm_y or 1
    rx, ry = cos(theta) / dx, sin(theta) / dy
    scale = min(note.width, note.height) * 0.38 / max(hypot(rx, ry), 1e-6)
    scale *= max(0.26, note.concentration or 0)
    ox, oy = note.x + note.width / 2, note.y + note.height / 2
    endpoints = [(ox - rx * scale, oy - ry * scale), (ox + rx * scale, oy + ry * scale)]
    painter.line(endpoints, fill=(34, 41, 42), width=line_size + 2)
    painter.line(endpoints, fill=color, width=line_size)


def paint_orientation(
    raster: Raster, field: OrientationField, tiles: list[TileDirection], settings: dict,
    groups: list[DirectionGroup] | None = None,
) -> Image.Image:
    groups = groups or []
    composite = symmetric_overlay(
        raster.rgb,
        [(raster.support & ~field.classified, (65, 69, 71), 0.55)],
        support=raster.support,
    )
    preview = Image.fromarray(composite)
    painter = ImageDraw.Draw(preview, "RGBA")
    line_size = max(1, min(4, settings["grid"] // 18))
    palette = {group.label: group.color for group in groups}
    for note in tiles:
        if note.angle is None or note.classified / note.supported < 0.20:
            continue
        if note.group == "mixed":
            color = (150, 152, 140)
        else:
            color = palette.get(note.group) or color_direction(radians(note.angle))
        _tile_glyph(painter, note, raster, settings, color, line_size)
    if groups:
        size = max(12, min(22, preview.width // 58))
        lines = [("Supported directions", None)]
        for group in groups:
            lines.append((
                f"{group.label}  {group.angle:.1f}°  {group.share * 100:.1f}% of measurable",
                group.color,
            ))
        draw_legend(painter, lines, size)
    return preview


def evidence_overlay(
    base: Image.Image, group: DirectionGroup, mask: np.ndarray, ambiguous: np.ndarray,
) -> np.ndarray:
    visual = np.asarray(base.convert("RGB")).astype(np.float32)
    color = np.asarray(group.color, dtype=np.float32)
    if not np.any(mask):
        return np.asarray(base.convert("RGB"))
    # Keep the standard orientation overlay underneath so the evidence can be
    # read against the directional texture it came from.
    visual[mask] = 0.44 * visual[mask] + 0.56 * color
    if np.any(ambiguous):
        visual[ambiguous] = 0.82 * visual[ambiguous] + 0.18 * np.asarray(AMBIGUOUS_COLOR, dtype=np.float32)
    edge = mask & ~ndi.binary_erosion(mask, iterations=1, border_value=0)
    if np.any(edge):
        visual[edge] = 0.18 * visual[edge] + 0.82 * color
    preview = Image.fromarray(np.rint(np.clip(visual, 0, 255)).astype(np.uint8))
    painter = ImageDraw.Draw(preview, "RGBA")
    size = max(12, min(20, preview.width // 62))
    lines = [
        (
            f"Group {group.label} · {group.angle:.1f}° · {group.share * 100:.1f}% of measurable area"
            f" · {group.pixels:,} px",
            group.color,
        ),
        ("Tinted: clear evidence for this direction.", None),
        ("Gray: ambiguous or not attributed. Unsupported areas are dimmed.", None),
    ]
    draw_legend(painter, lines, size)
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
    "record", "x", "y", "width", "height", "supported", "classified",
    "angle", "concentration", "contrast",
    "group", "group_angle", "group_share", "group_pixels", "measurable_pixels",
]
GRAIN_HEADERS = [
    "record", "x", "y", "width", "height", "supported_px", "classified_px",
    "angle_deg", "concentration", "contrast",
    "group", "group_angle_deg", "group_share_of_measurable", "group_px", "measurable_px",
]


def grain_rows(
    tiles: list[TileDirection], groups: list[DirectionGroup], measurable: int,
) -> list[list]:
    by_label = {group.label: group for group in groups}
    rows = []
    for group in groups:
        rows.append({
            "record": "direction_group",
            "group": group.label,
            "angle": group.angle,
            "concentration": group.concentration,
            "group_angle": group.angle,
            "group_share": group.share,
            "group_pixels": group.pixels,
            "measurable_pixels": measurable,
        })
    for tile in tiles:
        group = by_label.get(tile.group)
        rows.append({
            "record": "orientation_tile",
            "x": tile.x,
            "y": tile.y,
            "width": tile.width,
            "height": tile.height,
            "supported": tile.supported,
            "classified": tile.classified,
            "angle": tile.angle,
            "concentration": tile.concentration,
            "contrast": tile.contrast,
            "group": tile.group,
            "group_angle": group.angle if group else None,
            "group_share": group.share if group else None,
            "group_pixels": group.pixels if group else None,
            "measurable_pixels": measurable,
        })
    return [[row.get(name) for name in GRAIN_COLUMNS] for row in rows]


def table_document(tiles: list[TileDirection], groups: list[DirectionGroup], measurable: int) -> dict:
    return {"columns": GRAIN_HEADERS, "keys": GRAIN_COLUMNS, "rows": grain_rows(tiles, groups, measurable)}


def measure(raster: Raster, params: dict) -> tuple[dict, np.ndarray, list[list], dict[str, bytes]]:
    from .imaging import encode_png

    raw = tensor(raster, params)
    groups, masks, ambiguous, counts = direction_groups(raw, params)
    tiles = summarize_tiles(raster, raw, params["grid"], groups, masks)
    angles = raw.angle[raw.classified]
    coherence = raw.coherence[raw.classified]
    average, resultant = axial_average(angles, coherence)
    in_tolerance = axial_distance(angles, radians(params["nominal_axis"])) <= radians(params["tolerance"])
    counted = int(len(angles))
    total = int(raster.support.sum())
    measurable = counts["measurable"]
    grouped = counts["assigned"]
    unresolved = measurable - grouped
    messages = [
        "An orientation is axial: angles 0° and 180° are the same. "
        "Gray areas have insufficient support, contrast or coherence."
    ]
    if groups:
        summary = " and ".join(
            f"{group.label} {group.angle:.1f}° ({group.share * 100:.1f}% of measurable area)"
            for group in groups
        )
        messages.append("Supported directions: " + summary + ".")
        if len(groups) >= 2 and resultant is not None and resultant < 0.35:
            messages.append(
                "The principal angle is a weighted mean across competing directions and may not "
                "match either group in the photograph."
            )
    elif counted:
        messages.append("No direction group has enough clear, consistent evidence.")
    if measurable and unresolved > 0.15 * measurable:
        messages.append(
            "Ambiguous or out-of-band pixels are excluded from the group shares rather than "
            "assigned to a direction."
        )
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
            "direction_groups": len(groups),
            "grouped_fraction": safe_ratio(grouped, measurable),
            "unresolved_fraction": safe_ratio(unresolved, measurable),
        },
        "notices": messages,
        "unit": "°",
        "chart": angle_distribution(angles),
        "groups": [group.json() for group in groups],
        "measurable_pixels": measurable,
        "classified_pixels": counts["classified"],
        "assigned_pixels": grouped,
        "ambiguous_pixels": counts["ambiguous"],
        "unresolved_pixels": unresolved,
    }
    assets = {}
    base = paint_orientation(raster, raw, tiles, params, groups)
    for group in groups:
        assets[f"group-{group.index}.png"] = encode_png(
            evidence_overlay(base, group, masks[group.index], ambiguous)
        )
    return output, np.asarray(base), table_document(tiles, groups, measurable), assets
