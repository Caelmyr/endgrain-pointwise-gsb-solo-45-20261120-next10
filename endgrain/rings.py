from dataclasses import dataclass
from math import ceil
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from scipy.signal import find_peaks

from .errors import WorkbenchError
from .geometry import Plane, Raster, pixel_point
from .imaging import luminance, safe_ratio, symmetric_overlay
from .validation import choice, document, fields, integer, list_of, number


@dataclass(frozen=True)
class Signal:
    points: np.ndarray
    distance: np.ndarray
    population: np.ndarray
    raw: np.ndarray
    smoothed: np.ndarray
    response: np.ndarray
    valid: np.ndarray
    half_band: float


def normalize_params(values: Any, plane: Plane) -> dict:
    body = document(values, "annual ring settings")
    fields(body, {"endpoints", "band", "prominence", "minimum_gap", "polarity", "manual_crossings"}, "annual ring")
    w, h = plane.width, plane.height
    default = [[w * 0.46, h * 0.49], [w * 0.86, h * 0.54]]
    ends = [
        pixel_point(point, f"transect endpoint {idx}", w, h)
        for idx, point in enumerate(list_of(body.get("endpoints", default), "transect endpoints", 2, 2), 1)
    ]
    length = float(np.linalg.norm(np.asarray(ends[1]) - ends[0]))
    if length < 20:
        raise WorkbenchError("Place endpoints at least twenty measuring pixels apart.", "endpoints")
    crossings = list_of(body.get("manual_crossings", []), "manual ring crossings", 0, 120)
    return {
        "endpoints": ends,
        "band": integer(body.get("band", 7), "transect band in pixels", 1, 41),
        "prominence": number(body.get("prominence", 0.024), "boundary prominence", 0.003, 0.45),
        "minimum_gap": integer(body.get("minimum_gap", 6), "candidate minimum separation", 2, 200),
        "polarity": choice(body.get("polarity", "dark"), "boundary contrast", ("dark", "light")),
        "manual_crossings": sorted({number(value, "crossing fraction", 0, 1) for value in crossings}),
    }


def runs_of(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True))


def finite_smooth(for_signal: np.ndarray, valid: np.ndarray, sigma: float) -> np.ndarray:
    output = np.full(for_signal.shape, np.nan, dtype=np.float64)
    for start, end in runs_of(valid):
        if end - start >= 3:
            output[start:end] = ndi.gaussian_filter1d(for_signal[start:end], sigma, mode="nearest")
    return output


def transect_signal(raster: Raster, config: dict) -> Signal:
    p1, p2 = np.asarray(config["endpoints"], dtype=np.float64)
    delta = p2 - p1
    pixel_length = float(np.linalg.norm(delta))
    target_count = max(2, ceil(pixel_length) + 1)
    points = np.linspace(p1, p2, target_count)
    normal = np.array([-delta[1], delta[0]]) / pixel_length
    half = (config["band"] - 1) / 2
    offsets = np.linspace(-half, half, config["band"])
    bands = points[:, None, :] + offsets[None, :, None] * normal
    coords = np.stack([bands[..., 1], bands[..., 0]])
    coverage = ndi.map_coordinates(raster.support.astype(np.float32), coords, order=1, mode="constant", cval=0, prefilter=False)
    usable = coverage >= 0.9999
    lightness = luminance(raster.rgb)
    samples = ndi.map_coordinates(lightness, coords, order=1, mode="constant", cval=0, prefilter=False)
    votes = usable.sum(axis=1)
    enough = votes * 2 >= len(offsets)
    raw = np.empty(target_count, dtype=np.float64)
    raw.fill(np.nan)
    for index in np.flatnonzero(enough):
        raw[index] = np.median(samples[index, usable[index]])
    near = finite_smooth(raw, enough, 1)
    background = finite_smooth(raw, enough, max(2, min(config["minimum_gap"] * 0.66, 20)))
    response = background - near if config["polarity"] == "dark" else near - background
    distance = np.linspace(0, raster.plane.distance(p1, p2), target_count)
    return Signal(points, distance, votes, raw, near, response, np.isfinite(response), half)


def auto_crossings(signal: Signal, config: dict) -> list[dict]:
    candidates = []
    for left, right in runs_of(signal.valid):
        if right - left < 5:
            continue
        peaks, properties = find_peaks(
            signal.response[left:right],
            distance=config["minimum_gap"],
            prominence=config["prominence"],
        )
        for local_index, prominence in zip(peaks, properties["prominences"], strict=True):
            position = int(left + local_index)
            candidates.append({"sample": position, "prominence": float(prominence), "origin": "suggested"})
    return candidates


def manual_crossings(signal: Signal, config: dict) -> list[dict]:
    result = []
    seen = set()
    for value in config["manual_crossings"]:
        index = int(round(value * (len(signal.points) - 1)))
        if index in seen:
            continue
        if not signal.valid[index]:
            raise WorkbenchError("A manual crossing falls on an unsupported band position.", "manual_crossings")
        result.append({"sample": index, "prominence": None, "origin": "manual"})
        seen.add(index)
    return result


def add_positions(signal: Signal, events: list[dict]) -> list[dict]:
    prior = None
    out = []
    for item in events:
        index = item["sample"]
        pair_ok = prior is not None and signal.valid[prior:index + 1].all()
        step = float(signal.distance[index] - signal.distance[prior]) if pair_ok else None
        out.append({
            **item,
            "x": float(signal.points[index, 0]),
            "y": float(signal.points[index, 1]),
            "distance": float(signal.distance[index]),
            "gap_from_previous": step,
            "gap_has_missing_support": prior is not None and not pair_ok,
        })
        prior = index
    return out


def ring_preview(raster: Raster, signal: Signal, markings: list[dict]) -> np.ndarray:
    output = Image.fromarray(symmetric_overlay(raster.rgb, [], support=raster.support))
    pen = ImageDraw.Draw(output, mode="RGB")
    start, end = signal.points[[0, -1]]
    heading = end - start
    norm = np.linalg.norm(heading)
    tangent = np.array([-heading[1], heading[0]]) / norm
    size = max(3, int(round(max(raster.plane.width, raster.plane.height) / 360)))
    flank = max(4, signal.half_band)
    for side in (-1, 1):
        a, b = start + side * tangent * flank, end + side * tangent * flank
        pen.line([tuple(a), tuple(b)], fill=(232, 200, 129), width=1)
    pen.line([tuple(start), tuple(end)], fill=(231, 232, 220), width=size)
    for a, b in runs_of(~signal.valid):
        pen.line([tuple(signal.points[a]), tuple(signal.points[b - 1])], fill=(205, 102, 75), width=size + 2)
    for mark in markings:
        point = signal.points[mark["sample"]]
        a, b = point - tangent * (flank + 5), point + tangent * (flank + 5)
        color = (117, 214, 181) if mark["origin"] == "manual" else (255, 198, 97)
        pen.line([tuple(a), tuple(b)], fill=color, width=size + 1)
    for x, y in (start, end):
        r = size + 3
        pen.ellipse((x - r, y - r, x + r, y + r), fill=(252, 234, 183), outline=(55, 40, 28), width=2)
    return np.asarray(output)


def ring_rows(signal: Signal, crossings: list[dict]) -> list[list]:
    events = {entry["sample"]: entry for entry in crossings}
    rows = []
    for index, point in enumerate(signal.points):
        event = events.get(index, {})
        valid = bool(signal.valid[index])
        rows.append([
            index, float(point[0]), float(point[1]), float(signal.distance[index]),
            int(signal.population[index]), valid,
            float(signal.raw[index]) if valid else None,
            float(signal.response[index]) if valid else None,
            event.get("origin"),
            event.get("gap_from_previous"),
        ])
    return rows


RING_COLUMNS = [
    "sample", "x_px", "y_px", "distance", "supported_across_band", "usable",
    "median_luminance", "boundary_response", "crossing", "width_from_previous",
]


def measure(raster: Raster, params: dict) -> tuple[dict, np.ndarray, list[list]]:
    signal = transect_signal(raster, params)
    suggestions = auto_crossings(signal, params)
    crossings = manual_crossings(signal, params) if params["manual_crossings"] else suggestions
    marks = add_positions(signal, crossings)
    widths = [item["gap_from_previous"] for item in marks if item["gap_from_previous"] is not None]
    selected = np.linspace(0, len(signal.points) - 1, min(len(signal.points), 720), dtype=int)
    messages = ["Crossings are contrast candidates. Inspect the overlays before interpreting counts as annual growth."]
    if (~signal.valid).any():
        messages.append("Intervals crossing unsupported image samples are excluded from mean widths.")
    output = {
        "metrics": {
            "transect_length": float(signal.distance[-1]),
            "crossings": len(marks),
            "widths": len(widths),
            "mean_width": float(np.mean(widths)) if widths else None,
            "median_width": float(np.median(widths)) if widths else None,
            "valid_fraction": safe_ratio(int(signal.valid.sum()), len(signal.valid)),
        },
        "notices": messages,
        "chart": {
            "kind": "profile",
            "x": signal.distance[selected].tolist(),
            "y": [float(signal.raw[index]) if signal.valid[index] else None for index in selected],
            "markers": [item["distance"] for item in marks],
            "x_label": "Distance (" + raster.plane.unit + ")",
            "y_label": "Luminance",
        },
        "crossings": marks,
        "unit": raster.plane.unit,
    }
    return output, ring_preview(raster, signal, marks), ring_rows(signal, marks)
