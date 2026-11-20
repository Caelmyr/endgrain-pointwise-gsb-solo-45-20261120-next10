from dataclasses import dataclass
from math import cos, pi, sin
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from scipy.fft import rfft2, rfftfreq, fftfreq

from .geometry import Plane, Raster
from .imaging import luminance, masked_smooth, quantile, safe_ratio, symmetric_overlay
from .validation import document, fields, integer, number


@dataclass(frozen=True)
class FinishWindow:
    index: int
    bounds: tuple[int, int, int, int]
    coverage: float
    rms: float | None
    chatter_share: float | None
    peak_period: float | None
    peak_axis: float | None
    valid: bool
    reason: str
    radial_sum: np.ndarray | None

    def row(self) -> list:
        return [
            self.index,
            *self.bounds,
            self.coverage,
            self.valid,
            self.reason,
            self.rms,
            self.chatter_share,
            self.peak_period,
            self.peak_axis,
        ]


def normalize_params(raw: Any, plane: Plane) -> dict:
    body = document(raw, "finish settings")
    fields(body, {"window", "min_period", "max_period", "axis", "angle_band", "flag_share"}, "finish")
    minimum_period = number(body.get("min_period", 3), "shortest finish period", 2.5, 40)
    return {
        "window": integer(body.get("window", 72), "finish window pixels", 32, 256),
        "min_period": minimum_period,
        "max_period": number(body.get("max_period", max(18, minimum_period + 2)), "longest finish period", minimum_period + 0.1, 75),
        "axis": number(body.get("axis", 0), "tool mark axis in degrees", 0, 179.9999),
        "angle_band": number(body.get("angle_band", 20), "tool mark angle band", 5, 90),
        "flag_share": number(body.get("flag_share", 0.28), "flag band-power share", 0.025, 0.95),
    }


def tile_starts(size: int, window: int) -> list[int]:
    if size <= window:
        return [0]
    spans = list(range(0, size - window + 1, window // 2))
    if size - window - spans[-1] > window // 5:
        spans.append(size - window)
    return spans


def plane_remove(values: np.ndarray, support: np.ndarray) -> np.ndarray:
    y, x = np.mgrid[-1:1:complex(0, values.shape[0]), -1:1:complex(0, values.shape[1])]
    design = np.stack([np.ones(values.shape), x, y], axis=-1)
    fit, *_ = np.linalg.lstsq(design[support], values[support], rcond=None)
    return np.where(support, values - design @ fit, 0)


def axial_delta(angle: np.ndarray, axis: float) -> np.ndarray:
    return np.abs((angle - axis + pi / 2) % pi - pi / 2)


def frequency_templates(w: int, h: int, params: dict, plane: Plane) -> tuple:
    mm_x, mm_y = plane.mm_x or 1, plane.mm_y or 1
    average_step = (mm_x + mm_y) / 2
    fy = fftfreq(h, d=mm_y / average_step)
    fx = rfftfreq(w, d=mm_x / average_step)
    vx, vy = np.meshgrid(fx, fy)
    radius = np.hypot(vx, vy)
    line_axis = (np.arctan2(vy, vx) + pi / 2) % pi
    dx = min(abs(fx[1]), abs(fy[1]))
    eligible = (radius >= max(dx * 1.6, 1 / 90)) & (radius <= 0.44)
    in_band = (radius >= 1 / params["max_period"]) & (radius <= 1 / params["min_period"])
    aim = np.radians(params["axis"])
    relevant = eligible & in_band & (axial_delta(line_axis, aim) <= np.radians(params["angle_band"]))
    multiplicity = np.full(fx.shape, 2, dtype=np.float64)
    multiplicity[0] = 1
    if w % 2 == 0:
        multiplicity[-1] = 1
    window = np.outer(np.hanning(h), np.hanning(w))
    return window, radius, line_axis, eligible, relevant, multiplicity, average_step


def analyze_window(
    order: int,
    bounds: tuple[int, int, int, int],
    gray: np.ndarray,
    region: np.ndarray,
    frequency: tuple,
    bins: np.ndarray,
) -> FinishWindow:
    coverage = float(region.mean())
    if coverage < 0.86 or min(gray.shape) < 24:
        return FinishWindow(order, bounds, coverage, None, None, None, None, False, "insufficient support", None)
    window, frequencies, angles, eligible, relevant, multiplicity, average_step = frequency
    detrended = plane_remove(gray, region)
    whole, confidence = masked_smooth(detrended, region, 1.6)
    signal = np.where(region, detrended, np.where(confidence > 0.15, whole, 0)) * window
    spectrum = abs(rfft2(signal, workers=1)) ** 2 * multiplicity
    total = float(spectrum[eligible].sum())
    rms = float(np.sqrt(np.mean(detrended[region] ** 2)))
    if total <= 1e-8:
        return FinishWindow(order, bounds, coverage, rms, None, None, None, False, "no usable texture signal", None)
    if not relevant.any():
        return FinishWindow(order, bounds, coverage, rms, None, None, None, False, "period or angle unresolved at this window size", None)
    radial, _ = np.histogram(frequencies[eligible], bins=bins, weights=(spectrum[eligible] / total))
    band_fraction = float(spectrum[relevant].sum() / total)
    peak_index = np.unravel_index(np.argmax(np.where(relevant, spectrum, -1)), spectrum.shape)
    if spectrum[peak_index] > total * 0.003:
        period = float(average_step / frequencies[peak_index])
        angle = float(np.degrees(angles[peak_index]))
    else:
        period = None
        angle = None
    return FinishWindow(order, bounds, coverage, rms, band_fraction, period, angle, True, "", radial)


def measure_windows(raster: Raster, params: dict, bins: np.ndarray) -> list[FinishWindow]:
    image = luminance(raster.rgb)
    allowed = raster.support
    h, w = allowed.shape
    window = params["window"]
    templates = {}
    rows = []
    for y in tile_starts(h, window):
        for x in tile_starts(w, window):
            h_here, w_here = min(window, h - y), min(window, w - x)
            key = w_here, h_here
            if key not in templates:
                templates[key] = frequency_templates(w_here, h_here, params, raster.plane)
            block = np.s_[y:y + h_here, x:x + w_here]
            rows.append(analyze_window(
                len(rows) + 1, (x, y, x + w_here, y + h_here), image[block],
                allowed[block], templates[key], bins,
            ))
    return rows


def overlay(raster: Raster, tiles: list[FinishWindow], settings: dict) -> np.ndarray:
    preview = Image.fromarray(symmetric_overlay(raster.rgb, [], support=raster.support)).convert("RGBA")
    marking = Image.new("RGBA", preview.size)
    pen = ImageDraw.Draw(marking)
    flagged = sorted(
        (item for item in tiles if item.valid and (item.chatter_share or 0) >= settings["flag_share"]),
        key=lambda item: item.chatter_share,
    )
    for tile in flagged:
        x, y, right, bottom = tile.bounds
        fraction = tile.chatter_share or 0
        rgb = (251, 168, 91)
        pen.rectangle(tile.bounds, fill=(*rgb, int(34 + 116 * fraction)), outline=(*rgb, 205), width=2)
        if tile.peak_axis is None:
            continue
        ox, oy = (x + right) / 2, (y + bottom) / 2
        radius = min(right - x, bottom - y) * 0.32
        angle = np.radians(tile.peak_axis)
        px, py = radius * cos(angle), radius * sin(angle)
        pen.line([(ox - px, oy - py), (ox + px, oy + py)], fill=(254, 220, 150, 255), width=2)
    return np.asarray(Image.alpha_composite(preview, marking).convert("RGB"))


def spectrum_chart(ready: list[FinishWindow], bins: np.ndarray) -> dict:
    average = np.mean([row.radial_sum for row in ready], axis=0) if ready else np.zeros(len(bins) - 1)
    return {
        "kind": "bars",
        "rows": [
            {"x": float((a + b) / 2), "value": float(height)}
            for a, b, height in zip(bins[:-1], bins[1:], average, strict=True)
        ],
        "x_label": "Spatial frequency (cycles per measuring pixel)",
        "y_label": "Mean power fraction",
    }


FINISH_COLUMNS = [
    "window", "x0", "y0", "x1", "y1", "coverage", "valid", "reason", "rms", "band_power_share", "peak_period", "axis",
]


def measure(raster: Raster, params: dict) -> tuple[dict, np.ndarray, list[list]]:
    bins = np.linspace(0.012, 0.45, 30)
    tiles = measure_windows(raster, params, bins)
    ready = [row for row in tiles if row.valid]
    fraction = np.array([row.chatter_share for row in ready])
    peaks = np.array([row.peak_period for row in ready if row.peak_period is not None])
    contrast = np.array([row.rms for row in ready])
    notices = ["Warm outlines mark overlapping tiles dominated by periodic texture along the selected tool axis."]
    if len(ready) < len(tiles):
        notices.append("Some windows lack sufficient support, texture or resolution and are omitted from the finish summary.")
    result = {
        "metrics": {
            "analyzable_windows": len(ready),
            "total_windows": len(tiles),
            "flagged_window_fraction": safe_ratio(int(np.count_nonzero(fraction >= params["flag_share"])), len(ready)),
            "band_power_p90": quantile(fraction, 90),
            "peak_period_median": quantile(peaks, 50),
            "rms_p50": quantile(contrast, 50),
        },
        "notices": notices,
        "unit": raster.plane.unit,
        "chart": spectrum_chart(ready, bins),
    }
    return result, overlay(raster, tiles, params), [item.row() for item in tiles]
