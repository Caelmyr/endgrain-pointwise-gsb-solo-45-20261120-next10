from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from .geometry import Raster
from .imaging import luminance, masked_smooth, safe_ratio, symmetric_overlay


@dataclass(frozen=True)
class QualityReading:
    clip_shadow: np.ndarray
    clip_highlight: np.ndarray
    glare: np.ndarray
    soft: np.ndarray
    metrics: dict
    rows: list[list]
    rgb_histogram: dict[str, list[int]]
    notices: list[str]


def separation_masks(rgb: np.ndarray, allowed: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    maximum = rgb.max(axis=2)
    minimum = rgb.min(axis=2)
    shadows = allowed & (maximum < 12)
    clipped = allowed & (maximum >= 250)
    glare = clipped & (minimum >= 222) & ((maximum - minimum) < 39)
    return shadows, clipped, glare


def window_sharpness(light: np.ndarray, allowed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    clean, support = masked_smooth(light, allowed, 0.65)
    curvature = ndi.laplace(clean, mode="nearest")
    local_info, agree = masked_smooth(curvature ** 2, allowed & (support >= 0.93), 3.0)
    return np.sqrt(local_info), agree


def bins_for(rgb: np.ndarray, allowed: np.ndarray, count: int = 32) -> dict[str, list[int]]:
    elements = rgb[allowed]
    if not elements.size:
        return {key: [0] * count for key in ("r", "g", "b")}
    return {
        channel: np.histogram(elements[:, index], bins=count, range=(0, 256))[0].tolist()
        for index, channel in enumerate(("r", "g", "b"))
    }


def tile_measures(
    raster: Raster,
    shadows: np.ndarray,
    highlights: np.ndarray,
    glare: np.ndarray,
    energy: np.ndarray,
    soft: np.ndarray,
    grid: int,
) -> list[list]:
    height, width = raster.support.shape
    ys = np.linspace(0, height, num=grid + 1, dtype=int)
    xs = np.linspace(0, width, num=grid + 1, dtype=int)
    output = []
    for y in range(grid):
        for x in range(grid):
            region = np.s_[ys[y]:ys[y + 1], xs[x]:xs[x + 1]]
            usable = raster.support[region]
            count = int(np.count_nonzero(usable))
            if not count:
                continue
            output.append([
                y + 1,
                x + 1,
                int(xs[x]),
                int(ys[y]),
                count,
                int(usable.size),
                float(shadows[region].sum() / count),
                float(highlights[region].sum() / count),
                float(glare[region].sum() / count),
                float(np.median(energy[region][usable])),
                float(soft[region].sum() / count),
            ])
    return output


def measure_quality(raster: Raster, grid: int, softness_factor: float) -> QualityReading:
    allowed = raster.support
    shadow, highlight, glare = separation_masks(raster.rgb, allowed)
    light = luminance(raster.rgb)
    energy, confidence = window_sharpness(light, allowed)
    sharp_eligible = (confidence >= 0.80) & allowed & ~shadow & ~glare
    threshold = max(0.0015, float(np.median(energy[sharp_eligible])) * softness_factor) if sharp_eligible.any() else 0.0015
    soft = sharp_eligible & (energy < threshold)
    examined = int(allowed.sum())
    region_pixels = int(raster.inspected.sum())
    tonal_range = np.percentile(light[allowed], [5, 50, 95]) if examined else [None] * 3
    metrics = {
        "supported_pixels": examined,
        "region_pixels": region_pixels,
        "coverage": safe_ratio(examined, region_pixels),
        "shadow_fraction": safe_ratio(int(shadow.sum()), examined),
        "highlight_fraction": safe_ratio(int(highlight.sum()), examined),
        "glare_fraction": safe_ratio(int(glare.sum()), examined),
        "low_detail_fraction": safe_ratio(int(soft.sum()), int(sharp_eligible.sum())),
        "luminance_p05": tonal_range[0],
        "luminance_median": tonal_range[1],
        "luminance_p95": tonal_range[2],
        "low_detail_threshold": threshold,
    }
    notices = []
    if examined < 96:
        notices.append("Very little of this region is supported by source pixels.")
    if (metrics["glare_fraction"] or 0) > 0.025:
        notices.append("Specular-looking areas may conceal anatomy.")
    if (metrics["shadow_fraction"] or 0) > 0.04:
        notices.append("Deep shadows may conceal grain or cracks.")
    if (metrics["low_detail_fraction"] or 0) > 0.24:
        notices.append("More than one quarter of the eligible area has low local detail.")
    rows = tile_measures(raster, shadow, highlight, glare, energy, soft, grid)
    return QualityReading(shadow, highlight, glare, soft, metrics, rows, bins_for(raster.rgb, allowed), notices)


QUALITY_COLUMNS = [
    "row", "column", "x_px", "y_px", "support_px", "tile_px",
    "shadow_fraction", "highlight_fraction", "glare_fraction", "sharpness", "low_detail_fraction",
]


def quality_preview(raster: Raster, reading: QualityReading, mask: str) -> np.ndarray:
    chosen = {
        "all": [
            (reading.soft, (87, 167, 233), 0.56),
            (reading.clip_shadow, (160, 103, 202), 0.72),
            (reading.glare, (248, 208, 86), 0.7),
        ],
        "shadows": [(reading.clip_shadow, (160, 103, 202), 0.76)],
        "glare": [(reading.glare, (248, 208, 86), 0.76)],
        "softness": [(reading.soft, (87, 167, 233), 0.6)],
        "plain": [],
    }
    return symmetric_overlay(raster.rgb, chosen[mask], support=raster.support)


def prepare_download_image(raster: Raster, variant: str) -> Image.Image:
    if variant == "clean":
        alpha = raster.supported.astype(np.uint8) * 255
        return Image.fromarray(np.dstack((raster.rgb, alpha)), "RGBA")
    if variant == "region":
        alpha = raster.support.astype(np.uint8) * 255
        return Image.fromarray(np.dstack((raster.rgb, alpha)), "RGBA")
    return Image.fromarray(symmetric_overlay(raster.rgb, [], support=raster.support), "RGB")
