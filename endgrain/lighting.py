from dataclasses import dataclass
from typing import Any

import numpy as np

from .captures import decoded, reference_for
from .database import Database
from .errors import WorkbenchError
from .imaging import linear_to_srgb, luminance, masked_smooth, quantile, safe_ratio, srgb_to_linear
from .validation import document, fields, number, string


@dataclass(frozen=True)
class LightingResult:
    rgb: np.ndarray
    valid: np.ndarray
    gain: np.ndarray
    clipped: np.ndarray
    summary: dict


def normalize_lighting(raw: Any, db: Database, width: int, height: int) -> dict | None:
    if raw is None:
        return None
    req = document(raw, "illumination")
    fields(req, {"reference_id", "strength", "max_gain", "blur_px"}, "illumination")
    reference_id = string(req.get("reference_id"), "reference image", 64, required=True)
    reference_for(db, reference_id, width, height)
    return {
        "reference_id": reference_id,
        "strength": number(req.get("strength", 1), "correction strength", 0, 1),
        "max_gain": number(req.get("max_gain", 2), "maximum gain", 1, 4),
        "blur_px": number(req.get("blur_px", 18), "reference smoothing", 2, 100),
    }


def apply_flat_field(
    sample: np.ndarray,
    sample_support: np.ndarray,
    reference: np.ndarray,
    reference_support: np.ndarray,
    params: dict,
) -> LightingResult:
    linear_reference = luminance(reference, linear=True)
    referenced = reference_support & (linear_reference >= 0.012) & (linear_reference <= 0.97)
    blurred, support = masked_smooth(linear_reference, referenced, params["blur_px"])
    valid = (support >= 0.75) & referenced
    paired = valid & sample_support
    if np.count_nonzero(paired) < 96:
        raise WorkbenchError(
            "There are too few reliable reference pixels shared with the sample.",
            "reference_id",
        )
    anchor = float(np.median(blurred[paired]))
    raw_gain = anchor / np.clip(blurred, 0.005, 1.0)
    cap = params["max_gain"]
    gain = np.where(valid, 1 + (np.clip(raw_gain, 1 / cap, cap) - 1) * params["strength"], 1)
    adjusted = srgb_to_linear(sample) * gain[..., None]
    clipped = np.any(adjusted > 1, axis=2) & sample_support
    rgb = linear_to_srgb(adjusted)
    rgb[~valid] = sample[~valid]
    numerator = int(np.count_nonzero(paired))
    support_pixels = int(np.count_nonzero(sample_support))
    summary = {
        "shared_pixels": numerator,
        "sample_pixels": support_pixels,
        "reference_coverage": safe_ratio(numerator, support_pixels),
        "reference_anchor_linear": anchor,
        "mean_gain": float(np.mean(gain[paired])),
        "gain_p05": quantile(gain[paired], 5),
        "gain_p95": quantile(gain[paired], 95),
        "gain_cap_pixels": int(np.count_nonzero(((raw_gain > cap) | (raw_gain < 1 / cap)) & paired)),
        "would_clip_pixels": int(np.count_nonzero(clipped)),
    }
    return LightingResult(rgb, valid, gain, clipped, summary)


def field_for(
    db: Database, sample: np.ndarray, support: np.ndarray, adjustment: dict,
) -> tuple[LightingResult, dict]:
    reference, reference_support, record = decoded(db, adjustment["reference_id"])
    result = apply_flat_field(sample, support, reference, reference_support, adjustment)
    return result, record
