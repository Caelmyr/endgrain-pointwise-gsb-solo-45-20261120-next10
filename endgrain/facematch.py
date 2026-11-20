"""End-face re-photo matching.

The matcher looks for an old inspection photograph that could show the same
wood end face as a newly captured one.  Re-photographs are allowed to be
rotated, slightly cropped and differently exposed, so the search combines:

* illumination-normalized local texture,
* a rotation / translation / mild scale search with masked normalized
  cross-correlation on a shared canvas,
* vessel-pore correspondences whose geometry must stay consistent, and
* a small number of strong texture tile correspondences shown as evidence.

The verdict is deliberately conservative.  When the overlap is only a narrow
strip, when two candidates explain the image almost equally well, or when too
few landmarks line up, the answer is ``inconclusive`` rather than a forced
identification.  The machine never confirms a match on its own; an operator
records the final disposition separately.
"""

from dataclasses import dataclass
from math import cos, radians, sin
from threading import BoundedSemaphore

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy import ndimage as ndi
from scipy.signal import fftconvolve

from .database import Database
from .errors import Conflict
from .imaging import encode_png, luminance
from .pores import detect_components
from .workspaces import frame_for

# Two-stage search.  Coarse search ranks cheaply; only promising alignments
# are refined at full working resolution.
COARSE_LIMIT = 220
FINE_LIMIT = 420
COARSE_ANGLES = np.arange(0.0, 360.0, 24.0)
COARSE_SCALES = (0.94, 1.0, 1.06)
FINE_ANGLE_DELTA = 6.0
FINE_ANGLE_STEP = 2.0
FINE_SCALES = (0.98, 1.0, 1.02)
MAX_CANDIDATES = 12
PATCH = 31
HALF = PATCH // 2
NCC_FLOOR = 0.012

# Conservative evidence thresholds (tuned on synthetic re-photo pairs).
NCC_STRONG = 0.38
NCC_PLAUSIBLE = 0.24
PORE_MATCH_MIN = 3
PORE_STRONG = 5
TILE_STRONG = 4
NARROW_OVERLAP = 0.30
SMALL_OVERLAP = 0.16
NARROW_SPAN = 0.34
RIVAL_GAP = 0.08

VERDICTS = ("same_face", "different_face", "inconclusive")
DISPOSITIONS = ("confirmed", "rejected", "undecided")

# Registration is FFT-heavy; cap concurrent screenings to bound memory use.
SCREEN_SLOTS = BoundedSemaphore(2)


@dataclass(frozen=True)
class FaceImage:
    inspection_id: str
    title: str
    rgb: np.ndarray
    support: np.ndarray
    texture: np.ndarray
    pores: list[dict]
    canvas: np.ndarray
    canvas_mask: np.ndarray
    source_width: int
    source_height: int


@dataclass(frozen=True)
class Transform:
    angle: float
    scale: float
    tx: float
    ty: float
    ncc: float
    overlap_fraction: float
    narrowness: float = 1.0


def _resized(image: np.ndarray, support: np.ndarray, limit: int) -> tuple[np.ndarray, np.ndarray, float]:
    height, width = image.shape[:2]
    factor = min(1.0, limit / max(height, width))
    if factor == 1.0:
        return image, support, 1.0
    target = (max(2, round(width * factor)), max(2, round(height * factor)))
    rgb = np.asarray(Image.fromarray(image).resize(target, Image.Resampling.BILINEAR))
    mask = np.asarray(
        Image.fromarray(support.astype(np.uint8) * 255).resize(target, Image.Resampling.NEAREST)
    ).astype(bool)
    return rgb, mask, factor


def _texture(rgb: np.ndarray, support: np.ndarray) -> np.ndarray:
    """Locally normalized, illumination-insensitive contrast image."""
    light = luminance(rgb).astype(np.float32)
    sigma = max(1.0, HALF / 2.2)
    confidence = ndi.gaussian_filter(support.astype(np.float32), sigma, mode="constant")
    smoothed = ndi.gaussian_filter(
        np.where(support, light, 0.0).astype(np.float32), sigma, mode="constant"
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = np.where(confidence > 1e-4, smoothed / confidence, 0.0)
    squared = ndi.gaussian_filter(
        np.where(support, light * light, 0.0).astype(np.float32), sigma, mode="constant"
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_sq = np.where(confidence > 1e-4, squared / confidence, 0.0)
    local_std = np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))
    normalized = np.zeros_like(light)
    usable = support & (confidence > 0.6) & (local_std > NCC_FLOOR)
    normalized[usable] = (light[usable] - mean[usable]) / local_std[usable]
    return np.clip(normalized, -4.0, 4.0).astype(np.float32)


def _pore_records(rgb: np.ndarray, support: np.ndarray) -> list[dict]:
    from .geometry import Plane
    from .measurements import TOOLS

    height, width = rgb.shape[:2]
    plane = Plane(
        width, height, width, height, np.eye(3), np.eye(3), None, None, {"mode": "pixels"},
    )

    class _Raster:
        @property
        def support(self):
            return self.supported & self.inspected

    raster = _Raster()
    raster.rgb = rgb
    raster.supported = support
    raster.inspected = np.ones_like(support)
    raster.plane = plane
    settings = TOOLS["pores"].validator({}, plane)
    settings["min_area"] = min(settings["min_area"], 3)
    return detect_components(raster, settings).rows


def _canvas_for(rgb: np.ndarray, support: np.ndarray, texture: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Place a face centrally on a square canvas large enough for shifts."""
    height, width = rgb.shape[:2]
    side = int(np.ceil(max(height, width) * 1.62))
    if side % 2 == 0:
        side += 1
    canvas = np.zeros((side, side), dtype=np.float32)
    mask = np.zeros((side, side), dtype=bool)
    top = (side - height) // 2
    left = (side - width) // 2
    canvas[top: top + height, left: left + width] = texture
    mask[top: top + height, left: left + width] = support
    return canvas, mask


_FACE_CACHE: dict[tuple, FaceImage] = {}
_FACE_CACHE_MAX = 48


def load_face(db: Database, inspection: dict, limit: int, *, with_pores: bool) -> FaceImage:
    # A face without pores can be upgraded in place; pore-bearing entries are
    # reusable for every later call in the same screening run.
    key = (inspection["id"], inspection["revision"], limit, bool(with_pores))
    cached = _FACE_CACHE.get(key)
    if cached is not None:
        return cached
    frame = frame_for(db, inspection)
    resized, mask, factor = _resized(frame.raster.rgb, frame.raster.support, limit)
    texture = _texture(resized, mask)
    canvas, canvas_mask = _canvas_for(resized, mask, texture)
    pores = []
    if with_pores:
        pores = [
            {
                **pore,
                "x": pore["x"] * factor + (canvas.shape[1] - resized.shape[1]) / 2.0,
                "y": pore["y"] * factor + (canvas.shape[0] - resized.shape[0]) / 2.0,
            }
            for pore in _pore_records(resized, mask)
        ]
    meta = inspection["capture_metadata"]
    face = FaceImage(
        inspection["id"], inspection["title"], resized, mask, texture, pores,
        canvas, canvas_mask, meta["width"], meta["height"],
    )
    if len(_FACE_CACHE) >= _FACE_CACHE_MAX:
        _FACE_CACHE.clear()
    _FACE_CACHE[key] = face
    return face


def _reference_terms(reference: FaceImage) -> dict:
    """Precompute reference-side convolution terms shared across all warps."""
    ref_w = reference.canvas_mask.astype(np.float32)
    ref = np.where(reference.canvas_mask, reference.canvas, 0.0).astype(np.float32)
    reversed_rw = ref_w[::-1, ::-1]
    return {
        "ref_w": ref_w,
        "ref": ref,
        "reversed_rw": reversed_rw,
        "ref_dot_base": ref * ref_w,
        "ref_sq_base": ref * ref * ref_w,
    }


def masked_ncc_terms(terms: dict, tpl_texture: np.ndarray, tpl_mask: np.ndarray) -> np.ndarray:
    """NCC score map using precomputed reference terms (see :func:`masked_ncc`)."""
    ref_w = terms["ref_w"]
    ref = terms["ref"]
    reversed_rw = terms["reversed_rw"]
    side = ref.shape[0]
    center = side // 2
    tpl_w = tpl_mask.astype(np.float32)
    tpl = np.where(tpl_mask, tpl_texture, 0.0).astype(np.float32)

    reversed_tw = tpl_w[::-1, ::-1]
    weight = fftconvolve(ref_w, reversed_tw, mode="same")
    r_sum = fftconvolve(terms["ref_dot_base"], reversed_tw, mode="same")
    r_sq = fftconvolve(terms["ref_sq_base"], reversed_tw, mode="same")
    t_sum = fftconvolve(tpl * tpl_w, reversed_rw, mode="same")
    t_sq = fftconvolve(tpl * tpl * tpl_w, reversed_rw, mode="same")
    dot = fftconvolve(terms["ref_dot_base"], (tpl * tpl_w)[::-1, ::-1], mode="same")

    total_template_weight = float(tpl_w.sum())
    reach = int(side * 0.26)
    band = slice(center - reach, center + reach + 1)
    neg_band = slice(center + reach, center - reach - 1, -1)
    w = weight[band, band]
    safe_w = np.maximum(w, 1e-9)
    r_mean_map = r_sum[band, band] / safe_w
    t_mean_map = t_sum[neg_band, neg_band] / safe_w
    ref_var = np.maximum(r_sq[band, band] / safe_w - r_mean_map ** 2, 0.0)
    tpl_var = np.maximum(t_sq[neg_band, neg_band] / safe_w - t_mean_map ** 2, 0.0)
    covariance = dot[band, band] / safe_w - r_mean_map * t_mean_map
    usable = (
        (w >= 0.55 * total_template_weight)
        & (ref_var > NCC_FLOOR)
        & (tpl_var > NCC_FLOOR)
    )
    local = np.where(usable, covariance / np.maximum(np.sqrt(ref_var * tpl_var), 1e-9), np.nan)
    scores = np.full((side, side), np.nan, dtype=np.float32)
    scores[band, band] = local.astype(np.float32)
    return scores


def masked_ncc(reference: FaceImage, tpl_texture: np.ndarray, tpl_mask: np.ndarray) -> np.ndarray:
    """Vectorized masked normalized cross-correlation over shift positions.

    Both arrays share one odd-sized square canvas with faces centered.  At a
    shift ``d`` the overlap mask is the intersection of the reference mask and
    the shifted template mask.  Reference statistics convolve with the
    reversed template mask (indexed at ``d``); template statistics convolve
    with the reversed reference mask and are indexed at ``-d`` so both
    describe that same intersection.  The result is a Pearson correlation in
    [-1, 1] valid for rotated or cropped silhouettes.  The canvas center is
    zero translation.
    """
def masked_ncc(reference: FaceImage, tpl_texture: np.ndarray, tpl_mask: np.ndarray) -> np.ndarray:
    """Masked normalized cross-correlation score map (see :func:`masked_ncc_terms`)."""
    return masked_ncc_terms(_reference_terms(reference), tpl_texture, tpl_mask)


def coarse_score(terms: dict, tpl_texture: np.ndarray, tpl_mask: np.ndarray) -> np.ndarray:
    """Lightweight normalized dot-product map for coarse ranking.

    Uses three FFT convolutions instead of the full six.  It is monotone
    enough with the masked Pearson score to shortlist alignment candidates;
    the final verdict always comes from :func:`masked_ncc_terms`.
    """
    ref_w = terms["ref_w"]
    reversed_tw = tpl_mask.astype(np.float32)[::-1, ::-1]
    tpl_weighted = np.where(tpl_mask, tpl_texture, 0.0).astype(np.float32)
    weight = fftconvolve(ref_w, reversed_tw, mode="same")
    r_sq = fftconvolve(terms["ref_sq_base"], reversed_tw, mode="same")
    dot = fftconvolve(terms["ref_dot_base"], tpl_weighted[::-1, ::-1], mode="same")
    tpl_energy = float(np.sum(tpl_weighted * tpl_weighted))
    total = float(tpl_mask.sum())
    side = ref_w.shape[0]
    center = side // 2
    reach = int(side * 0.26)
    band = slice(center - reach, center + reach + 1)
    w = np.maximum(weight[band, band], 1e-9)
    denom = np.sqrt(np.maximum(r_sq[band, band], 1e-9) * max(tpl_energy, 1e-9))
    local = np.where(weight[band, band] >= 0.55 * total, dot[band, band] / denom, np.nan)
    scores = np.full((side, side), np.nan, dtype=np.float32)
    scores[band, band] = local.astype(np.float32)
    return scores


def _rotation(angle: float) -> np.ndarray:
    """2x2 rotation in scipy's (row, col) = (y, x) coordinate order."""
    theta = radians(angle)
    return np.array([[cos(theta), sin(theta)], [-sin(theta), cos(theta)]])


def _warp(face: FaceImage, angle: float, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Rotate and scale around the canvas center, keeping the center fixed."""
    center = face.canvas.shape[0] / 2.0
    matrix = scale * _rotation(angle)
    inverse = np.linalg.inv(matrix)
    center_point = np.array([center, center])
    offset = center_point - inverse @ center_point
    texture = ndi.affine_transform(
        face.canvas, inverse, offset=offset, order=1, mode="constant", cval=0.0,
    )
    mask = ndi.affine_transform(
        face.canvas_mask.astype(np.uint8), inverse, offset=offset,
        order=0, mode="constant", cval=0,
    ).astype(bool)
    texture[~mask] = 0.0
    return texture, mask


def _best(scores: np.ndarray) -> tuple[float, float, float]:
    valid = np.isfinite(scores)
    if not valid.any():
        return 0.0, 0.0, -1.0
    position = np.unravel_index(np.argmax(np.where(valid, scores, -2.0)), scores.shape)
    center = scores.shape[0] // 2
    # fftconvolve's "same" anchor is reversed: a peak offset (py, px) means
    # the template moved by (-px, -py) relative to the reference.
    return (
        float(center - position[1]),
        float(center - position[0]),
        float(scores[position]),
    )


def coarse_search(reference: FaceImage, candidate: FaceImage) -> list[Transform]:
    terms = _reference_terms(reference)
    found = []
    for scale in COARSE_SCALES:
        for angle in COARSE_ANGLES:
            texture, mask = _warp(candidate, angle, scale)
            tx, ty, score = _best(coarse_score(terms, texture, mask))
            found.append(Transform(float(angle), float(scale), tx, ty, score, 0.0))
    found.sort(key=lambda item: item.ncc, reverse=True)
    return found


def refine_search(reference: FaceImage, candidate: FaceImage, seeds: list[Transform]) -> Transform:
    terms = _reference_terms(reference)
    best = seeds[0]
    for seed in seeds[:1]:
        angles = np.arange(
            seed.angle - FINE_ANGLE_DELTA, seed.angle + FINE_ANGLE_DELTA + 0.1, FINE_ANGLE_STEP
        )
        for angle in angles:
            for scale in FINE_SCALES:
                texture, mask = _warp(candidate, angle % 360.0, scale)
                tx, ty, score = _best(masked_ncc_terms(terms, texture, mask))
                if score > best.ncc:
                    best = Transform(angle % 360.0, float(scale), tx, ty, score, 0.0)
    overlap, narrowness = _overlap_geometry(reference, candidate, best)
    return Transform(best.angle, best.scale, best.tx, best.ty, best.ncc, overlap, narrowness)


def _affine_for(face: FaceImage, transform: Transform) -> tuple[np.ndarray, np.ndarray]:
    """Forward affine mapping candidate-canvas points to reference-canvas.

    Coordinates are scipy's (row, col) order.  Rotation and scale happen
    around each canvas center, followed by the estimated translation.
    """
    center = face.canvas.shape[0] / 2.0
    matrix = transform.scale * _rotation(transform.angle)
    center_point = np.array([center, center])
    offset = center_point + np.array([transform.ty, transform.tx]) - matrix @ center_point
    return matrix, offset


def _inverse_warp_input_offset(matrix: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """Offset for scipy affine_transform given the forward map y = M x + o."""
    inverse = np.linalg.inv(matrix)
    return -inverse @ offset


def _overlap_geometry(
    reference: FaceImage, candidate: FaceImage, transform: Transform,
) -> tuple[float, float]:
    """Return (overlap fraction, spatial narrowness) for the shared region.

    ``overlap_fraction`` weights shared supported pixels against the smaller
    face.  ``narrowness`` is the shared bounding box's shortest side as a
    fraction of the reference face's shortest side, so a thin sliver of
    repeated ring bands stays detectable even when its pixel count alone looks
    large.
    """
    matrix, offset = _affine_for(candidate, transform)
    inverse = np.linalg.inv(matrix)
    input_offset = _inverse_warp_input_offset(matrix, offset)
    warped = ndi.affine_transform(
        candidate.canvas_mask.astype(np.uint8), inverse, offset=input_offset,
        output_shape=reference.canvas_mask.shape, order=0, mode="constant", cval=0,
    ).astype(bool)
    shared = warped & reference.canvas_mask
    smaller = min(np.count_nonzero(warped), np.count_nonzero(reference.canvas_mask))
    fraction = float(np.count_nonzero(shared) / smaller) if smaller else 0.0
    ref_rows = np.flatnonzero(reference.canvas_mask.any(axis=1))
    ref_cols = np.flatnonzero(reference.canvas_mask.any(axis=0))
    ref_short = min(ref_rows[-1] - ref_rows[0], ref_cols[-1] - ref_cols[0])
    rows = np.flatnonzero(shared.any(axis=1))
    cols = np.flatnonzero(shared.any(axis=0))
    if not rows.size or not cols.size or ref_short <= 0:
        return fraction, 0.0
    shared_short = min(rows[-1] - rows[0], cols[-1] - cols[0])
    return fraction, float(shared_short / ref_short)


def _pore_correspondences(reference: FaceImage, candidate: FaceImage, transform: Transform) -> list[dict]:
    matrix, offset = _affine_for(candidate, transform)
    if not candidate.pores or not reference.pores:
        return []
    # Pore coordinates are stored as (x, y); the affine matrix works in (y, x).
    candidate_xy = np.array([[pore["x"], pore["y"]] for pore in candidate.pores])
    projected_yx = candidate_xy[:, ::-1] @ matrix.T + offset
    ref_xy = np.array([[pore["x"], pore["y"]] for pore in reference.pores])
    ref_yx = ref_xy[:, ::-1]
    tolerance = 8.0 + 2.0 * abs(transform.scale - 1.0)
    matches = []
    taken = set()
    order = np.argsort([pore.get("diameter", 0.0) for pore in candidate.pores])[::-1]
    for candidate_index in order:
        point_yx = projected_yx[candidate_index]
        distances = np.linalg.norm(ref_yx - point_yx, axis=1)
        for ref_index in np.argsort(distances):
            if distances[ref_index] > tolerance:
                break
            if ref_index in taken:
                continue
            taken.add(ref_index)
            source_pore = candidate.pores[candidate_index]
            target_pore = reference.pores[ref_index]
            size_ratio = (
                min(source_pore["diameter"], target_pore["diameter"])
                / max(source_pore["diameter"], target_pore["diameter"])
            )
            if size_ratio < 0.35:
                continue
            matches.append({
                # Report endpoints back in (x, y) canvas coordinates.
                "candidate": [round(float(point_yx[1]), 2), round(float(point_yx[0]), 2)],
                "reference": [round(float(ref_xy[ref_index][0]), 2),
                              round(float(ref_xy[ref_index][1]), 2)],
                "distance_px": round(float(distances[ref_index]), 2),
                "size_ratio": round(float(size_ratio), 2),
            })
            break
    if len(matches) >= PORE_MATCH_MIN:
        matches = _geometry_filter(matches)
    return matches


def _geometry_filter(matches: list[dict]) -> list[dict]:
    candidate_pts = np.array([item["candidate"] for item in matches])[:, ::-1]
    reference_pts = np.array([item["reference"] for item in matches])[:, ::-1]
    keep = np.zeros(len(matches), dtype=bool)
    for index in range(len(matches)):
        delta_c = np.linalg.norm(candidate_pts - candidate_pts[index], axis=1)
        delta_r = np.linalg.norm(reference_pts - reference_pts[index], axis=1)
        compare = delta_c > 14.0
        if not compare.any():
            keep[index] = True
            continue
        relative = np.abs(delta_r[compare] - delta_c[compare]) / np.maximum(delta_c[compare], 1.0)
        keep[index] = float(np.median(relative)) <= 0.18
    survivors = [item for item, ok in zip(matches, keep) if ok]
    return survivors if len(survivors) >= PORE_MATCH_MIN else matches


def _warped_candidate(reference: FaceImage, candidate: FaceImage, transform: Transform) -> tuple[np.ndarray, np.ndarray]:
    matrix, offset = _affine_for(candidate, transform)
    inverse = np.linalg.inv(matrix)
    input_offset = _inverse_warp_input_offset(matrix, offset)
    texture = ndi.affine_transform(
        candidate.canvas,
        inverse,
        offset=input_offset,
        output_shape=reference.canvas.shape,
        order=1,
        mode="constant",
        cval=0.0,
    )
    mask = ndi.affine_transform(
        candidate.canvas_mask.astype(np.uint8),
        inverse,
        offset=input_offset,
        output_shape=reference.canvas.shape,
        order=0,
        mode="constant",
        cval=0,
    ).astype(bool)
    return texture, mask


def _texture_tiles(reference: FaceImage, candidate: FaceImage, transform: Transform) -> list[dict]:
    warped, warped_mask = _warped_candidate(reference, candidate, transform)
    matrix, offset = _affine_for(candidate, transform)
    inverse = np.linalg.inv(matrix)
    side = reference.canvas.shape[0]
    chosen = []
    blocked = np.zeros((side, side), dtype=bool)
    step = PATCH * 3
    for row in range(side // 2 - 260, side // 2 + 260, step):
        for col in range(side // 2 - 260, side // 2 + 260, step):
            y0, y1 = row - HALF, row + HALF + 1
            x0, x1 = col - HALF, col + HALF + 1
            if y0 < 0 or x0 < 0 or y1 > side or x1 > side:
                continue
            valid = reference.canvas_mask[y0:y1, x0:x1] & warped_mask[y0:y1, x0:x1]
            if valid.mean() < 0.8:
                continue
            window_r = reference.canvas[y0:y1, x0:x1]
            window_t = warped[y0:y1, x0:x1]
            if window_r[valid].std() < 0.35:
                continue
            centered_r = window_r - window_r[valid].mean()
            centered_t = window_t - window_t[valid].mean()
            denom = np.linalg.norm(centered_r[valid]) * np.linalg.norm(centered_t[valid])
            if denom < 1e-6:
                continue
            score = float(np.dot(centered_r[valid], centered_t[valid]) / denom)
            if score >= 0.62 and not blocked[row, col]:
                # Map the reference (row, col) point back into the candidate
                # frame, then report both endpoints in (x, y) order.
                source_yx = inverse @ (np.array([row, col], dtype=np.float64) - offset)
                chosen.append({
                    "reference": [int(col), int(row)],
                    "candidate": [round(float(source_yx[1]), 2), round(float(source_yx[0]), 2)],
                    "score": round(score, 3),
                })
                blocked[max(0, row - step): row + step, max(0, col - step): col + step] = True
    chosen.sort(key=lambda item: item["score"], reverse=True)
    return chosen[:8]


def decide(transform: Transform, pores: list[dict], tiles: list[dict], rival_gap: float) -> tuple[str, list[str]]:
    ncc, overlap = transform.ncc, transform.overlap_fraction
    narrowness = transform.narrowness
    if overlap < SMALL_OVERLAP:
        return "inconclusive", [
            f"Only {overlap:.0%} of the smaller face overlaps the old photo; the shared area is too small to identify it."
        ]
    if overlap < NARROW_OVERLAP or narrowness < NARROW_SPAN:
        return "inconclusive", [
            "Only a narrow strip of repeated growth rings is shared "
            f"({overlap:.0%} pixel overlap, shortest shared span {narrowness:.0%} of the face); "
            "repeated ring bands alone are not identifying evidence."
        ]
    if rival_gap < RIVAL_GAP:
        return "inconclusive", [
            f"The next candidate scores within {RIVAL_GAP:.2f} correlation of this one, so the faces are not distinguishable enough."
        ]
    strong_pores = len(pores) >= PORE_STRONG
    enough_pores = len(pores) >= PORE_MATCH_MIN
    strong_tiles = len(tiles) >= TILE_STRONG
    if ncc >= NCC_STRONG and (strong_pores or (enough_pores and strong_tiles)):
        return "same_face", [
            f"Strong texture agreement (r={ncc:.2f}) with {len(pores)} pore and {len(tiles)} texture correspondences across {overlap:.0%} overlap."
        ]
    if ncc < NCC_PLAUSIBLE and not enough_pores:
        return "different_face", [
            f"Texture correlation is weak (r={ncc:.2f}) and no pore layout repeats, so the faces do not match."
        ]
    if ncc < NCC_PLAUSIBLE:
        return "inconclusive", [
            f"Texture correlation is weak (r={ncc:.2f}) and only {len(pores)} pores line up; this is not enough to claim the same face."
        ]
    return "inconclusive", [
        f"Texture correlation r={ncc:.2f}, {len(pores)} pore correspondences and {len(tiles)} texture tiles do not meet the confirmation threshold."
    ]


def _document(
    query_inspection: dict, candidate_inspection: dict, query: FaceImage, candidate: FaceImage,
    transform: Transform, pores: list[dict], tiles: list[dict], verdict: str, reasons: list[str],
) -> dict:
    return {
        "query": {
            "inspection_id": query_inspection["id"],
            "title": query_inspection["title"],
            "lot": query_inspection["lot"],
            "capture_id": query_inspection["capture_id"],
        },
        "candidate": {
            "inspection_id": candidate_inspection["id"],
            "title": candidate_inspection["title"],
            "lot": candidate_inspection["lot"],
            "capture_id": candidate_inspection["capture_id"],
        },
        "transform": {
            "angle_deg": round(transform.angle % 360.0, 2),
            "scale": round(transform.scale, 4),
            "tx": round(transform.tx, 2),
            "ty": round(transform.ty, 2),
            "ncc": round(transform.ncc, 4),
            "overlap_fraction": round(transform.overlap_fraction, 4),
            "narrowness": round(transform.narrowness, 4),
        },
        "evidence": {
            "pore_matches": pores,
            "texture_tiles": tiles,
            "query_pore_count": len(query.pores),
            "candidate_pore_count": len(candidate.pores),
        },
        "verdict": verdict,
        "reasons": reasons,
    }


def _pair_analysis(
    db: Database, query_inspection: dict, candidate_inspection: dict,
    query_face: FaceImage | None = None, rival_gap: float = 1.0,
) -> tuple[dict, FaceImage, FaceImage]:
    coarse_query = query_face or load_face(db, query_inspection, COARSE_LIMIT, with_pores=False)
    coarse_candidate = load_face(db, candidate_inspection, COARSE_LIMIT, with_pores=False)
    seeds = coarse_search(coarse_query, coarse_candidate)
    if not seeds or seeds[0].ncc <= -1.0:
        fine_query = load_face(db, query_inspection, FINE_LIMIT, with_pores=True)
        fine_candidate = load_face(db, candidate_inspection, FINE_LIMIT, with_pores=True)
        transform = Transform(0.0, 1.0, 0.0, 0.0, -1.0, 0.0)
        verdict, reasons = "inconclusive", ["No overlapping textured region could be registered."]
        document = _document(query_inspection, candidate_inspection, fine_query, fine_candidate,
                             transform, [], [], verdict, reasons)
        return document, fine_query, fine_candidate
    fine_query = load_face(db, query_inspection, FINE_LIMIT, with_pores=True)
    fine_candidate = load_face(db, candidate_inspection, FINE_LIMIT, with_pores=True)
    # Map coarse alignment onto the fine canvas (both center their canvases).
    scale_ratio = fine_query.canvas.shape[0] / coarse_query.canvas.shape[0]
    fine_seeds = [
        Transform(item.angle, item.scale, item.tx * scale_ratio, item.ty * scale_ratio, item.ncc, 0.0)
        for item in seeds[:1]
    ]
    transform = refine_search(fine_query, fine_candidate, fine_seeds)
    pores = _pore_correspondences(fine_query, fine_candidate, transform)
    tiles = _texture_tiles(fine_query, fine_candidate, transform)
    verdict, reasons = decide(transform, pores, tiles, rival_gap)
    document = _document(query_inspection, candidate_inspection, fine_query, fine_candidate,
                         transform, pores, tiles, verdict, reasons)
    return document, fine_query, fine_candidate


def analyze_pair(db: Database, query_inspection: dict, candidate_inspection: dict) -> dict:
    document, _, _ = _pair_analysis(db, query_inspection, candidate_inspection)
    return document


def rank_candidates(db: Database, query_inspection: dict, candidates: list[dict]) -> list[dict]:
    candidates = [item for item in candidates if item["id"] != query_inspection["id"]][:MAX_CANDIDATES]
    if not SCREEN_SLOTS.acquire(blocking=False):
        raise Conflict("Two face screenings are already processing. Retry when one finishes.")
    try:
        coarse_query = load_face(db, query_inspection, COARSE_LIMIT, with_pores=False)
        coarse = []
        for candidate_inspection in candidates:
            face = load_face(db, candidate_inspection, COARSE_LIMIT, with_pores=False)
            seeds = coarse_search(coarse_query, face)
            best = seeds[0] if seeds else Transform(0.0, 1.0, 0.0, 0.0, -1.0, 0.0)
            coarse.append((candidate_inspection, face, best))
        coarse.sort(key=lambda item: item[2].ncc, reverse=True)
        best_score = coarse[0][2].ncc if coarse else -1.0
        documents = []
        for rank, (candidate_inspection, _, _) in enumerate(coarse):
            rival_gap = 1.0 if rank == 0 else max(0.0, best_score - coarse[rank][2].ncc)
            document, _, _ = _pair_analysis(
                db, query_inspection, candidate_inspection, query_face=coarse_query, rival_gap=rival_gap,
            )
            documents.append(document)
        return documents
    finally:
        SCREEN_SLOTS.release()


# ---------------------------------------------------------------------------
# Side-by-side evidence sheet
# ---------------------------------------------------------------------------

def evidence_sheet(db: Database, document: dict) -> bytes:
    query = db.inspection(document["query"]["inspection_id"])
    candidate = db.inspection(document["candidate"]["inspection_id"])
    query_face = load_face(db, query, FINE_LIMIT, with_pores=True)
    candidate_face = load_face(db, candidate, FINE_LIMIT, with_pores=True)
    transform = Transform(
        document["transform"]["angle_deg"], document["transform"]["scale"],
        document["transform"]["tx"], document["transform"]["ty"],
        document["transform"]["ncc"], document["transform"]["overlap_fraction"],
    )
    matrix, offset = _affine_for(candidate_face, transform)
    inverse = np.linalg.inv(matrix)

    left = _draw_panel(query_face, document, inverse=None, inverse_offset=None, side="reference")
    right = _draw_panel(candidate_face, document, inverse=inverse, inverse_offset=offset, side="candidate")
    panel_width = 560
    left_img = ImageOps.contain(Image.fromarray(left), (panel_width, 500), Image.Resampling.LANCZOS)
    right_img = ImageOps.contain(Image.fromarray(right), (panel_width, 500), Image.Resampling.LANCZOS)
    height = max(left_img.height, right_img.height) + 104
    sheet = Image.new("RGB", (panel_width * 2 + 24, height), (40, 43, 42))
    pen = ImageDraw.Draw(sheet)
    verdict_text = {
        "same_face": "Machine suggestion: likely the same end face",
        "different_face": "Machine suggestion: different end faces",
        "inconclusive": "Machine suggestion: evidence insufficient — do not assume identity",
    }[document["verdict"]]
    pen.text((14, 10), verdict_text, fill=(238, 227, 193))
    reason = "  /  ".join(document["reasons"])
    pen.text((14, 28), reason[:200], fill=(205, 217, 193))
    pen.text((14, 48), "Green numbered circles: matching vessel pores    amber squares: matching texture tiles",
             fill=(178, 190, 168))
    metric = (
        f"r={document['transform']['ncc']:.2f}   "
        f"overlap={document['transform']['overlap_fraction']:.0%}   "
        f"rotation={document['transform']['angle_deg']:.0f}°   "
        f"scale={document['transform']['scale']:.2f}"
    )
    pen.text((14, 66), metric, fill=(178, 190, 168))
    for index, (label, artwork) in enumerate((
        ("A / Re-photo: " + query_face.title, left_img),
        ("B / Old record: " + candidate_face.title, right_img),
    )):
        x0 = index * (panel_width + 24)
        pen.rectangle((x0 + 6, 88, x0 + panel_width - 6, 114), fill=(62, 64, 62))
        pen.text((x0 + 14, 94), label[:76], fill=(238, 227, 193))
        sheet.paste(artwork, (x0 + (panel_width - artwork.width) // 2, 120))
    return encode_png(np.asarray(sheet))


def _draw_panel(face: FaceImage, document: dict, inverse, inverse_offset, side: str) -> np.ndarray:
    image = Image.fromarray(face.rgb.copy())
    pen = ImageDraw.Draw(image, "RGBA")
    top = (face.canvas.shape[0] - face.rgb.shape[0]) // 2
    left = (face.canvas.shape[1] - face.rgb.shape[1]) // 2

    def place(canvas_xy):
        point_xy = np.array(canvas_xy, dtype=np.float64)
        if side == "candidate":
            # The inverse affine works in (row, col); convert from (x, y).
            source_yx = inverse @ (point_xy[::-1] - inverse_offset)
            point_xy = source_yx[::-1]
        return point_xy[0] - left, point_xy[1] - top

    for index, match in enumerate(document["evidence"]["pore_matches"], 1):
        # Stored points are in the reference canvas frame, (x, y) order.
        x, y = place(match["reference"])
        pen.ellipse((x - 8, y - 8, x + 8, y + 8), outline=(86, 226, 168, 255), width=3)
        pen.text((x + 10, y - 9), str(index), fill=(226, 255, 232, 255),
                 stroke_width=2, stroke_fill=(20, 40, 30, 255))
    for tile in document["evidence"]["texture_tiles"]:
        x, y = place(tile["reference"])
        pen.rectangle((x - 10, y - 10, x + 10, y + 10), outline=(245, 183, 92, 255), width=2)
    return np.asarray(image)
