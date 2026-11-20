"""Standalone checks for the end-face re-photo matcher.

Run with: python3 -I tests/test_facematch.py

The tests build small synthetic wood-like images (ring bands + dark vessel
pores + noise), then verify that:

1. a rotated, cropped and brightness-shifted copy of a face is suggested as
   the same face and the recovered transform points the right way,
2. a genuinely different texture (different ring phase and pore layout) is
   not confirmed as the same face,
3. a pair whose overlap is only a narrow strip of repeated rings returns
   ``inconclusive`` rather than a forced identity.
"""

import sys

sys.path.insert(0, "/opt/pointwise-env/site-packages")
sys.path.insert(0, "/workspace")

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from endgrain import facematch


def synthetic_face(seed: int, size=(300, 340), ring_gap=17.0, pore_count=120):
    rng = np.random.default_rng(seed)
    h, w = size
    y, x = np.mgrid[:h, :w].astype(np.float32)
    cx, cy = w * 0.47 + rng.normal(0, 4), h * 0.55 + rng.normal(0, 4)
    radial = np.hypot((x - cx) * 1.05, (y - cy) * 0.95)
    theta = np.arctan2(y - cy, x - cx)
    phase = (radial + 5.0 * np.sin(theta * 3 + 0.02 * radial) + seed) / ring_gap
    index = phase % 1.0
    rings = -28.0 * np.exp(-np.minimum(index, 1 - index) ** 2 / 0.008)
    rays = 7.0 * np.maximum(np.cos(37 * theta + 0.01 * radial), 0.0) ** 18
    noise = ndi.gaussian_filter(rng.normal(0, 1, size), 1.1) * 4.0
    base = np.clip(168.0 + rings + rays + noise, 40, 235)
    rgb = np.repeat(base[..., None], 3, axis=-1)
    pores = []
    for _ in range(pore_count):
        px = rng.uniform(18, w - 18)
        py = rng.uniform(18, h - 18)
        r = rng.uniform(2.0, 4.2)
        pores.append((px, py, r))
        yy, xx = np.ogrid[:h, :w]
        disk = (xx - px) ** 2 + (yy - py) ** 2 <= r * r
        rgb[disk] = np.clip(rgb[disk] - rng.uniform(70, 130), 0, 255)
    support = np.ones(size, dtype=bool)
    return rgb.astype(np.uint8), support, pores


def rephotograph(rgb, angle=24.0, scale=1.0, brightness=1.18, crop=0.16):
    rotated = ndi.rotate(rgb, angle, reshape=False, order=1, mode="constant", cval=0)
    if scale != 1.0:
        h0, w0 = rgb.shape[:2]
        zoomed = ndi.zoom(rotated, (scale, scale, 1.0), order=1)
        h1, w1 = zoomed.shape[:2]
        if h1 >= h0 and w1 >= w0:
            top, left = (h1 - h0) // 2, (w1 - w0) // 2
            rotated = zoomed[top: top + h0, left: left + w0]
        else:
            canvas = np.zeros_like(rgb)
            top, left = (h0 - h1) // 2, (w0 - w1) // 2
            canvas[top: top + h1, left: left + w1] = zoomed
            rotated = canvas
    adjusted = np.clip(rotated * brightness, 0, 255).astype(np.uint8)
    h, w = adjusted.shape[:2]
    # Support covers real wood only: exclude the black corners introduced by
    # rotation and the deliberately cropped left edge.
    keep = rotated.sum(axis=2) > 0
    cut = int(min(h, w) * crop)
    keep[:, : cut] = False
    adjusted[~keep] = 0
    return adjusted, keep


def face(inspection_id, title, rgb, support):
    resized, mask, _ = facematch._resized(rgb, support, facematch.FINE_LIMIT)
    texture = facematch._texture(resized, mask)
    canvas, canvas_mask = facematch._canvas_for(resized, mask, texture)
    pores = facematch._pore_records(resized, mask)
    top = (canvas.shape[0] - resized.shape[0]) // 2
    left = (canvas.shape[1] - resized.shape[1]) // 2
    pores = [{**p, "x": p["x"] + left, "y": p["y"] + top} for p in pores]
    return facematch.FaceImage(
        inspection_id, title, resized, mask, texture, pores, canvas, canvas_mask,
        resized.shape[1], resized.shape[0],
    )


def run_pair(reference, candidate, rival_gap=1.0):
    seeds = facematch.coarse_search(reference, candidate)
    assert seeds, "coarse search returned nothing"
    transform = facematch.refine_search(reference, candidate, seeds[:3])
    pores = facematch._pore_correspondences(reference, candidate, transform)
    tiles = facematch._texture_tiles(reference, candidate, transform)
    verdict, reasons = facematch.decide(transform, pores, tiles, rival_gap)
    return transform, pores, tiles, verdict, reasons


def main():
    print("building synthetic faces...")
    rgb_a, support_a, _ = synthetic_face(seed=11)
    rgb_b, support_b, _ = synthetic_face(seed=73)  # different face
    rep_rgb, rep_support = rephotograph(rgb_a, angle=24.0, brightness=1.18, crop=0.16)

    face_a = face("a", "face A", rgb_a, support_a)
    face_rep = face("a-rephoto", "face A rephoto", rep_rgb, rep_support)
    face_b = face("b", "face B", rgb_b, support_b)

    print("\n[1] same face, rotated + cropped + brighter")
    transform, pores, tiles, verdict, reasons = run_pair(face_a, face_rep)
    print(" transform:", transform)
    print(" pore matches:", len(pores), "tiles:", len(tiles))
    print(" verdict:", verdict, reasons)
    assert transform.ncc > 0.3, f"expected strong ncc, got {transform.ncc}"
    angle_error = abs(((transform.angle - 24.0 + 180) % 360) - 180)
    assert angle_error <= 8.0, f"rotation recovery off by {angle_error}"
    assert verdict == "same_face", f"expected same_face, got {verdict}"

    print("\n[2] genuinely different face")
    transform2, pores2, tiles2, verdict2, reasons2 = run_pair(face_a, face_b)
    print(" transform:", transform2)
    print(" pore matches:", len(pores2), "tiles:", len(tiles2))
    print(" verdict:", verdict2, reasons2)
    assert verdict2 in {"different_face", "inconclusive"}, verdict2
    assert verdict2 != "same_face", "different texture must not be confirmed"

    print("\n[3] narrow strip overlap (repeated rings only)")
    strip_support = np.zeros_like(support_a)
    strip_support[:, 120:200] = True
    face_strip = face("a-strip", "face A narrow strip", rgb_a, strip_support)
    transform3, pores3, tiles3, verdict3, reasons3 = run_pair(face_a, face_strip)
    print(" transform:", transform3)
    print(" narrowness:", transform3.narrowness, "verdict:", verdict3, reasons3)
    assert verdict3 == "inconclusive", f"narrow strip must be inconclusive, got {verdict3}"
    assert transform3.narrowness < facematch.NARROW_SPAN

    print("\n[4] two near-tied candidates")
    tied_reasons = None
    transform4, pores4, tiles4, verdict4, tied_reasons = run_pair(
        face_a, face_rep, rival_gap=0.02
    )
    print(" verdict with small rival gap:", verdict4, tied_reasons)
    assert verdict4 == "inconclusive"

    print("\nall matcher checks passed")


if __name__ == "__main__":
    main()
