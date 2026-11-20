from dataclasses import dataclass
from io import BytesIO
from math import isclose
from typing import Any

from PIL import Image, ImageDraw, ImageOps

from .database import Database
from .errors import WorkbenchError
from .imaging import encode_png
from .measurements import TOOLS, describe_run
from .validation import string


@dataclass(frozen=True)
class Pair:
    left: dict
    right: dict
    comparable: bool
    differences: list[str]
    metric_rows: list[dict]


def settings_diff(one: dict, two: dict) -> list[str]:
    return sorted(key for key in set(one) | set(two) if one.get(key) != two.get(key))


def compare_state(left: dict, right: dict) -> list[str]:
    warnings = []
    ls, rs = left["snapshot"], right["snapshot"]
    if ls["source_digest"] != rs["source_digest"]:
        warnings.append("The measurements use different source photographs.")
    if ls["state"]["region"] != rs["state"]["region"]:
        warnings.append("The selected measuring areas are different.")
    if ls["plane"]["calibration"] != rs["plane"]["calibration"]:
        warnings.append("Calibration differs across the measurements.")
    if ls["state"]["lighting"] != rs["state"]["lighting"] or ls["reference_digest"] != rs["reference_digest"]:
        warnings.append("Illumination correction differs across the measurements.")
    overrides = settings_diff(left["params"], right["params"])
    if overrides:
        warnings.append("Measurement settings changed: " + ", ".join(name.replace("_", " ") for name in overrides) + ".")
    return warnings


def units_match(left: dict, right: dict) -> bool:
    l, r = left["snapshot"]["plane"], right["snapshot"]["plane"]
    if l["unit"] != r["unit"]:
        return False
    if l["unit"] == "mm":
        return True
    if (l["width"], l["height"]) != (r["width"], r["height"]):
        return False
    return (l["source_width"], l["source_height"]) == (r["source_width"], r["source_height"])


def metric_rows(left: dict, right: dict, comparable: bool) -> list[dict]:
    lvals, rvals = left["result"]["metrics"], right["result"]["metrics"]
    tool = TOOLS[left["kind"]]
    output = []
    for field, (name, mode) in tool.metrics.items():
        a, b = lvals.get(field), rvals.get(field)
        numeric = isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool)
        difference = (b - a) if numeric and comparable else None
        if mode == "°" and difference is not None:
            difference = (difference + 90) % 180 - 90
        output.append({
            "key": field,
            "title": name,
            "mode": mode,
            "left": a,
            "right": b,
            "difference": difference,
        })
    return output


def pair(db: Database, inspection_id: str, left_id: Any, right_id: Any) -> Pair:
    left_id = string(left_id, "left measurement", 40, required=True)
    right_id = string(right_id, "right measurement", 40, required=True)
    if left_id == right_id:
        raise WorkbenchError("Choose two different saved measurements to compare.", "right")
    with db.transaction(write=False) as conn:
        left = db.run_on(conn, inspection_id, left_id)
        right = db.run_on(conn, inspection_id, right_id)
    if left["kind"] != right["kind"]:
        raise WorkbenchError("Use two measurements of the same kind.", "right")
    warnings = compare_state(left, right)
    comparable = units_match(left, right)
    if not comparable:
        warnings.append("These units or pixel grids are incompatible. Numeric deltas are hidden.")
    return Pair(describe_run(left), describe_run(right), comparable, warnings, metric_rows(left, right, comparable))


def pair_document(value: Pair) -> dict:
    return {
        "left": value.left,
        "right": value.right,
        "comparable": value.comparable,
        "notices": value.differences,
        "rows": value.metric_rows,
    }


def compare_sheet(db: Database, value: Pair) -> bytes:
    tiles = []
    for column, run in zip(("A", "B"), (value.left, value.right), strict=True):
        raw = db.run_attachment(run["inspection_id"], run["id"], "preview")
        with Image.open(BytesIO(raw)) as source:
            resized = ImageOps.contain(source.convert("RGB"), (625, 505), Image.Resampling.LANCZOS)
        tiles.append((column, run, resized))
    tile_width = 655
    tile_height = max(item[2].height for item in tiles) + 56
    sheet = Image.new("RGB", (tile_width * 2 + 12, tile_height), (40, 43, 42))
    pen = ImageDraw.Draw(sheet)
    for index, (label, run, image) in enumerate(tiles):
        offset = index * (tile_width + 12)
        x = offset + (tile_width - image.width) // 2
        y = 38 + (tile_height - 51 - image.height) // 2
        header = f"{label} / {run['title']} / revision {run['revision']} / {run['created_at'][:10]}"
        pen.rectangle((offset + 7, 5, offset + tile_width - 7, 31), fill=(62, 64, 62))
        pen.text((offset + 17, 13), header, fill=(238, 227, 193))
        sheet.paste(image, (x, y))
    import numpy as np
    return encode_png(np.asarray(sheet))


def same_pixel_frame(left: dict, right: dict) -> bool:
    first, second = left["snapshot"], right["snapshot"]
    return (
        first["source_digest"] == second["source_digest"]
        and first["plane"] == second["plane"]
        and first["state"] == second["state"]
    )


def delta_overlay(db: Database, output: Pair) -> bytes:
    if not same_pixel_frame(output.left, output.right):
        raise WorkbenchError("Image deltas require the same source, region, plane, and illumination.", "right")
    import numpy as np
    from .imaging import luminance, srgb_to_linear

    def loaded(run: dict) -> np.ndarray:
        image = db.run_attachment(run["inspection_id"], run["id"], "preview")
        with Image.open(BytesIO(image)) as reading:
            return np.array(reading.convert("RGB"), dtype=np.uint8)

    baseline = loaded(output.left)
    proposed = loaded(output.right)
    if baseline.shape != proposed.shape:
        raise WorkbenchError("The saved overlays have different dimensions.", "right")
    light_before = luminance(baseline, linear=True)
    light_after = luminance(proposed, linear=True)
    base = np.mean([light_before, light_after], axis=0)
    brightness = light_after - light_before
    difference = np.max(np.abs(srgb_to_linear(baseline) - srgb_to_linear(proposed)), axis=-1)
    keep = difference >= 0.014
    mag = np.clip(np.sqrt(difference) * 1.6, 0, 1)
    mag[~keep] = 0
    color = np.full(baseline.shape, (185, 204, 158), dtype=np.float32)
    color[brightness >= 0.028] = (250, 182, 92)
    color[brightness <= -0.028] = (118, 189, 231)
    neutral = np.stack([base * 110 + 43] * 3, axis=-1)
    visual = mag[..., None] * color + (1 - mag[..., None]) * neutral
    changed = float(np.mean(keep))
    ready = Image.fromarray(np.rint(visual).astype(np.uint8))
    fit = ImageOps.contain(ready, (1450, 1000), Image.Resampling.LANCZOS)
    size = (max(466, fit.width), fit.height + 35)
    canvas = Image.new("RGB", size, (43, 48, 41))
    canvas.paste(fit, ((size[0] - fit.width) // 2, 33))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 6), "Saved overlay color difference / visible change area {:.2%}".format(changed), fill=(239, 228, 187))
    draw.text((10, 19), "Warm: brighter B   /   Blue: darker B   /   Green: chromatic change", fill=(205, 217, 193))
    return encode_png(np.asarray(canvas))
