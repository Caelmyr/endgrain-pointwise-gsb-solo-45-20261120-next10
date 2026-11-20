from dataclasses import dataclass
from io import BytesIO
from math import hypot
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from .calibration_preview import positive_tick, readings
from .database import Database
from .imaging import encode_png, symmetric_overlay
from .workspaces import Frame, frame_for


@dataclass(frozen=True)
class Plate:
    rgb: Image.Image
    offset: tuple[int, int]
    scale: float
    original: tuple[int, int]


def contain(
    squares: np.ndarray, size: tuple[int, int], background: tuple[int, int, int] = (44, 47, 43),
) -> Plate:
    photo = Image.fromarray(squares)
    resized = ImageOps.contain(photo, size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, background)
    position = ((size[0] - resized.width) // 2, (size[1] - resized.height) // 2)
    canvas.paste(resized, position)
    return Plate(canvas, position, resized.width / photo.width, photo.size)


def canvas_position(point: list[float], layout: Plate) -> tuple[float, float]:
    x, y = point
    return (
        layout.offset[0] + (x + 0.5) * layout.scale,
        layout.offset[1] + (y + 0.5) * layout.scale,
    )


def sized_label_rgb(
    painting: ImageDraw.ImageDraw,
    pos: tuple[float, float],
    label: str,
    colors: tuple[tuple[int, int, int], tuple[int, int, int]],
    available: int,
) -> None:
    x, y = pos
    label_width = painting.textlength(label)
    if label_width > available:
        while label and painting.textlength(label + "…") > available:
            label = label[:-1]
        label += "…"
        label_width = painting.textlength(label)
    fill, text = colors
    painting.rectangle((x, y, x + label_width + 8, y + 15), fill=fill)
    painting.text((x + 4, y + 1), label, fill=text)


def calibration_positions(working: Frame) -> tuple[list, str]:
    cal = working.raster.plane.calibration
    if cal["mode"] == "ruler":
        return [cal["a"], cal["b"]], f"Ruler · {cal['length_mm']:g} mm"
    if cal["mode"] == "quadrilateral":
        return cal["corners"], f"Rectangle · {cal['width_mm']:g} × {cal['height_mm']:g} mm"
    return [], "Pixel scale"


def source_panel(frame: Frame, layout_size: tuple[int, int]) -> Image.Image:
    source = symmetric_overlay(frame.source, [(~frame.source_support, (125, 89, 82), 0.72)])
    plate = contain(source, layout_size)
    painter = ImageDraw.Draw(plate.rgb)
    anchors, label = calibration_positions(frame)
    points = [canvas_position(value, plate) for value in anchors]
    accent = (248, 217, 154)
    if points:
        path = points + points[:1] if len(points) == 4 else points
        painter.line(path, fill=accent, width=2)
    for index, (x, y) in enumerate(points, 1):
        painter.ellipse((x - 4, y - 4, x + 4, y + 4), fill=accent, outline=(34, 39, 33))
        pos = (min(x + 5, plate.rgb.width - 18), min(y + 5, plate.rgb.height - 16))
        sized_label_rgb(painter, pos, str(index), ((42, 49, 40), (239, 223, 177)), 22)
    sized_label_rgb(painter, (7, 6), label, ((42, 49, 40), (239, 223, 177)), plate.rgb.width - 18)
    return plate.rgb


def draw_region(plate: Plate, frame: Frame) -> None:
    points = frame.inspection["state"]["region"]
    if not points:
        return
    color = (221, 211, 156)
    entries = [canvas_position(point, plate) for point in points]
    painter = ImageDraw.Draw(plate.rgb)
    for start, stop in zip(entries, entries[1:] + entries[:1], strict=True):
        dx, dy = stop[0] - start[0], stop[1] - start[1]
        span = hypot(dx, dy)
        if not span:
            continue
        for piece in np.arange(0, span, 10.0):
            a, b = piece / span, min(span, piece + 5.0) / span
            painter.line(
                ((start[0] + a * dx, start[1] + a * dy), (start[0] + b * dx, start[1] + b * dy)),
                fill=color, width=2,
            )


def draw_scale(plate: Plate, frame: Frame) -> None:
    data = readings(frame.raster)
    plane = data.plane
    tick = positive_tick(data.extents[0] * 0.15)
    proportion = tick / data.extents[0]
    length = max(6, round(plate.original[0] * plate.scale * proportion))
    x = plate.offset[0] + 12
    y = min(plate.rgb.height - 12, plate.offset[1] + plate.original[1] * plate.scale - 12)
    painter = ImageDraw.Draw(plate.rgb)
    backing = (44, 50, 39)
    tint = (247, 224, 166)
    label = f"{tick:g} {plane.unit}"
    if len(label) * 8 > length:
        length_for_box = len(label) * 8
    else:
        length_for_box = length
    painter.rectangle((x - 5, y - 25, x + length_for_box + 5, y + 7), fill=backing)
    painter.text((x + 1, y - 22), label, fill=tint)
    painter.line((x, y, x + length, y), fill=tint, width=2)
    painter.line((x, y - 4, x, y + 3), fill=tint, width=2)
    painter.line((x + length, y - 4, x + length, y + 3), fill=tint, width=2)


def draw_markers(plate: Plate, marks: list[dict]) -> None:
    painter = ImageDraw.Draw(plate.rgb)
    color = (255, 200, 134)
    for index, mark in enumerate(marks, 1):
        x, y = canvas_position(mark["point"], plate)
        painter.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color, outline=(51, 41, 30), width=1)
        label = str(index) + "  " + mark["label"]
        tx = max(7, min(x + 7, plate.rgb.width - 130))
        ty = max(5, min(y - 16, plate.rgb.height - 17))
        sized_label_rgb(painter, (tx, ty), label, ((60, 50, 39), (252, 226, 183)), int(plate.rgb.width - tx - 8))


def plane_panel(frame: Frame, marks: list[dict], layout_size: tuple[int, int]) -> Image.Image:
    raster = frame.raster
    rendered = symmetric_overlay(
        raster.rgb, [(~raster.supported, (135, 88, 76), 0.69)], support=raster.inspected,
    )
    plate = contain(rendered, layout_size)
    draw_region(plate, frame)
    draw_markers(plate, marks)
    draw_scale(plate, frame)
    return plate.rgb


def report_plate(db: Database, report: dict) -> bytes:
    manifest = report["manifest"]
    current = frame_for(db, manifest["inspection"])
    marks = manifest["markers"]
    layout_size = (637, 471)
    original = source_panel(current, layout_size)
    projected = plane_panel(current, marks, layout_size)
    layout = Image.new("RGB", (1308, 557), (240, 237, 223))
    painter = ImageDraw.Draw(layout)
    base_labels = (
        ("SOURCE / RAW CAPTURE", original),
        ("WORKING / MEASURING SUPPORT", projected),
    )
    for index, (title, image) in enumerate(base_labels):
        start_x = 11 + index * (layout_size[0] + 12)
        layout.paste(image, (start_x, 55))
        painter.text((start_x, 32), title, fill=(66, 79, 53))
    spec = manifest["inspection"]
    name = ("[Synthetic] " if spec["capture_metadata"]["synthetic"] else "") + spec["title"]
    painter.text((10, 10), name, fill=(61, 68, 49))
    revision = spec["revision"]
    detail = f"Revision {revision}   /   {len(marks)} markers   /   Source {spec['source_digest'][:16]}"
    painter.text((10, 535), detail, fill=(61, 68, 49))
    return encode_png(np.asarray(layout))
