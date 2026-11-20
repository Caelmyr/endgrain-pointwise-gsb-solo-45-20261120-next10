from hashlib import sha256
from io import BytesIO
from pathlib import PurePath
from typing import Any

import numpy as np
from PIL import Image

from .database import Database
from .errors import WorkbenchError
from .imaging import ImageInput, canonical_png, read_image, thumbnail_png
from .validation import choice, string

SURFACES = ("sanded", "sawn", "polished", "unknown")
SOURCE_TYPES = ("sample", "reference")
MEDIA_EXTENSIONS = {"PNG": "png", "JPEG": "jpg", "TIFF": "tiff", "WEBP": "webp"}


def safe_stem(filename: str) -> str:
    basename = PurePath(filename.replace("\\", "/")).name
    filtered = "".join(char if ord(char) >= 32 else " " for char in PurePath(basename).stem)
    return filtered.strip()[:90] or "Untitled face"


def capture_payload(
    image: ImageInput,
    title: str,
    filename: str,
    *,
    synthetic: bool = False,
    source_type: str = "sample",
) -> dict[str, Any]:
    width, height = image.dimensions
    orig_w, orig_h = image.original_size
    return {
        "title": title,
        "digest": image.digest,
        "source": image.content,
        "canonical": canonical_png(image),
        "thumbnail": thumbnail_png(image),
        "metadata": {
            "width": width,
            "height": height,
            "file_width": orig_w,
            "file_height": orig_h,
            "format": image.format,
            "mime": image.media_type,
            "extension": MEDIA_EXTENSIONS[image.format],
            "filename": safe_stem(filename) + "." + MEDIA_EXTENSIONS[image.format],
            "bytes": len(image.content),
            "orientation": image.orientation,
            "color_profile": image.profile,
            "warnings": image.warnings,
            "support_fraction": round(float(image.available.mean()), 8),
            "synthetic": synthetic,
            "source_type": source_type,
        },
    }


def observation_fields(form: dict, fallback: str) -> dict[str, str | dict]:
    title = string(form.get("title") or fallback, "inspection name", 100, required=True)
    material = string(form.get("material"), "wood species", 80)
    lot = string(form.get("lot"), "lot identifier", 72)
    surface = choice(form.get("surface") or "unknown", "surface", SURFACES)
    return {
        "title": title,
        "material": material,
        "lot": lot,
        "surface": surface,
        "state": {"calibration": {"mode": "pixels"}, "region": None, "lighting": None},
    }


def ingest_inspection(db: Database, raw: bytes, filename: str, form: dict) -> dict:
    image = read_image(raw)
    name = safe_stem(filename)
    values = observation_fields(form, name)
    payload = capture_payload(image, name, filename)
    return db.import_inspection(payload, values)


def ingest_reference(db: Database, raw: bytes, filename: str, title: str | None) -> dict:
    image = read_image(raw)
    name = string(title or safe_stem(filename), "reference name", 100, required=True)
    payload = capture_payload(image, name, filename, source_type="reference")
    with db.transaction() as conn:
        capture_id = db.put_capture(conn, payload)
    return db.capture(capture_id)


def decoded(db: Database, capture_id: str) -> tuple[np.ndarray, np.ndarray, dict]:
    png, capture = db.capture_blob(capture_id, "canonical")
    with Image.open(BytesIO(png)) as image:
        rgba = np.asarray(image.convert("RGBA")).copy()
    available = rgba[..., 3] == 255
    pixels = np.ascontiguousarray(rgba[..., :3])
    return pixels, available, capture


def select_references(db: Database) -> list[dict]:
    return [entry for entry in db.captures() if entry["metadata"]["source_type"] == "reference"]


def reference_for(db: Database, reference_id: str, width: int, height: int) -> dict:
    candidate = db.capture(reference_id)
    metadata = candidate["metadata"]
    if metadata["source_type"] != "reference":
        raise WorkbenchError("Choose an illumination reference capture.", "reference_id")
    if (metadata["width"], metadata["height"]) != (width, height):
        raise WorkbenchError("The sample and reference must have the same oriented dimensions.", "reference_id")
    return candidate


def image_etag(capture: dict, body: bytes, variant: str) -> str:
    if variant == "original":
        return capture["digest"]
    return sha256(body).hexdigest()
