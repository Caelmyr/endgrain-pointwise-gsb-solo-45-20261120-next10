from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import warnings

import numpy as np
from PIL import ExifTags, Image, ImageCms, ImageOps, UnidentifiedImageError
from scipy import ndimage as ndi

from .errors import Limits, WorkbenchError

MAX_UPLOAD_BYTES = 32 * 1024 * 1024
MAX_PIXELS = 16_000_000
MIN_SIDE = 48
WORK_LIMIT = 1800
ACCEPTED_FORMATS = {"PNG", "JPEG", "TIFF", "WEBP"}
Image.MAX_IMAGE_PIXELS = MAX_PIXELS


@dataclass(frozen=True)
class ImageInput:
    content: bytes
    pixels: np.ndarray
    available: np.ndarray
    format: str
    media_type: str
    orientation: int
    profile: str
    warnings: tuple[str, ...]
    original_size: tuple[int, int]

    @property
    def digest(self) -> str:
        return sha256(self.content).hexdigest()

    @property
    def dimensions(self) -> tuple[int, int]:
        return self.pixels.shape[1], self.pixels.shape[0]


def convert_color(image: Image.Image, profile: bytes | None) -> tuple[Image.Image, str, list[str]]:
    rgb_input = image.convert("RGB") if image.mode == "RGBA" else image
    if not profile:
        return image.convert("RGB"), "assumed sRGB", []
    try:
        source = ImageCms.ImageCmsProfile(BytesIO(profile))
        output = ImageCms.profileToProfile(
            rgb_input if rgb_input.mode in {"RGB", "CMYK", "LAB", "L"} else rgb_input.convert("RGB"),
            source,
            ImageCms.createProfile("sRGB"),
            outputMode="RGB",
        )
    except (OSError, TypeError, ValueError, ImageCms.PyCMSError):
        return image.convert("RGB"), "assumed sRGB", ["Embedded color profile could not be applied."]
    return output, "converted to sRGB", []


def read_image(content: bytes) -> ImageInput:
    if not content:
        raise WorkbenchError("Choose an image to import.", "file")
    if len(content) > MAX_UPLOAD_BYTES:
        raise Limits("Upload an image no larger than 32 MiB.", "file")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(content)) as original:
                return _decode_open_image(original, content)
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise Limits("The image exceeds 16 megapixels.", "file") from None
    except (UnidentifiedImageError, OSError, SyntaxError):
        raise WorkbenchError("The image could not be decoded.", "file") from None


def _decode_open_image(original: Image.Image, content: bytes) -> ImageInput:
    file_format = original.format
    original_size = original.size
    if file_format not in ACCEPTED_FORMATS:
        raise WorkbenchError("Use a PNG, JPEG, TIFF or WebP image.", "file")
    if getattr(original, "n_frames", 1) != 1:
        raise WorkbenchError("Split multi-frame images before importing.", "file")
    if min(original_size) < MIN_SIDE:
        raise WorkbenchError(f"Both image sides must be at least {MIN_SIDE} pixels.", "file")
    if original_size[0] * original_size[1] > MAX_PIXELS:
        raise Limits("The image exceeds 16 megapixels.", "file")
    orientation = int(original.getexif().get(ExifTags.Base.Orientation, 1))
    oriented = ImageOps.exif_transpose(original)
    source_alpha = "A" in oriented.getbands() or "transparency" in oriented.info
    if source_alpha:
        available = np.asarray(oriented.convert("RGBA").getchannel("A")) == 255
    else:
        available = np.ones((oriented.height, oriented.width), dtype=bool)
    if np.count_nonzero(available) < MIN_SIDE * MIN_SIDE:
        raise WorkbenchError("The image has too few fully opaque pixels to inspect.", "file")
    converted, profile, notices = convert_color(oriented, oriented.info.get("icc_profile"))
    if source_alpha and not available.all():
        notices.append("Partially transparent pixels are outside the measuring support.")
    if oriented.mode in {"I", "I;16", "F"}:
        notices.append("High bit-depth input is converted to 8-bit preview values.")
        pixels = high_bit_gray_preview(oriented)
    else:
        pixels = np.array(converted, dtype=np.uint8)
    return ImageInput(
        content=content,
        pixels=pixels,
        available=available,
        format=file_format,
        media_type=Image.MIME[file_format],
        orientation=orientation,
        profile=profile,
        warnings=tuple(notices),
        original_size=original_size,
    )


def high_bit_gray_preview(source: Image.Image) -> np.ndarray:
    values = np.asarray(source, dtype=np.float64)
    if source.mode.startswith("I;16"):
        lower, upper = 0.0, 65535.0
    else:
        finite = values[np.isfinite(values)]
        if not finite.size:
            raise WorkbenchError("The source image has no finite pixel values.", "file")
        lower, upper = np.percentile(finite, [0.15, 99.85])
    if upper <= lower:
        scaled = np.zeros(values.shape, dtype=np.uint8)
    else:
        values = np.nan_to_num(values, nan=lower, neginf=lower, posinf=upper)
        scaled = np.clip(np.rint((values - lower) * (255.0 / (upper - lower))), 0, 255).astype(np.uint8)
    return np.repeat(scaled[:, :, None], 3, axis=2)


def encode_png(pixels: np.ndarray) -> bytes:
    out = BytesIO()
    Image.fromarray(pixels).save(out, format="PNG", compress_level=4)
    return out.getvalue()


def read_canonical(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with Image.open(path) as ready:
        rgba = np.array(ready.convert("RGBA"), dtype=np.uint8)
    return rgba[:, :, :3].copy(), rgba[:, :, 3] == 255


def canonical_png(image: ImageInput) -> bytes:
    if image.available.all():
        return encode_png(image.pixels)
    alpha = image.available.astype(np.uint8) * 255
    return encode_png(np.dstack((image.pixels, alpha)))


def thumbnail_png(image: ImageInput, longest: int = 480) -> bytes:
    thumbnail = Image.fromarray(image.pixels.copy())
    thumbnail.thumbnail((longest, longest), Image.Resampling.LANCZOS)
    return encode_png(np.asarray(thumbnail))


def rgb_float(pixels: np.ndarray) -> np.ndarray:
    pixels = np.asarray(pixels)
    if np.issubdtype(pixels.dtype, np.integer):
        return pixels.astype(np.float32) / 255.0
    return pixels.astype(np.float32, copy=False)


def srgb_to_linear(pixels: np.ndarray) -> np.ndarray:
    base = rgb_float(pixels)
    return np.where(base <= 0.04045, base / 12.92, ((base + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 0.0, 1.0)
    scaled = np.where(clipped <= 0.0031308, clipped * 12.92, 1.055 * clipped ** (1.0 / 2.4) - 0.055)
    return np.rint(scaled * 255).astype(np.uint8)


def luminance(rgb: np.ndarray, *, linear: bool = False) -> np.ndarray:
    working = srgb_to_linear(rgb) if linear else rgb_float(rgb)
    return (
        working[..., 0] * np.float32(0.2126)
        + working[..., 1] * np.float32(0.7152)
        + working[..., 2] * np.float32(0.0722)
    )


def masked_smooth(image: np.ndarray, support: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    support_float = support.astype(np.float32)
    confidence = ndi.gaussian_filter(support_float, sigma, mode="constant")
    smoothed = ndi.gaussian_filter(np.where(support, image, 0).astype(np.float32), sigma, mode="constant")
    np.divide(smoothed, confidence, out=smoothed, where=confidence > 0.0001)
    return smoothed, confidence


def symmetric_overlay(
    rgb: np.ndarray,
    masks: list[tuple[np.ndarray, tuple[int, int, int], float]],
    *,
    support: np.ndarray | None = None,
) -> np.ndarray:
    result = rgb.astype(np.float32)
    for mask, color, strength in masks:
        if np.any(mask):
            result[mask] = (1 - strength) * result[mask] + strength * np.asarray(color)
    if support is not None:
        result[~support] *= 0.42
    return np.rint(np.clip(result, 0, 255)).astype(np.uint8)


def rgba_layer(color: tuple[int, int, int], alpha: np.ndarray) -> np.ndarray:
    output = np.empty((*alpha.shape, 4), dtype=np.uint8)
    output[..., :3] = color
    output[..., 3] = alpha
    return output


def quantile(values: np.ndarray, limit: float) -> float | None:
    if values.size == 0:
        return None
    return float(np.percentile(values, limit))


def safe_ratio(top: int | float, bottom: int | float) -> float | None:
    return float(top / bottom) if bottom else None
