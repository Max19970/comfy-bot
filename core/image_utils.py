from __future__ import annotations

from io import BytesIO

from PIL import Image

TG_PHOTO_MAX_BYTES = 10 * 1024 * 1024
TG_PHOTO_MAX_SIDE = 4096
TG_PHOTO_MAX_SUM = 10000


_resampling = getattr(Image, "Resampling", None)
LANCZOS_RESAMPLE = getattr(_resampling, "LANCZOS", 1)


def _fit_dimensions(width: int, height: int) -> tuple[int, int]:
    ratio = 1.0
    if width > TG_PHOTO_MAX_SIDE:
        ratio = min(ratio, TG_PHOTO_MAX_SIDE / width)
    if height > TG_PHOTO_MAX_SIDE:
        ratio = min(ratio, TG_PHOTO_MAX_SIDE / height)
    if width * ratio + height * ratio > TG_PHOTO_MAX_SUM:
        ratio = min(ratio, TG_PHOTO_MAX_SUM / (width + height))
    return (int(width * ratio), int(height * ratio)) if ratio < 1.0 else (width, height)


def compress_for_photo(
    image_bytes: bytes,
    max_size: int = TG_PHOTO_MAX_BYTES,
) -> bytes:
    image = Image.open(BytesIO(image_bytes))
    target_w, target_h = _fit_dimensions(image.width, image.height)
    needs_resize = target_w != image.width or target_h != image.height
    if not needs_resize and len(image_bytes) <= max_size:
        return image_bytes

    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        image = background
    elif image.mode != "RGB":
        image = image.convert("RGB")

    if needs_resize:
        image = image.resize((target_w, target_h), LANCZOS_RESAMPLE)

    for quality in (95, 90, 85, 80, 70, 60, 50, 40):
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=quality, optimize=True)
        if buffer.tell() <= max_size:
            return buffer.getvalue()

    factor = 0.75
    buffer = BytesIO()
    while factor > 0.1:
        resized = image.resize(
            (int(image.width * factor), int(image.height * factor)),
            LANCZOS_RESAMPLE,
        )
        buffer = BytesIO()
        resized.save(buffer, format="JPEG", quality=50, optimize=True)
        if buffer.tell() <= max_size:
            return buffer.getvalue()
        factor -= 0.1

    return buffer.getvalue()


def image_dimensions(image_bytes: bytes) -> tuple[int, int]:
    with Image.open(BytesIO(image_bytes)) as image:
        return image.width, image.height
