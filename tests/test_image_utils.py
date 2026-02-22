from __future__ import annotations

from io import BytesIO

from PIL import Image

from core.image_utils import (
    compress_for_photo,
    image_dimensions,
    resize_image_by_percent,
    shrink_image_to_box,
)


def _png_bytes(width: int, height: int, *, mode: str = "RGB") -> bytes:
    if mode == "RGBA":
        image = Image.new(mode, (width, height), (255, 0, 0, 128))
    else:
        image = Image.new(mode, (width, height), (255, 0, 0))
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def test_compress_for_photo_returns_original_when_already_within_limits() -> None:
    payload = _png_bytes(256, 256)

    compressed = compress_for_photo(payload, max_size=len(payload) + 1024)

    assert compressed == payload


def test_compress_for_photo_converts_rgba_to_jpeg_when_compression_is_forced() -> None:
    payload = _png_bytes(512, 512, mode="RGBA")

    compressed = compress_for_photo(payload, max_size=300)

    assert compressed.startswith(b"\xff\xd8\xff")


def test_resize_image_by_percent_updates_dimensions() -> None:
    payload = _png_bytes(400, 200)

    resized = resize_image_by_percent(payload, 50)

    assert image_dimensions(resized) == (200, 100)


def test_shrink_image_to_box_reduces_large_image() -> None:
    payload = _png_bytes(1000, 800)

    resized = shrink_image_to_box(payload, 400, 400)

    assert image_dimensions(resized) == (400, 320)


def test_shrink_image_to_box_returns_same_payload_when_already_within_limits() -> None:
    payload = _png_bytes(128, 128)

    resized = shrink_image_to_box(payload, 512, 512)

    assert resized == payload
