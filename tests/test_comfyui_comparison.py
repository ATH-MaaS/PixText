"""Before/after crops must refer to exactly the same original-image area."""

import numpy as np
import pytest
from PIL import Image

from comfyui.comparison import compare_regions
from pixtext import TextEdit


def test_matching_crops_include_clamped_context_without_changing_aspect_ratio():
    pixels = np.arange(20 * 30 * 3, dtype=np.uint8).reshape(20, 30, 3)
    before = Image.fromarray(pixels)
    after = Image.fromarray(255 - pixels)
    _, pairs = compare_regions(before, after, [TextEdit((1.5, 2, 10.2, 12), "Hi")], padding=4)
    left, right = pairs[0]
    assert np.array_equal(np.asarray(left), pixels[0:16, 0:15])
    assert np.array_equal(np.asarray(right), (255 - pixels)[0:16, 0:15])
    assert left.size == right.size


def test_region_order_and_count_survive_sheet_layout():
    image = Image.new("RGB", (30, 20), "red")
    edits = [TextEdit((i, 0, i + 2, 2), str(i)) for i in range(13)]
    sheet, pairs = compare_regions(image, image, edits, padding=0)
    assert len(pairs) == 13
    assert all(a.size == b.size == (2, 2) for a, b in pairs)
    assert sheet.mode == "RGB" and sheet.height > sheet.width


def test_different_image_sizes_cannot_be_presented_as_matching_crops():
    with pytest.raises(ValueError, match="same dimensions"):
        compare_regions(
            Image.new("RGB", (20, 20)), Image.new("RGB", (10, 20)), [TextEdit((0, 0, 2, 2), "Hi")]
        )


def test_no_enabled_regions_produces_a_hint_instead_of_an_error():
    image = Image.new("RGB", (30, 20))
    sheet, pairs = compare_regions(image, image, [])
    assert pairs == [] and sheet.mode == "RGB"
