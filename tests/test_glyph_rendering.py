"""Exercise RAQM rendering with the installed Pillow and release fonts on CPU."""

import os
from pathlib import Path

import pytest
import torch
from PIL import features

from pixtext.preprocessing.geometry import BoundingBox, CropGeometry, ImageSize, ItemGeometry
from pixtext.preprocessing.glyphs import CanonicalItemGlyphRenderer


@pytest.fixture(scope="module")
def renderer():
    models = Path(
        os.environ.get("PIXTEXT_TEST_MODELS", Path(__file__).resolve().parents[1] / "models")
    )
    fonts = models / "PixText-MeanFlow" / "fonts"
    if not fonts.is_dir():
        pytest.skip("Set PIXTEXT_TEST_MODELS to the release model packages for font tests")
    assert features.check_feature("raqm"), "The installed Pillow must provide RAQM"
    result = CanonicalItemGlyphRenderer(
        font_path=fonts / "AlibabaPuHuiTi-3-55-Regular.ttf",
        fallback_font_path=fonts / "GoNotoCJKCore-v7.0.ttf",
    )
    yield result
    result.coverage.close()


@pytest.mark.parametrize("text,font_index", [("Nayeon", 0), ("喜", 0), ("A⠁", 1)])
def test_raqm_renders_english_chinese_and_fallback_glyphs(renderer, text, font_index):
    bbox = BoundingBox(24, 24, 168, 104)
    geometry = ItemGeometry(bbox, bbox, CropGeometry(0, 0, ImageSize(128, 192)))

    glyph = renderer.render(text, geometry)

    assert glyph.selected_font_index == font_index
    assert glyph.text == text
    assert glyph.image.shape == (1, 128, 192)
    assert glyph.image.dtype == torch.float32
    assert torch.isfinite(glyph.image).all()
    assert glyph.image.min() == 0
    assert 0 < glyph.image.max() <= 1
    ink = glyph.image[0].nonzero()
    assert ink[:, 0].min() >= bbox.y0
    assert ink[:, 0].max() < bbox.y1
    assert ink[:, 1].min() >= bbox.x0
    assert ink[:, 1].max() < bbox.x1
    assert torch.equal(glyph.image, renderer.render(text, geometry).image)
