"""Exercise the public editing seam using the real bundled release models."""

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from pixtext import PixTextPipeline, TextEdit

MODELS = Path(os.environ.get("PIXTEXT_TEST_MODELS", Path(__file__).resolve().parents[1] / "models"))
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


@pytest.fixture(scope="module", params=["PixText-Teacher", "PixText-MeanFlow"])
def pipeline(request):
    package = MODELS / request.param
    if not package.is_dir():
        pytest.skip("Set PIXTEXT_TEST_MODELS to the bundled release packages")
    result = PixTextPipeline.from_pretrained(package)
    yield result
    del result
    torch.cuda.empty_cache()


def image(width=128, height=128):
    rng = np.random.default_rng(17)
    return Image.fromarray(rng.integers(30, 220, size=(height, width, 3), dtype=np.uint8))


def test_single_image_preserves_pixels_outside_actual_crop(pipeline):
    source = image()
    result = pipeline.edit(source, [TextEdit((40, 40, 58, 62), "鹿")], seed=17)
    a, b = np.array(source), np.array(result.image)
    mask = np.zeros(a.shape[:2], bool)
    for wave in result.metadata["geometry"]:
        for x, y, w, h in wave["crops"]:
            mask[y : y + h, x : x + w] = True
    assert np.array_equal(a[~mask], b[~mask])
    assert np.any(a[mask] != b[mask])
    assert result.metadata["steps"] == (25 if pipeline.variant == "teacher" else 4)


def test_batch_matches_single_with_independent_seeds_and_multiple_targets(pipeline):
    sources = [image(), image()]
    targets = [
        [TextEdit((20, 20, 38, 44), "丰"), TextEdit((70, 70, 104, 96), "Hi")],
        [TextEdit((40, 40, 58, 62), "鹿")],
    ]
    batch = pipeline.edit_batch(sources, targets, seeds=[21, 22], sample_ids=["first", "second"])
    singles = [
        pipeline.edit(src, ts, seed=seed, sample_id=identity)
        for src, ts, seed, identity in zip(
            sources, targets, [21, 22], ["first", "second"], strict=True
        )
    ]
    for batched, single in zip(batch, singles, strict=True):
        # Packed GPU kernels can round differently with a different sequence length.
        difference = np.abs(
            np.array(batched.image).astype(float) - np.array(single.image).astype(float)
        )
        assert difference.mean() < 1.0
        assert batched.metadata["seed"] == single.metadata["seed"]


def test_unaligned_input_preserves_original_pixels_and_dimensions(pipeline):
    source = image(137, 119)
    result = pipeline.edit(source, [TextEdit((39, 40, 58, 62), "丰")])
    assert result.image.size == source.size
    assert np.array_equal(np.array(result.image)[:20], np.array(source)[:20])
    assert np.array_equal(np.array(result.image)[:, 100:], np.array(source)[:, 100:])


def test_empty_edits_return_original_image(pipeline):
    source = image()
    result = pipeline.edit(source, [])
    assert np.array_equal(np.array(result.image), np.array(source))


def test_invalid_regions_and_seeds_are_rejected(pipeline):
    with pytest.raises(ValueError, match="outside"):
        pipeline.edit(image(), [TextEdit((100, 100, 140, 140), "丰")])
    with pytest.raises(ValueError, match="Seeds"):
        pipeline.edit(image(), [], seed=True)


def test_more_than_twelve_edits_use_multiple_passes(pipeline):
    targets = [
        TextEdit((8 + 24 * (i % 4), 8 + 24 * (i // 4), 20 + 24 * (i % 4), 24 + 24 * (i // 4)), "丰")
        for i in range(13)
    ]
    result = pipeline.edit(image(), targets, seed=19)
    assert [wave["offset"] for wave in result.metadata["geometry"]] == [0, 12]
    assert sum(len(wave["crops"]) for wave in result.metadata["geometry"]) == 13
    assert result.image.size == (128, 128)


def test_glyph_coverage_error_identifies_target_text(pipeline):
    with pytest.raises(ValueError, match="glyph|font|cover"):
        pipeline.edit(image(), [TextEdit((40, 40, 58, 62), "\U0010ffff")])


def test_whole_string_font_fallback_is_available_without_user_font_configuration(pipeline):
    result = pipeline.edit(image(), [TextEdit((40, 40, 74, 70), "A⠁")])
    assert result.metadata["geometry"][0]["selected_font_indices"] == [1]
    assert result.image.size == (128, 128)
