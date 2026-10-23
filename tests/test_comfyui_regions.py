"""Validate editor documents without requiring a ComfyUI installation."""

import json
from pathlib import Path

import pytest

from comfyui.regions import parse_regions

ROOT = Path(__file__).resolve().parents[1]


def document(**updates):
    state = {
        "version": 1,
        "source_id": "file:test.png",
        "size": [100, 80],
        "regions": [{"bbox": [1.5, 2, 30, 40], "text": "喜", "enabled": True}],
    }
    state.update(updates)
    return json.dumps(state)


def test_editor_preserves_fractional_coordinates_and_whole_target_text():
    edits = parse_regions(document(), 100, 80, "file:test.png")
    assert edits[0].bbox == (1.5, 2, 30, 40)
    assert edits[0].text == "喜"


@pytest.mark.parametrize(
    "updates",
    [
        {"size": [80, 100]},
        {"source_id": "file:another.png"},
        {"version": 2},
    ],
)
def test_wrong_image_or_document_version_cannot_reuse_saved_regions(updates):
    with pytest.raises(ValueError):
        parse_regions(document(**updates), 100, 80, "file:test.png")


@pytest.mark.parametrize(
    "box,text",
    [
        ([1, 2, 101, 40], "a"),
        ([1, 2, 30, 40], " "),
        ([1, 2, float("nan"), 40], "a"),
        ([1, 2, 1, 40], "a"),
        ([True, 2, 30, 40], "a"),
    ],
)
def test_invalid_target_is_rejected_before_model_execution(box, text):
    value = document(regions=[{"bbox": box, "text": text}])
    with pytest.raises(ValueError):
        parse_regions(value, 100, 80, "file:test.png")


def test_disabled_regions_are_excluded_and_order_is_retained():
    rows = [
        {"bbox": [1, 2, 30, 40], "text": text, "enabled": enabled}
        for text, enabled in [("first", True), ("skip", False), ("last", True)]
    ]
    edits = parse_regions(document(regions=rows), 100, 80, "file:test.png")
    assert [edit.text for edit in edits] == ["first", "last"]


@pytest.mark.parametrize("variant", ["MeanFlow", "Teacher"])
def test_bundled_workflows_embed_the_exact_thirteen_chinese_targets(variant):
    workflow = json.loads((ROOT / f"example_workflows/PixText-Chinese-{variant}.json").read_text())
    editor = next(node for node in workflow["nodes"] if node["type"] == "PixTextRegionEditor")
    edits = parse_regions(editor["widgets_values"][1], 1024, 1024, "example:chinese")
    original = json.loads((ROOT / "examples/pipeline_case/edits.json").read_text())
    assert [{"bbox": list(edit.bbox), "text": edit.text} for edit in edits] == original
    assert len(edits) == 13
    refine = next(node for node in workflow["nodes"] if node["type"] == "PixTextRefine")
    assert refine["widgets_values"][2] == (4 if variant == "MeanFlow" else 25)
    assert any(node["type"] == "PixTextCompare" for node in workflow["nodes"])
