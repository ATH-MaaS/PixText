"""Persisted region format shared by the editor and headless workflows."""

from __future__ import annotations

import json

from pixtext import TextEdit


def parse_regions(value: str, width: int, height: int, source_id: str) -> list[TextEdit]:
    state = json.loads(value)
    if not isinstance(state, dict) or state.get("version") != 1:
        raise ValueError("Expected a PixText region document with version 1")
    if state.get("source_id") != source_id or state.get("size") != [width, height]:
        raise ValueError(
            "The image changed. Open the region editor and save regions for this image."
        )
    rows = state.get("regions")
    if not isinstance(rows, list) or len(rows) > 256:
        raise ValueError("The editor supports at most 256 regions")
    edits = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get("enabled", True), bool):
            raise ValueError(f"Region {index + 1} is invalid")
        if not row.get("enabled", True):
            continue
        try:
            edit = TextEdit(tuple(row["bbox"]), row["text"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Region {index + 1}: {exc}") from exc
        if edit.bbox[2] > width or edit.bbox[3] > height:
            raise ValueError(f"Region {index + 1} extends beyond the image")
        edits.append(edit)
    return edits
