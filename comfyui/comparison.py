"""Crop matching regions and build a portable, numbered comparison sheet."""

from __future__ import annotations

import math

from PIL import Image, ImageDraw, ImageOps


def compare_regions(before, after, edits, *, padding=6, tile_size=160):
    if before.size != after.size:
        raise ValueError("Before and after images must have the same dimensions")
    if not edits:
        sheet = Image.new("RGB", (480, 120), "#101820")
        ImageDraw.Draw(sheet).text(
            (24, 50), "Enable a target region to compare before and after.", fill="#82ead4"
        )
        return sheet, []
    pairs = []
    columns = min(2, len(edits))
    card_w, card_h = tile_size * 2 + 28, tile_size + 66
    sheet = Image.new(
        "RGB", (columns * card_w + 16, math.ceil(len(edits) / columns) * card_h + 16), "#101820"
    )
    draw = ImageDraw.Draw(sheet)
    for index, edit in enumerate(edits):
        x0, y0, x1, y1 = edit.bbox
        box = (
            max(0, math.floor(x0 - padding)),
            max(0, math.floor(y0 - padding)),
            min(before.width, math.ceil(x1 + padding)),
            min(before.height, math.ceil(y1 + padding)),
        )
        original, repaired = before.crop(box), after.crop(box)
        pairs.append((original, repaired))
        left = 8 + (index % columns) * card_w
        top = 8 + (index // columns) * card_h
        draw.rounded_rectangle(
            (left, top, left + card_w - 8, top + card_h - 8), radius=10, fill="#1c2633"
        )
        draw.text((left + 10, top + 8), f"REGION {index + 1:02d}", fill="#82ead4")
        for offset, (crop, label) in enumerate([(original, "BEFORE"), (repaired, "AFTER")]):
            x, y = left + 10 + offset * (tile_size + 8), top + 46
            draw.text((x, top + 28), label, fill="#c0ccda")
            tile = ImageOps.contain(crop, (tile_size, tile_size), Image.Resampling.LANCZOS)
            draw.rounded_rectangle(
                (x, y, x + tile_size - 1, y + tile_size - 1), radius=6, fill="#0b1018"
            )
            sheet.paste(
                tile, (x + (tile_size - tile.width) // 2, y + (tile_size - tile.height) // 2)
            )
    return sheet, pairs
