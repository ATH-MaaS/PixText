"""Build annotated SVG and PNG previews with boxes around the recorded edited footprints."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

from PIL import Image, ImageDraw

OUTLINE = (16, 59, 68)
HIGHLIGHT = (79, 244, 255)
SUPERSAMPLE = 4


def merged_footprints(metadata: dict) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    for group in metadata["geometry"]:
        for x, y, w, h in group["crops"]:
            boxes.append((x, y, x + w, y + h))
    # Merge intersecting footprints so adjacent repaired characters have one outline.
    merged: list[tuple[int, int, int, int]] = []
    while boxes:
        x0, y0, x1, y1 = boxes.pop(0)
        changed = True
        while changed:
            changed = False
            for other in boxes[:]:
                a, b, c, d = other
                if x0 <= c and a <= x1 and y0 <= d and b <= y1:
                    x0, y0, x1, y1 = min(x0, a), min(y0, b), max(x1, c), max(y1, d)
                    boxes.remove(other)
                    changed = True
        merged.append((x0, y0, x1, y1))
    return merged


def write_svg(
    case: Path, name: str, size: tuple[int, int], merged: list[tuple[int, int, int, int]]
) -> None:
    width, height = size
    encoded = base64.b64encode((case / f"{name}.png").read_bytes()).decode("ascii")
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">',
        f"<title>{name}: cyan boxes mark edited regions</title>",
        f'<image width="{width}" height="{height}" href="data:image/png;base64,{encoded}"/>',
    ]
    for x0, y0, x1, y1 in merged:
        for color, stroke in (("#103b44", 7), ("#4ff4ff", 4)):
            svg.append(
                f'<rect x="{x0}" y="{y0}" width="{x1 - x0}" height="{y1 - y0}" '
                f'fill="none" stroke="{color}" stroke-width="{stroke}" rx="3"/>'
            )
    svg.append("</svg>")
    output = case / f"{name}_boxed.svg"
    output.write_text("\n".join(svg) + "\n", encoding="utf-8")
    print(f"Saved {output.name}: {len(merged)} highlighted regions")


def write_boxed_png(
    case: Path, name: str, image: Image.Image, merged: list[tuple[int, int, int, int]]
) -> None:
    width, height = image.size
    overlay = Image.new("RGBA", (width * SUPERSAMPLE, height * SUPERSAMPLE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for x0, y0, x1, y1 in merged:
        for color, stroke in ((OUTLINE, 7), (HIGHLIGHT, 4)):
            half = stroke * SUPERSAMPLE / 2
            draw.rounded_rectangle(
                [
                    x0 * SUPERSAMPLE - half,
                    y0 * SUPERSAMPLE - half,
                    x1 * SUPERSAMPLE + half,
                    y1 * SUPERSAMPLE + half,
                ],
                radius=3 * SUPERSAMPLE,
                outline=(*color, 255),
                width=stroke * SUPERSAMPLE,
            )
    overlay = overlay.resize((width, height), Image.LANCZOS)
    boxed = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    output = case / f"{name}_boxed.png"
    boxed.save(output)
    print(f"Saved {output.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("pipeline", "english"), default="pipeline")
    args = parser.parse_args()
    case = Path(__file__).resolve().parent / f"{args.case}_case"
    for name in ("input", "meanflow_output", "teacher_output"):
        metadata_name = "meanflow_output" if name == "input" else name
        metadata = json.loads((case / f"{metadata_name}.json").read_text())
        merged = merged_footprints(metadata)
        image = Image.open(case / f"{name}.png").convert("RGB")
        write_svg(case, name, image.size, merged)
        write_boxed_png(case, name, image, merged)


if __name__ == "__main__":
    main()
