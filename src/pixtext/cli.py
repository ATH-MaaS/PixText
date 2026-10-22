"""Command-line adapter for the PixText editing interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pixtext import PixTextPipeline, TextEdit


def main() -> None:
    parser = argparse.ArgumentParser(description="Refine text in specified original-image regions")
    parser.add_argument(
        "--model", type=Path, required=True, help="Local self-contained model directory"
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--edits", type=Path, required=True, help="JSON list of {bbox: [x0,y0,x1,y1], text: ...}"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--guidance-scale", type=float)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--sample-id", default="image", help="Stable identity for per-image random streams"
    )
    args = parser.parse_args()
    if args.output.suffix.lower() != ".png":
        parser.error("--output must be PNG to preserve unedited pixels exactly")
    rows = json.loads(args.edits.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        parser.error("--edits must contain a JSON list")
    targets = [TextEdit(bbox=tuple(row["bbox"]), text=row["text"]) for row in rows]
    pipeline = PixTextPipeline.from_pretrained(args.model, device=args.device)
    result = pipeline.edit(
        args.image,
        targets,
        seed=args.seed,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        sample_id=args.sample_id,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.image.save(args.output)
    args.output.with_suffix(".json").write_text(json.dumps(result.metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
