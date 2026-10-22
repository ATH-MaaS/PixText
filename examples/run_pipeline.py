"""Run a bundled text refinement example using a local model package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pixtext import PixTextPipeline, TextEdit


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("pipeline", "english"), default="pipeline")
    parser.add_argument("--variant", choices=("meanflow", "teacher"), default="meanflow")
    parser.add_argument("--model", type=Path, help="Override the local model package directory")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    name = "PixText-MeanFlow" if args.variant == "meanflow" else "PixText-Teacher"
    model = args.model if args.model is not None else root / "models" / name
    if not model.is_dir():
        parser.error(f"Place the complete {name} model package at {model}, or use --model PATH")
    case_name = f"{args.case}_case"
    case = root / "examples" / case_name
    rows = json.loads((case / "edits.json").read_text(encoding="utf-8"))
    pipeline = PixTextPipeline.from_pretrained(model)
    result = pipeline.edit(
        case / "input.png",
        [TextEdit(bbox=tuple(row["bbox"]), text=row["text"]) for row in rows],
        seed=42,
    )
    output_dir = args.output_dir if args.output_dir is not None else root / "outputs" / case_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{args.variant}.png"
    result.image.save(output)
    output.with_suffix(".json").write_text(json.dumps(result.metadata, indent=2) + "\n")
    print(f"Saved image: {output.resolve()}")
    print(f"Saved reproduction metadata: {output.with_suffix('.json').resolve()}")


if __name__ == "__main__":
    main()
