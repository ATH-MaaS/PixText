"""Run with input.png in the current directory and a local model package."""

from pixtext import PixTextPipeline, TextEdit

pipeline = PixTextPipeline.from_pretrained("models/PixText-MeanFlow")
result = pipeline.edit("input.png", [TextEdit((120, 80, 160, 125), "鹿")], seed=42)
result.image.save("output.png")
