"""User-facing edit requests and results."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from PIL import Image


@dataclass(frozen=True)
class TextEdit:
    """Target text in an original-image half-open (x0, y0, x1, y1) box."""

    bbox: tuple[float, float, float, float]
    text: str

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("Target text must be a nonempty string")
        if len(self.bbox) != 4 or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
            for v in self.bbox
        ):
            raise ValueError("bbox must contain four finite coordinates")
        x0, y0, x1, y1 = self.bbox
        if x0 < 0 or y0 < 0 or x1 <= x0 or y1 <= y0:
            raise ValueError("bbox must have nonnegative origin and positive area")


@dataclass(frozen=True)
class EditResult:
    image: Image.Image
    metadata: dict[str, Any] = field(default_factory=dict)
