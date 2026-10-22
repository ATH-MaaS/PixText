"""Deterministic geometry planning for group-image JiT inputs.

This module computes pixel-coordinate transforms only.  Image interpolation and
tensor slicing are deliberately left to data adapters so the geometry contract
does not depend on PIL, NumPy, or PyTorch.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction
from numbers import Real


@dataclass(frozen=True, slots=True)
class ImageSize:
    """An image size in ``(height, width)`` order."""

    height: int
    width: int

    def __post_init__(self) -> None:
        for name, value in (("height", self.height), ("width", self.width)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """A half-open pixel-boundary box in ``[x0, y0, x1, y1)`` form."""

    x0: Real
    y0: Real
    x1: Real
    y1: Real

    def __post_init__(self) -> None:
        for name, value in (("x0", self.x0), ("y0", self.y0), ("x1", self.x1), ("y1", self.y1)):
            if isinstance(value, bool) or not isinstance(value, Real) or (not math.isfinite(value)):
                raise ValueError(f"{name} must be a finite number, got {value!r}")
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError(f"bbox must be non-degenerate, got {self!r}")


@dataclass(frozen=True, slots=True)
class ReferenceGeometry:
    """The reference canvas and its scale from the logical canvas."""

    size: ImageSize
    scale_h: float
    scale_w: float


@dataclass(frozen=True, slots=True)
class MainImageGeometry:
    """The main image size and its actual scale from the reference canvas."""

    size: ImageSize
    scale_h: float
    scale_w: float


@dataclass(frozen=True, slots=True)
class CropGeometry:
    """An integer slice rectangle on the reference image."""

    top: int
    left: int
    size: ImageSize


@dataclass(frozen=True, slots=True)
class ItemGeometry:
    """One item's bbox in logical and reference coordinates."""

    logical_bbox: BoundingBox
    reference_bbox: BoundingBox
    crop: CropGeometry


@dataclass(frozen=True, slots=True)
class GroupGeometry:
    """All deterministic geometry needed to prepare one group."""

    logical_size: ImageSize
    reference: ReferenceGeometry
    main: MainImageGeometry
    items: tuple[ItemGeometry, ...]


def _round_to_nearest_patch(value: float, patch_size: int) -> int:
    return patch_size * max(1, math.floor(value / patch_size + 0.5))


def _validate_bbox_in_canvas(bbox: BoundingBox, canvas: ImageSize) -> None:
    if bbox.x0 < 0 or bbox.y0 < 0 or bbox.x1 > canvas.width or (bbox.y1 > canvas.height):
        raise ValueError(f"bbox {bbox!r} lies outside logical canvas {canvas!r}")


def _as_exact_coordinate(coordinate: Real) -> Fraction:
    if isinstance(coordinate, Fraction):
        return coordinate
    if isinstance(coordinate, int):
        return Fraction(coordinate)
    return Fraction.from_float(float(coordinate))


def _scale_coordinate_exact(
    coordinate: float, logical_extent: int, reference_extent: int
) -> Fraction:
    return _as_exact_coordinate(coordinate) * reference_extent / logical_extent


def _aligned_axis_crop(
    bbox_start: Fraction, bbox_end: Fraction, canvas_extent: int, patch_size: int
) -> tuple[int, int]:
    pixel_start = math.floor(bbox_start)
    pixel_end = math.ceil(bbox_end)
    required_extent = pixel_end - pixel_start
    crop_extent = (required_extent + patch_size - 1) // patch_size * patch_size
    feasible_start = max(0, pixel_end - crop_extent)
    feasible_end = min(pixel_start, canvas_extent - crop_extent)
    if feasible_start > feasible_end:
        raise RuntimeError("no in-canvas crop can contain a valid bbox")
    centered_start = math.floor((bbox_start + bbox_end - crop_extent) / 2 + Fraction(1, 2))
    crop_start = min(max(centered_start, feasible_start), feasible_end)
    return (crop_start, crop_extent)


def plan_group_geometry(
    logical_size: ImageSize,
    item_bboxes: Iterable[BoundingBox] = (),
    *,
    patch_size: int = 16,
    main_target_area: float = 1048576,
    clamp_main_to_reference: bool = True,
) -> GroupGeometry:
    """Plan the reference canvas and main image for one group.

    ``clamp_main_to_reference`` caps the main scale at 1.0 so the main is never
    upsampled past the reference canvas, which keeps the RoPE main-grid step at
    or above the item-grid step of 1.
    """
    if isinstance(patch_size, bool) or not isinstance(patch_size, int) or patch_size <= 0:
        raise ValueError(f"patch_size must be a positive integer, got {patch_size!r}")
    if (
        isinstance(main_target_area, bool)
        or not isinstance(main_target_area, (int, float))
        or (not math.isfinite(main_target_area))
        or (main_target_area <= 0)
    ):
        raise ValueError(
            f"main_target_area must be a positive finite number, got {main_target_area!r}"
        )
    if not isinstance(clamp_main_to_reference, bool):
        raise TypeError("clamp_main_to_reference must be a bool")
    reference_size = ImageSize(
        height=_round_to_nearest_patch(logical_size.height, patch_size),
        width=_round_to_nearest_patch(logical_size.width, patch_size),
    )
    main_scale = math.sqrt(main_target_area / (reference_size.height * reference_size.width))
    if clamp_main_to_reference:
        main_scale = min(main_scale, 1.0)
    main_size = ImageSize(
        height=_round_to_nearest_patch(reference_size.height * main_scale, patch_size),
        width=_round_to_nearest_patch(reference_size.width * main_scale, patch_size),
    )
    items_list = []
    for bbox in item_bboxes:
        _validate_bbox_in_canvas(bbox, logical_size)
        reference_x0 = _scale_coordinate_exact(bbox.x0, logical_size.width, reference_size.width)
        reference_y0 = _scale_coordinate_exact(bbox.y0, logical_size.height, reference_size.height)
        reference_x1 = _scale_coordinate_exact(bbox.x1, logical_size.width, reference_size.width)
        reference_y1 = _scale_coordinate_exact(bbox.y1, logical_size.height, reference_size.height)
        reference_bbox = BoundingBox(
            x0=reference_x0, y0=reference_y0, x1=reference_x1, y1=reference_y1
        )
        (crop_left, crop_width) = _aligned_axis_crop(
            reference_x0, reference_x1, reference_size.width, patch_size
        )
        (crop_top, crop_height) = _aligned_axis_crop(
            reference_y0, reference_y1, reference_size.height, patch_size
        )
        items_list.append(
            ItemGeometry(
                logical_bbox=bbox,
                reference_bbox=reference_bbox,
                crop=CropGeometry(
                    top=crop_top,
                    left=crop_left,
                    size=ImageSize(height=crop_height, width=crop_width),
                ),
            )
        )
    items = tuple(items_list)
    return GroupGeometry(
        logical_size=logical_size,
        reference=ReferenceGeometry(
            size=reference_size,
            scale_h=reference_size.height / logical_size.height,
            scale_w=reference_size.width / logical_size.width,
        ),
        main=MainImageGeometry(
            size=main_size,
            scale_h=main_size.height / reference_size.height,
            scale_w=main_size.width / reference_size.width,
        ),
        items=items,
    )
