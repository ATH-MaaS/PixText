"""Apply deterministic group geometry plans to RGB image tensors."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize

from pixtext.preprocessing.geometry import GroupGeometry


@dataclass(frozen=True, slots=True)
class PreparedGroupImages:
    """Reference, main, and ordered item tensors for one group."""

    reference: torch.Tensor
    main: torch.Tensor
    items: tuple[torch.Tensor, ...]


def _resize_bicubic(image: torch.Tensor, height: int, width: int) -> torch.Tensor:
    return resize(
        image, [height, width], interpolation=InterpolationMode.BICUBIC, antialias=True
    ).contiguous()


def prepare_group_images(image: torch.Tensor, geometry: GroupGeometry) -> PreparedGroupImages:
    """Resize and crop one RGB image according to a geometry plan."""
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"image must be a torch.Tensor, got {type(image).__name__}")
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(
            f"image must be one RGB tensor in (C, H, W) layout, got shape {tuple(image.shape)}"
        )
    actual_size = tuple(image.shape[-2:])
    expected_size = (geometry.logical_size.height, geometry.logical_size.width)
    if actual_size != expected_size:
        raise ValueError(
            f"image size does not match the geometry logical canvas: got {actual_size}, expected {expected_size}"
        )
    reference = _resize_bicubic(
        image, geometry.reference.size.height, geometry.reference.size.width
    )
    main = _resize_bicubic(reference, geometry.main.size.height, geometry.main.size.width)
    items = tuple(
        reference[
            :,
            item.crop.top : item.crop.top + item.crop.size.height,
            item.crop.left : item.crop.left + item.crop.size.width,
        ].contiguous()
        for item in geometry.items
    )
    return PreparedGroupImages(reference=reference, main=main, items=items)
