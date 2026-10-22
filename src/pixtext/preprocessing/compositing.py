from __future__ import annotations

from collections.abc import Sequence

import torch

from pixtext.preprocessing.geometry import GroupGeometry, ImageSize


def _validate_generated_image(image: torch.Tensor, *, expected_size: ImageSize, label: str) -> None:
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"{label} must be a torch.Tensor")
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(
            f"{label} must be one RGB tensor in (C, H, W) layout, got shape {tuple(image.shape)}"
        )
    actual_size = tuple(image.shape[-2:])
    planned_size = (expected_size.height, expected_size.width)
    if actual_size != planned_size:
        raise ValueError(
            f"{label} size does not match its geometry: got {actual_size}, expected {planned_size}"
        )


def compose_native_edit_items(
    clean_reference: torch.Tensor, generated_items: Sequence[torch.Tensor], geometry: GroupGeometry
) -> torch.Tensor:
    """Paste generated item crops on an undegraded reference canvas."""
    _validate_generated_image(
        clean_reference, expected_size=geometry.reference.size, label="clean reference image"
    )
    if len(generated_items) != len(geometry.items):
        raise ValueError(
            f"generated item count does not match geometry: got {len(generated_items)}, expected {len(geometry.items)}"
        )
    composite = clean_reference.detach().float().clone().contiguous()
    for item_index, (generated_item, item_geometry) in enumerate(
        zip(generated_items, geometry.items, strict=True), start=1
    ):
        _validate_generated_image(
            generated_item,
            expected_size=item_geometry.crop.size,
            label=f"generated item {item_index}",
        )
        if generated_item.device != composite.device:
            raise ValueError(
                f"generated item {item_index} is on {generated_item.device}, but the generated main image is on {composite.device}"
            )
        top = item_geometry.crop.top
        left = item_geometry.crop.left
        bottom = top + item_geometry.crop.size.height
        right = left + item_geometry.crop.size.width
        composite[:, top:bottom, left:right] = generated_item.detach().float()
    return composite
