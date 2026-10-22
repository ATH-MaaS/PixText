from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn

from pixtext.preprocessing.geometry import ImageSize


def _validate_positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


@dataclass(frozen=True, slots=True)
class PatchLayout:
    """Explicit metadata required to reconstruct one rectangular image."""

    image_size: ImageSize
    patch_size: int
    channels: int

    def __post_init__(self) -> None:
        _validate_positive_integer("patch_size", self.patch_size)
        _validate_positive_integer("channels", self.channels)
        if (
            self.image_size.height % self.patch_size != 0
            or self.image_size.width % self.patch_size != 0
        ):
            raise ValueError(
                f"image height and width must be divisible by patch_size: got {self.image_size!r} and patch_size={self.patch_size}"
            )

    @property
    def grid_size(self) -> ImageSize:
        return ImageSize(
            self.image_size.height // self.patch_size, self.image_size.width // self.patch_size
        )

    @property
    def num_patches(self) -> int:
        grid = self.grid_size
        return grid.height * grid.width


@dataclass(frozen=True, slots=True)
class PatchifiedImage:
    """Patch vectors together with their explicit rectangular layout."""

    patches: torch.Tensor
    layout: PatchLayout


@dataclass(frozen=True, slots=True)
class PackedImagePatches:
    """Raw image patches and routing metadata for a rank-local group batch."""

    patches: torch.Tensor
    sample_ids: torch.Tensor
    image_ids: torch.Tensor
    sample_offsets: torch.Tensor
    image_offsets: torch.Tensor
    layouts: tuple[PatchLayout, ...]
    images_per_sample: tuple[int, ...]

    @property
    def image_shapes(self) -> tuple[ImageSize, ...]:
        return tuple(layout.image_size for layout in self.layouts)

    def sample_token_range(self, sample_start: int, sample_end: int) -> tuple[int, int]:
        if (
            isinstance(sample_start, bool)
            or not isinstance(sample_start, int)
            or isinstance(sample_end, bool)
            or (not isinstance(sample_end, int))
            or (not 0 <= sample_start < sample_end <= len(self.images_per_sample))
        ):
            raise IndexError("packed image sample range is out of bounds")
        image_start = sum(self.images_per_sample[:sample_start])
        image_end = image_start + sum(self.images_per_sample[sample_start:sample_end])
        return (
            sum(layout.num_patches for layout in self.layouts[:image_start]),
            sum(layout.num_patches for layout in self.layouts[:image_end]),
        )

    def select_samples(self, sample_start: int, sample_end: int) -> PackedImagePatches:
        """Return a contiguous sample range with local packed routing."""
        (token_start, token_end) = self.sample_token_range(sample_start, sample_end)
        images_per_sample = self.images_per_sample[sample_start:sample_end]
        image_start = sum(self.images_per_sample[:sample_start])
        image_end = image_start + sum(images_per_sample)
        layouts = self.layouts[image_start:image_end]
        image_offsets = [0]
        for layout in layouts:
            image_offsets.append(image_offsets[-1] + layout.num_patches)
        sample_offsets = [0]
        layout_index = 0
        for image_count in images_per_sample:
            sample_offsets.append(
                sample_offsets[-1]
                + sum(
                    layout.num_patches
                    for layout in layouts[layout_index : layout_index + image_count]
                )
            )
            layout_index += image_count
        device = self.patches.device
        return PackedImagePatches(
            patches=self.patches[token_start:token_end],
            sample_ids=self.sample_ids[token_start:token_end] - sample_start,
            image_ids=self.image_ids[token_start:token_end],
            sample_offsets=torch.tensor(sample_offsets, dtype=torch.int64, device=device),
            image_offsets=torch.tensor(image_offsets, dtype=torch.int64, device=device),
            layouts=layouts,
            images_per_sample=images_per_sample,
        )


class DirectPatchProjection(nn.Module):
    """Project flattened RGB patches directly into JiT token features."""

    def __init__(
        self, *, patch_size: int = 16, in_channels: int = 3, hidden_size: int = 1280
    ) -> None:
        super().__init__()
        for name, value in (
            ("patch_size", patch_size),
            ("in_channels", in_channels),
            ("hidden_size", hidden_size),
        ):
            _validate_positive_integer(name, value)
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.proj = nn.Linear(in_channels * patch_size**2, hidden_size, bias=True)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        if not isinstance(patches, torch.Tensor):
            raise TypeError(f"patches must be a torch.Tensor, got {type(patches).__name__}")
        if patches.ndim != 2:
            raise ValueError(
                f"patches must use (tokens, patch_width) layout, got shape {tuple(patches.shape)}"
            )
        if patches.shape[1] != self.proj.in_features:
            raise ValueError(
                f"patch width does not match projection: got {patches.shape[1]}, expected {self.proj.in_features}"
            )
        return self.proj(patches)


def patchify(image: torch.Tensor, patch_size: int = 16) -> PatchifiedImage:
    """Convert one ``[C,H,W]`` image into row-major flattened patches."""
    _validate_positive_integer("patch_size", patch_size)
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"image must be a torch.Tensor, got {type(image).__name__}")
    if image.ndim != 3:
        raise ValueError(f"image must use (C, H, W) layout, got shape {tuple(image.shape)}")
    (channels, height, width) = image.shape
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(
            f"image height and width must be divisible by patch_size: got {(height, width)} and patch_size={patch_size}"
        )
    grid_height = height // patch_size
    grid_width = width // patch_size
    patches = (
        image.reshape(channels, grid_height, patch_size, grid_width, patch_size)
        .permute(1, 3, 0, 2, 4)
        .reshape(grid_height * grid_width, channels * patch_size**2)
    )
    return PatchifiedImage(
        patches=patches,
        layout=PatchLayout(
            image_size=ImageSize(height, width), patch_size=patch_size, channels=channels
        ),
    )


def unpatchify(patches: torch.Tensor, layout: PatchLayout) -> torch.Tensor:
    """Reconstruct one image using explicit rectangular layout metadata."""
    if not isinstance(patches, torch.Tensor):
        raise TypeError(f"patches must be a torch.Tensor, got {type(patches).__name__}")
    if patches.ndim != 2:
        raise ValueError(
            f"patches must use (tokens, patch_width) layout, got shape {tuple(patches.shape)}"
        )
    grid = layout.grid_size
    if patches.shape[0] != layout.num_patches:
        raise ValueError(
            f"patch token count does not match layout: got {patches.shape[0]}, expected {layout.num_patches}"
        )
    expected_patch_width = layout.channels * layout.patch_size**2
    if patches.shape[1] != expected_patch_width:
        raise ValueError(
            f"patch width does not match layout: got {patches.shape[1]}, expected {expected_patch_width}"
        )
    image = (
        patches.reshape(
            grid.height, grid.width, layout.channels, layout.patch_size, layout.patch_size
        )
        .permute(2, 0, 3, 1, 4)
        .reshape(layout.channels, layout.image_size.height, layout.image_size.width)
    )
    return image


def pack_image_groups(
    groups: Sequence[Sequence[torch.Tensor]], patch_size: int = 16
) -> PackedImagePatches:
    """Patchify and concatenate ``main + ordered items`` for every group."""
    _validate_positive_integer("patch_size", patch_size)
    if len(groups) == 0:
        raise ValueError("packed image batch must contain at least one group")
    patch_parts = []
    sample_id_values = []
    image_id_values = []
    layouts = []
    images_per_sample = []
    sample_offsets = [0]
    image_offsets = [0]
    expected_dtype = None
    expected_device = None
    expected_channels = None
    for sample_id, group in enumerate(groups):
        if len(group) == 0:
            raise ValueError(f"group {sample_id} must contain at least one image (the main image)")
        images_per_sample.append(len(group))
        for image_id, image in enumerate(group):
            patchified = patchify(image, patch_size)
            if expected_dtype is None:
                expected_dtype = patchified.patches.dtype
                expected_device = patchified.patches.device
                expected_channels = patchified.layout.channels
            else:
                location = f"group {sample_id} image {image_id}"
                if patchified.patches.dtype != expected_dtype:
                    raise ValueError(
                        f"{location} dtype {patchified.patches.dtype} does not match packed dtype {expected_dtype}"
                    )
                if patchified.patches.device != expected_device:
                    raise ValueError(
                        f"{location} device {patchified.patches.device} does not match packed device {expected_device}"
                    )
                if patchified.layout.channels != expected_channels:
                    raise ValueError(
                        f"{location} channels {patchified.layout.channels} do not match packed channels {expected_channels}"
                    )
            patch_parts.append(patchified.patches)
            layouts.append(patchified.layout)
            token_count = patchified.layout.num_patches
            sample_id_values.extend([sample_id] * token_count)
            image_id_values.extend([image_id] * token_count)
            image_offsets.append(image_offsets[-1] + token_count)
        sample_offsets.append(image_offsets[-1])
    patches = torch.cat(patch_parts)
    device = patches.device
    return PackedImagePatches(
        patches=patches,
        sample_ids=torch.tensor(sample_id_values, dtype=torch.int64, device=device),
        image_ids=torch.tensor(image_id_values, dtype=torch.int64, device=device),
        sample_offsets=torch.tensor(sample_offsets, dtype=torch.int64, device=device),
        image_offsets=torch.tensor(image_offsets, dtype=torch.int64, device=device),
        layouts=tuple(layouts),
        images_per_sample=tuple(images_per_sample),
    )


def unpack_image_patches(
    patches: torch.Tensor, packed: PackedImagePatches
) -> tuple[tuple[torch.Tensor, ...], ...]:
    """Restore packed patch pixels to the original group/image structure."""
    if not isinstance(patches, torch.Tensor):
        raise TypeError(f"patches must be a torch.Tensor, got {type(patches).__name__}")
    if patches.ndim != 2:
        raise ValueError(
            f"patches must use (tokens, patch_width) layout, got shape {tuple(patches.shape)}"
        )
    expected_token_count = sum(layout.num_patches for layout in packed.layouts)
    if patches.shape[0] != expected_token_count:
        raise ValueError(
            f"packed total token count mismatch: got {patches.shape[0]}, expected {expected_token_count}"
        )
    if packed.image_offsets.shape != (len(packed.layouts) + 1,):
        raise ValueError("packed image offsets do not match the image layouts")
    flat_images = []
    start = 0
    for layout in packed.layouts:
        end = start + layout.num_patches
        flat_images.append(unpatchify(patches[start:end], layout))
        start = end
    groups = []
    image_index = 0
    for image_count in packed.images_per_sample:
        groups.append(tuple(flat_images[image_index : image_index + image_count]))
        image_index += image_count
    return tuple(groups)
