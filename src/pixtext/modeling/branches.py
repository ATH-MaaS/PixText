"""Post-projection packing for mixed RGB and canonical-glyph branches."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn

from pixtext.modeling.patches import (
    DirectPatchProjection,
    PackedImagePatches,
    PatchLayout,
    patchify,
)
from pixtext.preprocessing.geometry import GroupGeometry
from pixtext.preprocessing.glyphs import GlyphCondition

MAIN_BRANCH = 0
ITEM_BRANCH = 1
GLYPH_BRANCH = 2


@dataclass(frozen=True, slots=True)
class BranchDescriptor:
    kind: str
    slot_id: int | None

    def __post_init__(self) -> None:
        if self.kind not in {"main", "item", "glyph"}:
            raise ValueError(f"unknown branch kind: {self.kind!r}")
        if self.kind == "main":
            if self.slot_id is not None:
                raise ValueError("main branch slot_id must be None")
        elif (
            isinstance(self.slot_id, bool) or not isinstance(self.slot_id, int) or self.slot_id < 0
        ):
            raise ValueError("item/glyph slot_id must be a non-negative integer")

    @property
    def kind_id(self) -> int:
        return {"main": MAIN_BRANCH, "item": ITEM_BRANCH, "glyph": GLYPH_BRANCH}[self.kind]


class DirectGlyphPatchProjection(nn.Module):
    """Independent direct projection for one-channel [0,1] coverage patches."""

    def __init__(self, *, patch_size: int = 16, hidden_size: int = 1280) -> None:
        super().__init__()
        if patch_size <= 0 or hidden_size <= 0:
            raise ValueError("patch_size and hidden_size must be positive")
        self.patch_size = patch_size
        self.in_channels = 1
        self.proj = nn.Linear(patch_size**2, hidden_size, bias=True)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        if patches.ndim != 2 or patches.shape[1] != self.proj.in_features:
            raise ValueError("glyph patches must use (tokens, patch_size**2) layout")
        return self.proj(patches)


@dataclass(frozen=True, slots=True)
class PackedBranchTokens:
    tokens: torch.Tensor
    descriptors: tuple[BranchDescriptor, ...]
    layouts: tuple[PatchLayout, ...]
    sample_ids: torch.Tensor
    image_ids: torch.Tensor
    branch_kinds: torch.Tensor
    slot_ids: torch.Tensor
    image_offsets: torch.Tensor
    sample_offsets: torch.Tensor
    branches_per_sample: tuple[int, ...]
    rgb_token_indices: torch.Tensor
    rgb_branch_indices: tuple[int, ...]
    rgb_images_per_sample: tuple[int, ...]


class PackedBranchTokenizer:
    """Hide mixed raw widths and merge RGB/glyphs after projection."""

    def __init__(
        self, *, rgb_projection: DirectPatchProjection, glyph_projection: DirectGlyphPatchProjection
    ) -> None:
        if not isinstance(rgb_projection, DirectPatchProjection):
            raise TypeError("rgb_projection must be DirectPatchProjection")
        if not isinstance(glyph_projection, DirectGlyphPatchProjection):
            raise TypeError("glyph_projection must be DirectGlyphPatchProjection")
        if rgb_projection.patch_size != glyph_projection.patch_size:
            raise ValueError("RGB and glyph patch sizes must match")
        if rgb_projection.proj.out_features != glyph_projection.proj.out_features:
            raise ValueError("RGB and glyph projections must share hidden size")
        self.rgb_projection = rgb_projection
        self.glyph_projection = glyph_projection
        self.patch_size = rgb_projection.patch_size

    def tokenize(
        self,
        *,
        packed_rgb: PackedImagePatches,
        glyphs: Sequence[Sequence[GlyphCondition | None]],
        geometries: Sequence[GroupGeometry],
    ) -> PackedBranchTokens:
        group_count = len(packed_rgb.images_per_sample)
        if len(glyphs) != group_count or len(geometries) != group_count:
            raise ValueError("RGB, glyph, and geometry group counts must match")
        if any(
            len(group_glyphs) != len(geometry.items)
            for (group_glyphs, geometry) in zip(glyphs, geometries, strict=True)
        ):
            raise ValueError("glyph slots must match geometry items")
        torch._assert_async(
            torch.isfinite(packed_rgb.patches).all(), "RGB branches must remain finite"
        )
        main_patches = packed_rgb.patches[packed_rgb.image_ids.eq(MAIN_BRANCH)]
        torch._assert_async(
            main_patches.ge(-1).all() & main_patches.le(1).all(),
            "clean main RGB branches must remain within [-1, 1]",
        )
        rgb_tokens = self.rgb_projection(packed_rgb.patches)
        token_parts: list[torch.Tensor] = []
        descriptors: list[BranchDescriptor] = []
        layouts: list[PatchLayout] = []
        sample_values: list[int] = []
        image_values: list[int] = []
        kind_values: list[int] = []
        slot_values: list[int] = []
        image_offsets = [0]
        sample_offsets = [0]
        branches_per_sample: list[int] = []
        rgb_token_indices: list[torch.Tensor] = []
        rgb_branch_indices: list[int] = []
        flat_rgb = 0
        compact_rgb_offset = 0
        device = packed_rgb.patches.device
        dtype = rgb_tokens.dtype
        for sample_id, (rgb_count, group_glyphs, geometry) in enumerate(
            zip(packed_rgb.images_per_sample, glyphs, geometries, strict=True)
        ):
            if rgb_count != 1 + len(geometry.items):
                raise ValueError("packed RGB topology does not match geometry")
            group_branch_count = 0
            for rgb_image_id in range(rgb_count):
                layout = packed_rgb.layouts[flat_rgb]
                token_count = layout.num_patches
                rgb_part = rgb_tokens[compact_rgb_offset : compact_rgb_offset + token_count]
                descriptor = BranchDescriptor(
                    "main" if rgb_image_id == 0 else "item",
                    None if rgb_image_id == 0 else rgb_image_id - 1,
                )
                physical_id = group_branch_count
                token_parts.append(rgb_part)
                descriptors.append(descriptor)
                layouts.append(layout)
                sample_values.extend([sample_id] * token_count)
                image_values.extend([physical_id] * token_count)
                kind_values.extend([descriptor.kind_id] * token_count)
                slot_values.extend(
                    [-1 if descriptor.slot_id is None else descriptor.slot_id] * token_count
                )
                start = image_offsets[-1]
                image_offsets.append(start + token_count)
                rgb_token_indices.append(torch.arange(start, start + token_count, device=device))
                rgb_branch_indices.append(len(descriptors) - 1)
                compact_rgb_offset += token_count
                flat_rgb += 1
                group_branch_count += 1
                if rgb_image_id == 0:
                    continue
                glyph = group_glyphs[rgb_image_id - 1]
                if glyph is None:
                    continue
                if glyph.aligned_crop != geometry.items[rgb_image_id - 1].crop:
                    raise ValueError("glyph crop provenance does not match geometry")
                pixels = glyph.image.to(device=device, dtype=dtype, non_blocking=True)
                torch._assert_async(
                    torch.isfinite(pixels).all() & pixels.ge(0).all() & pixels.le(1).all(),
                    "glyph coverage must remain finite and within [0, 1]",
                )
                patchified = patchify(pixels, self.patch_size)
                if patchified.layout.num_patches != token_count:
                    raise ValueError("glyph and paired item token counts must match")
                glyph_tokens = self.glyph_projection(patchified.patches)
                glyph_descriptor = BranchDescriptor("glyph", rgb_image_id - 1)
                glyph_physical_id = group_branch_count
                token_parts.append(glyph_tokens)
                descriptors.append(glyph_descriptor)
                layouts.append(patchified.layout)
                sample_values.extend([sample_id] * token_count)
                image_values.extend([glyph_physical_id] * token_count)
                kind_values.extend([GLYPH_BRANCH] * token_count)
                slot_values.extend([rgb_image_id - 1] * token_count)
                image_offsets.append(image_offsets[-1] + token_count)
                group_branch_count += 1
            branches_per_sample.append(group_branch_count)
            sample_offsets.append(image_offsets[-1])
        tokens = torch.cat(token_parts)
        return PackedBranchTokens(
            tokens=tokens,
            descriptors=tuple(descriptors),
            layouts=tuple(layouts),
            sample_ids=torch.tensor(sample_values, dtype=torch.int64, device=device),
            image_ids=torch.tensor(image_values, dtype=torch.int64, device=device),
            branch_kinds=torch.tensor(kind_values, dtype=torch.int64, device=device),
            slot_ids=torch.tensor(slot_values, dtype=torch.int64, device=device),
            image_offsets=torch.tensor(image_offsets, dtype=torch.int64, device=device),
            sample_offsets=torch.tensor(sample_offsets, dtype=torch.int64, device=device),
            branches_per_sample=tuple(branches_per_sample),
            rgb_token_indices=torch.cat(rgb_token_indices),
            rgb_branch_indices=tuple(rgb_branch_indices),
            rgb_images_per_sample=packed_rgb.images_per_sample,
        )


def _branch_position_ids(
    packed: PackedBranchTokens, geometries: Sequence[GroupGeometry]
) -> torch.Tensor:
    """Build main/item/glyph role positions from the shared descriptor order."""
    parts: list[torch.Tensor] = []
    branch_index = 0
    device = packed.tokens.device
    for sample_id, (geometry, branch_count) in enumerate(
        zip(geometries, packed.branches_per_sample, strict=True)
    ):
        reference = geometry.reference.size
        center_h = reference.height / (2 * packed.layouts[branch_index].patch_size)
        center_w = reference.width / (2 * packed.layouts[branch_index].patch_size)
        for image_id in range(branch_count):
            descriptor = packed.descriptors[branch_index]
            layout = packed.layouts[branch_index]
            grid = layout.grid_size
            rows = torch.arange(grid.height, dtype=torch.float32, device=device)
            columns = torch.arange(grid.width, dtype=torch.float32, device=device)
            (rows, columns) = torch.meshgrid(rows, columns, indexing="ij")
            if descriptor.kind == "main":
                h = (rows + 0.5) / geometry.main.scale_h - center_h
                w = (columns + 0.5) / geometry.main.scale_w - center_w
                n = torch.zeros_like(h)
            else:
                assert descriptor.slot_id is not None
                item = geometry.items[descriptor.slot_id]
                if layout.image_size != item.crop.size:
                    raise ValueError(
                        f"group {sample_id} branch {image_id} layout does not match slot"
                    )
                h = rows + 0.5 + item.crop.top / layout.patch_size - center_h
                w = columns + 0.5 + item.crop.left / layout.patch_size - center_w
                n = torch.full_like(h, 1.0 if descriptor.kind == "item" else 2.0)
            parts.append(torch.stack((n, h, w), dim=-1).reshape(-1, 3))
            branch_index += 1
    positions = torch.cat(parts)
    if positions.shape != (packed.tokens.shape[0], 3):
        raise RuntimeError("branch position count does not match token count")
    return positions


def pack_branch_group_sequences(
    *, branches: PackedBranchTokens, prompts, geometries: Sequence[GroupGeometry]
):
    from pixtext.modeling.conditioning import IMAGE_TOKEN_TYPE, PackedGroupSequence

    "Assemble optional prompts with physical RGB/glyph branches.\n\n    ``prompts=None`` is the real no-text adapter: it emits no placeholder\n    tokens or prompt-owned topology and derives every offset from branches.\n    "
    if prompts is None:
        if len(geometries) != len(branches.branches_per_sample):
            raise ValueError("branch and geometry group counts must match")
        positions = _branch_position_ids(branches, geometries)
        device = branches.tokens.device
        token_count = branches.tokens.shape[0]
        empty = torch.empty(0, dtype=torch.int64, device=device)
        return PackedGroupSequence(
            tokens=branches.tokens,
            position_ids=positions,
            sample_ids=branches.sample_ids,
            image_ids=branches.image_ids,
            token_types=torch.full(
                (token_count,), IMAGE_TOKEN_TYPE, dtype=torch.int64, device=device
            ),
            timestep_roles=branches.branch_kinds.ne(MAIN_BRANCH).to(torch.int64),
            sample_offsets=branches.sample_offsets,
            image_token_indices=torch.arange(token_count, dtype=torch.int64, device=device),
            prompt_token_indices=empty,
            image_offsets=branches.image_offsets,
            prompt_offsets=torch.zeros(1, dtype=torch.int64, device=device),
            layouts=branches.layouts,
            images_per_sample=branches.branches_per_sample,
            timestep_token_routing=None,
            branch_kinds=branches.branch_kinds,
            slot_ids=branches.slot_ids,
            branch_descriptors=branches.descriptors,
        )
    raise ValueError("PixText does not use prompt tokens")
