from __future__ import annotations

from dataclasses import dataclass

import torch

from pixtext.modeling.modes import TimestepTokenRoutingMode
from pixtext.modeling.patches import PatchLayout
from pixtext.preprocessing.geometry import ImageSize

IMAGE_TOKEN_TYPE = 0
PROMPT_TOKEN_TYPE = 1
TIME_TOKEN_TYPE = 2


@dataclass(frozen=True, slots=True)
class PackedGroupSequence:
    """Group-contiguous image and prompt tokens ready for single-stream attention."""

    tokens: torch.Tensor
    position_ids: torch.Tensor
    sample_ids: torch.Tensor
    image_ids: torch.Tensor
    token_types: torch.Tensor
    timestep_roles: torch.Tensor
    sample_offsets: torch.Tensor
    image_token_indices: torch.Tensor
    prompt_token_indices: torch.Tensor
    image_offsets: torch.Tensor
    prompt_offsets: torch.Tensor
    layouts: tuple[PatchLayout, ...]
    images_per_sample: tuple[int, ...]
    timestep_token_routing: TimestepTokenRoutingMode | None
    branch_kinds: torch.Tensor | None = None
    slot_ids: torch.Tensor | None = None
    branch_descriptors: tuple[object, ...] = ()
    sample_token_counts: tuple[int, ...] = ()

    @property
    def image_shapes(self) -> tuple[ImageSize, ...]:
        return tuple(layout.image_size for layout in self.layouts)


def extract_image_tokens(
    sequence_tokens: torch.Tensor, packed: PackedGroupSequence
) -> torch.Tensor:
    """Restore compact image-token order from a transformed group sequence."""
    if not isinstance(sequence_tokens, torch.Tensor):
        raise TypeError(
            f"sequence_tokens must be a torch.Tensor, got {type(sequence_tokens).__name__}"
        )
    if sequence_tokens.ndim != 2:
        raise ValueError(
            f"sequence_tokens must use (tokens, hidden_size) layout, got shape {tuple(sequence_tokens.shape)}"
        )
    if sequence_tokens.shape[0] != packed.tokens.shape[0]:
        raise ValueError(
            f"sequence token count does not match packed sequence: got {sequence_tokens.shape[0]}, expected {packed.tokens.shape[0]}"
        )
    if sequence_tokens.device != packed.image_token_indices.device:
        raise ValueError("sequence tokens and packed indices must share a device")
    return sequence_tokens.index_select(0, packed.image_token_indices)
