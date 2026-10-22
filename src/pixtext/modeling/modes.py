from __future__ import annotations

from enum import Enum


class ImageAttentionRoutingMode(str, Enum):
    """Control direct attention edges between images in the same group."""

    ALL_IMAGES = "all_images"
    MAIN_IMAGE_HUB = "main_image_hub"
    ITEMS_READ_MAIN = "items_read_main"
    ITEMS_INTERREAD_GLYPH_READS_OWN_ITEM = "items_interread_glyph_reads_own_item"
    ITEMS_INTERREAD_GLYPH_READS_OWN_ITEM_BIDIRECTIONAL_MAIN = (
        "items_interread_glyph_reads_own_item_bidirectional_main"
    )

    @classmethod
    def require(
        cls, value: object, *, name: str = "image_attention_routing"
    ) -> ImageAttentionRoutingMode:
        if not isinstance(value, cls):
            raise TypeError(f"{name} must be an ImageAttentionRoutingMode, got {value!r}")
        return value


class TextConditioningMode(str, Enum):
    """Choose whether the packed Transformer contains prompt branches."""

    PACKED_PROMPT = "packed_prompt"
    NONE = "none"

    @classmethod
    def require(cls, value: object, *, name: str = "text_conditioning") -> TextConditioningMode:
        if not isinstance(value, cls):
            raise TypeError(f"{name} must be a TextConditioningMode, got {value!r}")
        return value


class TimestepTokenRoutingMode(str, Enum):
    """Choose the topology and directional routing of iMF timestep banks."""

    SHARED_ROLE_ONE_WAY = "shared_role_one_way"
    PER_IMAGE_CONTEXTUAL = "per_image_contextual"

    @classmethod
    def require(
        cls, value: object, *, name: str = "timestep_token_routing"
    ) -> TimestepTokenRoutingMode:
        if not isinstance(value, cls):
            raise TypeError(f"{name} must be a TimestepTokenRoutingMode, got {value!r}")
        return value


class NAxisMode(str, Enum):
    """Assign the RoPE n coordinate from image role or group-local identity."""

    ROLE = "role"
    IMAGE_ID = "image_id"

    @classmethod
    def require(cls, value: object, *, name: str = "n_axis_mode") -> NAxisMode:
        if not isinstance(value, cls):
            raise TypeError(f"{name} must be an NAxisMode, got {value!r}")
        return value

    def position_for(self, image_id: int) -> float:
        if self is NAxisMode.ROLE:
            return float(image_id > 0)
        return float(image_id)
