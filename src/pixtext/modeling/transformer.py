from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.flex_attention import BlockMask

from pixtext.modeling.attention import GroupSelfAttention, build_flex_attention_block_mask
from pixtext.modeling.branches import (
    GLYPH_BRANCH,
    ITEM_BRANCH,
    DirectGlyphPatchProjection,
    PackedBranchTokenizer,
    pack_branch_group_sequences,
)
from pixtext.modeling.conditioning import extract_image_tokens
from pixtext.modeling.modes import ImageAttentionRoutingMode, NAxisMode, TextConditioningMode
from pixtext.modeling.norm import RMSNorm
from pixtext.modeling.patches import DirectPatchProjection, PackedImagePatches, unpack_image_patches
from pixtext.modeling.rope import DEFAULT_ROPE_AXES_DIMS, DEFAULT_ROPE_THETA, RotaryEmbedding3D
from pixtext.preprocessing.geometry import GroupGeometry
from pixtext.preprocessing.glyphs import GlyphCondition
from pixtext.sampling.timesteps import GroupTimesteps


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def _validate_positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def _validate_probability(name: str, value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or (not math.isfinite(value))
        or (not 0 <= value < 1)
    ):
        raise ValueError(f"{name} must be finite and in [0, 1), got {value!r}")
    return float(value)


def _validate_packed_timestep_routing(
    tokens: torch.Tensor,
    timestep_embeddings: torch.Tensor,
    conditioning_ids: torch.Tensor,
    *,
    hidden_size: int,
) -> None:
    if not isinstance(tokens, torch.Tensor):
        raise TypeError(f"tokens must be a torch.Tensor, got {type(tokens).__name__}")
    if tokens.ndim != 2 or tokens.shape[1] != hidden_size:
        raise ValueError(
            f"tokens must use (tokens, hidden_size) layout with hidden_size {hidden_size}, got shape {tuple(tokens.shape)}"
        )
    if not tokens.is_floating_point():
        raise ValueError(f"tokens must be floating point, got {tokens.dtype}")
    if not isinstance(timestep_embeddings, torch.Tensor):
        raise TypeError(
            f"timestep_embeddings must be a torch.Tensor, got {type(timestep_embeddings).__name__}"
        )
    if (
        timestep_embeddings.ndim != 2
        or timestep_embeddings.shape[0] == 0
        or timestep_embeddings.shape[1] != hidden_size
    ):
        raise ValueError(
            f"timestep_embeddings must use (images, hidden_size) layout with at least one image and hidden_size {hidden_size}, got shape {tuple(timestep_embeddings.shape)}"
        )
    if not timestep_embeddings.is_floating_point():
        raise ValueError(
            f"timestep_embeddings must be floating point, got {timestep_embeddings.dtype}"
        )
    if not isinstance(conditioning_ids, torch.Tensor):
        raise TypeError(
            f"conditioning_ids must be a torch.Tensor, got {type(conditioning_ids).__name__}"
        )
    if conditioning_ids.shape != (tokens.shape[0],):
        raise ValueError(
            f"conditioning_ids must contain {tokens.shape[0]} entries, got shape {tuple(conditioning_ids.shape)}"
        )
    if conditioning_ids.dtype != torch.int64:
        raise ValueError(f"conditioning_ids must use torch.int64, got {conditioning_ids.dtype}")
    if tokens.dtype != timestep_embeddings.dtype:
        raise ValueError(
            f"tokens and timestep_embeddings must share a dtype, got {tokens.dtype} and {timestep_embeddings.dtype}"
        )
    if tokens.device != timestep_embeddings.device or tokens.device != conditioning_ids.device:
        raise ValueError("tokens, timestep_embeddings, and conditioning_ids must share a device")


def _route_packed_timestep_embeddings(
    tokens: torch.Tensor,
    timestep_embeddings: torch.Tensor,
    conditioning_ids: torch.Tensor,
    *,
    hidden_size: int,
) -> torch.Tensor:
    _validate_packed_timestep_routing(
        tokens, timestep_embeddings, conditioning_ids, hidden_size=hidden_size
    )
    return timestep_embeddings.index_select(0, conditioning_ids)


def _route_packed_adaln_parameters(
    tokens: torch.Tensor,
    image_conditions: torch.Tensor,
    conditioning_ids: torch.Tensor,
    *,
    modulation: nn.Module,
    hidden_size: int,
    parameter_count: int,
) -> tuple[torch.Tensor, ...]:
    """Compute AdaLN once per image, then route its parameters to tokens."""
    _validate_packed_timestep_routing(
        tokens, image_conditions, conditioning_ids, hidden_size=hidden_size
    )
    image_parameters = modulation(image_conditions)
    expected_shape = (image_conditions.shape[0], parameter_count * hidden_size)
    if image_parameters.shape != expected_shape:
        raise ValueError(
            f"AdaLN modulation must return {expected_shape}, got {tuple(image_parameters.shape)}"
        )
    return image_parameters.index_select(0, conditioning_ids).chunk(parameter_count, dim=-1)


def _flat_image_conditioning_ids(
    sample_ids: torch.Tensor, image_ids: torch.Tensor, images_per_sample: tuple[int, ...]
) -> torch.Tensor:
    """Map group-local image IDs to the stable flat image order."""
    group_offsets = []
    next_offset = 0
    for image_count in images_per_sample:
        group_offsets.append(next_offset)
        next_offset += image_count
    offset_tensor = torch.tensor(group_offsets, device=sample_ids.device, dtype=torch.int64)
    return offset_tensor.index_select(0, sample_ids) + image_ids


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000.0):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_freq = t_freq.to(self.mlp[0].weight.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class IntervalEmbedder(TimestepEmbedder):
    """Zero-initialized additive embedding for MeanFlow interval lengths."""

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__(hidden_size, frequency_embedding_size)
        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)

    def forward(self, interval):
        raw = super().forward(interval)
        zeros = torch.zeros_like(interval)
        return raw - super().forward(zeros)


class SwiGLUFFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop=0.0, bias=True) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, x):
        x12 = self.w12(x)
        (x1, x2) = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(self.ffn_dropout(hidden))


class PackedJiTBlock(nn.Module):
    """Timestep-conditioned JiT block for one rank-local packed sequence."""

    def __init__(
        self,
        *,
        hidden_size: int = 1280,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        axes_dims: Sequence[int] = DEFAULT_ROPE_AXES_DIMS,
        theta: float = DEFAULT_ROPE_THETA,
        proj_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if (
            isinstance(mlp_ratio, bool)
            or not isinstance(mlp_ratio, (int, float))
            or (not math.isfinite(mlp_ratio))
            or (mlp_ratio <= 0)
        ):
            raise ValueError(f"mlp_ratio must be positive and finite, got {mlp_ratio!r}")
        self.hidden_size = hidden_size
        self.proj_dropout = _validate_probability("proj_dropout", proj_dropout)
        self.norm1 = RMSNorm(hidden_size, eps=1e-06)
        self.attn = GroupSelfAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            axes_dims=axes_dims,
            theta=theta,
            proj_dropout=self.proj_dropout,
        )
        self.norm2 = RMSNorm(hidden_size, eps=1e-06)
        self.mlp = SwiGLUFFN(hidden_size, int(hidden_size * mlp_ratio), drop=self.proj_dropout)
        for linear in (self.mlp.w12, self.mlp.w3):
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def _timestep_modulations(
        self,
        tokens: torch.Tensor,
        timestep_embeddings: torch.Tensor,
        conditioning_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        return _route_packed_adaln_parameters(
            tokens,
            timestep_embeddings,
            conditioning_ids,
            modulation=self.adaLN_modulation,
            hidden_size=self.hidden_size,
            parameter_count=6,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        timestep_embeddings: torch.Tensor,
        conditioning_ids: torch.Tensor,
        position_ids: torch.Tensor,
        block_mask: BlockMask,
        rotary_factors: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Transform packed tokens using per-image timesteps and fixed routing."""
        (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) = (
            self._timestep_modulations(tokens, timestep_embeddings, conditioning_ids)
        )
        attention_input = self.norm1(tokens) * (1 + scale_msa) + shift_msa
        tokens = tokens + gate_msa * self.attn(
            attention_input, position_ids, block_mask, rotary_factors
        )
        mlp_input = self.norm2(tokens) * (1 + scale_mlp) + shift_mlp
        return tokens + gate_mlp * self.mlp(mlp_input)


class PackedFinalLayer(nn.Module):
    """Predict flattened pixel patches from packed image tokens."""

    def __init__(
        self, *, hidden_size: int = 1280, patch_size: int = 16, out_channels: int = 3
    ) -> None:
        super().__init__()
        for name, value in (
            ("hidden_size", hidden_size),
            ("patch_size", patch_size),
            ("out_channels", out_channels),
        ):
            _validate_positive_integer(name, value)
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.norm_final = RMSNorm(hidden_size, eps=1e-06)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(
        self,
        tokens: torch.Tensor,
        timestep_embeddings: torch.Tensor,
        conditioning_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Predict one flattened pixel patch per packed image token."""
        (shift, scale) = _route_packed_adaln_parameters(
            tokens,
            timestep_embeddings,
            conditioning_ids,
            modulation=self.adaLN_modulation,
            hidden_size=self.hidden_size,
            parameter_count=2,
        )
        normalized = self.norm_final(tokens) * (1 + scale) + shift
        return self.linear(normalized)


class GroupJiT(nn.Module):
    """The glyph-conditioned Group JiT architecture used by both releases."""

    def __init__(
        self,
        *,
        patch_size=16,
        in_channels=3,
        hidden_size=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4.0,
        axes_dims=(12, 34, 34),
        theta=10000.0,
        interval_conditioning=False,
        image_attention_routing="items_interread_glyph_reads_own_item_bidirectional_main",
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.depth = depth
        self.text_conditioning = TextConditioningMode.NONE
        self.n_axis_mode = NAxisMode.ROLE
        self.image_attention_routing = ImageAttentionRoutingMode(image_attention_routing)
        self.interval_conditioning = interval_conditioning
        self.cfg_conditioning = False
        self.cfg_embedder = None
        self.rotary = RotaryEmbedding3D(axes_dims=axes_dims, theta=theta)
        self.patch_projection = DirectPatchProjection(
            patch_size=patch_size, in_channels=in_channels, hidden_size=hidden_size
        )
        self.glyph_projection = DirectGlyphPatchProjection(
            patch_size=patch_size, hidden_size=hidden_size
        )
        self.branch_tokenizer = PackedBranchTokenizer(
            rgb_projection=self.patch_projection, glyph_projection=self.glyph_projection
        )
        self.timestep_embedder = TimestepEmbedder(hidden_size)
        self.interval_embedder = IntervalEmbedder(hidden_size) if interval_conditioning else None
        self.blocks = nn.ModuleList(
            (
                PackedJiTBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    axes_dims=axes_dims,
                    theta=theta,
                )
                for _ in range(depth)
            )
        )
        self.final_layer = PackedFinalLayer(
            hidden_size=hidden_size, patch_size=patch_size, out_channels=in_channels
        )

    def forward(
        self,
        *,
        packed_images,
        prompts,
        geometries,
        timesteps,
        glyph_conditions,
        image_intervals=None,
        return_packed_patches=False,
    ):
        if prompts is not None:
            raise ValueError("PixText target text is specified by glyph images")
        if glyph_conditions is None:
            raise ValueError("glyph_conditions are required")
        if self.interval_conditioning:
            if image_intervals is None or image_intervals.shape != timesteps.images.shape:
                raise ValueError("MeanFlow requires one interval per RGB image")
        elif image_intervals is not None:
            raise ValueError("Teacher does not accept interval conditioning")
        return self._forward_canonical_glyph(
            packed_images=packed_images,
            prompts=None,
            geometries=geometries,
            timesteps=timesteps,
            glyph_conditions=glyph_conditions,
            return_packed_patches=return_packed_patches,
            image_intervals=image_intervals,
            cfg_scales=None,
        )

    def _forward_canonical_glyph(
        self,
        *,
        packed_images: PackedImagePatches,
        prompts: None,
        geometries: Sequence[GroupGeometry],
        timesteps: GroupTimesteps,
        glyph_conditions: Sequence[Sequence[GlyphCondition | None]],
        return_packed_patches: bool,
        image_intervals: torch.Tensor | None,
        cfg_scales: torch.Tensor | None,
    ):
        """Run mixed RGB/glyph branches while returning only RGB predictions."""
        if self.branch_tokenizer is None or self.glyph_projection is None:
            raise RuntimeError("canonical glyph tokenizer is not configured")
        branches = self.branch_tokenizer.tokenize(
            packed_rgb=packed_images, glyphs=glyph_conditions, geometries=geometries
        )
        projected_prompts = None
        sequence = pack_branch_group_sequences(
            branches=branches, prompts=projected_prompts, geometries=geometries
        )
        assert sequence.branch_kinds is not None
        assert sequence.slot_ids is not None
        sequence_timesteps = torch.ones(
            sequence.tokens.shape[0], dtype=timesteps.images.dtype, device=sequence.tokens.device
        )
        flat_image_offset = 0
        for sample_id, image_count in enumerate(packed_images.images_per_sample):
            sample_mask = sequence.sample_ids.eq(sample_id)
            for slot_id in range(image_count - 1):
                item_mask = (
                    sample_mask
                    & sequence.branch_kinds.eq(ITEM_BRANCH)
                    & sequence.slot_ids.eq(slot_id)
                )
                sequence_timesteps[item_mask] = timesteps.images[flat_image_offset + slot_id + 1]
            flat_image_offset += image_count
        if not bool(sequence_timesteps[sequence.branch_kinds.eq(GLYPH_BRANCH)].eq(1).all()):
            raise RuntimeError("glyph timestep must remain fixed at 1")
        timestep_embeddings = self.timestep_embedder(sequence_timesteps)
        if self.interval_embedder is not None:
            if image_intervals is None:
                raise RuntimeError("interval-aware glyph model requires intervals")
            sequence_intervals = torch.zeros_like(sequence_timesteps)
            flat_image_offset = 0
            for sample_id, image_count in enumerate(packed_images.images_per_sample):
                sample_mask = sequence.sample_ids.eq(sample_id)
                for slot_id in range(image_count - 1):
                    item_mask = (
                        sample_mask
                        & sequence.branch_kinds.eq(ITEM_BRANCH)
                        & sequence.slot_ids.eq(slot_id)
                    )
                    sequence_intervals[item_mask] = image_intervals[flat_image_offset + slot_id + 1]
                flat_image_offset += image_count
            timestep_embeddings = timestep_embeddings + self.interval_embedder(sequence_intervals)
        if self.cfg_embedder is not None:
            timestep_embeddings = timestep_embeddings + self.cfg_embedder(
                cfg_scales[sequence.sample_ids]
            )
        conditioning_ids = torch.arange(
            sequence.tokens.shape[0], dtype=torch.int64, device=sequence.tokens.device
        )
        sequence_mask = build_flex_attention_block_mask(
            sequence, image_attention_routing=self.image_attention_routing
        )
        rotary_factors = self.rotary.prepare(sequence.position_ids)
        sequence_tokens = sequence.tokens
        for block in self.blocks:
            sequence_tokens = block(
                sequence_tokens,
                timestep_embeddings,
                conditioning_ids,
                sequence.position_ids,
                sequence_mask,
                rotary_factors,
            )
        branch_image_tokens = extract_image_tokens(sequence_tokens, sequence)
        rgb_tokens = branch_image_tokens.index_select(0, branches.rgb_token_indices)
        rgb_timestep_embeddings = self.timestep_embedder(timesteps.images)
        if self.interval_embedder is not None:
            assert image_intervals is not None
            rgb_timestep_embeddings = rgb_timestep_embeddings + self.interval_embedder(
                image_intervals
            )
        rgb_conditioning_ids = _flat_image_conditioning_ids(
            packed_images.sample_ids, packed_images.image_ids, packed_images.images_per_sample
        )
        if self.final_layer is None:
            raise RuntimeError("glyph conditioning requires the standard final layer")
        predicted_patches = self.final_layer(
            rgb_tokens, rgb_timestep_embeddings, rgb_conditioning_ids
        )
        if return_packed_patches:
            return predicted_patches
        return unpack_image_patches(predicted_patches, packed_images)
