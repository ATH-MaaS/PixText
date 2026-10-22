from __future__ import annotations

import math
from collections.abc import Sequence
from contextlib import contextmanager
from contextvars import ContextVar

import torch
from torch import nn
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention

from pixtext.modeling.conditioning import (
    IMAGE_TOKEN_TYPE,
    PROMPT_TOKEN_TYPE,
    TIME_TOKEN_TYPE,
    PackedGroupSequence,
)
from pixtext.modeling.modes import ImageAttentionRoutingMode, TimestepTokenRoutingMode
from pixtext.modeling.patches import PackedImagePatches
from pixtext.modeling.rope import (
    DEFAULT_ROPE_AXES_DIMS,
    DEFAULT_ROPE_THETA,
    RotaryEmbedding3D,
    _apply_rotary_factors,
)

_compiled_flex_attention = torch.compile(flex_attention, dynamic=True)
_FORCE_STRICT_FP32_FLEX_ATTENTION = ContextVar("force_strict_fp32_flex_attention", default=False)
_FLEX_ATTENTION_FORWARD_NUM_STAGES = ContextVar("flex_attention_forward_num_stages", default=None)
_INTERREAD = ImageAttentionRoutingMode.ITEMS_INTERREAD_GLYPH_READS_OWN_ITEM
_INTERREAD_BIDIRECTIONAL_MAIN = (
    ImageAttentionRoutingMode.ITEMS_INTERREAD_GLYPH_READS_OWN_ITEM_BIDIRECTIONAL_MAIN
)


@contextmanager
def force_strict_fp32_flex_attention():
    """Keep FP32 FlexAttention strict inside a broader TF32 forward policy."""
    token = _FORCE_STRICT_FP32_FLEX_ATTENTION.set(True)
    try:
        yield
    finally:
        _FORCE_STRICT_FP32_FLEX_ATTENTION.reset(token)


@contextmanager
def use_flex_attention_forward_num_stages(num_stages: int):
    """Select a scoped Triton forward pipeline depth for FlexAttention."""
    if isinstance(num_stages, bool) or not isinstance(num_stages, int) or num_stages <= 0:
        raise ValueError("FlexAttention num_stages must be a positive integer")
    token = _FLEX_ATTENTION_FORWARD_NUM_STAGES.set(num_stages)
    try:
        yield
    finally:
        _FLEX_ATTENTION_FORWARD_NUM_STAGES.reset(token)


def _normalize_and_apply_rotary(
    values: torch.Tensor, weight: torch.Tensor, eps: float, cosine: torch.Tensor, sine: torch.Tensor
) -> torch.Tensor:
    """Run Q/K RMSNorm and precomputed RoPE as one compilable subgraph."""
    output_dtype = values.dtype
    variance = values.float().square().mean(dim=-1, keepdim=True)
    normalized = values.float() * torch.rsqrt(variance + eps)
    normalized = (normalized * weight.float()).to(output_dtype)
    return _apply_rotary_factors(normalized, cosine, sine)


_compiled_normalize_and_apply_rotary = torch.compile(_normalize_and_apply_rotary, dynamic=True)


class _RMSNorm(nn.Module):
    """Last-dimension RMSNorm with FP32 reduction statistics."""

    def __init__(self, hidden_size: int, eps: float = 1e-06) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        output_dtype = values.dtype
        variance = values.float().square().mean(dim=-1, keepdim=True)
        normalized = values.float() * torch.rsqrt(variance + self.eps)
        return (normalized * self.weight.float()).to(output_dtype)


def _validate_routing(
    *,
    sample_ids: torch.Tensor,
    image_ids: torch.Tensor,
    token_types: torch.Tensor,
    timestep_roles: torch.Tensor,
    branch_kinds: torch.Tensor,
    slot_ids: torch.Tensor,
    token_count: int,
    device: torch.device,
) -> None:
    for name, values in (
        ("sample_ids", sample_ids),
        ("image_ids", image_ids),
        ("token_types", token_types),
        ("timestep_roles", timestep_roles),
        ("branch_kinds", branch_kinds),
        ("slot_ids", slot_ids),
    ):
        if not isinstance(values, torch.Tensor):
            raise TypeError(f"routing {name} must be a torch.Tensor")
        if values.shape != (token_count,):
            raise ValueError(
                f"routing {name} must contain {token_count} entries, got shape {tuple(values.shape)}"
            )
        if values.dtype != torch.int64:
            raise ValueError(f"routing {name} must use torch.int64, got {values.dtype}")
        if values.device != device:
            raise ValueError(f"routing {name} must be on {device}, got {values.device}")
    valid_token_types = (
        (token_types == IMAGE_TOKEN_TYPE)
        | (token_types == PROMPT_TOKEN_TYPE)
        | (token_types == TIME_TOKEN_TYPE)
    )
    if not bool(valid_token_types.all()):
        raise ValueError("routing token_types may only contain image, prompt, or timestep IDs")
    valid_timestep_roles = (timestep_roles == 0) | (timestep_roles == 1)
    if not bool(valid_timestep_roles.all()):
        raise ValueError("routing timestep_roles may only contain 0 or 1")
    if not bool(((branch_kinds >= -1) & (branch_kinds <= 2)).all()):
        raise ValueError("routing branch_kinds may only contain -1, 0, 1, or 2")
    if not bool((slot_ids >= -1).all()):
        raise ValueError("routing slot_ids must be at least -1")


def _validated_routing(
    packed: PackedGroupSequence | PackedImagePatches,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
    torch.device,
    TimestepTokenRoutingMode | None,
    torch.Tensor,
    torch.Tensor,
    bool,
]:
    if isinstance(packed, PackedGroupSequence):
        sample_ids = packed.sample_ids
        image_ids = packed.image_ids
        token_types = packed.token_types
        timestep_roles = packed.timestep_roles
        token_count = packed.tokens.shape[0]
        device = packed.tokens.device
        timestep_token_routing = packed.timestep_token_routing
        explicit_branches = packed.branch_kinds is not None
        if explicit_branches:
            assert packed.branch_kinds is not None
            assert packed.slot_ids is not None
            branch_kinds = packed.branch_kinds
            slot_ids = packed.slot_ids
        else:
            branch_kinds = torch.where(
                image_ids.eq(0), torch.zeros_like(image_ids), torch.ones_like(image_ids)
            )
            slot_ids = torch.where(image_ids.gt(0), image_ids - 1, -1)
    elif isinstance(packed, PackedImagePatches):
        sample_ids = packed.sample_ids
        image_ids = packed.image_ids
        token_types = torch.full_like(image_ids, IMAGE_TOKEN_TYPE)
        timestep_roles = image_ids.gt(0).to(torch.int64)
        token_count = packed.patches.shape[0]
        device = packed.patches.device
        timestep_token_routing = None
        explicit_branches = False
        branch_kinds = torch.where(
            image_ids.eq(0), torch.zeros_like(image_ids), torch.ones_like(image_ids)
        )
        slot_ids = torch.where(image_ids.gt(0), image_ids - 1, -1)
    else:
        raise TypeError(
            f"packed must be PackedGroupSequence or PackedImagePatches, got {type(packed).__name__}"
        )
    _validate_routing(
        sample_ids=sample_ids,
        image_ids=image_ids,
        token_types=token_types,
        timestep_roles=timestep_roles,
        branch_kinds=branch_kinds,
        slot_ids=slot_ids,
        token_count=token_count,
        device=device,
    )
    contains_time_tokens = bool((token_types == TIME_TOKEN_TYPE).any())
    if contains_time_tokens != (timestep_token_routing is not None):
        raise ValueError(
            "timestep token routing must be present exactly when timestep tokens are packed"
        )
    if timestep_token_routing is not None:
        TimestepTokenRoutingMode.require(timestep_token_routing)
        if timestep_token_routing is TimestepTokenRoutingMode.PER_IMAGE_CONTEXTUAL and bool(
            (image_ids[token_types == TIME_TOKEN_TYPE] < 0).any()
        ):
            raise ValueError("per-image contextual timestep tokens require image IDs")
    return (
        sample_ids,
        image_ids,
        token_types,
        timestep_roles,
        token_count,
        device,
        timestep_token_routing,
        branch_kinds,
        slot_ids,
        explicit_branches,
    )


def _allowed_image_pair(
    *,
    query_image_ids: torch.Tensor,
    key_image_ids: torch.Tensor,
    query_token_types: torch.Tensor,
    key_token_types: torch.Tensor,
    image_attention_routing: ImageAttentionRoutingMode,
) -> torch.Tensor:
    allowed = (query_token_types == IMAGE_TOKEN_TYPE) & (key_token_types == IMAGE_TOKEN_TYPE)
    if image_attention_routing is ImageAttentionRoutingMode.ALL_IMAGES:
        return allowed
    if image_attention_routing is ImageAttentionRoutingMode.MAIN_IMAGE_HUB:
        return allowed & ((query_image_ids == 0) | (key_image_ids == 0))
    if image_attention_routing is ImageAttentionRoutingMode.ITEMS_READ_MAIN:
        return allowed & (query_image_ids > 0) & (key_image_ids == 0)
    if image_attention_routing in (_INTERREAD, _INTERREAD_BIDIRECTIONAL_MAIN):
        if image_attention_routing is _INTERREAD_BIDIRECTIONAL_MAIN:
            return allowed & ((query_image_ids > 0) | (key_image_ids > 0))
        return allowed & (query_image_ids > 0)
    raise AssertionError(f"unhandled image attention routing: {image_attention_routing!r}")


def _allowed_routing_pair(
    *,
    query_image_ids: torch.Tensor,
    key_image_ids: torch.Tensor,
    query_token_types: torch.Tensor,
    key_token_types: torch.Tensor,
    query_timestep_roles: torch.Tensor,
    key_timestep_roles: torch.Tensor,
    image_attention_routing: ImageAttentionRoutingMode,
    per_image_contextual_time: bool,
    query_branch_kinds: torch.Tensor,
    key_branch_kinds: torch.Tensor,
    query_slot_ids: torch.Tensor,
    key_slot_ids: torch.Tensor,
    explicit_branches: bool,
) -> torch.Tensor:
    query_is_time = query_token_types == TIME_TOKEN_TYPE
    key_is_time = key_token_types == TIME_TOKEN_TYPE
    content_pair = ~query_is_time & ~key_is_time
    same_image = query_image_ids == key_image_ids
    if explicit_branches:
        query_is_image = query_token_types == IMAGE_TOKEN_TYPE
        key_is_image = key_token_types == IMAGE_TOKEN_TYPE
        image_pair = query_is_image & key_is_image
        if image_attention_routing is ImageAttentionRoutingMode.ALL_IMAGES:
            allowed_image_pair = image_pair
        elif image_attention_routing is ImageAttentionRoutingMode.MAIN_IMAGE_HUB:
            allowed_image_pair = image_pair & (query_branch_kinds.eq(0) | key_branch_kinds.eq(0))
        elif image_attention_routing is ImageAttentionRoutingMode.ITEMS_READ_MAIN:
            item_reads_main = image_pair & query_branch_kinds.eq(1) & key_branch_kinds.eq(0)
            item_reads_own_glyph = (
                image_pair
                & query_branch_kinds.eq(1)
                & key_branch_kinds.eq(2)
                & query_slot_ids.eq(key_slot_ids)
            )
            allowed_image_pair = item_reads_main | item_reads_own_glyph
        elif image_attention_routing in (_INTERREAD, _INTERREAD_BIDIRECTIONAL_MAIN):
            item_reads_main = image_pair & query_branch_kinds.eq(1) & key_branch_kinds.eq(0)
            item_reads_items = image_pair & query_branch_kinds.eq(1) & key_branch_kinds.eq(1)
            item_reads_own_glyph = (
                image_pair
                & query_branch_kinds.eq(1)
                & key_branch_kinds.eq(2)
                & query_slot_ids.eq(key_slot_ids)
            )
            glyph_reads_own_item = (
                image_pair
                & query_branch_kinds.eq(2)
                & key_branch_kinds.eq(1)
                & query_slot_ids.eq(key_slot_ids)
            )
            allowed_image_pair = (
                item_reads_main | item_reads_items | item_reads_own_glyph | glyph_reads_own_item
            )
            if image_attention_routing is _INTERREAD_BIDIRECTIONAL_MAIN:
                allowed_image_pair = allowed_image_pair | image_pair & query_branch_kinds.eq(
                    0
                ) & key_branch_kinds.eq(1)
        else:
            raise AssertionError(f"unhandled image attention routing: {image_attention_routing!r}")
    else:
        allowed_image_pair = _allowed_image_pair(
            query_image_ids=query_image_ids,
            key_image_ids=key_image_ids,
            query_token_types=query_token_types,
            key_token_types=key_token_types,
            image_attention_routing=image_attention_routing,
        )
    if per_image_contextual_time:
        reads_own_time_bank = key_is_time & same_image
        time_reads_own_image = query_is_time & (key_token_types == IMAGE_TOKEN_TYPE) & same_image
    else:
        same_timestep_role = query_timestep_roles == key_timestep_roles
        reads_own_time_bank = key_is_time & same_timestep_role
        time_reads_own_image = torch.zeros_like(reads_own_time_bank)
    return (
        content_pair & (same_image | allowed_image_pair)
        | reads_own_time_bank
        | time_reads_own_image
    )


def build_dense_attention_mask(
    packed: PackedGroupSequence | PackedImagePatches,
    *,
    image_attention_routing: ImageAttentionRoutingMode = ImageAttentionRoutingMode.ITEMS_INTERREAD_GLYPH_READS_OWN_ITEM_BIDIRECTIONAL_MAIN,
) -> torch.Tensor:
    """Return a ``[query, key]`` bool mask where ``True`` means allowed."""
    routing = ImageAttentionRoutingMode.require(image_attention_routing)
    (
        sample_ids,
        image_ids,
        token_types,
        timestep_roles,
        _,
        _,
        timestep_token_routing,
        branch_kinds,
        slot_ids,
        explicit_branches,
    ) = _validated_routing(packed)
    same_group = sample_ids[:, None] == sample_ids[None, :]
    allowed_pair = _allowed_routing_pair(
        query_image_ids=image_ids[:, None],
        key_image_ids=image_ids[None, :],
        query_token_types=token_types[:, None],
        key_token_types=token_types[None, :],
        query_timestep_roles=timestep_roles[:, None],
        key_timestep_roles=timestep_roles[None, :],
        image_attention_routing=routing,
        per_image_contextual_time=timestep_token_routing
        is TimestepTokenRoutingMode.PER_IMAGE_CONTEXTUAL,
        query_branch_kinds=branch_kinds[:, None],
        key_branch_kinds=branch_kinds[None, :],
        query_slot_ids=slot_ids[:, None],
        key_slot_ids=slot_ids[None, :],
        explicit_branches=explicit_branches,
    )
    return same_group & allowed_pair


def build_flex_attention_block_mask(
    packed: PackedGroupSequence | PackedImagePatches,
    *,
    image_attention_routing: ImageAttentionRoutingMode = ImageAttentionRoutingMode.ITEMS_INTERREAD_GLYPH_READS_OWN_ITEM_BIDIRECTIONAL_MAIN,
) -> BlockMask:
    """Build the packed routing mask consumed by PyTorch FlexAttention."""
    routing = ImageAttentionRoutingMode.require(image_attention_routing)
    (
        sample_ids,
        image_ids,
        token_types,
        timestep_roles,
        token_count,
        device,
        timestep_token_routing,
        branch_kinds,
        slot_ids,
        explicit_branches,
    ) = _validated_routing(packed)
    if token_count == 0:
        raise ValueError("FlexAttention requires at least one packed token")
    per_image_contextual_time = (
        timestep_token_routing is TimestepTokenRoutingMode.PER_IMAGE_CONTEXTUAL
    )

    def mask_mod(
        batch_index: torch.Tensor,
        head_index: torch.Tensor,
        query_index: torch.Tensor,
        key_index: torch.Tensor,
    ) -> torch.Tensor:
        del batch_index, head_index
        query_in_bounds = query_index < token_count
        key_in_bounds = key_index < token_count
        safe_query_index = query_index.clamp(max=token_count - 1)
        safe_key_index = key_index.clamp(max=token_count - 1)
        same_group = sample_ids[safe_query_index] == sample_ids[safe_key_index]
        allowed_pair = _allowed_routing_pair(
            query_image_ids=image_ids[safe_query_index],
            key_image_ids=image_ids[safe_key_index],
            query_token_types=token_types[safe_query_index],
            key_token_types=token_types[safe_key_index],
            query_timestep_roles=timestep_roles[safe_query_index],
            key_timestep_roles=timestep_roles[safe_key_index],
            image_attention_routing=routing,
            per_image_contextual_time=per_image_contextual_time,
            query_branch_kinds=branch_kinds[safe_query_index],
            key_branch_kinds=branch_kinds[safe_key_index],
            query_slot_ids=slot_ids[safe_query_index],
            key_slot_ids=slot_ids[safe_key_index],
            explicit_branches=explicit_branches,
        )
        return query_in_bounds & key_in_bounds & same_group & allowed_pair

    return create_block_mask(
        mask_mod, B=None, H=None, Q_LEN=token_count, KV_LEN=token_count, device=str(device)
    )


def run_flex_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, *, block_mask: BlockMask
) -> torch.Tensor:
    """Run packed CUDA self-attention without changing the model head dimension."""
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, and value must use (B, H, T, D) layout")
    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError("packed self-attention requires identical query, key, and value shapes")
    if query.device.type != "cuda":
        raise RuntimeError("FlexAttention requires CUDA tensors")
    force_strict = query.dtype is torch.float32 and _FORCE_STRICT_FP32_FLEX_ATTENTION.get()
    old_matmul_tf32 = None
    if force_strict:
        old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
    kernel_options = {"FORCE_USE_FLEX_ATTENTION": True}
    forward_num_stages = _FLEX_ATTENTION_FORWARD_NUM_STAGES.get()
    if forward_num_stages is not None:
        kernel_options["num_stages"] = forward_num_stages
    try:
        return _compiled_flex_attention(
            query, key, value, block_mask=block_mask, kernel_options=kernel_options
        )
    finally:
        if force_strict:
            assert old_matmul_tf32 is not None
            torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32


class GroupSelfAttention(nn.Module):
    """QKV, QK normalization, 3D RoPE, and packed FlexAttention."""

    def __init__(
        self,
        *,
        hidden_size: int = 1280,
        num_heads: int = 16,
        axes_dims: Sequence[int] = DEFAULT_ROPE_AXES_DIMS,
        theta: float = DEFAULT_ROPE_THETA,
        proj_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        for name, value in (("hidden_size", hidden_size), ("num_heads", num_heads)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size must be divisible by num_heads: got {hidden_size} and {num_heads}"
            )
        if (
            isinstance(proj_dropout, bool)
            or not isinstance(proj_dropout, (int, float))
            or (not math.isfinite(proj_dropout))
            or (not 0 <= proj_dropout < 1)
        ):
            raise ValueError(f"proj_dropout must be finite and in [0, 1), got {proj_dropout!r}")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.rotary = RotaryEmbedding3D(axes_dims=axes_dims, theta=theta)
        if self.rotary.head_dim != self.head_dim:
            raise ValueError(
                f"axes_dims must sum to the attention head dimension: got {self.rotary.head_dim}, expected {self.head_dim}"
            )
        self.q_norm = _RMSNorm(self.head_dim)
        self.k_norm = _RMSNorm(self.head_dim)
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.proj_dropout = nn.Dropout(float(proj_dropout))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.ones_(self.q_norm.weight)
        nn.init.ones_(self.k_norm.weight)
        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.zeros_(self.qkv.bias)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(
        self,
        tokens: torch.Tensor,
        position_ids: torch.Tensor,
        block_mask: BlockMask,
        rotary_factors: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Transform packed ``[tokens, hidden_size]`` values without padding."""
        if not isinstance(tokens, torch.Tensor):
            raise TypeError(f"tokens must be a torch.Tensor, got {type(tokens).__name__}")
        if tokens.ndim != 2 or tokens.shape[1] != self.hidden_size:
            raise ValueError(
                f"tokens must use (tokens, hidden_size) layout with hidden_size {self.hidden_size}, got shape {tuple(tokens.shape)}"
            )
        if not tokens.is_floating_point():
            raise ValueError(f"tokens must be floating point, got {tokens.dtype}")
        if not isinstance(block_mask, BlockMask):
            raise TypeError(f"block_mask must be a BlockMask, got {type(block_mask).__name__}")
        token_count = tokens.shape[0]
        if block_mask.shape[-2:] != (token_count, token_count):
            raise ValueError(
                f"BlockMask query/key lengths must match the packed token count: got {block_mask.shape[-2:]}, expected {(token_count, token_count)}"
            )
        qkv = self.qkv(tokens).reshape(token_count, 3, self.num_heads, self.head_dim)
        (query, key, value) = qkv.permute(1, 2, 0, 3).unbind(dim=0)
        if rotary_factors is None:
            rotary_factors = self.rotary.prepare(position_ids)
        normalize_and_rotate = (
            _compiled_normalize_and_apply_rotary
            if tokens.device.type == "cuda"
            else _normalize_and_apply_rotary
        )
        (cosine, sine) = rotary_factors
        query = normalize_and_rotate(
            query, self.q_norm.weight, self.q_norm.eps, cosine, sine
        ).unsqueeze(0)
        key = normalize_and_rotate(
            key, self.k_norm.weight, self.k_norm.eps, cosine, sine
        ).unsqueeze(0)
        value = value.unsqueeze(0)
        attended = run_flex_attention(query, key, value, block_mask=block_mask)
        merged = attended.transpose(1, 2).reshape(token_count, self.hidden_size)
        return self.proj_dropout(self.proj(merged))
