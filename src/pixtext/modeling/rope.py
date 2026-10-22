"""Geometry-aware 3D rotary positions for variable-length image groups."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn

from pixtext.modeling.modes import NAxisMode
from pixtext.modeling.patches import PackedImagePatches, PatchLayout
from pixtext.preprocessing.geometry import GroupGeometry, ImageSize

DEFAULT_ROPE_AXES_DIMS = (12, 34, 34)
DEFAULT_ROPE_THETA = 10000.0


def _apply_rotary_factors(
    values: torch.Tensor, cosine: torch.Tensor, sine: torch.Tensor
) -> torch.Tensor:
    """Apply already-materialized pairwise rotations."""
    output_dtype = values.dtype
    pair_count = values.shape[-1] // 2
    paired = values.float().reshape(*values.shape[:-1], pair_count, 2)
    (first, second) = paired.unbind(dim=-1)
    factor_shape = (*[1] * (values.ndim - 2), values.shape[-2], pair_count)
    cosine = cosine.reshape(factor_shape)
    sine = sine.reshape(factor_shape)
    return (
        torch.stack((first * cosine - second * sine, first * sine + second * cosine), dim=-1)
        .flatten(start_dim=-2)
        .to(output_dtype)
    )


_compiled_apply_rotary_factors = torch.compile(_apply_rotary_factors, dynamic=True)


class RotaryEmbedding3D(nn.Module):
    """Apply dynamic 3D RoPE to the last dimension of Q or K tensors."""

    def __init__(
        self, axes_dims: Sequence[int] = DEFAULT_ROPE_AXES_DIMS, theta: float = DEFAULT_ROPE_THETA
    ) -> None:
        super().__init__()
        if len(axes_dims) != 3:
            raise ValueError(f"axes_dims must contain exactly three dimensions, got {axes_dims!r}")
        if any(
            isinstance(axis_dim, bool)
            or not isinstance(axis_dim, int)
            or axis_dim <= 0
            or (axis_dim % 2 != 0)
            for axis_dim in axes_dims
        ):
            raise ValueError(
                f"every axes_dims entry must be a positive even integer, got {axes_dims!r}"
            )
        if (
            isinstance(theta, bool)
            or not isinstance(theta, (int, float))
            or (not math.isfinite(theta))
            or (theta <= 0)
        ):
            raise ValueError(f"theta must be positive and finite, got {theta!r}")
        self.axes_dims = tuple(axes_dims)
        self.theta = float(theta)
        self.head_dim = sum(self.axes_dims)
        for axis_name, axis_dim in zip(("n", "h", "w"), self.axes_dims, strict=True):
            inverse_frequency = 1.0 / self.theta ** (
                torch.arange(0, axis_dim, 2, dtype=torch.float32, device="cpu") / axis_dim
            )
            self.register_buffer(f"inv_freq_{axis_name}", inverse_frequency, persistent=False)

    def _apply(self, fn, recurse: bool = True):
        """Move buffers with the module while retaining FP32 frequencies."""
        super()._apply(fn, recurse)
        self.inv_freq_n = self.inv_freq_n.float()
        self.inv_freq_h = self.inv_freq_h.float()
        self.inv_freq_w = self.inv_freq_w.float()
        return self

    def forward(self, values: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """Rotate ``values[..., tokens, head_dim]`` using ``[tokens, 3]`` IDs."""
        if not isinstance(values, torch.Tensor):
            raise TypeError(f"values must be a torch.Tensor, got {type(values).__name__}")
        if values.ndim < 2:
            raise ValueError(
                f"values must have at least (tokens, head_dim) dimensions, got shape {tuple(values.shape)}"
            )
        if not values.is_floating_point():
            raise ValueError(f"values must be floating point, got {values.dtype}")
        factors = self.prepare(position_ids)
        return self.apply_precomputed(values, factors)

    def prepare(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Materialize FP32 cosine/sine factors once for a packed sequence."""
        if not isinstance(position_ids, torch.Tensor):
            raise TypeError(
                f"position_ids must be a torch.Tensor, got {type(position_ids).__name__}"
            )
        if position_ids.ndim != 2 or position_ids.shape[1] != 3:
            raise ValueError(
                f"position_ids must use (tokens, 3) layout, got shape {tuple(position_ids.shape)}"
            )
        float_positions = position_ids.float()
        angles = []
        for axis, inverse_frequency in enumerate(
            (self.inv_freq_n, self.inv_freq_h, self.inv_freq_w)
        ):
            if inverse_frequency.device != position_ids.device:
                raise ValueError(
                    f"RotaryEmbedding3D buffers and position_ids must be on the same device: got {inverse_frequency.device} and {position_ids.device}"
                )
            angles.append(float_positions[:, axis, None] * inverse_frequency[None, :])
        packed_angles = torch.cat(angles, dim=-1)
        return (packed_angles.cos(), packed_angles.sin())

    def apply_precomputed(
        self, values: torch.Tensor, factors: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Rotate values using factors returned by :meth:`prepare`."""
        if not isinstance(values, torch.Tensor):
            raise TypeError(f"values must be a torch.Tensor, got {type(values).__name__}")
        if values.ndim < 2:
            raise ValueError(
                f"values must have at least (tokens, head_dim) dimensions, got shape {tuple(values.shape)}"
            )
        if not values.is_floating_point():
            raise ValueError(f"values must be floating point, got {values.dtype}")
        if (
            not isinstance(factors, tuple)
            or len(factors) != 2
            or (not all(isinstance(factor, torch.Tensor) for factor in factors))
        ):
            raise TypeError("factors must be the (cosine, sine) tensor pair")
        (cosine, sine) = factors
        if values.shape[-1] != self.head_dim:
            raise ValueError(
                f"values head dimension does not match axes_dims: got {values.shape[-1]}, expected {self.head_dim}"
            )
        if cosine.ndim != 2 or sine.ndim != 2:
            raise ValueError("rotary factors must use (tokens, head_dim / 2) layout")
        if cosine.shape[0] != values.shape[-2] or sine.shape[0] != values.shape[-2]:
            raise ValueError(
                f"position token count does not match values: got {cosine.shape[0]} and {sine.shape[0]}, expected {values.shape[-2]}"
            )
        expected_factor_shape = (values.shape[-2], self.head_dim // 2)
        if cosine.shape != expected_factor_shape or sine.shape != expected_factor_shape:
            raise ValueError(
                f"rotary factors must use (tokens, head_dim / 2) layout: got {tuple(cosine.shape)} and {tuple(sine.shape)}, expected {expected_factor_shape}"
            )
        if cosine.device != values.device or sine.device != values.device:
            raise ValueError("rotary factors and values must be on the same device")
        apply = (
            _compiled_apply_rotary_factors
            if values.device.type == "cuda"
            else _apply_rotary_factors
        )
        return apply(values, cosine, sine)


def _validate_nonnegative_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")


def _validate_positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def _validate_layout(
    layout: PatchLayout,
    expected_size: ImageSize,
    expected_patch_size: int,
    *,
    sample_id: int,
    image_id: int,
) -> None:
    if layout.patch_size != expected_patch_size:
        raise ValueError(
            f"group {sample_id} image {image_id} patch_size {layout.patch_size} does not match position patch_size {expected_patch_size}"
        )
    if layout.image_size != expected_size:
        raise ValueError(
            f"group {sample_id} image {image_id} layout size {layout.image_size!r} does not match geometry size {expected_size!r}"
        )


def _spatial_grid(
    layout: PatchLayout,
    *,
    offset_h: float,
    offset_w: float,
    scale_h: float = 1.0,
    scale_w: float = 1.0,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    grid = layout.grid_size
    rows = torch.arange(grid.height, dtype=torch.float32, device=device)
    columns = torch.arange(grid.width, dtype=torch.float32, device=device)
    (rows, columns) = torch.meshgrid(rows, columns, indexing="ij")
    return ((rows + 0.5) / scale_h + offset_h, (columns + 0.5) / scale_w + offset_w)


def build_image_position_ids(
    packed: PackedImagePatches,
    geometries: Sequence[GroupGeometry],
    *,
    patch_size: int = 16,
    n_axis_mode: NAxisMode = NAxisMode.ROLE,
) -> torch.Tensor:
    """Build packed ``[n, h, w]`` coordinates on each reference canvas."""
    _validate_positive_integer("patch_size", patch_size)
    n_axis_mode = NAxisMode.require(n_axis_mode)
    if len(geometries) != len(packed.images_per_sample):
        raise ValueError(
            f"geometry count does not match packed group count: got {len(geometries)}, expected {len(packed.images_per_sample)}"
        )
    device = packed.patches.device
    position_parts: list[torch.Tensor] = []
    flat_image_id = 0
    for sample_id, (geometry, image_count) in enumerate(
        zip(geometries, packed.images_per_sample, strict=True)
    ):
        expected_image_count = 1 + len(geometry.items)
        if image_count != expected_image_count:
            raise ValueError(
                f"group {sample_id} has {image_count} packed images but geometry describes {expected_image_count}"
            )
        main_layout = packed.layouts[flat_image_id]
        _validate_layout(
            main_layout, geometry.main.size, patch_size, sample_id=sample_id, image_id=0
        )
        reference = geometry.reference.size
        center_h = reference.height / (2 * patch_size)
        center_w = reference.width / (2 * patch_size)
        (h, w) = _spatial_grid(
            main_layout,
            offset_h=-center_h,
            offset_w=-center_w,
            scale_h=geometry.main.scale_h,
            scale_w=geometry.main.scale_w,
            device=device,
        )
        position_parts.append(torch.stack((torch.zeros_like(h), h, w), dim=-1).reshape(-1, 3))
        flat_image_id += 1
        for image_id, item in enumerate(geometry.items, start=1):
            layout = packed.layouts[flat_image_id]
            _validate_layout(
                layout, item.crop.size, patch_size, sample_id=sample_id, image_id=image_id
            )
            (h, w) = _spatial_grid(
                layout,
                offset_h=item.crop.top / patch_size - center_h,
                offset_w=item.crop.left / patch_size - center_w,
                device=device,
            )
            n = torch.full_like(h, n_axis_mode.position_for(image_id))
            position_parts.append(torch.stack((n, h, w), dim=-1).reshape(-1, 3))
            flat_image_id += 1
    positions = torch.cat(position_parts)
    if positions.shape[0] != packed.patches.shape[0]:
        raise ValueError(
            f"generated position count does not match packed token count: got {positions.shape[0]}, expected {packed.patches.shape[0]}"
        )
    return positions


def build_prompt_position_ids(
    prompt_length: int,
    image_id: int,
    geometry: GroupGeometry,
    *,
    patch_size: int = 16,
    n_axis_mode: NAxisMode = NAxisMode.ROLE,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Build one prompt's independent sequential ``[n, h, w]`` positions."""
    _validate_nonnegative_integer("prompt_length", prompt_length)
    _validate_nonnegative_integer("image_id", image_id)
    _validate_positive_integer("patch_size", patch_size)
    n_axis_mode = NAxisMode.require(n_axis_mode)
    if image_id > len(geometry.items):
        raise ValueError(
            f"image_id {image_id} is outside geometry image range [0, {len(geometry.items)}]"
        )
    reference = geometry.reference.size
    if reference.height % patch_size != 0 or reference.width % patch_size != 0:
        raise ValueError(
            f"reference size must be divisible by patch_size: got {reference!r} and patch_size={patch_size}"
        )
    text_base = math.ceil(
        max(reference.height / (2 * patch_size), reference.width / (2 * patch_size))
    )
    sequence = torch.arange(prompt_length, dtype=torch.float32, device=device).add_(text_base)
    image_axis = torch.full_like(sequence, n_axis_mode.position_for(image_id))
    return torch.stack((image_axis, sequence, sequence), dim=-1)


def build_timestep_position_ids(
    geometry: GroupGeometry,
    *,
    bank_image_ids: Sequence[int],
    tokens_per_bank: int = 4,
    patch_size: int = 16,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Place ordered timestep banks before the spatial coordinate range.

    Multiple item banks intentionally share the item-role ``n`` coordinate;
    their exact image identity is enforced by attention routing rather than by
    adding an arbitrary item-index position.
    """
    if not isinstance(geometry, GroupGeometry):
        raise TypeError(f"geometry must be a GroupGeometry, got {type(geometry).__name__}")
    if not isinstance(bank_image_ids, Sequence) or isinstance(bank_image_ids, (str, bytes)):
        raise TypeError("bank_image_ids must be a sequence of image IDs")
    bank_image_ids = tuple(bank_image_ids)
    if not bank_image_ids:
        raise ValueError("bank_image_ids must contain at least the main image")
    if any(
        isinstance(image_id, bool)
        or not isinstance(image_id, int)
        or (not 0 <= image_id <= len(geometry.items))
        for image_id in bank_image_ids
    ):
        raise ValueError("bank_image_ids must lie within the geometry image range")
    if bank_image_ids[0] != 0:
        raise ValueError("the first timestep bank must target the main image")
    _validate_positive_integer("tokens_per_bank", tokens_per_bank)
    _validate_positive_integer("patch_size", patch_size)
    reference = geometry.reference.size
    if reference.height % patch_size != 0 or reference.width % patch_size != 0:
        raise ValueError(
            f"reference size must be divisible by patch_size: got {reference!r} and patch_size={patch_size}"
        )
    sequence_base = math.ceil(
        max(reference.height / (2 * patch_size), reference.width / (2 * patch_size))
    )
    sequence = torch.arange(tokens_per_bank, dtype=torch.float32, device=device).add_(
        -sequence_base - tokens_per_bank
    )
    banks = []
    for image_id in bank_image_ids:
        role = float(image_id > 0)
        role_axis = torch.full_like(sequence, role)
        banks.append(torch.stack((role_axis, sequence, sequence), dim=-1))
    return torch.cat(banks)
