from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class TimestepRangeStats:
    """Device-independent ranges for timestep observability."""

    main_min: float
    main_max: float
    item_min: float | None
    item_max: float | None
    shift_mu_min: float
    shift_mu_max: float
    effective_step_min: float | None = None
    effective_step_max: float | None = None

    @classmethod
    def merge(cls, values: Sequence[TimestepRangeStats]) -> TimestepRangeStats:
        """Merge already-synchronized ranges without device work."""
        if not values:
            raise ValueError("at least one TimestepRangeStats value is required")

        def optional_min(name: str) -> float | None:
            present = [getattr(value, name) for value in values]
            finite = [value for value in present if value is not None]
            return min(finite) if finite else None

        def optional_max(name: str) -> float | None:
            present = [getattr(value, name) for value in values]
            finite = [value for value in present if value is not None]
            return max(finite) if finite else None

        return cls(
            main_min=min(value.main_min for value in values),
            main_max=max(value.main_max for value in values),
            item_min=optional_min("item_min"),
            item_max=optional_max("item_max"),
            shift_mu_min=min(value.shift_mu_min for value in values),
            shift_mu_max=max(value.shift_mu_max for value in values),
            effective_step_min=optional_min("effective_step_min"),
            effective_step_max=optional_max("effective_step_max"),
        )

    @classmethod
    def from_timesteps(
        cls, values: Sequence[GroupTimesteps], *, effective_steps: Sequence[torch.Tensor] = ()
    ) -> TimestepRangeStats:
        """Summarize one or more schedule evaluations with one device sync."""
        if len(values) == 0:
            raise ValueError("at least one GroupTimesteps value is required")
        main = torch.cat(tuple(value.main for value in values))
        shift_mu = torch.cat(tuple(value.shift_mu for value in values))
        item_parts = tuple(
            item_values for value in values if (item_values := value.item_values()) is not None
        )
        items = torch.cat(item_parts) if item_parts else None
        steps = torch.cat(tuple(effective_steps)) if effective_steps else None
        missing = main.new_full((), math.nan)
        summary_parts = [main.min(), main.max()]
        summary_parts.extend((items.min(), items.max()) if items is not None else (missing,) * 2)
        summary_parts.extend((shift_mu.min(), shift_mu.max()))
        summary_parts.extend((steps.min(), steps.max()) if steps is not None else (missing,) * 2)
        summary = torch.stack(summary_parts).detach().cpu().tolist()
        required_indices = [0, 1, 4, 5]
        if items is not None:
            required_indices.extend((2, 3))
        if steps is not None:
            required_indices.extend((6, 7))
        if any(not math.isfinite(summary[index]) for index in required_indices):
            raise FloatingPointError("timestep range contains non-finite values")

        def optional(value: float) -> float | None:
            return None if math.isnan(value) else value

        return cls(
            main_min=summary[0],
            main_max=summary[1],
            item_min=optional(summary[2]),
            item_max=optional(summary[3]),
            shift_mu_min=summary[4],
            shift_mu_max=summary[5],
            effective_step_min=optional(summary[6]),
            effective_step_max=optional(summary[7]),
        )

    def to_dict(self) -> dict[str, float | None]:
        """Return stable field names for logs and sampling metadata."""
        return {
            "main_min": self.main_min,
            "main_max": self.main_max,
            "item_min": self.item_min,
            "item_max": self.item_max,
            "shift_mu_min": self.shift_mu_min,
            "shift_mu_max": self.shift_mu_max,
            "effective_step_min": self.effective_step_min,
            "effective_step_max": self.effective_step_max,
        }


@dataclass(frozen=True, slots=True)
class GroupTimesteps:
    """Main and effective image timesteps in stable packed-image order."""

    main: torch.Tensor
    images: torch.Tensor
    images_per_sample: tuple[int, ...]
    shift_mu: torch.Tensor

    @property
    def applied_shift_mu(self) -> torch.Tensor:
        """Policy-neutral name for the exponent shift used by the schedule."""
        return self.shift_mu

    def item_values(self) -> torch.Tensor | None:
        """Return packed item-only timesteps, or None for main-only batches."""
        item_parts = []
        offset = 0
        for image_count in self.images_per_sample:
            if image_count > 1:
                item_parts.append(self.images[offset + 1 : offset + image_count])
            offset += image_count
        return torch.cat(item_parts) if item_parts else None

    def range_stats(self, *, effective_steps: torch.Tensor | None = None) -> TimestepRangeStats:
        """Synchronize compact timestep ranges for logs and metadata."""
        if effective_steps is not None:
            if effective_steps.shape != self.images.shape:
                raise ValueError(
                    f"effective_steps must have one value per image, got {tuple(effective_steps.shape)} and {tuple(self.images.shape)}"
                )
        return TimestepRangeStats.from_timesteps(
            (self,), effective_steps=(effective_steps,) if effective_steps is not None else ()
        )


class GroupTimestepSchedule:
    """Keep main context at t=1 and advance items on an unshifted grid."""

    is_native_edit = True

    def __init__(self, patch_size=16):
        self.patch_size = patch_size

    def __call__(self, group_progress, geometries, *, validate_values=True):
        if group_progress.shape != (len(geometries),) or not geometries:
            raise ValueError("Need one progress value per image group")
        progress = group_progress.float()
        if validate_values and not bool(
            (torch.isfinite(progress) & (progress >= 0) & (progress <= 1)).all()
        ):
            raise ValueError("Progress must be finite and within [0, 1]")
        counts = tuple(1 + len(g.items) for g in geometries)
        parts = []
        for index, count in enumerate(counts):
            parts.extend(
                (torch.ones_like(progress[index : index + 1]), progress[index].expand(count - 1))
            )
        return GroupTimesteps(
            torch.ones_like(progress), torch.cat(parts), counts, torch.zeros_like(progress)
        )


def resolve_group_timestep_schedule(schedule, *, patch_size):
    if schedule is None:
        return GroupTimestepSchedule(patch_size)
    if schedule.patch_size != patch_size:
        raise ValueError("Schedule and model patch sizes must match")
    return schedule
