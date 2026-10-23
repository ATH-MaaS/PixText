from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from pixtext.modeling.patches import pack_image_groups
from pixtext.sampling.flow import GroupFlowSampler
from pixtext.sampling.timesteps import GroupTimesteps, TimestepRangeStats


@dataclass(frozen=True, slots=True)
class GroupMeanFlowCondition:
    """Validated start/end schedule expansion in packed image order."""

    start: GroupTimesteps
    end: GroupTimesteps
    image_intervals: torch.Tensor

    @classmethod
    def from_timesteps(
        cls, start: GroupTimesteps, end: GroupTimesteps, *, validate_values: bool = True
    ) -> GroupMeanFlowCondition:
        if start.images_per_sample != end.images_per_sample:
            raise ValueError("MeanFlow endpoint topology must match")
        if start.images.shape != end.images.shape:
            raise ValueError("MeanFlow endpoint image vectors must match")
        intervals = end.images - start.images
        if validate_values:
            valid = torch.isfinite(intervals).all() & (intervals >= 0).all()
            offset = 0
            for image_count in start.images_per_sample:
                group = intervals[offset : offset + image_count]
                valid = valid & (group[0] == 0)
                if image_count > 2:
                    valid = valid & (group[1:] == group[1]).all()
                offset += image_count
            torch._assert_async(valid, "invalid MeanFlow image intervals")
        return cls(start=start, end=end, image_intervals=intervals)


class GroupMeanFlowSampler(GroupFlowSampler):
    """Uniform clean-grid Euler sampler for interval-aware MeanFlow students."""

    def __init__(
        self,
        model: nn.Module,
        *,
        cfg_embedding_scale: float | None = None,
        cfg_all_steps: bool = False,
        progress_grid: tuple[float, ...] | None = None,
        **kwargs,
    ) -> None:
        if not isinstance(cfg_all_steps, bool):
            raise TypeError("cfg_all_steps must be a bool")
        if cfg_all_steps and cfg_embedding_scale is not None:
            raise ValueError("cfg_all_steps is only for two-branch CFG")
        if cfg_embedding_scale is not None and (
            not math.isfinite(cfg_embedding_scale) or not getattr(model, "cfg_conditioning", False)
        ):
            raise ValueError(
                "cfg_embedding_scale requires a CFG-conditioned model and finite value"
            )
        self.cfg_embedding_scale = cfg_embedding_scale
        self.cfg_all_steps = cfg_all_steps
        kwargs.pop("solver", None)
        super().__init__(model, solver="euler", **kwargs)
        if progress_grid is not None:
            if (
                not isinstance(progress_grid, tuple)
                or len(progress_grid) != self.steps + 1
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or (not math.isfinite(value))
                    for value in progress_grid
                )
                or (progress_grid[0] != 0)
                or (progress_grid[-1] != 1)
                or any(left >= right for (left, right) in zip(progress_grid, progress_grid[1:]))
            ):
                raise ValueError(
                    "progress_grid must strictly increase from 0 to 1 with steps + 1 points"
                )
            progress_grid = tuple(float(value) for value in progress_grid)
        self.progress_grid = progress_grid
        if not getattr(model, "interval_conditioning", False):
            raise ValueError("MeanFlow sampler requires an interval-aware student")
        self.solver = "meanflow_interval_euler"

    def _interval_prediction(self, images, prompts, geometries, start, end, glyph_conditions=None):
        packed = pack_image_groups(images, patch_size=self.patch_size)
        condition = GroupMeanFlowCondition.from_timesteps(start, end)
        return self.model(
            packed_images=packed,
            prompts=prompts,
            geometries=geometries,
            timesteps=start,
            image_intervals=condition.image_intervals,
            **{"cfg_scales": start.main.new_full((len(images),), self.cfg_embedding_scale)}
            if self.cfg_embedding_scale is not None
            else {},
            glyph_conditions=glyph_conditions,
        )

    def _guided_interval_prediction(
        self, images, conditional, unconditional, geometries, start, end, c, glyph_conditions=None
    ):
        conditional_prediction = self._interval_prediction(
            images, conditional, geometries, start, end, glyph_conditions
        )
        (low, high) = self.cfg_interval
        scale = self.cfg_scale if self.cfg_all_steps or low < c < high else 1.0
        if scale == 1.0:
            return conditional_prediction
        unconditional_prediction = self._interval_prediction(
            images,
            unconditional,
            geometries,
            start,
            end,
            None
            if glyph_conditions is None
            else tuple((None,) * len(group) for group in glyph_conditions),
        )
        return tuple(
            tuple(
                uncond + scale * (cond - uncond) for (cond, uncond) in zip(cond_group, uncond_group)
            )
            for (cond_group, uncond_group) in zip(conditional_prediction, unconditional_prediction)
        )

    @torch.no_grad()
    def forward(
        self,
        conditional,
        unconditional,
        geometries,
        *,
        generator=None,
        main_conditions=None,
        glyph_conditions=None,
        progress_callback=None,
    ):
        self._validate_inputs(conditional, unconditional, geometries, generator, main_conditions)
        images = self._initial_noise(geometries, conditional, generator, main_conditions)
        runtime_tensor = conditional.tokens if conditional is not None else main_conditions[0]
        timestep_runs = []
        effective_steps = []
        for step_index in range(self.steps):
            if progress_callback is not None:
                progress_callback(step_index, self.steps)
            c = (
                self.progress_grid[step_index]
                if self.progress_grid is not None
                else step_index / self.steps
            )
            q = (
                self.progress_grid[step_index + 1]
                if self.progress_grid is not None
                else (step_index + 1) / self.steps
            )
            start_progress = torch.full((len(geometries),), c, device=runtime_tensor.device)
            end_progress = torch.full((len(geometries),), q, device=runtime_tensor.device)
            start = self.timestep_schedule(start_progress, geometries)
            end = self.timestep_schedule(end_progress, geometries)
            prediction = self._guided_interval_prediction(
                images, conditional, unconditional, geometries, start, end, c, glyph_conditions
            )
            velocities = self._velocities_from_predictions(
                images, prediction, start, fixed_main=True
            )
            step_sizes = end.images - start.images
            images = self._advance(images, velocities, step_sizes, fixed_main=True)
            timestep_runs.extend((start, end))
            effective_steps.append(step_sizes)
        if progress_callback is not None:
            progress_callback(self.steps, self.steps)
        self.last_timestep_stats = TimestepRangeStats.from_timesteps(
            timestep_runs, effective_steps=effective_steps
        )
        return images
