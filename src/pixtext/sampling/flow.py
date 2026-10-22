"""Joint ODE sampling for variable-length grouped image JiT outputs."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import torch
from torch import nn

from pixtext.modeling.patches import pack_image_groups
from pixtext.preprocessing.geometry import GroupGeometry
from pixtext.preprocessing.glyphs import GlyphCondition
from pixtext.sampling.timesteps import (
    GroupTimesteps,
    GroupTimestepSchedule,
    TimestepRangeStats,
    resolve_group_timestep_schedule,
)


class GroupFlowSampler(nn.Module):
    """Jointly sample complete image groups with group-level CFG."""

    def __init__(
        self,
        model: nn.Module,
        *,
        solver: str = "heun",
        steps: int = 50,
        cfg_scale: float = 1.0,
        cfg_interval: tuple[float, float] = (0.1, 1.0),
        noise_scale: float = 1.0,
        timestep_schedule: GroupTimestepSchedule | None = None,
        compute_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        if solver not in {"euler", "heun"}:
            raise ValueError(f"solver must be 'euler' or 'heun', got {solver!r}")
        if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
            raise ValueError(f"steps must be a positive integer, got {steps!r}")
        for name, value, positive in (
            ("cfg_scale", cfg_scale, False),
            ("noise_scale", noise_scale, False),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or (not math.isfinite(value))
                or (value <= 0 if positive else value < 0)
            ):
                qualifier = "positive" if positive else "non-negative"
                raise ValueError(f"{name} must be finite and {qualifier}, got {value!r}")
        if (
            not isinstance(cfg_interval, tuple)
            or len(cfg_interval) != 2
            or any(
                (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or (not math.isfinite(value))
                    for value in cfg_interval
                )
            )
            or (not 0 <= cfg_interval[0] < cfg_interval[1] <= 1)
        ):
            raise ValueError(
                f"cfg_interval must be a finite (low, high) pair within [0, 1], got {cfg_interval!r}"
            )
        patch_size = getattr(model, "patch_size", None)
        if isinstance(patch_size, bool) or not isinstance(patch_size, int) or patch_size <= 0:
            raise ValueError(f"model.patch_size must be a positive integer, got {patch_size!r}")
        in_channels = getattr(model, "in_channels", None)
        if isinstance(in_channels, bool) or not isinstance(in_channels, int) or in_channels <= 0:
            raise ValueError(f"model.in_channels must be a positive integer, got {in_channels!r}")
        self.model = model
        self.solver = solver
        self.steps = steps
        self.cfg_scale = float(cfg_scale)
        self.cfg_interval = (float(cfg_interval[0]), float(cfg_interval[1]))
        self.noise_scale = float(noise_scale)
        self.patch_size = patch_size
        self.in_channels = in_channels
        if compute_dtype is not None and (
            not isinstance(compute_dtype, torch.dtype) or not compute_dtype.is_floating_point
        ):
            raise TypeError("compute_dtype must be a floating torch.dtype or None")
        self.compute_dtype = compute_dtype
        self.timestep_schedule = resolve_group_timestep_schedule(
            timestep_schedule, patch_size=self.patch_size
        )
        self.last_timestep_stats: TimestepRangeStats | None = None

    def _validate_inputs(
        self,
        conditional: None,
        unconditional: None,
        geometries: Sequence[GroupGeometry],
        generator: torch.Generator | Sequence[torch.Generator] | None,
        main_conditions: Sequence[torch.Tensor] | None,
    ) -> None:
        if self.model.training:
            raise RuntimeError("sampling model must be in eval mode")
        if conditional is not None or unconditional is not None:
            raise ValueError("no-text sampler requires prompt views to be None")
        group_count = len(geometries)
        if group_count == 0:
            raise ValueError("sampling batch must contain at least one group")
        if generator is not None and (not isinstance(generator, torch.Generator)):
            if (
                not isinstance(generator, Sequence)
                or len(generator) != group_count
                or any((not isinstance(item, torch.Generator) for item in generator))
            ):
                raise TypeError("generator must be one torch.Generator, one per group, or None")
        image_counts = []
        for group_id, geometry in enumerate(geometries):
            if not isinstance(geometry, GroupGeometry):
                raise TypeError(
                    f"geometry {group_id} must be GroupGeometry, got {type(geometry).__name__}"
                )
            image_counts.append(1 + len(geometry.items))
        image_counts = tuple(image_counts)
        if self.timestep_schedule.is_native_edit:
            if main_conditions is None or len(main_conditions) != group_count:
                raise ValueError("native edit sampling requires one main condition per group")
            for group_id, (condition, geometry) in enumerate(
                zip(main_conditions, geometries, strict=True)
            ):
                expected_shape = (
                    self.in_channels,
                    geometry.main.size.height,
                    geometry.main.size.width,
                )
                if not isinstance(condition, torch.Tensor) or condition.shape != expected_shape:
                    raise ValueError(
                        f"main condition {group_id} must match shape, device, and dtype {expected_shape}, {condition.device}, and {condition.dtype}"
                    )
        elif main_conditions is not None:
            raise ValueError("main_conditions are only valid for native edit sampling")
        floating_parameters = tuple(
            (parameter for parameter in self.model.parameters() if parameter.is_floating_point())
        )
        runtime_tensor = main_conditions[0]
        if floating_parameters and any(
            (parameter.device != runtime_tensor.device for parameter in floating_parameters)
        ):
            raise ValueError("all floating model parameters and main conditions must share device")
        if self.compute_dtype is None:
            if floating_parameters and any(
                (parameter.dtype != runtime_tensor.dtype for parameter in floating_parameters)
            ):
                raise ValueError(
                    "all floating model parameters and main conditions must share dtype"
                )
        elif runtime_tensor.dtype != self.compute_dtype:
            raise ValueError(
                f"main condition dtype must match the configured model compute dtype {self.compute_dtype}"
            )

    def _initial_noise(
        self,
        geometries: Sequence[GroupGeometry],
        prompts: None,
        generator: torch.Generator | Sequence[torch.Generator] | None,
        main_conditions: Sequence[torch.Tensor] | None,
    ) -> tuple[tuple[torch.Tensor, ...], ...]:
        groups = []
        runtime_tensor = main_conditions[0]
        if isinstance(generator, Sequence):
            group_generators = tuple(generator)
        else:
            group_generators = (generator,) * len(geometries)
        for group_id, (geometry, group_generator) in enumerate(
            zip(geometries, group_generators, strict=True)
        ):
            item_sizes = tuple((item.crop.size for item in geometry.items))
            if self.timestep_schedule.is_native_edit:
                assert main_conditions is not None
                group = (main_conditions[group_id],) + tuple(
                    (
                        torch.randn(
                            self.in_channels,
                            size.height,
                            size.width,
                            device=runtime_tensor.device,
                            dtype=runtime_tensor.dtype,
                            generator=group_generator,
                        )
                        * self.noise_scale
                        for size in item_sizes
                    )
                )
            else:
                sizes = (geometry.main.size,) + item_sizes
                group = tuple(
                    (
                        torch.randn(
                            self.in_channels,
                            size.height,
                            size.width,
                            device=runtime_tensor.device,
                            dtype=runtime_tensor.dtype,
                            generator=group_generator,
                        )
                        * self.noise_scale
                        for size in sizes
                    )
                )
            groups.append(group)
        return tuple(groups)

    def _model_prediction(
        self,
        images: tuple[tuple[torch.Tensor, ...], ...],
        prompts: None,
        geometries: Sequence[GroupGeometry],
        timesteps: GroupTimesteps,
        glyph_conditions: Sequence[Sequence[GlyphCondition | None]] | None = None,
    ) -> tuple[tuple[torch.Tensor, ...], ...]:
        packed_images = pack_image_groups(images, patch_size=self.patch_size)
        model_kwargs = dict(
            packed_images=packed_images, prompts=prompts, geometries=geometries, timesteps=timesteps
        )
        if glyph_conditions is not None:
            model_kwargs["glyph_conditions"] = glyph_conditions
        predictions = self.model(**model_kwargs)
        if len(predictions) != len(images):
            raise ValueError("model prediction group count must match sampling state")
        clean_predictions = []
        for group_id, (image_group, prediction_group) in enumerate(zip(images, predictions)):
            if len(prediction_group) != len(image_group):
                raise ValueError(
                    f"model prediction count for group {group_id} must match sampling images"
                )
            clean_prediction_group = []
            for image_id, (image, prediction) in enumerate(zip(image_group, prediction_group)):
                if not isinstance(prediction, torch.Tensor):
                    raise TypeError(
                        f"model prediction for group {group_id} image {image_id} must be a torch.Tensor"
                    )
                if (
                    prediction.shape != image.shape
                    or prediction.device != image.device
                    or prediction.dtype != image.dtype
                ):
                    raise ValueError(
                        f"model prediction for group {group_id} image {image_id} must match sampling shape, device, and dtype"
                    )
                clean_prediction_group.append(prediction)
            clean_predictions.append(tuple(clean_prediction_group))
        return tuple(clean_predictions)

    def _guided_prediction(
        self,
        images: tuple[tuple[torch.Tensor, ...], ...],
        conditional: None,
        unconditional: None,
        geometries: Sequence[GroupGeometry],
        timesteps: GroupTimesteps,
        time_value: float,
        glyph_conditions: Sequence[Sequence[GlyphCondition | None]] | None = None,
    ) -> tuple[tuple[torch.Tensor, ...], ...]:
        conditional_prediction = self._model_prediction(
            images, conditional, geometries, timesteps, glyph_conditions
        )
        (low, high) = self.cfg_interval
        scale = self.cfg_scale if low < time_value < high else 1.0
        if conditional is None and scale == 1.0:
            return conditional_prediction
        unconditional_glyphs = glyph_conditions
        if conditional is None:
            if glyph_conditions is None:
                raise ValueError("glyph guidance requires conditional glyphs")
            unconditional_glyphs = tuple(((None,) * len(group) for group in glyph_conditions))
        unconditional_prediction = self._model_prediction(
            images, unconditional, geometries, timesteps, unconditional_glyphs
        )
        if scale == 1.0:
            return conditional_prediction
        return tuple(
            (
                tuple(
                    (
                        uncond + scale * (cond - uncond)
                        for (cond, uncond) in zip(cond_group, uncond_group)
                    )
                )
                for (cond_group, uncond_group) in zip(
                    conditional_prediction, unconditional_prediction
                )
            )
        )

    @staticmethod
    def _velocities_from_predictions(
        images: tuple[tuple[torch.Tensor, ...], ...],
        predictions: tuple[tuple[torch.Tensor, ...], ...],
        timesteps: GroupTimesteps,
        *,
        fixed_main: bool = False,
    ) -> tuple[tuple[torch.Tensor, ...], ...]:
        velocities = []
        flat_image_id = 0
        for image_group, prediction_group in zip(images, predictions, strict=True):
            velocity_group = []
            for image_id, (image, prediction) in enumerate(
                zip(image_group, prediction_group, strict=True)
            ):
                if fixed_main and image_id == 0:
                    velocity_group.append(torch.zeros_like(image))
                else:
                    denominator = (1 - timesteps.images[flat_image_id]).to(dtype=image.dtype)
                    velocity_group.append((prediction - image) / denominator)
                flat_image_id += 1
            velocities.append(tuple(velocity_group))
        return tuple(velocities)

    @staticmethod
    def _advance_toward_predictions(
        images: tuple[tuple[torch.Tensor, ...], ...],
        predictions: tuple[tuple[torch.Tensor, ...], ...],
        timesteps: GroupTimesteps,
        next_timesteps: GroupTimesteps,
        *,
        fixed_main: bool = False,
    ) -> tuple[tuple[torch.Tensor, ...], ...]:
        """Apply x_next = x + ((t_next - t) / (1 - t)) * (x0 - x)."""
        advanced_groups = []
        flat_image_id = 0
        for image_group, prediction_group in zip(images, predictions, strict=True):
            advanced_group = []
            for image_id, (image, prediction) in enumerate(
                zip(image_group, prediction_group, strict=True)
            ):
                if fixed_main and image_id == 0:
                    advanced_group.append(image)
                else:
                    t = timesteps.images[flat_image_id]
                    t_next = next_timesteps.images[flat_image_id]
                    fraction = ((t_next - t) / (1 - t)).to(dtype=image.dtype)
                    advanced_group.append(
                        torch.where(
                            t_next == 1, prediction, image + fraction * (prediction - image)
                        )
                    )
                flat_image_id += 1
            advanced_groups.append(tuple(advanced_group))
        return tuple(advanced_groups)

    @staticmethod
    def _advance(
        images: tuple[tuple[torch.Tensor, ...], ...],
        velocities: tuple[tuple[torch.Tensor, ...], ...],
        step_sizes: torch.Tensor,
        *,
        fixed_main: bool = False,
    ) -> tuple[tuple[torch.Tensor, ...], ...]:
        advanced_groups = []
        flat_image_id = 0
        for image_group, velocity_group in zip(images, velocities):
            advanced_group = []
            for image_id, (image, velocity) in enumerate(zip(image_group, velocity_group)):
                step_size = step_sizes[flat_image_id].to(dtype=image.dtype)
                advanced_group.append(
                    image if fixed_main and image_id == 0 else image + step_size * velocity
                )
                flat_image_id += 1
            advanced_groups.append(tuple(advanced_group))
        return tuple(advanced_groups)

    @staticmethod
    def _average_velocities(
        first: tuple[tuple[torch.Tensor, ...], ...], second: tuple[tuple[torch.Tensor, ...], ...]
    ) -> tuple[tuple[torch.Tensor, ...], ...]:
        return tuple(
            (
                tuple(
                    (
                        0.5 * (first_velocity + second_velocity)
                        for (first_velocity, second_velocity) in zip(first_group, second_group)
                    )
                )
                for (first_group, second_group) in zip(first, second)
            )
        )

    @torch.no_grad()
    def forward(
        self,
        conditional: None,
        unconditional: None,
        geometries: Sequence[GroupGeometry],
        *,
        generator: torch.Generator | Sequence[torch.Generator] | None = None,
        main_conditions: Sequence[torch.Tensor] | None = None,
        glyph_conditions: Sequence[Sequence[GlyphCondition | None]] | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> tuple[tuple[torch.Tensor, ...], ...]:
        """Generate all images in every group from independent Gaussian noise."""
        self._validate_inputs(conditional, unconditional, geometries, generator, main_conditions)
        images = self._initial_noise(geometries, conditional, generator, main_conditions)
        timestep_runs = []
        effective_steps = []
        for step_index in range(self.steps):
            if progress_callback is not None:
                progress_callback(step_index, self.steps)
            time_value = step_index / self.steps
            runtime_tensor = main_conditions[0]
            main_timesteps = torch.full(
                (len(geometries),), time_value, device=runtime_tensor.device, dtype=torch.float32
            )
            timesteps = self.timestep_schedule(main_timesteps, geometries)
            next_time_value = (step_index + 1) / self.steps
            next_main_timesteps = torch.full(
                (len(geometries),),
                next_time_value,
                device=runtime_tensor.device,
                dtype=torch.float32,
            )
            next_timesteps = self.timestep_schedule(next_main_timesteps, geometries)
            step_sizes = next_timesteps.images - timesteps.images
            timestep_runs.extend((timesteps, next_timesteps))
            effective_steps.append(step_sizes)
            prediction = self._guided_prediction(
                images,
                conditional,
                unconditional,
                geometries,
                timesteps,
                time_value,
                glyph_conditions,
            )
            if self.solver == "euler" or step_index == self.steps - 1:
                images = self._advance_toward_predictions(
                    images,
                    prediction,
                    timesteps,
                    next_timesteps,
                    fixed_main=self.timestep_schedule.is_native_edit,
                )
                continue
            euler_prediction = self._advance_toward_predictions(
                images,
                prediction,
                timesteps,
                next_timesteps,
                fixed_main=self.timestep_schedule.is_native_edit,
            )
            endpoint_prediction = self._guided_prediction(
                euler_prediction,
                conditional,
                unconditional,
                geometries,
                next_timesteps,
                next_time_value,
                glyph_conditions,
            )
            velocity = self._velocities_from_predictions(
                images, prediction, timesteps, fixed_main=self.timestep_schedule.is_native_edit
            )
            endpoint_velocity = self._velocities_from_predictions(
                euler_prediction,
                endpoint_prediction,
                next_timesteps,
                fixed_main=self.timestep_schedule.is_native_edit,
            )
            images = self._advance(
                images,
                self._average_velocities(velocity, endpoint_velocity),
                step_sizes,
                fixed_main=self.timestep_schedule.is_native_edit,
            )
        if progress_callback is not None:
            progress_callback(self.steps, self.steps)
        self.last_timestep_stats = TimestepRangeStats.from_timesteps(
            timestep_runs, effective_steps=effective_steps
        )
        return images
