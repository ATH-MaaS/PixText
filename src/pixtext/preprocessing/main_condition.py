from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
import torch

from pixtext.preprocessing.geometry import CropGeometry, GroupGeometry, ImageSize, ItemGeometry


@dataclass(frozen=True)
class NativeEditConfig:
    degradation_region: str = "strict_ocr_bbox"
    operators: tuple[str, ...] = ("defocus", "gaussian_blur")
    required_operators: tuple[str, ...] = ()
    min_operators: int = 2
    max_operators: int = 2
    severity_min: float = 0.7
    severity_max: float = 1.0


class NativeEditMainConditioner:
    def __init__(self, config: NativeEditConfig):
        self.config = config

    def __call__(self, clean_main, geometry, rng):
        self._validate_inputs(clean_main, geometry)
        cv2.setNumThreads(1)
        seed = int(torch.randint(0, 2**63 - 1, (), dtype=torch.int64, generator=rng).item())
        local_rng = torch.Generator(device="cpu").manual_seed(seed)
        (operators, severities) = self._sample_operator_plan(local_rng)
        operator_seeds = torch.randint(
            0, 2**31 - 1, (len(operators),), generator=local_rng
        ).tolist()
        original = clean_main.permute(1, 2, 0).contiguous().numpy()
        candidate = original.copy()
        for operator, severity, operator_seed in zip(
            operators, severities, operator_seeds, strict=True
        ):
            candidate = self._apply_operator(candidate, operator, severity, operator_seed)
        footprint = self._replacement_footprint(geometry)
        result = np.where(footprint[:, :, None], candidate, original).astype(np.uint8, copy=False)
        return torch.from_numpy(np.ascontiguousarray(result)).permute(2, 0, 1).contiguous()

    @staticmethod
    def _validate_inputs(clean_main: torch.Tensor, geometry: GroupGeometry) -> None:
        if not isinstance(clean_main, torch.Tensor):
            raise TypeError("clean_main must be a torch.Tensor")
        if not isinstance(geometry, GroupGeometry):
            raise TypeError("geometry must be a GroupGeometry")
        expected_shape = (3, geometry.main.size.height, geometry.main.size.width)
        if clean_main.shape != expected_shape or clean_main.dtype is not torch.uint8:
            raise ValueError(
                f"clean_main must be uint8 RGB matching geometry.main.size, got {tuple(clean_main.shape)} and {clean_main.dtype}"
            )
        if clean_main.device.type != "cpu":
            raise ValueError("native edit CPU conditioning requires a CPU tensor")
        if len(geometry.items) == 0:
            raise ValueError("native edit conditioning requires at least one item")

    def _sample_operator_plan(
        self, rng: torch.Generator
    ) -> tuple[tuple[str, ...], tuple[float, ...]]:
        required = self.config.required_operators
        if required:
            minimum_count = max(self.config.min_operators, len(required))
            operator_count = int(
                torch.randint(
                    minimum_count, self.config.max_operators + 1, (), generator=rng
                ).item()
            )
            optional = tuple(
                operator for operator in self.config.operators if operator not in required
            )
            optional_count = operator_count - len(required)
            optional_permutation = torch.randperm(len(optional), generator=rng)[
                :optional_count
            ].tolist()
            selected = required + tuple(optional[index] for index in optional_permutation)
            order = torch.randperm(len(selected), generator=rng).tolist()
            operators = tuple(selected[index] for index in order)
        else:
            operator_count = int(
                torch.randint(
                    self.config.min_operators, self.config.max_operators + 1, (), generator=rng
                ).item()
            )
            permutation = torch.randperm(len(self.config.operators), generator=rng)[
                :operator_count
            ].tolist()
            operators = tuple(self.config.operators[index] for index in permutation)
        severities = tuple(
            float(value)
            for value in (
                torch.rand(operator_count, generator=rng)
                * (self.config.severity_max - self.config.severity_min)
                + self.config.severity_min
            ).tolist()
        )
        return (operators, severities)

    def _footprint_crop(self, item: ItemGeometry) -> CropGeometry:
        """Resolve the reference-space rectangle this item degrades."""
        if self.config.degradation_region == "aligned_item_crop":
            return item.crop
        bbox = item.reference_bbox
        top = math.floor(bbox.y0)
        left = math.floor(bbox.x0)
        return CropGeometry(
            top=top,
            left=left,
            size=ImageSize(height=math.ceil(bbox.y1) - top, width=math.ceil(bbox.x1) - left),
        )

    @staticmethod
    def _projected_main_box(
        crop: CropGeometry, geometry: GroupGeometry, main_h: int, main_w: int
    ) -> tuple[int, int, int, int]:
        reference_h = geometry.reference.size.height
        reference_w = geometry.reference.size.width
        left = math.floor(crop.left * main_w / reference_w)
        top = math.floor(crop.top * main_h / reference_h)
        right = math.ceil((crop.left + crop.size.width) * main_w / reference_w)
        bottom = math.ceil((crop.top + crop.size.height) * main_h / reference_h)
        return (
            min(max(left, 0), main_w),
            min(max(top, 0), main_h),
            min(max(right, 0), main_w),
            min(max(bottom, 0), main_h),
        )

    def _replacement_footprint(self, geometry: GroupGeometry) -> np.ndarray:
        main_h = geometry.main.size.height
        main_w = geometry.main.size.width
        mask = np.zeros((main_h, main_w), dtype=np.bool_)
        for item in geometry.items:
            (left, top, right, bottom) = self._projected_main_box(
                self._footprint_crop(item), geometry, main_h, main_w
            )
            mask[top:bottom, left:right] = True
        return mask

    @staticmethod
    def _apply_operator(
        image: np.ndarray, operator: str, severity: float, seed: int, scale: float | None = None
    ) -> np.ndarray:
        if operator == "solid_black":
            return np.zeros_like(image)
        (height, width) = image.shape[:2]
        if scale is None:
            scale = min(height, width)
        if operator == "gaussian_blur":
            sigma = scale * (0.005 + 0.025 * severity)
            radius = max(1, math.ceil(3 * sigma))
            kernel = 2 * radius + 1
            return cv2.GaussianBlur(
                image,
                (kernel, kernel),
                sigmaX=sigma,
                sigmaY=sigma,
                borderType=cv2.BORDER_REFLECT_101,
            )
        if operator == "defocus":
            radius = max(1, math.ceil(scale * (0.005 + 0.025 * severity)))
            coordinates = np.arange(-radius, radius + 1, dtype=np.float32)
            (grid_y, grid_x) = np.meshgrid(coordinates, coordinates)
            kernel = (grid_x**2 + grid_y**2 <= radius**2).astype(np.float32)
            kernel /= kernel.sum()
            alias_sigma = 0.1 + 0.4 * severity
            kernel = cv2.GaussianBlur(
                kernel, (3, 3), sigmaX=alias_sigma, borderType=cv2.BORDER_REFLECT_101
            )
            kernel /= kernel.sum()
            return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT_101)
        raise ValueError(f"Unsupported blur operator: {operator}")
