"""Local text editing behind one interface shared by Python and the CLI."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps
from safetensors.torch import load_file

from pixtext.loading import package_file, read_package
from pixtext.modeling.transformer import GroupJiT
from pixtext.preprocessing.compositing import compose_native_edit_items
from pixtext.preprocessing.geometry import BoundingBox, ImageSize, plan_group_geometry
from pixtext.preprocessing.glyphs import CanonicalItemGlyphRenderer
from pixtext.preprocessing.images import prepare_group_images
from pixtext.preprocessing.main_condition import NativeEditConfig, NativeEditMainConditioner
from pixtext.sampling.flow import GroupFlowSampler
from pixtext.sampling.meanflow import GroupMeanFlowSampler
from pixtext.sampling.timesteps import GroupTimestepSchedule
from pixtext.types import EditResult, TextEdit


def _stable_seed(seed: int, identity: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}\0{identity}".encode()).digest()[:8], "big") & (
        2**63 - 1
    )


def _normalized(image: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return (
        image.to(device=device, dtype=torch.float32).div(127.5).sub(1).to(dtype=dtype).contiguous()
    )


def _uint8(image: torch.Tensor) -> torch.Tensor:
    return (
        image.detach()
        .float()
        .add(1)
        .mul(127.5)
        .round()
        .clamp(0, 255)
        .to(device="cpu", dtype=torch.uint8)
        .contiguous()
    )


class PixTextPipeline:
    """Load one model package and edit original-image regions with target text."""

    def __init__(
        self,
        model: GroupJiT,
        renderer: CanonicalItemGlyphRenderer,
        inference: dict[str, Any],
        *,
        variant: str,
        device: torch.device,
        dtype: torch.dtype,
        package_id: str,
    ):
        self.model = model.eval().requires_grad_(False)
        self.renderer = renderer
        self.inference = inference
        self.variant = variant
        self.device = device
        self.dtype = dtype
        self.package_id = package_id
        self.conditioner = NativeEditMainConditioner(
            NativeEditConfig(**inference["main_condition"])
        )

    @classmethod
    def from_pretrained(
        cls, path: str | Path, *, device: str = "cuda", dtype: torch.dtype = torch.bfloat16
    ) -> PixTextPipeline:
        root, config, inference, manifest = read_package(path)
        glyph = inference["glyph"]
        renderer = CanonicalItemGlyphRenderer(
            font_path=package_file(root, glyph["font_path"]),
            expected_font_sha256=glyph["font_sha256"],
            fallback_font_path=package_file(root, glyph["fallback_font_path"]),
            expected_fallback_font_sha256=glyph["fallback_font_sha256"],
            occupancy=glyph["occupancy"],
        )
        target = torch.device(device)
        if target.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; PixText editing requires a CUDA GPU")
        if dtype not in (torch.bfloat16, torch.float32):
            raise ValueError("dtype must be torch.bfloat16 or torch.float32")
        with torch.device("meta"):
            model = GroupJiT(**config["model"])
        model.load_state_dict(
            load_file(str(root / "model.safetensors"), device="cpu"), strict=True, assign=True
        )
        model = model.to(device=target, dtype=dtype)
        return cls(
            model,
            renderer,
            inference,
            variant=config["variant"],
            device=target,
            dtype=dtype,
            package_id=manifest["model_id"],
        )

    def edit(
        self,
        image: Image.Image | str | Path,
        edits: Sequence[TextEdit],
        *,
        seed: int = 42,
        steps: int | None = None,
        guidance_scale: float | None = None,
        sample_id: str = "image",
    ) -> EditResult:
        return self.edit_batch(
            [image],
            [edits],
            seeds=[seed],
            steps=steps,
            guidance_scale=guidance_scale,
            sample_ids=[sample_id],
        )[0]

    @torch.inference_mode()
    def edit_batch(
        self,
        images: Sequence[Image.Image | str | Path],
        edits: Sequence[Sequence[TextEdit]],
        *,
        seeds: Sequence[int] | None = None,
        steps: int | None = None,
        guidance_scale: float | None = None,
        sample_ids: Sequence[str] | None = None,
    ) -> list[EditResult]:
        if not images or len(images) != len(edits):
            raise ValueError("A batch requires matching nonempty image and edit lists")
        seeds = list(seeds) if seeds is not None else [42] * len(images)
        sample_ids = list(sample_ids) if sample_ids is not None else ["image"] * len(images)
        if len(seeds) != len(images) or len(sample_ids) != len(images):
            raise ValueError("Need one seed and sample_id per image")
        if any(isinstance(s, bool) or not isinstance(s, int) or not 0 <= s < 2**63 for s in seeds):
            raise ValueError("Seeds must be integers within [0, 2**63)")
        if any(not isinstance(s, str) or not s for s in sample_ids):
            raise ValueError("sample_ids must be nonempty strings")
        sources = [self._image(image) for image in images]
        for source, targets in zip(sources, edits, strict=True):
            for target in targets:
                if not isinstance(target, TextEdit):
                    raise TypeError("edits must contain TextEdit objects")
                if target.bbox[2] > source.width or target.bbox[3] > source.height:
                    raise ValueError("Edit bbox lies outside the original image")
        settings = self.inference["sampling"]
        actual_steps = settings["steps"] if steps is None else steps
        cfg = settings["guidance_scale"] if guidance_scale is None else guidance_scale
        sampler_type = GroupMeanFlowSampler if self.variant == "meanflow" else GroupFlowSampler
        sampler = sampler_type(
            self.model,
            steps=actual_steps,
            cfg_scale=cfg,
            cfg_interval=tuple(settings["guidance_interval"]),
            noise_scale=settings["noise_scale"],
            timestep_schedule=GroupTimestepSchedule(self.model.patch_size),
            compute_dtype=self.dtype,
            solver="euler",
        ).eval()
        original = [
            torch.from_numpy(np.array(source).copy()).permute(2, 0, 1) for source in sources
        ]
        results = [x.clone() for x in original]
        geometries_by_image: list[list[dict[str, Any]]] = [[] for _ in sources]
        max_items = settings["max_items_per_pass"]
        for offset in range(0, max(map(len, edits)), max_items):
            active, geometries, mains, glyphs, generators = [], [], [], [], []
            for index, (source, targets) in enumerate(zip(sources, edits, strict=True)):
                current = targets[offset : offset + max_items]
                if not current:
                    continue
                geometry = plan_group_geometry(
                    ImageSize(height=source.height, width=source.width),
                    tuple(BoundingBox(*edit.bbox) for edit in current),
                    patch_size=self.model.patch_size,
                    main_target_area=self.inference["geometry"]["main_target_area"],
                    clamp_main_to_reference=True,
                )
                run_seed = _stable_seed(seeds[index], f"{sample_ids[index]}:{offset}")
                prepared = prepare_group_images(original[index].float(), geometry)
                main = prepared.main.round().clamp(0, 255).to(torch.uint8).contiguous()
                conditioned = self.conditioner(
                    main, geometry, torch.Generator(device="cpu").manual_seed(run_seed)
                )
                active.append(index)
                geometries.append(geometry)
                mains.append(_normalized(conditioned, self.device, self.dtype))
                glyphs.append(
                    tuple(
                        self.renderer.render(edit.text, item)
                        for edit, item in zip(current, geometry.items, strict=True)
                    )
                )
                generators.append(torch.Generator(device=self.device).manual_seed(run_seed))
                geometries_by_image[index].append(
                    {
                        "offset": offset,
                        "selected_font_indices": [g.selected_font_index for g in glyphs[-1]],
                        "reference_size": [
                            geometry.reference.size.width,
                            geometry.reference.size.height,
                        ],
                        "crops": [
                            [
                                item.crop.left,
                                item.crop.top,
                                item.crop.size.width,
                                item.crop.size.height,
                            ]
                            for item in geometry.items
                        ],
                    }
                )
            if self.device.type != "cuda":
                raise RuntimeError(
                    "PixText editing requires CUDA FlexAttention; CPU supports package loading only"
                )
            generated = sampler(
                None,
                None,
                tuple(geometries),
                generator=tuple(generators),
                main_conditions=tuple(mains),
                glyph_conditions=tuple(glyphs),
            )
            for index, geometry, group in zip(active, geometries, generated, strict=True):
                # Compose at the reference resolution; restore only the edited footprints.
                # Keeping the original canvas avoids resampling unedited pixels.
                reference = prepare_group_images(results[index].float(), geometry).reference
                composite = _uint8(
                    compose_native_edit_items(
                        _normalized(reference, self.device, self.dtype).float(), group[1:], geometry
                    )
                )
                restored = Image.fromarray(composite.permute(1, 2, 0).numpy()).resize(
                    sources[index].size, Image.Resampling.LANCZOS
                )
                plane = torch.from_numpy(np.array(restored).copy()).permute(2, 0, 1)

                for item in geometry.items:
                    x0 = math.floor(item.crop.left / geometry.reference.scale_w)
                    y0 = math.floor(item.crop.top / geometry.reference.scale_h)
                    x1 = min(
                        sources[index].width,
                        math.ceil(
                            (item.crop.left + item.crop.size.width) / geometry.reference.scale_w
                        ),
                    )
                    y1 = min(
                        sources[index].height,
                        math.ceil(
                            (item.crop.top + item.crop.size.height) / geometry.reference.scale_h
                        ),
                    )
                    results[index][:, y0:y1, x0:x1] = plane[:, y0:y1, x0:x1]
        return [
            EditResult(
                Image.fromarray(result.permute(1, 2, 0).numpy()),
                {
                    "model_id": self.package_id,
                    "variant": self.variant,
                    "seed": seed,
                    "sample_id": identity,
                    "steps": actual_steps,
                    "guidance_scale": cfg,
                    "guidance_interval": settings["guidance_interval"],
                    "precision": str(self.dtype),
                    "geometry": geometry,
                },
            )
            for result, seed, identity, geometry in zip(
                results, seeds, sample_ids, geometries_by_image, strict=True
            )
        ]

    @staticmethod
    def _image(image: Image.Image | str | Path) -> Image.Image:
        if isinstance(image, (str, Path)):
            with Image.open(image) as opened:
                return ImageOps.exif_transpose(opened).convert("RGB")
        if isinstance(image, Image.Image):
            return ImageOps.exif_transpose(image).convert("RGB")
        raise TypeError("image must be a Pillow image or an image path")
