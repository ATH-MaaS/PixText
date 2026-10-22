"""V3 nodes: package loading, visual region editing, and native-resolution repair."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import folder_paths
import numpy as np
import torch
from aiohttp import web
from comfy import model_management
from comfy.model_patcher import ModelPatcher
from comfy.utils import ProgressBar
from comfy_api.latest import ComfyExtension, io, ui
from comfy_execution.graph_utils import ExecutionBlocker
from PIL import Image, ImageDraw, ImageOps
from server import PromptServer

from pixtext import PixTextPipeline

from .comparison import compare_regions
from .ocr import detect_words, model_identity
from .regions import parse_regions

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = "example: Chinese New Year"
ENGLISH_EXAMPLE = "example: English caption"
DETECTIONS = io.Custom("PIXTEXT_DETECTIONS")
MODEL = io.Custom("PIXTEXT_MODEL")
TARGETS = io.Custom("PIXTEXT_TARGETS")
COLORS = ("#5eead4", "#fbbf24", "#a5b4fc", "#fb7185", "#7dd3fc", "#d8b4fe")
folder_paths.add_model_folder_path("pixtext", str(Path(folder_paths.models_dir) / "pixtext"))


def packages() -> dict[str, Path]:
    found = {}
    for directory in folder_paths.get_folder_paths("pixtext"):
        root = Path(directory)
        if root.is_dir():
            for manifest in sorted(root.glob("*/manifest.json")):
                if (manifest.parent / "model.safetensors").is_file():
                    found.setdefault(manifest.parent.name, manifest.parent)
    return found


def file_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_image_path(name: str) -> Path:
    path = Path(folder_paths.get_annotated_filepath(name)).resolve()
    if not path.is_relative_to(Path(folder_paths.get_input_directory()).resolve()):
        raise ValueError("Select an image from ComfyUI's input directory")
    return path


def tensor_image(value: torch.Tensor) -> Image.Image:
    if value.ndim != 4 or value.shape[0] != 1 or value.shape[-1] not in (3, 4):
        raise ValueError("PixText Region Editor expects one RGB/RGBA image, not an image batch")
    pixels = value[0, ..., :3].detach().float().cpu().clamp(0, 1).mul(255).round()
    return Image.fromarray(pixels.to(torch.uint8).numpy())


def image_tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image, copy=True)).float().div(255).unsqueeze(0)


def annotated(image: Image.Image, edits) -> Image.Image:
    result = image.copy()
    draw = ImageDraw.Draw(result)
    line = max(2, round(min(result.size) / 400))
    for index, edit in enumerate(edits):
        color = COLORS[index % len(COLORS)]
        draw.rectangle(edit.bbox, outline=color, width=line)
        x, y = edit.bbox[:2]
        draw.rectangle((x, max(0, y - 15), x + 23, max(0, y - 15) + 15), fill=color)
        draw.text((x + 3, max(0, y - 15)), str(index + 1), fill="#101820")
    return result


@dataclass
class ModelHandle:
    path: Path
    dtype: torch.dtype
    pipeline: PixTextPipeline | None = None
    patcher: ModelPatcher | None = None

    def load(self) -> PixTextPipeline:
        device = model_management.get_torch_device()
        if device.type != "cuda":
            raise RuntimeError("PixText requires an NVIDIA CUDA GPU with FlexAttention support")
        if self.pipeline is None:
            self.pipeline = PixTextPipeline.from_pretrained(
                self.path, device="cpu", dtype=self.dtype
            )
            self.patcher = ModelPatcher(self.pipeline.model, device, torch.device("cpu"))
        # Native PixText modules do not implement ComfyUI's partial weight casting.
        model_management.load_models_gpu([self.patcher], force_full_load=True)
        self.pipeline.device = device
        return self.pipeline


class PixTextModelLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PixTextModelLoader",
            display_name="Load PixText Model",
            category="PixText",
            description="Load a complete MeanFlow or Teacher package from models/pixtext.",
            inputs=[
                io.Combo.Input(
                    "model_name",
                    options=list(packages()) or ["(install a model package)"],
                    tooltip="Complete package folder, including configuration and fonts.",
                ),
                io.Combo.Input(
                    "precision", options=["bfloat16", "float32"], default="bfloat16", advanced=True
                ),
            ],
            outputs=[MODEL.Output("model")],
        )

    @classmethod
    def execute(cls, model_name, precision):
        if model_name not in packages():
            raise ValueError(
                "Place a complete PixText package in ComfyUI/models/pixtext and refresh."
            )
        dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[precision]
        return io.NodeOutput(ModelHandle(packages()[model_name], dtype))

    @classmethod
    def fingerprint_inputs(cls, model_name, precision):
        path = packages().get(model_name)
        if path is None:
            return model_name
        return tuple(
            (p.name, p.stat().st_size, p.stat().st_mtime_ns)
            for p in sorted(path.rglob("*"))
            if p.is_file()
        )


class PixTextAutoDetect(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PixTextAutoDetect",
            display_name="Detect Words · PixText",
            category="PixText",
            description="Local CRAFT word boxes + PP-OCRv6 recognition. Run once, review suggestions in the Region Editor, then apply.",
            inputs=[
                io.Combo.Input(
                    "image",
                    options=[ENGLISH_EXAMPLE]
                    + sorted(
                        str(p.relative_to(folder_paths.get_input_directory()))
                        for p in Path(folder_paths.get_input_directory()).rglob("*")
                        if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
                    ),
                    default=ENGLISH_EXAMPLE,
                    upload=io.UploadType.image,
                ),
                io.Float.Input(
                    "text_threshold", default=0.5, min=0.1, max=0.95, step=0.05, advanced=True
                ),
                io.Float.Input(
                    "link_threshold", default=0.2, min=0.1, max=0.95, step=0.05, advanced=True
                ),
                io.Image.Input("image_in", optional=True),
            ],
            outputs=[
                io.Image.Output("image"),
                DETECTIONS.Output("detections"),
                io.Image.Output("words_preview"),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, image, text_threshold, link_threshold, image_in=None):
        if image_in is not None:
            source = tensor_image(image_in)
        else:
            path = (
                ROOT / "examples/english_case/input.png"
                if image == ENGLISH_EXAMPLE
                else input_image_path(image)
            )
            with Image.open(path) as opened:
                source = ImageOps.exif_transpose(opened).convert("RGB")
        document = detect_words(
            source, Path(folder_paths.models_dir) / "pixtext_ocr", text_threshold, link_threshold
        )
        from pixtext import TextEdit

        overlay = image_tensor(
            annotated(
                source,
                [TextEdit(bbox=row["bbox"], text=row["text"]) for row in document["regions"]],
            )
        )
        return io.NodeOutput(
            image_tensor(source), document, overlay, ui=ui.PreviewImage(overlay, cls=cls)
        )

    @classmethod
    def fingerprint_inputs(cls, image, text_threshold, link_threshold, image_in=None):
        identity = model_identity(Path(folder_paths.models_dir) / "pixtext_ocr")
        if image_in is not None:
            return identity
        path = (
            ROOT / "examples/english_case/input.png"
            if image == ENGLISH_EXAMPLE
            else input_image_path(image)
        )
        return identity, file_fingerprint(path)


class PixTextRegionEditor(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        files = folder_paths.filter_files_content_types(
            folder_paths.get_filename_list("input")
            if "input" in folder_paths.folder_names_and_paths
            else [
                str(p.relative_to(folder_paths.get_input_directory()))
                for p in Path(folder_paths.get_input_directory()).rglob("*")
                if p.is_file()
            ],
            ["image"],
        )
        return io.Schema(
            node_id="PixTextRegionEditor",
            display_name="PixText Region Editor",
            category="PixText",
            description="Upload an image, draw rectangles and enter target text. Regions persist in the workflow.",
            inputs=[
                io.Combo.Input(
                    "image",
                    options=[EXAMPLE] + sorted(files),
                    default=EXAMPLE,
                    upload=io.UploadType.image,
                ),
                io.String.Input("region_data", default="{}", multiline=True, dynamic_prompts=False),
                io.Image.Input(
                    "image_in",
                    optional=True,
                    tooltip="Optional upstream image. Run once to refresh its editor preview.",
                ),
                DETECTIONS.Input(
                    "detections",
                    optional=True,
                    tooltip="Connect detections and image from Detect Words. Review and apply before refinement.",
                ),
            ],
            outputs=[
                io.Image.Output("image"),
                TARGETS.Output("targets"),
                io.Image.Output("regions_preview"),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, image, region_data, image_in=None, detections=None):
        if image_in is not None:
            source = tensor_image(image_in)
            source_id = "pixels:" + hashlib.sha256(source.tobytes()).hexdigest()
        else:
            path = (
                ROOT / "examples/pipeline_case/input.png"
                if image == EXAMPLE
                else input_image_path(image)
            )
            if image != EXAMPLE and not folder_paths.exists_annotated_filepath(image):
                raise ValueError("Upload an image before editing regions")
            with Image.open(path) as opened:
                source = ImageOps.exif_transpose(opened).convert("RGB")
            source_id = "example:chinese" if image == EXAMPLE else "file:" + image
        pixels = image_tensor(source)
        source_ui = ui.PreviewImage(pixels, cls=cls).as_dict()
        ui_data = {
            "source_image": source_ui["images"],
            "source_id": [source_id],
            "source_size": [list(source.size)],
        }
        if detections is not None:
            if (
                image_in is None
                or detections["source_id"] != source_id
                or detections["size"] != list(source.size)
            ):
                raise ValueError("Connect the image and detections from the same Detect Words node")
            ui_data["ocr_suggestions"] = [detections]
        try:
            edits = parse_regions(region_data, *source.size, source_id)
            if (
                detections is not None
                and json.loads(region_data).get("detection_id") != detections["detection_id"]
            ):
                raise ValueError(
                    "Review OCR suggestions in the Region Editor and Apply regions before refinement."
                )
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            ui_data["region_message"] = [
                f"{len(detections['regions'])} words detected. Open Region Editor, review targets and Apply regions."
                if detections is not None and region_data.strip() == "{}"
                else str(exc)
            ]
            return io.NodeOutput(pixels, ExecutionBlocker(None), pixels, ui=ui_data)
        overlay = image_tensor(annotated(source, edits))
        ui_data.update(ui.PreviewImage(overlay, cls=cls).as_dict())
        target_data = {
            "edits": edits,
            "size": source.size,
            "source_digest": hashlib.sha256(source.tobytes()).hexdigest(),
        }
        return io.NodeOutput(pixels, target_data, overlay, ui=ui_data)

    @classmethod
    def fingerprint_inputs(cls, image, region_data, image_in=None, detections=None):
        if image_in is not None:
            return None  # Upstream IMAGE dependency is already part of ComfyUI's cache key.
        path = (
            ROOT / "examples/pipeline_case/input.png"
            if image == EXAMPLE
            else input_image_path(image)
        )
        return file_fingerprint(path) if path.is_file() else "missing"


class PixTextRefine(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PixTextRefine",
            display_name="Refine Text · PixText",
            category="PixText",
            description="Repair enabled regions at the original resolution. More than 12 targets are grouped automatically.",
            inputs=[
                MODEL.Input("model"),
                io.Image.Input("image"),
                TARGETS.Input("targets"),
                io.Int.Input("seed", default=42, min=0, max=2**63 - 1, control_after_generate=True),
                io.Int.Input(
                    "steps",
                    default=4,
                    min=1,
                    max=100,
                    advanced=True,
                    tooltip="Actual sampling steps. Recommended: MeanFlow 4, Teacher 25.",
                ),
                io.Float.Input("guidance", default=3.3, min=0, max=20, step=0.1, advanced=True),
            ],
            outputs=[
                io.Image.Output("image"),
                io.Mask.Output("edit_footprint"),
                io.String.Output("metadata"),
            ],
        )

    @classmethod
    def execute(cls, model, image, targets, seed, steps, guidance):
        source = tensor_image(image)
        if tuple(targets["size"]) != source.size:
            raise ValueError("Image and target regions have different dimensions")
        if targets["source_digest"] != hashlib.sha256(source.tobytes()).hexdigest():
            raise ValueError("Connect the image and targets from the same PixText Region Editor")
        edits = targets["edits"]
        if not edits:
            return io.NodeOutput(
                image,
                torch.zeros((1, source.height, source.width)),
                json.dumps({"status": "no_enabled_regions"}),
            )
        pipeline = model.load()
        effective_steps = steps
        groups = math.ceil(len(edits) / pipeline.inference["sampling"]["max_items_per_pass"])
        progress = ProgressBar(effective_steps * groups)

        def callback(done, total):
            model_management.throw_exception_if_processing_interrupted()
            progress.update_absolute(done, total)

        result = pipeline.edit(
            source,
            edits,
            seed=seed,
            steps=steps,
            guidance_scale=guidance,
            progress_callback=callback,
        )
        footprint = torch.zeros((1, source.height, source.width))
        for group in result.metadata["geometry"]:
            rw, rh = group["reference_size"]
            for x, y, w, h in group["crops"]:
                x0, y0 = math.floor(x * source.width / rw), math.floor(y * source.height / rh)
                x1 = min(source.width, math.ceil((x + w) * source.width / rw))
                y1 = min(source.height, math.ceil((y + h) * source.height / rh))
                footprint[:, y0:y1, x0:x1] = 1
        output = image_tensor(result.image)
        return io.NodeOutput(
            output,
            footprint,
            json.dumps(result.metadata, ensure_ascii=False),
            ui=ui.PreviewImage(output, cls=cls),
        )


class PixTextCompare(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PixTextCompare",
            display_name="Compare Regions · PixText",
            category="PixText",
            description="Compare every enabled target before and after repair, with matching crops and target labels.",
            inputs=[
                io.Image.Input("before"),
                io.Image.Input("after"),
                TARGETS.Input("targets"),
                io.Int.Input(
                    "padding",
                    default=6,
                    min=0,
                    max=64,
                    advanced=True,
                    tooltip="Context pixels around each target in both crops.",
                ),
            ],
            outputs=[io.Image.Output("contact_sheet")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, before, after, targets, padding):
        original, repaired = tensor_image(before), tensor_image(after)
        if tuple(targets["size"]) != original.size:
            raise ValueError("Target regions and input image have different dimensions")
        if targets["source_digest"] != hashlib.sha256(original.tobytes()).hexdigest():
            raise ValueError("Use the original image from the same PixText Region Editor")
        sheet, pairs = compare_regions(original, repaired, targets["edits"], padding=padding)
        cards = []
        for index, (edit, (left, right)) in enumerate(zip(targets["edits"], pairs, strict=True)):
            cards.append(
                {
                    "index": index + 1,
                    "text": edit.text,
                    "before": ui.PreviewImage(image_tensor(left), cls=cls).as_dict()["images"][0],
                    "after": ui.PreviewImage(image_tensor(right), cls=cls).as_dict()["images"][0],
                }
            )
        return io.NodeOutput(image_tensor(sheet), ui={"pixtext_comparison": [{"cards": cards}]})


class PixTextExtension(ComfyExtension):
    async def get_node_list(self):
        return [
            PixTextModelLoader,
            PixTextAutoDetect,
            PixTextRegionEditor,
            PixTextRefine,
            PixTextCompare,
        ]


@PromptServer.instance.routes.get("/pixtext/example/chinese")
async def chinese_example(request):
    return web.FileResponse(ROOT / "examples/pipeline_case/input.png")


@PromptServer.instance.routes.get("/pixtext/example/chinese/regions")
async def chinese_regions(request):
    with Image.open(ROOT / "examples/pipeline_case/input.png") as image:
        size = list(image.size)
    rows = json.loads((ROOT / "examples/pipeline_case/edits.json").read_text())
    return web.json_response(
        {
            "version": 1,
            "source_id": "example:chinese",
            "size": size,
            "regions": [dict(row, id=str(i + 1), enabled=True) for i, row in enumerate(rows)],
        }
    )
