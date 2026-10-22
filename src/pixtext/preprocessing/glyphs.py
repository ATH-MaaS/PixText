"""Canonical per-item glyph rendering and font coverage validation.

The renderer owns the inference glyph seam.  It deliberately owns shaping identity, font
measurement, bbox-local placement, rasterization, and failure behaviour.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import hashlib
import json
import math
import unicodedata
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, features

from pixtext.preprocessing.geometry import BoundingBox, CropGeometry, ItemGeometry

CANONICAL_GLYPH_POLICY = "whole_item_glyph"
CANONICAL_FONT_RESOLUTION_POLICY = "first_font_covering_entire_item"
CANONICAL_FONT_PATH = (
    Path(__file__).resolve().parents[2] / "assets/fonts/canonical/AlibabaPuHuiTi-3-55-Regular.ttf"
)
CANONICAL_FONT_SHA256 = "be33bc8d45ced30fa5c542f6f056565baf85003d4e079522110d76016f8a04e2"
CANONICAL_FALLBACK_FONT_PATH = (
    Path(__file__).resolve().parents[2] / "assets/fonts/canonical/GoNotoCJKCore-v7.0.ttf"
)
CANONICAL_FALLBACK_FONT_SHA256 = "bfe18b47612e58ff28b568f32b55e35fde4505d6137581bf11d2363e257e2ad4"
CANONICAL_LANGUAGE = "zh-CN"
CANONICAL_FEATURES = ("locl",)
CANONICAL_OCCUPANCY = 0.85


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class FontCoverageResult:
    """Observable shaping/coverage verdict for one unchanged Unicode string."""

    text: str
    missing_codepoints: tuple[int, ...]
    notdef_glyph_count: int
    expected_visible_ink: bool
    rendered_ink_nonempty: bool

    @property
    def covered(self) -> bool:
        return (
            not self.missing_codepoints
            and self.notdef_glyph_count == 0
            and (not self.expected_visible_ink or self.rendered_ink_nonempty)
        )


@dataclass(frozen=True, slots=True)
class GlyphFontIdentity:
    index: int
    role: str
    path: str
    sha256: str


@dataclass(frozen=True, slots=True)
class GlyphRendererIdentity:
    policy: str
    font_resolution_policy: str
    font_chain: tuple[GlyphFontIdentity, ...]
    occupancy: float
    rasterization: str
    layout_engine: str
    language: str
    opentype_features: tuple[str, ...]
    pillow_version: str
    freetype_version: str
    raqm_version: str
    polarity: str
    orientation: str

    @property
    def digest(self) -> str:
        payload = json.dumps(
            asdict(self), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class GlyphCondition:
    """The exact coverage tensor consumed by the model plus its provenance."""

    image: torch.Tensor
    text: str
    pre_alignment_bbox: BoundingBox
    aligned_crop: CropGeometry
    renderer_identity: str
    selected_font_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.image, torch.Tensor):
            raise TypeError("glyph image must be a torch.Tensor")
        expected = (1, self.aligned_crop.size.height, self.aligned_crop.size.width)
        if self.image.shape != expected:
            raise ValueError(
                f"glyph image shape {tuple(self.image.shape)} does not match {expected}"
            )
        if not self.image.is_floating_point():
            raise ValueError("glyph image must use a floating-point dtype")
        torch._assert_async(
            torch.isfinite(self.image).all() & self.image.ge(0).all() & self.image.le(1).all(),
            "glyph coverage must be finite and lie in [0, 1]",
        )
        if not isinstance(self.text, str):
            raise ValueError("glyph text must be a string")
        if not self.text:
            if not self.renderer_identity.endswith(":blank-v1"):
                raise ValueError("empty glyph text requires explicit blank conditioning")
            torch._assert_async(self.image.eq(0).all(), "blank glyph must have zero ink")
        if not isinstance(self.renderer_identity, str) or not self.renderer_identity:
            raise ValueError("renderer_identity must be non-empty")
        if (
            isinstance(self.selected_font_index, bool)
            or not isinstance(self.selected_font_index, int)
            or self.selected_font_index < 0
        ):
            raise ValueError("selected_font_index must be a non-negative integer")

    def pin_memory(self) -> GlyphCondition:
        if self.image.device.type != "cpu":
            raise ValueError("only CPU glyph conditions can be pinned")
        return GlyphCondition(
            image=self.image.pin_memory(),
            text=self.text,
            pre_alignment_bbox=self.pre_alignment_bbox,
            aligned_crop=self.aligned_crop,
            renderer_identity=self.renderer_identity,
            selected_font_index=self.selected_font_index,
        )


class _HarfBuzzGlyphInfo(ctypes.Structure):
    _fields_ = (
        ("codepoint", ctypes.c_uint32),
        ("mask", ctypes.c_uint32),
        ("cluster", ctypes.c_uint32),
        ("var1", ctypes.c_uint32),
        ("var2", ctypes.c_uint32),
    )


class FontCoverageChecker:
    """Check cmap and shaped glyph IDs through the renderer's pinned font."""

    def __init__(self, font_path: Path = CANONICAL_FONT_PATH) -> None:
        self.font_path = Path(font_path)
        library_path = ctypes.util.find_library("harfbuzz")
        if library_path is None:
            raise RuntimeError("HarfBuzz is required for canonical glyph coverage")
        self._hb = ctypes.CDLL(library_path)
        self._bind_harfbuzz()
        self._font_bytes = self.font_path.read_bytes()
        self._font_buffer = ctypes.create_string_buffer(self._font_bytes)
        self._blob = self._hb.hb_blob_create(
            self._font_buffer, len(self._font_bytes), 2, None, None
        )
        if not self._blob:
            raise RuntimeError(f"HarfBuzz could not read font: {self.font_path}")
        self._face = self._hb.hb_face_create(self._blob, 0)
        self._font = self._hb.hb_font_create(self._face)
        if not self._face or not self._font:
            self.close()
            raise RuntimeError("HarfBuzz could not create the canonical font")
        self._hb.hb_ot_font_set_funcs(self._font)

    def _bind_harfbuzz(self) -> None:
        hb = self._hb
        hb.hb_blob_create.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        hb.hb_blob_create.restype = ctypes.c_void_p
        hb.hb_blob_destroy.argtypes = [ctypes.c_void_p]
        hb.hb_face_create.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        hb.hb_face_create.restype = ctypes.c_void_p
        hb.hb_face_destroy.argtypes = [ctypes.c_void_p]
        hb.hb_font_create.argtypes = [ctypes.c_void_p]
        hb.hb_font_create.restype = ctypes.c_void_p
        hb.hb_font_destroy.argtypes = [ctypes.c_void_p]
        hb.hb_ot_font_set_funcs.argtypes = [ctypes.c_void_p]
        hb.hb_font_get_nominal_glyph.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
        ]
        hb.hb_font_get_nominal_glyph.restype = ctypes.c_int
        hb.hb_buffer_create.restype = ctypes.c_void_p
        hb.hb_buffer_destroy.argtypes = [ctypes.c_void_p]
        hb.hb_buffer_add_utf8.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_uint,
            ctypes.c_int,
        ]
        hb.hb_buffer_guess_segment_properties.argtypes = [ctypes.c_void_p]
        hb.hb_language_from_string.argtypes = [ctypes.c_char_p, ctypes.c_int]
        hb.hb_language_from_string.restype = ctypes.c_void_p
        hb.hb_buffer_set_language.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        hb.hb_feature_from_string.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p]
        hb.hb_feature_from_string.restype = ctypes.c_int
        hb.hb_shape.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint]
        hb.hb_buffer_get_glyph_infos.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)]
        hb.hb_buffer_get_glyph_infos.restype = ctypes.POINTER(_HarfBuzzGlyphInfo)

    @staticmethod
    def _requires_nominal_glyph(character: str) -> bool:
        codepoint = ord(character)
        category = unicodedata.category(character)
        return not (
            unicodedata.combining(character)
            or category in {"Cf", "Cc", "Cs", "Zl", "Zp"}
            or 65024 <= codepoint <= 65039
            or (917760 <= codepoint <= 917999)
            or (codepoint == 8205)
        )

    @staticmethod
    def _expects_visible_ink(text: str) -> bool:
        return any(
            not character.isspace() and unicodedata.category(character)[0] not in {"C", "Z"}
            for character in text
        )

    @lru_cache(maxsize=65536)
    def check(self, text: str) -> FontCoverageResult:
        if not isinstance(text, str) or not text:
            raise ValueError("coverage text must be a non-empty string")
        missing: list[int] = []
        for character in text:
            if not self._requires_nominal_glyph(character):
                continue
            glyph = ctypes.c_uint32()
            if (
                not self._hb.hb_font_get_nominal_glyph(
                    self._font, ord(character), ctypes.byref(glyph)
                )
                or glyph.value == 0
            ):
                missing.append(ord(character))
        encoded = text.encode("utf-8")
        buffer = self._hb.hb_buffer_create()
        if not buffer:
            raise RuntimeError("HarfBuzz could not allocate a shaping buffer")
        try:
            self._hb.hb_buffer_add_utf8(buffer, encoded, len(encoded), 0, len(encoded))
            self._hb.hb_buffer_guess_segment_properties(buffer)
            language = self._hb.hb_language_from_string(CANONICAL_LANGUAGE.encode(), -1)
            self._hb.hb_buffer_set_language(buffer, language)
            feature = (ctypes.c_uint32 * 4)()
            if not self._hb.hb_feature_from_string(b"locl=1", -1, ctypes.byref(feature)):
                raise RuntimeError("HarfBuzz rejected the pinned locl feature")
            self._hb.hb_shape(self._font, buffer, ctypes.byref(feature), 1)
            length = ctypes.c_uint()
            infos = self._hb.hb_buffer_get_glyph_infos(buffer, ctypes.byref(length))
            notdef_count = sum(infos[index].codepoint == 0 for index in range(length.value))
        finally:
            self._hb.hb_buffer_destroy(buffer)
        expected_ink = self._expects_visible_ink(text)
        rendered_ink = True
        if expected_ink:
            font = ImageFont.truetype(
                str(self.font_path), size=128, layout_engine=ImageFont.Layout.RAQM
            )
            mask = font.getmask(
                text, mode="L", language=CANONICAL_LANGUAGE, features=list(CANONICAL_FEATURES)
            )
            rendered_ink = mask.getbbox() is not None
        return FontCoverageResult(
            text=text,
            missing_codepoints=tuple(sorted(set(missing))),
            notdef_glyph_count=notdef_count,
            expected_visible_ink=expected_ink,
            rendered_ink_nonempty=rendered_ink,
        )

    def close(self) -> None:
        font = getattr(self, "_font", None)
        face = getattr(self, "_face", None)
        blob = getattr(self, "_blob", None)
        if font:
            self._hb.hb_font_destroy(font)
            self._font = None
        if face:
            self._hb.hb_face_destroy(face)
            self._face = None
        if blob:
            self._hb.hb_blob_destroy(blob)
            self._blob = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


@dataclass(frozen=True, slots=True)
class FontResolution:
    """Whole-item result from an ordered, pinned font chain."""

    text: str
    selected_font_index: int | None
    attempts: tuple[FontCoverageResult, ...]

    @property
    def covered(self) -> bool:
        return self.selected_font_index is not None

    @property
    def selected_coverage(self) -> FontCoverageResult | None:
        if self.selected_font_index is None:
            return None
        return self.attempts[self.selected_font_index]


class PinnedFontResolver:
    """Select the first font covering the entire unchanged item string."""

    def __init__(self, font_paths: tuple[Path, ...]) -> None:
        if not font_paths:
            raise ValueError("canonical glyph font chain must not be empty")
        self.checkers = tuple(FontCoverageChecker(path) for path in font_paths)

    @lru_cache(maxsize=65536)
    def resolve(self, text: str) -> FontResolution:
        attempts: list[FontCoverageResult] = []
        for index, checker in enumerate(self.checkers):
            result = checker.check(text)
            attempts.append(result)
            if result.covered:
                return FontResolution(
                    text=text, selected_font_index=index, attempts=tuple(attempts)
                )
        return FontResolution(text=text, selected_font_index=None, attempts=tuple(attempts))

    def check(self, text: str) -> FontResolution:
        """Compatibility alias for callers interested only in chain coverage."""
        return self.resolve(text)

    def close(self) -> None:
        for checker in self.checkers:
            checker.close()


class CanonicalItemGlyphRenderer:
    """Render upright, bbox-centered 1x grayscale canonical glyphs."""

    def __init__(
        self,
        *,
        font_path: Path = CANONICAL_FONT_PATH,
        expected_font_sha256: str = CANONICAL_FONT_SHA256,
        fallback_font_path: Path = CANONICAL_FALLBACK_FONT_PATH,
        expected_fallback_font_sha256: str = CANONICAL_FALLBACK_FONT_SHA256,
        font_resolution_policy: str = CANONICAL_FONT_RESOLUTION_POLICY,
        occupancy: float = CANONICAL_OCCUPANCY,
    ) -> None:
        self.font_paths = (Path(font_path), Path(fallback_font_path))
        expected_sha256s = (expected_font_sha256, expected_fallback_font_sha256)
        actual_sha256s: list[str] = []
        for index, (path, expected_sha256) in enumerate(
            zip(self.font_paths, expected_sha256s, strict=True)
        ):
            if not path.is_file():
                raise ValueError(f"canonical font {index} is not readable: {path}")
            actual_sha256 = _sha256_file(path)
            if actual_sha256 != expected_sha256:
                raise ValueError(
                    f"canonical font {index} SHA-256 mismatch: got {actual_sha256}, expected {expected_sha256}"
                )
            actual_sha256s.append(actual_sha256)
        if font_resolution_policy != CANONICAL_FONT_RESOLUTION_POLICY:
            raise ValueError("unsupported canonical font resolution policy")
        if not math.isfinite(occupancy) or not 0 < occupancy <= 1:
            raise ValueError("glyph occupancy must lie in (0, 1]")
        if not features.check_feature("raqm"):
            raise RuntimeError("canonical glyph rendering requires Pillow RAQM")
        self.occupancy = float(occupancy)
        self.coverage = PinnedFontResolver(self.font_paths)
        import PIL

        self.identity = GlyphRendererIdentity(
            policy=CANONICAL_GLYPH_POLICY,
            font_resolution_policy=font_resolution_policy,
            font_chain=tuple(
                GlyphFontIdentity(
                    index=index, role=role, path=str(path.resolve()), sha256=actual_sha256s[index]
                )
                for (index, (role, path)) in enumerate(
                    zip(("primary", "fallback"), self.font_paths, strict=True)
                )
            ),
            occupancy=self.occupancy,
            rasterization="direct_1x_grayscale",
            layout_engine="RAQM",
            language=CANONICAL_LANGUAGE,
            opentype_features=CANONICAL_FEATURES,
            pillow_version=PIL.__version__,
            freetype_version=features.version_module("freetype2") or "unknown",
            raqm_version=features.version_feature("raqm") or "unknown",
            polarity="black_0_white_1",
            orientation="horizontal_upright_uniform_scale",
        )

    @lru_cache(maxsize=1024)
    def _font(self, font_index: int, size: int) -> ImageFont.FreeTypeFont:
        return ImageFont.truetype(
            str(self.font_paths[font_index]), size=max(1, size), layout_engine=ImageFont.Layout.RAQM
        )

    @staticmethod
    def _ink_bbox(font: ImageFont.FreeTypeFont, text: str) -> tuple[int, int, int, int]:
        (mask, offset) = font.getmask2(
            text, mode="L", language=CANONICAL_LANGUAGE, features=list(CANONICAL_FEATURES)
        )
        mask_bbox = mask.getbbox()
        if mask_bbox is None:
            raise ValueError(f"canonical glyph has an empty ink bbox: {text!r}")
        return (
            offset[0] + mask_bbox[0],
            offset[1] + mask_bbox[1],
            offset[0] + mask_bbox[2],
            offset[1] + mask_bbox[3],
        )

    def render_blank(self, item_geometry: ItemGeometry) -> GlyphCondition:
        """Present, zero-ink condition for erasure; distinct from absent CFG glyphs."""
        crop = item_geometry.crop
        return GlyphCondition(
            image=torch.zeros((1, crop.size.height, crop.size.width)),
            text="",
            pre_alignment_bbox=item_geometry.reference_bbox,
            aligned_crop=crop,
            renderer_identity=self.identity.digest + ":blank-v1",
            selected_font_index=0,
        )

    def render(self, text: str, item_geometry: ItemGeometry) -> GlyphCondition:
        if not isinstance(text, str) or not text:
            raise ValueError("canonical glyph text must be non-empty")
        if not isinstance(item_geometry, ItemGeometry):
            raise TypeError("item_geometry must be an ItemGeometry")
        resolution = self.coverage.resolve(text)
        if not resolution.covered:
            attempts = tuple(
                {
                    "font_index": index,
                    "missing": tuple(f"U+{value:04X}" for value in result.missing_codepoints),
                    "notdef": result.notdef_glyph_count,
                    "nonempty": result.rendered_ink_nonempty,
                }
                for (index, result) in enumerate(resolution.attempts)
            )
            raise ValueError(f"canonical font chain coverage failed for {text!r}: {attempts}")
        font_index = resolution.selected_font_index
        assert font_index is not None
        coverage = resolution.selected_coverage
        assert coverage is not None
        crop = item_geometry.crop
        bbox = item_geometry.reference_bbox
        local_x0 = float(bbox.x0) - crop.left
        local_y0 = float(bbox.y0) - crop.top
        local_x1 = float(bbox.x1) - crop.left
        local_y1 = float(bbox.y1) - crop.top
        target_width = (local_x1 - local_x0) * self.occupancy
        target_height = (local_y1 - local_y0) * self.occupancy
        if target_width <= 0 or target_height <= 0:
            raise ValueError("canonical glyph bbox is degenerate after localization")
        probe_size = 128
        probe_bbox = self._ink_bbox(self._font(font_index, probe_size), text)
        probe_width = probe_bbox[2] - probe_bbox[0]
        probe_height = probe_bbox[3] - probe_bbox[1]
        fitted_size = max(
            1,
            math.floor(probe_size * min(target_width / probe_width, target_height / probe_height)),
        )
        while fitted_size > 1:
            font = self._font(font_index, fitted_size)
            (left, top, right, bottom) = self._ink_bbox(font, text)
            if right - left <= target_width and bottom - top <= target_height:
                break
            fitted_size -= 1
        else:
            font = self._font(font_index, 1)
            (left, top, right, bottom) = self._ink_bbox(font, text)
        if right - left > target_width or bottom - top > target_height:
            raise ValueError(
                f"canonical glyph cannot fit the bbox occupancy at the minimum font size: text={text!r}, target={target_width}x{target_height}"
            )
        while True:
            next_font = self._font(font_index, fitted_size + 1)
            next_bbox = self._ink_bbox(next_font, text)
            if (
                next_bbox[2] - next_bbox[0] > target_width
                or next_bbox[3] - next_bbox[1] > target_height
            ):
                break
            fitted_size += 1
            font = next_font
            (left, top, right, bottom) = next_bbox
        center_x = (local_x0 + local_x1) / 2
        center_y = (local_y0 + local_y1) / 2
        draw_x = center_x - (left + right) / 2
        draw_y = center_y - (top + bottom) / 2
        canvas = Image.new("L", (crop.size.width, crop.size.height), 0)
        ImageDraw.Draw(canvas).text(
            (draw_x, draw_y),
            text,
            font=font,
            fill=255,
            language=CANONICAL_LANGUAGE,
            features=list(CANONICAL_FEATURES),
        )
        array = np.asarray(canvas, dtype=np.float32)
        image = torch.from_numpy(array.copy()).unsqueeze(0).div_(255.0)
        if coverage.expected_visible_ink and (not bool(image.gt(0).any())):
            raise ValueError(f"canonical glyph rasterized to empty ink: {text!r}")
        return GlyphCondition(
            image=image.contiguous(),
            text=text,
            pre_alignment_bbox=bbox,
            aligned_crop=crop,
            renderer_identity=self.identity.digest,
            selected_font_index=font_index,
        )
