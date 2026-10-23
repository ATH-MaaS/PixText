# PixText: Efficient Scene-Text Refinement via Pixel-Space Grouped Image Generation

[![arXiv](https://img.shields.io/badge/arXiv-2610.18284-b31b1b.svg)](https://arxiv.org/abs/2610.18284)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Model weights](https://img.shields.io/badge/🤗%20Hugging%20Face-Model%20Weights-yellow)](https://huggingface.co/ATH-MaaS/PixText)
[![Benchmark](https://img.shields.io/badge/🤗%20Hugging%20Face-TextRefinementBench-yellow)](https://huggingface.co/datasets/ATH-MaaS/TextRefinementBench)

[![ComfyUI](https://img.shields.io/badge/ComfyUI-Registry-2D8C83)](https://registry.comfy.org/publishers/yaoliliu/nodes/pixtext)

![PixText pipeline](docs/assets/pipeline.png)

**English** | [简体中文](README.zh-CN.md)

PixText refines local text in pixel space. Provide an image, the regions to edit, and the target text. The model generates those regions using the image as context and preserves pixels outside the actual edited footprints. Multiple targets are generated jointly.

| Model | Default sampling | Glyph guidance |
| --- | --- | --- |
| PixText-Teacher | 25-step Euler | 3.3 |
| PixText-MeanFlow | 4-step interval Euler | 3.3 |

MeanFlow is the default quick-start choice. Each model package contains its weights, configuration, fonts and SHA-256 checksums. MeanFlow needs no additional text encoder, OCR model, VAE, or teacher.

## Example

Two input images and their targets are bundled in [`examples/`](examples). Cyan boxes mark the edited regions. Click an image to enlarge it.

| Input | MeanFlow · 4 steps | Teacher · 25 steps |
| --- | --- | --- |
| [<img src="examples/pipeline_case/input_boxed.png" width="220" alt="Input with edit regions">](examples/pipeline_case/input_boxed.png) | [<img src="examples/pipeline_case/meanflow_output_boxed.png" width="220" alt="MeanFlow output with edit regions">](examples/pipeline_case/meanflow_output_boxed.png) | [<img src="examples/pipeline_case/teacher_output_boxed.png" width="220" alt="Teacher output with edit regions">](examples/pipeline_case/teacher_output_boxed.png) |

Each row pairs two targets; every target shows input, MeanFlow output and Teacher output.

| # | Input | MeanFlow · 4 steps | Teacher · 25 steps | # | Input | MeanFlow · 4 steps | Teacher · 25 steps |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1&nbsp;喜 | <img src="examples/pipeline_case/details/p01_input.png" width="82" alt="喜 input"> | <img src="examples/pipeline_case/details/p01_meanflow.png" width="82" alt="喜 meanflow"> | <img src="examples/pipeline_case/details/p01_teacher.png" width="82" alt="喜 teacher"> | 8&nbsp;给 | <img src="examples/pipeline_case/details/p08_input.png" width="82" alt="给 input"> | <img src="examples/pipeline_case/details/p08_meanflow.png" width="82" alt="给 meanflow"> | <img src="examples/pipeline_case/details/p08_teacher.png" width="82" alt="给 teacher"> |
| 2&nbsp;发 | <img src="examples/pipeline_case/details/p02_input.png" width="82" alt="发 input"> | <img src="examples/pipeline_case/details/p02_meanflow.png" width="82" alt="发 meanflow"> | <img src="examples/pipeline_case/details/p02_teacher.png" width="82" alt="发 teacher"> | 9&nbsp;解 | <img src="examples/pipeline_case/details/p09_input.png" width="82" alt="解 input"> | <img src="examples/pipeline_case/details/p09_meanflow.png" width="82" alt="解 meanflow"> | <img src="examples/pipeline_case/details/p09_teacher.png" width="82" alt="解 teacher"> |
| 3&nbsp;甜 | <img src="examples/pipeline_case/details/p03_input.png" width="82" alt="甜 input"> | <img src="examples/pipeline_case/details/p03_meanflow.png" width="82" alt="甜 meanflow"> | <img src="examples/pipeline_case/details/p03_teacher.png" width="82" alt="甜 teacher"> | 10&nbsp;馋 | <img src="examples/pipeline_case/details/p10_input.png" width="82" alt="馋 input"> | <img src="examples/pipeline_case/details/p10_meanflow.png" width="82" alt="馋 meanflow"> | <img src="examples/pipeline_case/details/p10_teacher.png" width="82" alt="馋 teacher"> |
| 4&nbsp;初 | <img src="examples/pipeline_case/details/p04_input.png" width="82" alt="初 input"> | <img src="examples/pipeline_case/details/p04_meanflow.png" width="82" alt="初 meanflow"> | <img src="examples/pipeline_case/details/p04_teacher.png" width="82" alt="初 teacher"> | 11&nbsp;满 | <img src="examples/pipeline_case/details/p11_input.png" width="82" alt="满 input"> | <img src="examples/pipeline_case/details/p11_meanflow.png" width="82" alt="满 meanflow"> | <img src="examples/pipeline_case/details/p11_teacher.png" width="82" alt="满 teacher"> |
| 5&nbsp;恋 | <img src="examples/pipeline_case/details/p05_input.png" width="82" alt="恋 input"> | <img src="examples/pipeline_case/details/p05_meanflow.png" width="82" alt="恋 meanflow"> | <img src="examples/pipeline_case/details/p05_teacher.png" width="82" alt="恋 teacher"> | 12&nbsp;满 | <img src="examples/pipeline_case/details/p12_input.png" width="82" alt="满 input"> | <img src="examples/pipeline_case/details/p12_meanflow.png" width="82" alt="满 meanflow"> | <img src="examples/pipeline_case/details/p12_teacher.png" width="82" alt="满 teacher"> |
| 6&nbsp;鞭 | <img src="examples/pipeline_case/details/p06_input.png" width="82" alt="鞭 input"> | <img src="examples/pipeline_case/details/p06_meanflow.png" width="82" alt="鞭 meanflow"> | <img src="examples/pipeline_case/details/p06_teacher.png" width="82" alt="鞭 teacher"> | 13&nbsp;畜 | <img src="examples/pipeline_case/details/p13_input.png" width="82" alt="畜 input"> | <img src="examples/pipeline_case/details/p13_meanflow.png" width="82" alt="畜 meanflow"> | <img src="examples/pipeline_case/details/p13_teacher.png" width="82" alt="畜 teacher"> |
| 7&nbsp;炮 | <img src="examples/pipeline_case/details/p07_input.png" width="82" alt="炮 input"> | <img src="examples/pipeline_case/details/p07_meanflow.png" width="82" alt="炮 meanflow"> | <img src="examples/pipeline_case/details/p07_teacher.png" width="82" alt="炮 teacher"> |  |  |  |  |

**English caption repair.** [`examples/english_case`](examples/english_case) repairs four damaged words in a printed caption, including a misspelled name.

| Input | MeanFlow · 4 steps | Teacher · 25 steps |
| --- | --- | --- |
| [<img src="examples/english_case/input_boxed.png" width="220" alt="Caption input with edit regions">](examples/english_case/input_boxed.png) | [<img src="examples/english_case/meanflow_output_boxed.png" width="220" alt="Caption MeanFlow output with edit regions">](examples/english_case/meanflow_output_boxed.png) | [<img src="examples/english_case/teacher_output_boxed.png" width="220" alt="Caption Teacher output with edit regions">](examples/english_case/teacher_output_boxed.png) |

Each row shows one repaired word: input, MeanFlow output and Teacher output.

| Word | Input | MeanFlow · 4 steps | Teacher · 25 steps |
| --- | --- | --- | --- |
| Nayeen → Nayeon | <img src="examples/english_case/details/nayeon_input.png" width="180" alt="Nayeen input"> | <img src="examples/english_case/details/nayeon_meanflow.png" width="180" alt="Nayeon MeanFlow"> | <img src="examples/english_case/details/nayeon_teacher.png" width="180" alt="Nayeon Teacher"> |
| Nicola | <img src="examples/english_case/details/nicola_input.png" width="180" alt="Nicola input"> | <img src="examples/english_case/details/nicola_meanflow.png" width="180" alt="Nicola MeanFlow"> | <img src="examples/english_case/details/nicola_teacher.png" width="180" alt="Nicola Teacher"> |
| crossbody | <img src="examples/english_case/details/crossbody_input.png" width="180" alt="crossbody input"> | <img src="examples/english_case/details/crossbody_meanflow.png" width="180" alt="crossbody MeanFlow"> | <img src="examples/english_case/details/crossbody_teacher.png" width="180" alt="crossbody Teacher"> |
| fashionable | <img src="examples/english_case/details/fashionable_input.png" width="180" alt="fashionable input"> | <img src="examples/english_case/details/fashionable_meanflow.png" width="180" alt="fashionable MeanFlow"> | <img src="examples/english_case/details/fashionable_teacher.png" width="180" alt="fashionable Teacher"> |

After [installation](#installation), place the complete MeanFlow model package in `models/PixText-MeanFlow` as described in [Model packages](#model-packages), then run:

```bash
python examples/run_pipeline.py                  # pipeline_case
python examples/run_pipeline.py --case english   # english_case
```

Results are saved to `outputs/<case>/meanflow.png`, with reproduction metadata in the adjacent JSON file. To run Teacher, prepare `models/PixText-Teacher` and add `--variant teacher`. Use `--model PATH` for a package stored elsewhere and `--output-dir PATH` to choose the output directory. All cases use seed 42.

Full-resolution images: [pipeline input](examples/pipeline_case/input.png), [MeanFlow](examples/pipeline_case/meanflow_output.png), [Teacher](examples/pipeline_case/teacher_output.png); [caption input](examples/english_case/input.png), [MeanFlow](examples/english_case/meanflow_output.png), [Teacher](examples/english_case/teacher_output.png). Run `python examples/annotate_pipeline.py --case <pipeline|english>` to rebuild the annotated SVG and PNG images.

## ComfyUI

Install the Registry package **`pixtext`** (publisher: **`yaoliliu`**) from ComfyUI Manager, or run `comfy node install pixtext`. Then download a model package as described below. The source installation steps remain available for development.

Draw rectangles directly on your image and enter the intended words in the **PixText Region Editor**. Five PixText nodes handle model loading, optional word detection, region editing, refinement and per-region comparison. Results connect to standard ComfyUI image nodes.

![PixText Region Editor](docs/assets/comfyui-editor.jpg)

1. In your ComfyUI environment, install the custom nodes:

   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/ATH-MaaS/PixText.git
   python -m pip install -r PixText/requirements.txt
   ```

   Use the Python executable that runs ComfyUI. Its existing PyTorch, torchvision, NumPy, Pillow and safetensors installations are reused. A CUDA GPU and a working Triton compiler are required.
2. Download a complete package from [Hugging Face](https://huggingface.co/ATH-MaaS/PixText) into `ComfyUI/models/pixtext/PixText-MeanFlow/` or `ComfyUI/models/pixtext/PixText-Teacher/`. Include its configuration, manifest and fonts alongside the weights.
3. Restart ComfyUI and refresh your browser tab. Open **Templates → PixText**, or drag in [Chinese · MeanFlow](example_workflows/PixText-Chinese-MeanFlow.json) or [Chinese · Teacher](example_workflows/PixText-Chinese-Teacher.json). Both workflows contain the Chinese example image and all 13 targets; no image upload is needed for the example.
4. Click **Open Region Editor**. Draw a box, type its target text, and choose **Apply regions**. Save the workflow to retain your changes, then click **Run**.

The editor supports zoom, pan, moving/resizing boxes, undo/redo, per-region enable switches, coordinate adjustment and JSON import/export. **B** draws, **V** selects, **Space** pans, and the mouse wheel zooms. Clicking any existing box selects it and scrolls the matching target card into view; **Focus region** centers the selected box. Targets use original-image coordinates even while zoomed. The **Example regions** button restores the 13 Chinese targets. Uploading another image starts a separate annotation set.

For upstream images, connect an `IMAGE` to the editor's optional `image_in` input, run once to refresh its preview, then open the editor and apply regions. Connect the editor's `image` and `targets` outputs to **Refine Text · PixText**. It accepts one image at a time; use the Python API for image batches. No enabled regions returns the input unchanged.

The Refine node shows its own live sampling progress, including completed steps and the total across grouped passes. The workflow displays the actual sampling steps: **4 for MeanFlow, 25 for Teacher**. Switching between these standard model packages updates the recommended value; custom step counts are retained. More than 12 targets are grouped automatically. `edit_footprint` is a mask of the actual processed area, which can extend beyond the drawn boxes; pixels outside it are preserved. Models participate in ComfyUI's loading/offloading, and running jobs can be interrupted between sampling steps.

**Compare Regions · PixText** shows matching before/after crops for every enabled target, labeled with its number and target text. Use **Open comparison** for a larger gallery. The optional `padding` adds the same context to both crops; its `contact_sheet` output can connect to Save Image. The example workflows already include this node.

![PixText region comparison gallery](docs/assets/comfyui-comparison.jpg)

### English workflow with automatic word boxes

[English · Auto Detect](example_workflows/PixText-English-AutoDetect.json) uses the caption example above. **Detect Words · PixText** runs [CRAFT](https://github.com/clovaai/CRAFT-pytorch) through EasyOCR's detector and the official [PP-OCRv6 medium ONNX recognizer](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_rec_onnx) locally on CPU. Install the optional dependencies and models once, using ComfyUI's Python:

```bash
cd ComfyUI/custom_nodes/PixText
python -m pip install -r requirements-ocr.txt
python scripts/download_ocr_models.py --output ../../models/pixtext_ocr
```

Restart ComfyUI and refresh the browser, then load the English workflow:

1. **Run** once to detect and recognize words. Refinement waits for your review. The supplied image produces 36 candidate regions with the workflow's thresholds; detections can include false positives or split words.
2. **Open Region Editor**. The detected boxes and recognized text appear automatically, with confidence values. Search for a word to find its card; use **Focus region** to inspect it closely; correct the target text, adjust its box, delete unwanted detections, or draw a missing region. Editing an OCR target automatically enables it; switches also work manually. Unchanged detections start disabled.
3. For this example, change `Nayeen` → `Nayeon`, `Niicola` → `Nicola`, `crossBody` → `crossbody`, and the word starting `fashion` → `fashionable`. OCR confidence measures recognition of the visible image, not whether the spelling is correct. For this tightly spaced caption, set the `Nayeon` box’s `y1` to 853 and the `fashionable` box’s `x1` to 344 to exclude the next line and the trailing comma. Keep the remaining regions disabled.
4. **Apply regions**, save the workflow, then **Run** again. The comparison node shows each selected word before and after repair. Later runs retain your saved boxes and target text. **Reload OCR suggestions** explicitly replaces the editor's current list and supports Undo.

For another image, upload it in **Detect Words**, or connect its optional `image_in`. Connect both its `image` and `detections` outputs to the editor. New images or changed OCR results require review again. A lower `link_threshold` joins more character fragments; raise it if adjacent words merge. The English example uses 0.2 to keep damaged words together. Boxes are axis-aligned repair targets derived from CRAFT quadrilaterals; rotated text may need manual adjustment. This workflow offers word-level suggestions; the Chinese workflow retains its manually reviewed character targets. Model downloads are explicit and checksum-verified; inference sends no images to an external OCR service.

![English OCR suggestions reviewed in the PixText editor](docs/assets/comfyui-english-ocr.jpg)

![Four words repaired with automatically detected boxes](docs/assets/comfyui-english-comparison.jpg)

For headless OCR, submit the [English API workflow](comfyui/workflows/english-autodetect-api.json) once, then copy node 2’s `ocr_suggestions[0]` from the history UI data into its `region_data` JSON string. Set the intended target text and `enabled` flags, retain its `detection_id`, and resubmit for refinement.

Headless examples: [MeanFlow API workflow](comfyui/workflows/chinese-meanflow-api.json), [Teacher API workflow](comfyui/workflows/chinese-teacher-api.json). Tested with ComfyUI 0.42.0, frontend 1.57.1, PyTorch 2.8.0 and Pillow 13.0.0.

## Installation

Use Python 3.11 or newer and a CUDA GPU. The dependencies are specified in [`pyproject.toml`](pyproject.toml), including PyTorch 2.8.0, torchvision 0.23.0 and Pillow 13.0.0. Install a CUDA build of PyTorch and torchvision appropriate for your machine, then install PixText:

```bash
pip install -e .
```

Pillow must support RAQM. The system must provide HarfBuzz (`libharfbuzz0` on Debian/Ubuntu). Font files are loaded from the model package; no system font installation is required.

## Model packages

Download model weights from [Hugging Face](https://huggingface.co/ATH-MaaS/PixText). The examples require complete local model packages arranged as follows:

```text
models/
  PixText-Teacher/
  PixText-MeanFlow/
    model.safetensors
    config.json
    inference_config.json
    manifest.json
    README.md
    fonts/
      AlibabaPuHuiTi-3-55-Regular.ttf
      GoNotoCJKCore-v7.0.ttf
```

Teacher uses the same package layout. [`model_index.json`](model_index.json) lists the model identifiers and weight checksums. `from_pretrained` loads a local directory and checks the files listed in its manifest, including the weights, configuration and fonts.

## Python

```python
from pixtext import PixTextPipeline, TextEdit

pipeline = PixTextPipeline.from_pretrained("models/PixText-MeanFlow")
result = pipeline.edit(
    "input.png",
    edits=[TextEdit(bbox=(120, 80, 160, 125), text="鹿")],
    seed=42,
)
result.image.save("output.png")
```

Use `models/PixText-Teacher` to load Teacher. Both variants can coexist in one process. Results contain `image` and `metadata`, including model identity, seed, sample identity, sampling settings, and crop geometry.

## Command line

Create `edits.json`:

```json
[{"bbox": [120, 80, 160, 125], "text": "鹿"}]
```

```bash
pixtext --model models/PixText-MeanFlow \
  --image input.png --edits edits.json --output output.png --seed 42
```

The CLI saves a PNG and a matching JSON metadata file. `python -m pixtext` provides the same command. `--steps` and `--guidance-scale` override the package defaults.

## Batch editing

```python
results = pipeline.edit_batch(
    ["first.png", "second.png"],
    [
        [TextEdit((120, 80, 160, 125), "鹿")],
        [TextEdit((20, 30, 100, 70), "PixText")],
    ],
    seeds=[42, 43],
    sample_ids=["first", "second"],
)
```

Each image has independent noise and blur streams.

## Inputs and pixel preservation

- Inputs are Pillow images or local image paths, converted to upright RGB using EXIF orientation. Coordinates refer to that upright image.
- Boxes use original-image pixel coordinates `(x0, y0, x1, y1)` and half-open intervals.
- Supply one character or a text line per target.
- Crops align to 16-pixel patches. The edited footprint may extend beyond the requested box. Pixels outside those footprints remain identical to the original RGB image.
- Arbitrary image dimensions are supported.
- Targets are processed in input order, up to 12 per pass.
- The renderer selects the first bundled font covering the entire target string, using PuHuiTi then Go Noto.

Glyph-present and glyph-absent predictions provide guidance on the open interval `(0.1, 1.0)`. At the default guidance setting, four MeanFlow updates use seven model forwards.

## Development

```bash
pip install -e '.[dev]'
ruff check src tests scripts
ruff format --check src tests scripts
mypy
pytest
python3 scripts/check_secrets.py --tracked
```

GPU tests require CUDA and local model packages. Set `PIXTEXT_TEST_MODELS` if packages are stored elsewhere.

## License

PixText is released under the Apache License 2.0. See [LICENSE](LICENSE).
