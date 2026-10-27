# PixText: Efficient Scene-Text Refinement via Pixel-Space Grouped Image Generation

[![arXiv](https://img.shields.io/badge/arXiv-2610.18284-b31b1b.svg)](https://arxiv.org/abs/2610.18284)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Model weights](https://img.shields.io/badge/🤗%20Hugging%20Face-Model%20Weights-yellow)](https://huggingface.co/ATH-MaaS/PixText)
[![Benchmark](https://img.shields.io/badge/🤗%20Hugging%20Face-TextRefinementBench-yellow)](https://huggingface.co/datasets/ATH-MaaS/TextRefinementBench)

[![ComfyUI](https://img.shields.io/badge/ComfyUI-Registry-2D8C83)](https://registry.comfy.org/publishers/yaoliliu/nodes/pixtext)

![PixText pipeline](docs/assets/pipeline.png)

**English** | [简体中文](README.zh-CN.md)

PixText fixes words and characters in images. Give it an image, boxes around the text you want to change, and the target text. It refines the selected regions together in pixel space, using the surrounding image as context. Pixels outside the actual edit areas stay unchanged.

| Model | Default sampling | Glyph guidance |
| --- | --- | --- |
| PixText-Teacher | 25-step Euler | 3.3 |
| PixText-MeanFlow | 4-step interval Euler | 3.3 |

Start with MeanFlow. Each model package includes weights, configuration, fonts and SHA-256 checksums. MeanFlow runs without an additional text encoder, OCR model, VAE or teacher.

## Example

The two examples in [`examples/`](examples) include the input images and target text. Cyan boxes mark the edits. Click an image to view it full size.

| Input | MeanFlow · 4 steps | Teacher · 25 steps |
| --- | --- | --- |
| [<img src="examples/pipeline_case/input_boxed.png" width="220" alt="Input with edit regions">](examples/pipeline_case/input_boxed.png) | [<img src="examples/pipeline_case/meanflow_output_boxed.png" width="220" alt="MeanFlow output with edit regions">](examples/pipeline_case/meanflow_output_boxed.png) | [<img src="examples/pipeline_case/teacher_output_boxed.png" width="220" alt="Teacher output with edit regions">](examples/pipeline_case/teacher_output_boxed.png) |

| # | Input | MeanFlow · 4 steps | Teacher · 25 steps | # | Input | MeanFlow · 4 steps | Teacher · 25 steps |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1&nbsp;喜 | <img src="examples/pipeline_case/details/p01_input.png" width="82" alt="喜 input"> | <img src="examples/pipeline_case/details/p01_meanflow.png" width="82" alt="喜 meanflow"> | <img src="examples/pipeline_case/details/p01_teacher.png" width="82" alt="喜 teacher"> | 8&nbsp;给 | <img src="examples/pipeline_case/details/p08_input.png" width="82" alt="给 input"> | <img src="examples/pipeline_case/details/p08_meanflow.png" width="82" alt="给 meanflow"> | <img src="examples/pipeline_case/details/p08_teacher.png" width="82" alt="给 teacher"> |
| 2&nbsp;发 | <img src="examples/pipeline_case/details/p02_input.png" width="82" alt="发 input"> | <img src="examples/pipeline_case/details/p02_meanflow.png" width="82" alt="发 meanflow"> | <img src="examples/pipeline_case/details/p02_teacher.png" width="82" alt="发 teacher"> | 9&nbsp;解 | <img src="examples/pipeline_case/details/p09_input.png" width="82" alt="解 input"> | <img src="examples/pipeline_case/details/p09_meanflow.png" width="82" alt="解 meanflow"> | <img src="examples/pipeline_case/details/p09_teacher.png" width="82" alt="解 teacher"> |
| 3&nbsp;甜 | <img src="examples/pipeline_case/details/p03_input.png" width="82" alt="甜 input"> | <img src="examples/pipeline_case/details/p03_meanflow.png" width="82" alt="甜 meanflow"> | <img src="examples/pipeline_case/details/p03_teacher.png" width="82" alt="甜 teacher"> | 10&nbsp;馋 | <img src="examples/pipeline_case/details/p10_input.png" width="82" alt="馋 input"> | <img src="examples/pipeline_case/details/p10_meanflow.png" width="82" alt="馋 meanflow"> | <img src="examples/pipeline_case/details/p10_teacher.png" width="82" alt="馋 teacher"> |
| 4&nbsp;初 | <img src="examples/pipeline_case/details/p04_input.png" width="82" alt="初 input"> | <img src="examples/pipeline_case/details/p04_meanflow.png" width="82" alt="初 meanflow"> | <img src="examples/pipeline_case/details/p04_teacher.png" width="82" alt="初 teacher"> | 11&nbsp;满 | <img src="examples/pipeline_case/details/p11_input.png" width="82" alt="满 input"> | <img src="examples/pipeline_case/details/p11_meanflow.png" width="82" alt="满 meanflow"> | <img src="examples/pipeline_case/details/p11_teacher.png" width="82" alt="满 teacher"> |
| 5&nbsp;恋 | <img src="examples/pipeline_case/details/p05_input.png" width="82" alt="恋 input"> | <img src="examples/pipeline_case/details/p05_meanflow.png" width="82" alt="恋 meanflow"> | <img src="examples/pipeline_case/details/p05_teacher.png" width="82" alt="恋 teacher"> | 12&nbsp;满 | <img src="examples/pipeline_case/details/p12_input.png" width="82" alt="满 input"> | <img src="examples/pipeline_case/details/p12_meanflow.png" width="82" alt="满 meanflow"> | <img src="examples/pipeline_case/details/p12_teacher.png" width="82" alt="满 teacher"> |
| 6&nbsp;鞭 | <img src="examples/pipeline_case/details/p06_input.png" width="82" alt="鞭 input"> | <img src="examples/pipeline_case/details/p06_meanflow.png" width="82" alt="鞭 meanflow"> | <img src="examples/pipeline_case/details/p06_teacher.png" width="82" alt="鞭 teacher"> | 13&nbsp;畜 | <img src="examples/pipeline_case/details/p13_input.png" width="82" alt="畜 input"> | <img src="examples/pipeline_case/details/p13_meanflow.png" width="82" alt="畜 meanflow"> | <img src="examples/pipeline_case/details/p13_teacher.png" width="82" alt="畜 teacher"> |
| 7&nbsp;炮 | <img src="examples/pipeline_case/details/p07_input.png" width="82" alt="炮 input"> | <img src="examples/pipeline_case/details/p07_meanflow.png" width="82" alt="炮 meanflow"> | <img src="examples/pipeline_case/details/p07_teacher.png" width="82" alt="炮 teacher"> |  |  |  |  |

The English example in [`examples/english_case`](examples/english_case) repairs four damaged words in a printed caption, including a misspelled name.

| Input | MeanFlow · 4 steps | Teacher · 25 steps |
| --- | --- | --- |
| [<img src="examples/english_case/input_boxed.png" width="220" alt="Caption input with edit regions">](examples/english_case/input_boxed.png) | [<img src="examples/english_case/meanflow_output_boxed.png" width="220" alt="Caption MeanFlow output with edit regions">](examples/english_case/meanflow_output_boxed.png) | [<img src="examples/english_case/teacher_output_boxed.png" width="220" alt="Caption Teacher output with edit regions">](examples/english_case/teacher_output_boxed.png) |

| Word | Input | MeanFlow · 4 steps | Teacher · 25 steps |
| --- | --- | --- | --- |
| Nayeen → Nayeon | <img src="examples/english_case/details/nayeon_input.png" width="180" alt="Nayeen input"> | <img src="examples/english_case/details/nayeon_meanflow.png" width="180" alt="Nayeon MeanFlow"> | <img src="examples/english_case/details/nayeon_teacher.png" width="180" alt="Nayeon Teacher"> |
| Nicola | <img src="examples/english_case/details/nicola_input.png" width="180" alt="Nicola input"> | <img src="examples/english_case/details/nicola_meanflow.png" width="180" alt="Nicola MeanFlow"> | <img src="examples/english_case/details/nicola_teacher.png" width="180" alt="Nicola Teacher"> |
| crossbody | <img src="examples/english_case/details/crossbody_input.png" width="180" alt="crossbody input"> | <img src="examples/english_case/details/crossbody_meanflow.png" width="180" alt="crossbody MeanFlow"> | <img src="examples/english_case/details/crossbody_teacher.png" width="180" alt="crossbody Teacher"> |
| fashionable | <img src="examples/english_case/details/fashionable_input.png" width="180" alt="fashionable input"> | <img src="examples/english_case/details/fashionable_meanflow.png" width="180" alt="fashionable MeanFlow"> | <img src="examples/english_case/details/fashionable_teacher.png" width="180" alt="fashionable Teacher"> |

To run either example, [install PixText](#installation) and place the complete MeanFlow model package in `models/PixText-MeanFlow` (see [Model packages](#model-packages)). Then run:

```bash
python examples/run_pipeline.py                  # pipeline_case
python examples/run_pipeline.py --case english   # english_case
```

The script saves the result to `outputs/<case>/meanflow.png` and writes reproduction metadata to a JSON file beside it. For Teacher, prepare `models/PixText-Teacher` and add `--variant teacher`. Use `--model PATH` if your model package is elsewhere, or `--output-dir PATH` to change where results are saved. Both examples use seed 42.

Full-size images: [pipeline input](examples/pipeline_case/input.png), [MeanFlow](examples/pipeline_case/meanflow_output.png), [Teacher](examples/pipeline_case/teacher_output.png); [caption input](examples/english_case/input.png), [MeanFlow](examples/english_case/meanflow_output.png), [Teacher](examples/english_case/teacher_output.png). To regenerate the annotated SVG and PNG images, run `python examples/annotate_pipeline.py --case <pipeline|english>`.

## ComfyUI

Draw boxes around the text to fix, enter the correct words, and view each repair in ComfyUI.

### Install nodes

1. Clone the nodes and install their dependencies in ComfyUI's Python environment:

   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/ATH-MaaS/PixText.git
   python -m pip install -r PixText/requirements.txt
   ```

2. Download a complete model package from [Hugging Face](https://huggingface.co/ATH-MaaS/PixText) and place it in `ComfyUI/models/pixtext/PixText-MeanFlow/` or `ComfyUI/models/pixtext/PixText-Teacher/`. The loader also finds these package folders directly under `ComfyUI/models/`. See [Model packages](#model-packages) for the directory layout. Refinement requires an NVIDIA CUDA GPU.
3. Restart ComfyUI and refresh your browser. Open **Templates → PixText**, or drag one of the workflow files below onto the canvas.

### Chinese · Draw and refine

[Download workflow · MeanFlow](example_workflows/PixText-Chinese-MeanFlow.json) · [Teacher variant](example_workflows/PixText-Chinese-Teacher.json)

![Chinese text refinement workflow: model loader, region editor, refiner and before/after comparison](docs/assets/comfyui-workflow-chinese.jpg)

1. Choose a model in **Load PixText Model**. The example comes with an image and 13 text regions. MeanFlow defaults to 4 steps; Teacher defaults to 25.
2. Click **Open Region Editor** to draw or adjust the boxes and enter the target text. Click **Apply regions** when you are done.
3. Click **Run**, then check each repair in **Compare Regions**. Click **Open comparison** for the full gallery. Saving the workflow keeps your boxes and target text.

### English · Detect, adjust and refine

[Download workflow · Auto Detect](example_workflows/PixText-English-AutoDetect.json)

![English text refinement workflow with automatic word detection and editable regions](docs/assets/comfyui-workflow-english.jpg)

Install the OCR dependencies once in ComfyUI's Python environment:

```bash
cd ComfyUI/custom_nodes/PixText
python -m pip install -r requirements-ocr.txt
```

Restart ComfyUI, then open the English workflow:

1. In **Detect Words**, click **Prepare OCR models** to download and verify the files on your ComfyUI server. When the status reads **Verified**, click **Run** to detect word boxes and read their text. [OCR setup](docs/ocr-models.md) covers manual installation.
2. Open **Region Editor** and adjust the boxes and target words. The example targets are `Nayeon`, `Nicola`, `crossbody` and `fashionable`. Enable the regions you want to repair, then click **Apply regions**.
3. Click **Run** again to repair the selected words and compare the results. To try another image, upload it in **Detect Words**.

## Installation

PixText requires Python 3.11 or newer and a CUDA GPU. [`pyproject.toml`](pyproject.toml) lists the dependencies, including PyTorch 2.8.0, torchvision 0.23.0 and Pillow 11.3.0 or newer. Install the PyTorch and torchvision builds for your CUDA environment, then install PixText:

```bash
pip install -e .
```

PixText uses Pillow with RAQM to lay out target text and render the glyph images supplied to the model. The system also needs HarfBuzz (`libharfbuzz0` on Debian/Ubuntu). PixText loads fonts from the model package, so you do not need to install them as system fonts.

## Model packages

Download a complete model package from [Hugging Face](https://huggingface.co/ATH-MaaS/PixText). For the Python examples, use this directory layout:

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

Teacher uses the same layout. [`model_index.json`](model_index.json) lists model identifiers and weight checksums. When `from_pretrained` loads a local package, it checks the weights, configuration, fonts and other files listed in the manifest.

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

For Teacher, use `models/PixText-Teacher`. You can load both models in one process. Each result contains an `image` and `metadata` with the model identity, seed, sample identity, sampling settings and crop geometry.

## Command line

Create `edits.json`:

```json
[{"bbox": [120, 80, 160, 125], "text": "鹿"}]
```

```bash
pixtext --model models/PixText-MeanFlow \
  --image input.png --edits edits.json --output output.png --seed 42
```

The command saves a PNG and a JSON metadata file with the same base name. You can also call it with `python -m pixtext`. Set `--steps` and `--guidance-scale` to override the model package defaults.

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

Each image uses its own noise and blur streams.

## Inputs and pixel preservation

- Pass a Pillow image or a local image path. PixText uses EXIF orientation to turn it upright and converts it to RGB. Box coordinates refer to this upright image.
- Specify boxes in original-image pixels as `(x0, y0, x1, y1)`, using half-open intervals.
- Each target can be one character or a line of text.
- Crops align to 16-pixel patches, so the actual edit area may extend beyond the box. Pixels outside that area remain identical to the original RGB image.
- Images can have arbitrary dimensions.
- Targets run in input order, with up to 12 per pass.
- For each target, the renderer uses the first bundled font that covers the whole string: PuHuiTi, then Go Noto.

Glyph guidance combines predictions with and without glyphs on the open interval `(0.1, 1.0)`. With the default guidance setting, four MeanFlow updates take seven model forwards.

## Development

```bash
pip install -e '.[dev]'
ruff check src tests scripts
ruff format --check src tests scripts
mypy
pytest
python3 scripts/check_secrets.py --tracked
```

GPU tests need CUDA and local model packages. Set `PIXTEXT_TEST_MODELS` if your packages are in another directory.

## Citation

If you use PixText in your research, please cite:

```bibtex
@misc{liu2026efficientscenetextrefinementpixelspace,
  title={Efficient Scene-Text Refinement via Pixel-Space Grouped Image Generation},
  author={Yaoli Liu and Guo-Hua Wang and Yiyang Chen and Qing-Guo Chen and Zhao Xu and Weihua Luo and Yao-Xiang Ding},
  year={2026},
  eprint={2610.18284},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2610.18284},
}
```

## License

PixText is released under the Apache License 2.0. See [LICENSE](LICENSE).
