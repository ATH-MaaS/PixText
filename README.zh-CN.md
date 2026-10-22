# PixText

[![arXiv](https://img.shields.io/badge/arXiv-2610.18284-b31b1b.svg)](https://arxiv.org/abs/2610.18284)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

[English](README.md) | **简体中文**

PixText 在像素空间中修复图像中的局部文字。提供图像、待编辑区域和目标文字后，模型会结合图像上下文生成这些区域，并保持实际编辑范围之外的像素不变。多个目标可以联合生成。

| 模型 | 默认采样方式 | 字形引导强度 |
| --- | --- | --- |
| PixText-Teacher | 25 步 Euler | 3.3 |
| PixText-MeanFlow | 4 步区间 Euler | 3.3 |

快速体验默认使用 MeanFlow。每个模型包包含权重、配置、字体和 SHA-256 校验信息。运行 MeanFlow 不需要额外的文本编码器、OCR 模型、VAE 或 Teacher。

## 示例

两个内置示例的输入与目标均位于 [`examples/`](examples)。青色方框标出编辑区域，点击图片可查看大图。

| 输入 | MeanFlow · 4 步 | Teacher · 25 步 |
| --- | --- | --- |
| [<img src="examples/pipeline_case/input_boxed.png" width="220" alt="输入图像及编辑区域">](examples/pipeline_case/input_boxed.png) | [<img src="examples/pipeline_case/meanflow_output_boxed.png" width="220" alt="MeanFlow 输出及编辑区域">](examples/pipeline_case/meanflow_output_boxed.png) | [<img src="examples/pipeline_case/teacher_output_boxed.png" width="220" alt="Teacher 输出及编辑区域">](examples/pipeline_case/teacher_output_boxed.png) |

每行并排两个目标，每个目标展示输入、MeanFlow 输出与 Teacher 输出。

| # | Input | MeanFlow · 4 steps | Teacher · 25 steps | # | Input | MeanFlow · 4 steps | Teacher · 25 steps |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1&nbsp;喜 | <img src="examples/pipeline_case/details/p01_input.png" width="82" alt="喜 input"> | <img src="examples/pipeline_case/details/p01_meanflow.png" width="82" alt="喜 meanflow"> | <img src="examples/pipeline_case/details/p01_teacher.png" width="82" alt="喜 teacher"> | 8&nbsp;给 | <img src="examples/pipeline_case/details/p08_input.png" width="82" alt="给 input"> | <img src="examples/pipeline_case/details/p08_meanflow.png" width="82" alt="给 meanflow"> | <img src="examples/pipeline_case/details/p08_teacher.png" width="82" alt="给 teacher"> |
| 2&nbsp;发 | <img src="examples/pipeline_case/details/p02_input.png" width="82" alt="发 input"> | <img src="examples/pipeline_case/details/p02_meanflow.png" width="82" alt="发 meanflow"> | <img src="examples/pipeline_case/details/p02_teacher.png" width="82" alt="发 teacher"> | 9&nbsp;解 | <img src="examples/pipeline_case/details/p09_input.png" width="82" alt="解 input"> | <img src="examples/pipeline_case/details/p09_meanflow.png" width="82" alt="解 meanflow"> | <img src="examples/pipeline_case/details/p09_teacher.png" width="82" alt="解 teacher"> |
| 3&nbsp;甜 | <img src="examples/pipeline_case/details/p03_input.png" width="82" alt="甜 input"> | <img src="examples/pipeline_case/details/p03_meanflow.png" width="82" alt="甜 meanflow"> | <img src="examples/pipeline_case/details/p03_teacher.png" width="82" alt="甜 teacher"> | 10&nbsp;馋 | <img src="examples/pipeline_case/details/p10_input.png" width="82" alt="馋 input"> | <img src="examples/pipeline_case/details/p10_meanflow.png" width="82" alt="馋 meanflow"> | <img src="examples/pipeline_case/details/p10_teacher.png" width="82" alt="馋 teacher"> |
| 4&nbsp;初 | <img src="examples/pipeline_case/details/p04_input.png" width="82" alt="初 input"> | <img src="examples/pipeline_case/details/p04_meanflow.png" width="82" alt="初 meanflow"> | <img src="examples/pipeline_case/details/p04_teacher.png" width="82" alt="初 teacher"> | 11&nbsp;满 | <img src="examples/pipeline_case/details/p11_input.png" width="82" alt="满 input"> | <img src="examples/pipeline_case/details/p11_meanflow.png" width="82" alt="满 meanflow"> | <img src="examples/pipeline_case/details/p11_teacher.png" width="82" alt="满 teacher"> |
| 5&nbsp;恋 | <img src="examples/pipeline_case/details/p05_input.png" width="82" alt="恋 input"> | <img src="examples/pipeline_case/details/p05_meanflow.png" width="82" alt="恋 meanflow"> | <img src="examples/pipeline_case/details/p05_teacher.png" width="82" alt="恋 teacher"> | 12&nbsp;满 | <img src="examples/pipeline_case/details/p12_input.png" width="82" alt="满 input"> | <img src="examples/pipeline_case/details/p12_meanflow.png" width="82" alt="满 meanflow"> | <img src="examples/pipeline_case/details/p12_teacher.png" width="82" alt="满 teacher"> |
| 6&nbsp;鞭 | <img src="examples/pipeline_case/details/p06_input.png" width="82" alt="鞭 input"> | <img src="examples/pipeline_case/details/p06_meanflow.png" width="82" alt="鞭 meanflow"> | <img src="examples/pipeline_case/details/p06_teacher.png" width="82" alt="鞭 teacher"> | 13&nbsp;畜 | <img src="examples/pipeline_case/details/p13_input.png" width="82" alt="畜 input"> | <img src="examples/pipeline_case/details/p13_meanflow.png" width="82" alt="畜 meanflow"> | <img src="examples/pipeline_case/details/p13_teacher.png" width="82" alt="畜 teacher"> |
| 7&nbsp;炮 | <img src="examples/pipeline_case/details/p07_input.png" width="82" alt="炮 input"> | <img src="examples/pipeline_case/details/p07_meanflow.png" width="82" alt="炮 meanflow"> | <img src="examples/pipeline_case/details/p07_teacher.png" width="82" alt="炮 teacher"> |  |  |  |  |

**英文标语修复。** [`examples/english_case`](examples/english_case) 修复印刷标语中的三处损坏：一个断裂字母和两个单词，周围 logo 保持不变。

| 输入 | MeanFlow · 4 步 | Teacher · 25 步 |
| --- | --- | --- |
| [<img src="examples/english_case/input_boxed.png" width="220" alt="标语输入及编辑区域">](examples/english_case/input_boxed.png) | [<img src="examples/english_case/meanflow_output_boxed.png" width="220" alt="标语 MeanFlow 输出及编辑区域">](examples/english_case/meanflow_output_boxed.png) | [<img src="examples/english_case/teacher_output_boxed.png" width="220" alt="标语 Teacher 输出及编辑区域">](examples/english_case/teacher_output_boxed.png) |

每个修复位置一行：输入、MeanFlow 输出与 Teacher 输出。

| 位置 | 输入 | MeanFlow · 4 步 | Teacher · 25 步 |
| --- | --- | --- | --- |
| Worldwide | <img src="examples/english_case/details/worldwide_input.png" width="180" alt="Worldwide input"> | <img src="examples/english_case/details/worldwide_meanflow.png" width="180" alt="Worldwide MeanFlow"> | <img src="examples/english_case/details/worldwide_teacher.png" width="180" alt="Worldwide Teacher"> |
| Excellence | <img src="examples/english_case/details/excellence_input.png" width="180" alt="Excellence input"> | <img src="examples/english_case/details/excellence_meanflow.png" width="180" alt="Excellence MeanFlow"> | <img src="examples/english_case/details/excellence_teacher.png" width="180" alt="Excellence Teacher"> |
| Quality | <img src="examples/english_case/details/quality_input.png" width="180" alt="Quality input"> | <img src="examples/english_case/details/quality_meanflow.png" width="180" alt="Quality MeanFlow"> | <img src="examples/english_case/details/quality_teacher.png" width="180" alt="Quality Teacher"> |

完成[安装](#安装)后，按照[模型包](#模型包)说明，将完整 MeanFlow 模型包放在 `models/PixText-MeanFlow`，然后运行：

```bash
python examples/run_pipeline.py                  # pipeline_case
python examples/run_pipeline.py --case english   # english_case
```

结果保存到 `outputs/<case>/meanflow.png`，同目录的 JSON 文件记录复现信息。使用 Teacher 时准备好 `models/PixText-Teacher` 并加上 `--variant teacher`。可使用 `--model PATH` 指定其他位置的模型包，使用 `--output-dir PATH` 指定输出目录。两个示例均使用随机种子 42。

原始分辨率图片：[pipeline 输入](examples/pipeline_case/input.png)、[MeanFlow 输出](examples/pipeline_case/meanflow_output.png)、[Teacher 输出](examples/pipeline_case/teacher_output.png)；[标语输入](examples/english_case/input.png)、[MeanFlow 输出](examples/english_case/meanflow_output.png)、[Teacher 输出](examples/english_case/teacher_output.png)。运行 `python examples/annotate_pipeline.py --case <pipeline|english>` 可重新生成带标注的 SVG 与 PNG 图片。

## 方法

![PixText 流程](docs/assets/pipeline.png)

## 安装

需要 Python 3.11 或更高版本，以及支持 CUDA 的 GPU。依赖见 [`pyproject.toml`](pyproject.toml)，包括 PyTorch 2.8.0、torchvision 0.23.0 和 Pillow 13.0.0。先安装适合本机 CUDA 环境的 PyTorch 和 torchvision，再安装 PixText：

```bash
pip install -e .
```

Pillow 需要支持 RAQM，系统需要提供 HarfBuzz（Debian/Ubuntu 中为 `libharfbuzz0`）。字体从模型包加载，无需单独安装到系统。

## 模型包

本仓库不包含模型权重文件。运行示例前，需要准备完整的本地模型包，目录结构如下：

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

Teacher 使用相同的模型包结构。[`model_index.json`](model_index.json) 列出了模型标识和权重校验值。`from_pretrained` 从本地目录加载模型，检查清单中列出的权重、配置和字体。

## Python 接口

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

加载 Teacher 时，将路径替换为 `models/PixText-Teacher`。两种模型可以在同一进程中使用。返回结果包含 `image` 和 `metadata`，记录模型标识、随机种子、样本标识、采样设置及裁剪区域。

## 命令行

创建 `edits.json`：

```json
[{"bbox": [120, 80, 160, 125], "text": "鹿"}]
```

```bash
pixtext --model models/PixText-MeanFlow \
  --image input.png --edits edits.json --output output.png --seed 42
```

命令行会保存 PNG 图片及同名 JSON 元数据文件，也可以使用 `python -m pixtext` 调用。`--steps` 和 `--guidance-scale` 可覆盖模型包的默认设置。

## 批量编辑

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

每张图片使用独立的噪声和模糊随机流。

## 输入与像素保留规则

- 输入可以是 Pillow 图片或本地图片路径，会根据 EXIF 方向转为正向 RGB。编辑坐标以转正后的图片为准。
- 编辑框使用原图像素坐标 `(x0, y0, x1, y1)`，采用左闭右开的区间。
- 每个目标可以是一个字符或一行文字。
- 裁剪区域按 16 像素的 patch 对齐，因此实际编辑范围可能超出指定框。实际编辑范围之外的像素与原始 RGB 图片完全一致。
- 支持任意图片尺寸。
- 按输入顺序处理目标，每轮最多 12 个。
- 字体按普惠体、Go Noto 的顺序选择，要求单个字体覆盖整条目标文字。

字形引导在开区间 `(0.1, 1.0)` 内组合有字形和无字形条件的预测。默认引导设置下，MeanFlow 的 4 次更新需要 7 次模型前向计算。

## 开发

```bash
pip install -e '.[dev]'
ruff check src tests scripts
ruff format --check src tests scripts
mypy
pytest
python3 scripts/check_secrets.py --tracked
```

GPU 测试需要 CUDA 和本地模型包。模型包位于其他目录时，可设置 `PIXTEXT_TEST_MODELS`。

## 许可

PixText 以 Apache License 2.0 发布，详见 [LICENSE](LICENSE)。
