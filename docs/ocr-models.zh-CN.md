# OCR 模型安装

[English](ocr-models.md)

英文 Auto Detect 工作流使用 CRAFT 检测单词框、PP-OCRv6 识别文字。安装 `requirements-ocr.txt` 后重启 ComfyUI，打开工作流并点击 **Detect Words** 中的 **Prepare OCR models**。节点会显示下载与 SHA-256 校验进度，完成后显示 **Verified**。文件保存在运行 ComfyUI 的机器上，远程服务器也适用。准备完成后直接点击 **运行**，无需再次重启。

所有 Detect Words 节点和浏览器标签共享同一个准备任务。重复点击会复用任务，已校验通过的文件也会复用。失败时点击 **Retry OCR setup**，只重试缺失或校验不通过的文件。

## 手动安装

下载以下文件并放入 ComfyUI 的模型目录：

| 下载文件 | 在 `ComfyUI/models/pixtext_ocr/` 下的位置 |
| --- | --- |
| [CRAFT 模型 ZIP](https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.zip)，解压其中的 `craft_mlt_25k.pth` | `craft_mlt_25k.pth` |
| [PP-OCRv6 权重](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_rec_onnx/resolve/50c7eacafc52fa7bcf4194e8cd08e46f8558504b/inference.onnx?download=true) | `PP-OCRv6_medium_rec_onnx/inference.onnx` |
| [PP-OCRv6 配置](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_rec_onnx/resolve/50c7eacafc52fa7bcf4194e8cd08e46f8558504b/inference.yml?download=true) | `PP-OCRv6_medium_rec_onnx/inference.yml` |

在 PixText 节点目录中，用运行 ComfyUI 的 Python 校验文件：

```bash
python scripts/verify_ocr_models.py --models ../../models/pixtext_ocr
```

三个文件的 SHA-256 均匹配时，命令会分别输出 `Verified`。节点会自动刷新本地模型状态。

## 命令行准备

也可以通过命令行调用同一套准备工具：

```bash
python scripts/download_ocr_models.py --output ../../models/pixtext_ocr
```

工具会下载并校验同一组固定版本的模型文件。按钮、命令行下载工具和本地校验工具均随源码与 Registry 包提供。
