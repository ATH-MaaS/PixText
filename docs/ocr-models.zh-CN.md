# OCR 模型安装

[English](ocr-models.md)

英文 Auto Detect 工作流使用 CRAFT 检测单词框、PP-OCRv6 识别文字。下载以下文件并放入 ComfyUI 的模型目录：

| 下载文件 | 在 `ComfyUI/models/pixtext_ocr/` 下的位置 |
| --- | --- |
| [CRAFT 模型 ZIP](https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.zip)，解压其中的 `craft_mlt_25k.pth` | `craft_mlt_25k.pth` |
| [PP-OCRv6 权重](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_rec_onnx/resolve/50c7eacafc52fa7bcf4194e8cd08e46f8558504b/inference.onnx?download=true) | `PP-OCRv6_medium_rec_onnx/inference.onnx` |
| [PP-OCRv6 配置](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_rec_onnx/resolve/50c7eacafc52fa7bcf4194e8cd08e46f8558504b/inference.yml?download=true) | `PP-OCRv6_medium_rec_onnx/inference.yml` |

在 PixText 节点目录中，用运行 ComfyUI 的 Python 校验文件：

```bash
python scripts/verify_ocr_models.py --models ../../models/pixtext_ocr
```

三个文件的 SHA-256 均匹配时，命令会分别输出 `Verified`。随后重启 ComfyUI、刷新浏览器并打开英文 Auto Detect 工作流。

## 源码安装快捷方式

从 GitHub 克隆的源码还包含可选下载工具：

```bash
python scripts/download_ocr_models.py --output ../../models/pixtext_ocr
```

工具会下载并校验同一组固定版本的模型文件。它随 GitHub 源码维护，与 Registry 运行包分开；Registry 安装请使用上表中的下载链接。两种安装方式均包含本地校验命令。
