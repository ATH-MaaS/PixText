# OCR 模型安装

[English](ocr-models.md)

英文 Auto Detect 工作流使用 CRAFT 检测单词框、PP-OCRv6 识别文字。在运行 ComfyUI 的 Python 环境中安装 `requirements-ocr.txt` 后重启 ComfyUI。工作流中的 **OCR model downloads** 备注列出了下载链接和每个文件的保存位置。

连接远程 ComfyUI 时，要把文件放到运行 ComfyUI 的服务器上。浏览器下载的文件会保存在自己的电脑上。

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

三个文件的 SHA-256 均匹配时，命令会分别输出 `Verified`。文件就位后运行工作流即可。
