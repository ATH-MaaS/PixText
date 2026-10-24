# OCR model setup

[简体中文](ocr-models.zh-CN.md)

The English Auto Detect workflow uses CRAFT for word boxes and PP-OCRv6 for recognition. Download these files and place them in your ComfyUI model directory:

| Download | Destination under `ComfyUI/models/pixtext_ocr/` |
| --- | --- |
| [CRAFT model ZIP](https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.zip) — extract `craft_mlt_25k.pth` | `craft_mlt_25k.pth` |
| [PP-OCRv6 weights](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_rec_onnx/resolve/50c7eacafc52fa7bcf4194e8cd08e46f8558504b/inference.onnx?download=true) | `PP-OCRv6_medium_rec_onnx/inference.onnx` |
| [PP-OCRv6 configuration](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_rec_onnx/resolve/50c7eacafc52fa7bcf4194e8cd08e46f8558504b/inference.yml?download=true) | `PP-OCRv6_medium_rec_onnx/inference.yml` |

From the PixText node folder, verify the installed files with ComfyUI's Python:

```bash
python scripts/verify_ocr_models.py --models ../../models/pixtext_ocr
```

The command prints `Verified` for all three files when their SHA-256 checksums match. Restart ComfyUI, refresh the browser, and open the English Auto Detect workflow.

## Source-install shortcut

GitHub source checkouts also include an optional download helper:

```bash
python scripts/download_ocr_models.py --output ../../models/pixtext_ocr
```

The helper downloads the same pinned files and verifies their checksums. It is maintained in the source repository separately from the Registry runtime package; Registry installations use the downloads above. The local verification command is included in both distributions.
