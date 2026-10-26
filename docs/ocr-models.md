# OCR model setup

[简体中文](ocr-models.zh-CN.md)

The English Auto Detect workflow uses CRAFT for word boxes and PP-OCRv6 for recognition. After installing `requirements-ocr.txt`, restart ComfyUI and open the workflow. Click **Prepare OCR models** in **Detect Words**. The node shows download and SHA-256 verification progress, then **Verified**. Models are saved on the machine running ComfyUI, including remote servers. Click **Run** to detect words; no restart is needed after preparing the models.

All Detect Words nodes and browser tabs share a preparation job. Repeated clicks reuse that job; existing verified files are reused. If setup fails, **Retry OCR setup** retries only files that are missing or fail verification.

## Manual installation

Download these files and place them in your ComfyUI model directory:

| Download | Destination under `ComfyUI/models/pixtext_ocr/` |
| --- | --- |
| [CRAFT model ZIP](https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.zip) — extract `craft_mlt_25k.pth` | `craft_mlt_25k.pth` |
| [PP-OCRv6 weights](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_rec_onnx/resolve/50c7eacafc52fa7bcf4194e8cd08e46f8558504b/inference.onnx?download=true) | `PP-OCRv6_medium_rec_onnx/inference.onnx` |
| [PP-OCRv6 configuration](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_rec_onnx/resolve/50c7eacafc52fa7bcf4194e8cd08e46f8558504b/inference.yml?download=true) | `PP-OCRv6_medium_rec_onnx/inference.yml` |

From the PixText node folder, verify the installed files with ComfyUI's Python:

```bash
python scripts/verify_ocr_models.py --models ../../models/pixtext_ocr
```

The command prints `Verified` for all three files when their SHA-256 checksums match. The node refreshes its local model status automatically.

## Command-line setup

The same preparation engine is available from the command line:

```bash
python scripts/download_ocr_models.py --output ../../models/pixtext_ocr
```

The helper downloads the same pinned files and verifies their checksums. The button, command-line helper and local verifier are included in both source and Registry packages.
