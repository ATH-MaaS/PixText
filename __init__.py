"""ComfyUI custom-node entry point; the Python API lives in src/pixtext."""

WEB_DIRECTORY = "./comfyui/web"


async def comfy_entrypoint():
    # ComfyUI clones custom nodes directly, without installing their Python package.
    import sys
    from pathlib import Path

    source = str(Path(__file__).resolve().parent / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    from .comfyui.nodes import PixTextExtension

    return PixTextExtension()
