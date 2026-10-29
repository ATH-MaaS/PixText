// ComfyUI owns the missing-model dialog and download dispatch.
export const MODEL_DOWNLOADS = [
  {
    name: "PixText-MeanFlow.safetensors",
    url: "https://huggingface.co/ATH-MaaS/PixText/resolve/main/models/PixText-MeanFlow/model.safetensors",
    directory: "diffusion_models",
    hash: "d622bb403165c17853bcb9050e6df97330617df7d22516840f2be719240a08ea",
    hash_type: "SHA256",
  },
  {
    name: "PixText-Teacher.safetensors",
    url: "https://huggingface.co/ATH-MaaS/PixText/resolve/main/models/PixText-Teacher/model.safetensors",
    directory: "diffusion_models",
    hash: "4717506a7934334fd04755e5d5f22f398fd31c170cc6f3ea76b48cd99458ec28",
    hash_type: "SHA256",
  },
];

export function setupModelDownloads(node) {
  const input = node.widgets?.find((widget) => widget.name === "model_name");
  if (!input || node.ptModelDownloads) return;
  node.ptModelDownloads = true;
  const update = () => {
    if (input.value === "(install a model package)") {
      input.value = MODEL_DOWNLOADS[0].name;
    }
    const legacy = MODEL_DOWNLOADS.find((model) => model.name === `${input.value}.safetensors`);
    if (legacy) input.value = legacy.name;
    const selected = MODEL_DOWNLOADS.find((model) => model.name === input.value);
    node.properties ??= {};
    node.properties.models = selected ? [{ ...selected }] : [];
  };
  const callback = input.callback;
  input.callback = function (...args) {
    const result = callback?.apply(this, args);
    update();
    return result;
  };
  const configure = node.onConfigure;
  node.onConfigure = function (...args) {
    const result = configure?.apply(this, args);
    update();
    return result;
  };
  update();
}
