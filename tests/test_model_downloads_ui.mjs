import assert from "node:assert/strict";
import test from "node:test";
import { MODEL_DOWNLOADS, setupModelDownloads } from "../comfyui/web/model_downloads.js";

function loader(value) {
  const node = { widgets: [{ name: "model_name", value }], properties: {} };
  setupModelDownloads(node);
  return node;
}

test("new empty loaders carry a native missing-model requirement", () => {
  const node = loader("(install a model package)");
  assert.equal(node.widgets[0].value, "PixText-MeanFlow.safetensors");
  assert.deepEqual(node.properties.models, [MODEL_DOWNLOADS[0]]);
});

test("switching models updates the declared URL and hash", () => {
  const node = loader("PixText-MeanFlow.safetensors");
  node.widgets[0].value = "PixText-Teacher.safetensors";
  node.widgets[0].callback(node.widgets[0].value);
  assert.deepEqual(node.properties.models, [MODEL_DOWNLOADS[1]]);
});

test("legacy workflow package names migrate on configure", () => {
  const node = loader("PixText-MeanFlow.safetensors");
  node.widgets[0].value = "PixText-Teacher";
  node.onConfigure({});
  assert.equal(node.widgets[0].value, "PixText-Teacher.safetensors");
  assert.deepEqual(node.properties.models, [MODEL_DOWNLOADS[1]]);
});

test("custom complete packages do not advertise unrelated downloads", () => {
  const node = loader("my-local-package");
  assert.deepEqual(node.properties.models, []);
});
