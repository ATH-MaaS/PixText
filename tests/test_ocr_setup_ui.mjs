// Run with: node --experimental-vm-modules tests/test_ocr_setup_ui.mjs
import assert from "node:assert/strict";
import fs from "node:fs/promises";
import vm from "node:vm";
import { test } from "node:test";

await test("two nodes share polling and queue one click behind a status request", async () => {
  const requests = [], timers = new Map();
  let timerId = 0;
  class Element {
    constructor() { this.children = []; this.listeners = {}; this.style = {}; this.dataset = {}; }
    append(...children) { this.children.push(...children); }
    setAttribute() {}
    addEventListener(name, fn) { this.listeners[name] = fn; }
  }
  const api = { fetchApi(path, options) {
    return new Promise((resolve) => requests.push({ path, method: options.method, resolve }));
  } };
  const context = vm.createContext({
    document: { createElement: () => new Element() },
    setTimeout: (fn) => { timers.set(++timerId, fn); return timerId; },
    clearTimeout: (id) => timers.delete(id),
  });
  const module = new vm.SourceTextModule(
    await fs.readFile(new URL("../comfyui/web/ocr_setup.js", import.meta.url), "utf8"),
    { context },
  );
  await module.link(() => new vm.SyntheticModule(["api"], function () {
    this.setExport("api", api);
  }, { context }));
  await module.evaluate();
  const node = () => ({
    size: [350, 650], addDOMWidget: () => ({}), setSize() {},
  });
  const first = node(), second = node();
  module.namespace.setupOCR(first); module.namespace.setupOCR(second);
  const action = (n) => n.ptOCRSetup.children[4];
  const tick = () => new Promise((resolve) => setImmediate(resolve));
  const respond = async (index, state) => {
    requests[index].resolve({ ok: true, json: async () => state }); await tick();
  };
  const poll = () => {
    assert.equal(timers.size, 1);
    const [id, fn] = timers.entries().next().value;
    timers.delete(id); fn();
  };
  const missing = { state: "missing", busy: false, progress: 0, message: "Prepare models" };
  assert.equal(requests.length, 1, "one initial GET for both nodes");
  await respond(0, missing);
  assert.equal(action(first).disabled, false);
  poll(); // A slow status GET overlaps the user's click.
  action(first).listeners.click(); action(second).listeners.click();
  assert.equal(action(first).disabled, true);
  assert.equal(action(second).disabled, true);
  await respond(1, missing);
  assert.equal(requests.length, 3);
  assert.equal(requests[2].method, "POST");
  await respond(2, { state: "downloading", busy: true, progress: 25, message: "Downloading" });
  assert.equal(action(first).textContent, "Preparing · 25%");
  assert.equal(action(second).textContent, "Preparing · 25%");
  poll();
  await respond(3, { state: "ready", busy: false, progress: 100, message: "Verified" });
  assert.equal(action(first).textContent, "✓  OCR models ready");
  assert.equal(action(second).disabled, true);
  action(first).listeners.click();
  assert.equal(requests.length, 4, "ready models cannot start another preparation");
  first.onRemoved(); second.onRemoved();
  assert.equal(timers.size, 0, "removed nodes stop polling");
});
