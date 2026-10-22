import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const stylesheet = document.createElement("link");
stylesheet.rel = "stylesheet";
stylesheet.href = new URL("./pixtext.css", import.meta.url).href;
document.head.append(stylesheet);
const EXAMPLE = "example: Chinese New Year";
const COLORS = ["#5eead4", "#fbbf24", "#a5b4fc", "#fb7185", "#7dd3fc", "#d8b4fe"];
const clone = (value) => JSON.parse(JSON.stringify(value));
const widget = (node, name) => node.widgets?.find((item) => item.name === name);
const element = (tag, className, text) => {
  const item = document.createElement(tag);
  if (className) item.className = className;
  if (text !== undefined) item.textContent = text;
  return item;
};
const button = (text, action, className = "") => {
  const item = element("button", className, text);
  item.type = "button";
  item.addEventListener("click", action);
  return item;
};
function readState(node) {
  try { return JSON.parse(widget(node, "region_data")?.value || "{}"); }
  catch { return {}; }
}
function writeState(node, state) {
  app.graph?.beforeChange();
  const target = widget(node, "region_data");
  target.value = JSON.stringify(state);
  target.callback?.(target.value);
  updateSummary(node);
  app.graph?.afterChange();
  app.graph?.change();
  app.graph?.setDirtyCanvas(true, true);
}
function viewURL(ref) {
  return api.apiURL(`/view?${new URLSearchParams(ref)}`);
}
function uploadedURL(name) {
  const split = name.lastIndexOf("/");
  return viewURL({ filename: name.slice(split + 1), subfolder: split < 0 ? "" : name.slice(0, split), type: "input" });
}
function loadImage(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("Image could not be loaded. Refresh the source image and retry."));
    image.src = url;
  });
}
function toast(message) {
  app.extensionManager?.toast?.add({ severity: "warn", summary: "PixText", detail: message, life: 6000 });
}
function connected(node) {
  return node.inputs?.some((input) => input.name === "image_in" && input.link != null);
}
function updateSummary(node) {
  if (!node.ptSummary) return;
  const state = readState(node);
  const rows = state.regions || [];
  node.ptSummary.count.textContent = `${rows.filter((row) => row.enabled !== false).length} active regions${node.ptDetections ? ` · ${node.ptDetections.regions.length} detected` : ""}`;
  node.ptSummary.detail.textContent = state.size ? `${state.size[0]} × ${state.size[1]} · saved in workflow` : "Draw a box, then type the target text";
}
let closeActiveEditor = null;

async function openEditor(node) {
  const imageName = widget(node, "image").value;
  let imageURL, sourceId;
  if (connected(node)) {
    if (!node.ptSource) return toast("Run the workflow once to refresh the connected image, then open the editor.");
    imageURL = viewURL(node.ptSource.image);
    sourceId = node.ptSource.id;
  } else {
    imageURL = imageName === EXAMPLE ? api.apiURL("/pixtext/example/chinese") : uploadedURL(imageName);
    sourceId = imageName === EXAMPLE ? "example:chinese" : `file:${imageName}`;
  }
  let image;
  try { image = await loadImage(imageURL); }
  catch (error) { return toast(error.message); }
  closeActiveEditor?.();
  const saved = readState(node);
  const suggestions = node.ptDetections?.source_id === sourceId ? node.ptDetections : null;
  const compatible = saved.source_id === sourceId && saved.size?.[0] === image.width && saved.size?.[1] === image.height;
  let rows = compatible ? clone(saved.regions || []) : clone(suggestions?.regions || []);
  let selected = rows[0]?.id ?? null;
  let tool = "draw", scale = 1, tx = 0, ty = 0, gesture = null, draft = null, space = false;
  let undo = [], redo = [], dirty = false, closed = false;
  const previousFocus = document.activeElement;
  const backdrop = element("div", "pt-backdrop");
  const dialog = element("section", "pt-editor");
  dialog.setAttribute("role", "dialog");
  dialog.setAttribute("aria-modal", "true");
  dialog.setAttribute("aria-label", "PixText Region Editor");
  const header = element("header", "pt-header");
  const title = element("div", "pt-title");
  title.append(element("span", "pt-brand", "PixText"), document.createTextNode("Region Editor"));
  header.append(title, element("span", "pt-subtitle", suggestions ? "Detect. Review. Refine." : "Draw precisely. Tell it what to say."), element("div", "pt-spacer"));
  const close = () => {
    if (dirty && !window.confirm("Discard unsaved region changes?")) return;
    dispose();
  };
  const status = element("span", "pt-status", suggestions ? "OCR suggestions · choose words to repair and verify target text" : compatible ? "All regions loaded" : "Ready to draw");
  const save = () => {
    const invalid = rows.find((row) => row.enabled !== false && !row.text.trim());
    if (invalid) { selected = invalid.id; renderList(true); setStatus("Enter target text for every enabled region.", true); return; }
    writeState(node, { version: 1, source_id: sourceId, size: [image.width, image.height], regions: rows, ...(suggestions ? { detection_id: suggestions.detection_id } : {}) });
    dirty = false;
    dispose();
  };
  header.append(button("Cancel", close), button("Apply regions", save, "pt-primary"));
  const toolbar = element("div", "pt-toolbar");
  const drawButton = button("▢  Draw", () => setTool("draw"));
  const selectButton = button("↖  Select", () => setTool("select"));
  const panButton = button("✥  Pan", () => setTool("pan"));
  const undoButton = button("↶", () => history(-1)); undoButton.title = "Undo (Ctrl/Cmd + Z)";
  const redoButton = button("↷", () => history(1)); redoButton.title = "Redo (Ctrl/Cmd + Shift + Z)";
  const zoomLabel = element("span", "pt-hint");
  toolbar.append(drawButton, selectButton, panButton, element("div", "pt-divider"), undoButton, redoButton,
    element("div", "pt-divider"), button("−", () => zoom(1 / 1.25)), zoomLabel,
    button("+", () => zoom(1.25)), button("Fit", () => fit()), button("Focus region", () => focusRegion()), element("div", "pt-spacer"),
    element("span", "pt-hint", "B draw · V select · Space pan · Scroll zoom"));
  if (imageName === EXAMPLE && !connected(node)) {
    toolbar.append(button("Example regions", async () => {
      try {
        const response = await api.fetchApi("/pixtext/example/chinese/regions");
        if (!response.ok) throw new Error("Example regions are unavailable.");
        const preset = await response.json();
        snapshot(); rows = preset.regions; selected = rows[0]?.id ?? null;
        markDirty(); renderList(); paint();
      } catch (error) { setStatus(error.message, true); }
    }));
  }
  if (suggestions) {
    toolbar.append(button("Reload OCR suggestions", () => {
      snapshot(); rows = clone(suggestions.regions); selected = rows[0]?.id ?? null;
      markDirty(); renderList(); paint();
    }));
  }
  const body = element("div", "pt-body");
  const stage = element("div", "pt-stage");
  const canvas = element("canvas"); canvas.tabIndex = 0; canvas.setAttribute("aria-label", "Image region drawing canvas");
  const ctx = canvas.getContext("2d"); stage.append(canvas);
  const sidebar = element("aside", "pt-sidebar");
  const sidehead = element("div", "pt-sidehead");
  const counter = element("span", "pt-counter");
  sidehead.append(element("strong", "", "Target regions"), counter, element("div", "pt-spacer"));
  const importInput = element("input"); importInput.type = "file"; importInput.accept = ".json,application/json"; importInput.hidden = true;
  sidehead.append(button("Import", () => importInput.click()), button("Export", exportJSON), importInput);
  const search = element("input", "pt-region-search"); search.type = "search";
  search.placeholder = "Find a word or target…"; search.setAttribute("aria-label", "Find region by text");
  search.addEventListener("input", () => renderList());
  const list = element("div", "pt-list"); sidebar.append(sidehead, search, list);
  body.append(stage, sidebar);
  const footer = element("footer", "pt-footer");
  footer.append(status, element("div", "pt-spacer"), element("span", "", `${image.width} × ${image.height} · original-image coordinates`));
  dialog.append(header, toolbar, body, footer); document.body.append(backdrop, dialog);
  const listeners = [];
  const listen = (target, name, fn, options) => { target.addEventListener(name, fn, options); listeners.push(() => target.removeEventListener(name, fn, options)); };
  function dispose() {
    if (closed) return;
    closed = true; listeners.forEach((fn) => fn()); observer.disconnect();
    backdrop.remove(); dialog.remove(); closeActiveEditor = null; previousFocus?.focus?.();
  }
  closeActiveEditor = dispose;
  function setStatus(message, error = false) { status.textContent = message; status.className = error ? "pt-error" : "pt-status"; }
  function markDirty() { dirty = true; setStatus("Unsaved changes"); }
  function snapshot() { undo.push(clone(rows)); if (undo.length > 80) undo.shift(); redo = []; updateHistory(); }
  function updateHistory() { undoButton.disabled = !undo.length; redoButton.disabled = !redo.length; }
  function history(direction) {
    const from = direction < 0 ? undo : redo, to = direction < 0 ? redo : undo;
    if (!from.length) return;
    to.push(clone(rows)); rows = from.pop(); selected = rows.some((r) => r.id === selected) ? selected : rows[0]?.id ?? null;
    markDirty(); updateHistory(); renderList(); paint();
  }
  function setTool(value) {
    tool = value;
    [[drawButton, "draw"], [selectButton, "select"], [panButton, "pan"]].forEach(([btn, mode]) => btn.setAttribute("aria-pressed", String(mode === tool)));
    canvas.style.cursor = tool === "pan" ? "grab" : tool === "draw" ? "crosshair" : "default";
  }
  function fit() {
    scale = Math.min((stage.clientWidth - 56) / image.width, (stage.clientHeight - 56) / image.height);
    tx = (stage.clientWidth - image.width * scale) / 2; ty = (stage.clientHeight - image.height * scale) / 2; paint();
  }
  function focusRegion() {
    const row = rows.find((item) => item.id === selected);
    if (!row) return;
    const [x0, y0, x1, y1] = row.bbox;
    scale = Math.min(6, (stage.clientWidth - 80) / (x1 - x0 + 120), (stage.clientHeight - 80) / (y1 - y0 + 120));
    tx = stage.clientWidth / 2 - (x0 + x1) / 2 * scale;
    ty = stage.clientHeight / 2 - (y0 + y1) / 2 * scale;
    paint();
  }
  function zoom(factor, x = stage.clientWidth / 2, y = stage.clientHeight / 2) {
    const next = Math.max(.05, Math.min(12, scale * factor));
    tx = x - (x - tx) * next / scale; ty = y - (y - ty) * next / scale; scale = next; paint();
  }
  function point(event) {
    const rect = canvas.getBoundingClientRect();
    return { sx: event.clientX - rect.left, sy: event.clientY - rect.top,
      x: Math.max(0, Math.min(image.width, (event.clientX - rect.left - tx) / scale)),
      y: Math.max(0, Math.min(image.height, (event.clientY - rect.top - ty) / scale)) };
  }
  function handles(box) {
    const [a, b, c, d] = box;
    return [[a, b], [(a + c) / 2, b], [c, b], [c, (b + d) / 2], [c, d], [(a + c) / 2, d], [a, d], [a, (b + d) / 2]];
  }
  function hit(p) {
    const chosen = rows.find((r) => r.id === selected);
    const handle = chosen ? handles(chosen.bbox).findIndex(([x, y]) => Math.hypot((x - p.x) * scale, (y - p.y) * scale) < 9) : -1;
    if (handle >= 0) return { row: chosen, handle };
    const row = [...rows].reverse().find((r) => p.x >= r.bbox[0] && p.x <= r.bbox[2] && p.y >= r.bbox[1] && p.y <= r.bbox[3]);
    return row ? { row, handle: -1 } : null;
  }
  function paint() {
    const dpr = window.devicePixelRatio || 1;
    const width = stage.clientWidth, height = stage.clientHeight;
    if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) { canvas.width = Math.round(width * dpr); canvas.height = Math.round(height * dpr); }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, width, height);
    ctx.save(); ctx.translate(tx, ty); ctx.scale(scale, scale); ctx.drawImage(image, 0, 0); ctx.restore();
    rows.forEach((row, index) => drawBox(row.bbox, index, row.id === selected, row.enabled !== false));
    if (draft) drawBox(draft, rows.length, true, true);
    zoomLabel.textContent = `${Math.round(scale * 100)}%`;
  }
  function drawBox(box, index, active, enabled) {
    const color = COLORS[index % COLORS.length], [x0, y0, x1, y1] = box;
    const x = tx + x0 * scale, y = ty + y0 * scale, w = (x1 - x0) * scale, h = (y1 - y0) * scale;
    ctx.save(); ctx.globalAlpha = enabled ? 1 : .4;
    ctx.strokeStyle = color; ctx.lineWidth = active ? 2 : 1.3;
    ctx.setLineDash(enabled ? [] : [5, 3]); ctx.fillStyle = active ? color + "22" : color + "09";
    ctx.fillRect(x, y, w, h); ctx.strokeRect(x, y, w, h); ctx.setLineDash([]);
    const labelY = y > 22 ? y - 21 : y + 2;
    ctx.fillStyle = color; ctx.beginPath(); ctx.roundRect(x, labelY, 25, 19, 4); ctx.fill();
    ctx.fillStyle = "#102027"; ctx.font = "bold 11px system-ui"; ctx.fillText(String(index + 1).padStart(2, "0"), x + 5, labelY + 13);
    if (active) handles(box).forEach(([hx, hy]) => { ctx.fillStyle = "#111a24"; ctx.fillRect(tx + hx * scale - 3, ty + hy * scale - 3, 6, 6); ctx.strokeRect(tx + hx * scale - 3, ty + hy * scale - 3, 6, 6); });
    ctx.restore();
  }
  function select(id, focus = false) { selected = id; search.value = ""; renderList(focus); paint(); }
  function remove(id) { snapshot(); rows = rows.filter((r) => r.id !== id); selected = rows[0]?.id ?? null; markDirty(); renderList(); paint(); }
  function renderList(focus = false) {
    const scroll = list.scrollTop; list.replaceChildren(); counter.textContent = `${rows.length}`;
    if (!rows.length) list.append(element("div", "pt-empty", "Draw a rectangle over the text you want to repair. Its target text will appear here."));
    rows.forEach((row, index) => {
      const query = search.value.trim().toLocaleLowerCase();
      if (query && !`${row.text} ${row.recognized_text || ""}`.toLocaleLowerCase().includes(query)) return;
      const card = element("div", "pt-card"); card.dataset.id = row.id; card.dataset.enabled = String(row.enabled !== false);
      card.setAttribute("aria-selected", String(row.id === selected));
      card.addEventListener("click", () => { if (selected !== row.id) select(row.id); });
      const top = element("div", "pt-cardtop"), crop = element("canvas", "pt-crop"); crop.width = 100; crop.height = 88;
      const [x0, y0, x1, y1] = row.bbox, cw = x1 - x0, ch = y1 - y0;
      const ratio = Math.min(100 / cw, 88 / ch);
      crop.getContext("2d").drawImage(image, x0, y0, cw, ch, (100 - cw * ratio) / 2, (88 - ch * ratio) / 2, cw * ratio, ch * ratio);
      const number = element("span", "pt-number", `Region ${String(index + 1).padStart(2, "0")}`); number.style.color = COLORS[index % COLORS.length];
      const enabled = element("input"); enabled.type = "checkbox"; enabled.checked = row.enabled !== false; enabled.setAttribute("aria-label", `Enable region ${index + 1}`);
      enabled.addEventListener("click", (e) => e.stopPropagation());
      enabled.addEventListener("change", () => { snapshot(); row.enabled = enabled.checked; markDirty(); renderList(); paint(); });
      const removeButton = button("×", (event) => { event.stopPropagation(); remove(row.id); }); removeButton.title = "Delete region";
      top.append(crop, number, element("div", "pt-spacer"), enabled, removeButton);
      const input = element("input", "pt-textinput"); input.value = row.text; input.placeholder = "Target text / 目标文字";
      input.setAttribute("aria-label", `Target text for region ${index + 1}`);
      input.addEventListener("click", (e) => e.stopPropagation());
      input.addEventListener("focus", () => { selected = row.id; snapshot(); [...list.children].forEach((el) => el.setAttribute("aria-selected", String(el === card))); paint(); });
      input.addEventListener("input", () => {
        row.text = input.value;
        if (row.origin === "ocr") { row.enabled = true; enabled.checked = true; card.dataset.enabled = "true"; }
        markDirty(); paint();
      });
      const recognition = row.origin === "ocr" ? element("div", "pt-recognition") : null;
      if (recognition) {
        const score = Number.isFinite(row.confidence) ? `${Math.round(row.confidence * 100)}%` : "—";
        recognition.textContent = `OCR: ${row.recognized_text} · ${score} confidence`;
        recognition.dataset.low = String(row.confidence < .85);
      }
      const coords = element("details", "pt-coords");
      coords.addEventListener("click", (event) => event.stopPropagation());
      coords.append(element("summary", "", row.bbox.map((v) => Math.round(v * 10) / 10).join(" · ")));
      const fields = element("div", "pt-coordinate-fields");
      ["x0", "y0", "x1", "y1"].forEach((name, k) => {
        const field = element("input"); field.type = "number"; field.step = "0.1"; field.value = String(row.bbox[k]); field.title = name; field.setAttribute("aria-label", `${name} region ${index + 1}`);
        field.addEventListener("focus", snapshot);
        field.addEventListener("input", () => {
          const values = [...fields.children].map((f) => f.value === "" ? NaN : Number(f.value));
          if (validBox(values)) { row.bbox = values; coords.querySelector("summary").textContent = values.map((v) => Math.round(v * 10) / 10).join(" · "); markDirty(); paint(); }
        });
        field.addEventListener("change", () => {
          const values = [...fields.children].map((f) => f.value === "" ? NaN : Number(f.value));
          if (!validBox(values)) { setStatus("Coordinates must form a valid box inside the image.", true); field.value = String(row.bbox[k]); }
        });
        fields.append(field);
      }); coords.append(fields); card.append(top, input); if (recognition) card.append(recognition); card.append(coords); list.append(card);
      if (focus && row.id === selected) requestAnimationFrame(() => { input.focus(); card.scrollIntoView({ block: "nearest" }); });
    });
    list.scrollTop = scroll;
  }
  function validBox(box) {
    return box.length === 4 && box.every(Number.isFinite) && box[0] >= 0 && box[1] >= 0 && box[2] <= image.width && box[3] <= image.height && box[2] > box[0] && box[3] > box[1];
  }
  function exportJSON() {
    const data = JSON.stringify({ version: 1, source_id: sourceId, size: [image.width, image.height], regions: rows }, null, 2);
    const url = URL.createObjectURL(new Blob([data], { type: "application/json" }));
    const a = element("a"); a.href = url; a.download = "pixtext-regions.json"; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  listen(importInput, "change", async () => {
    try {
      const value = JSON.parse(await importInput.files[0].text());
      const incoming = Array.isArray(value) ? value : value.regions;
      if (!Array.isArray(incoming) || incoming.length > 256) throw new Error("Expected up to 256 region entries.");
      if (!Array.isArray(value) && (value.size?.[0] !== image.width || value.size?.[1] !== image.height)) throw new Error("Imported image dimensions do not match.");
      const next = incoming.map((row, i) => {
        if (!validBox(row.bbox) || typeof row.text !== "string") throw new Error(`Invalid region ${i + 1}.`);
        return { id: crypto.randomUUID(), bbox: row.bbox, text: row.text, enabled: row.enabled !== false };
      }); snapshot(); rows = next; selected = rows[0]?.id ?? null; markDirty(); renderList(); paint();
    } catch (error) { setStatus(error.message, true); }
    importInput.value = "";
  });
  listen(canvas, "wheel", (event) => { event.preventDefault(); const p = point(event); zoom(Math.exp(-event.deltaY * .0015), p.sx, p.sy); }, { passive: false });
  listen(canvas, "pointerdown", (event) => {
    if (event.button !== 0 && event.button !== 1) return;
    event.preventDefault(); canvas.focus(); canvas.setPointerCapture(event.pointerId);
    const p = point(event);
    if (space || tool === "pan" || event.button === 1) { gesture = { type: "pan", p, tx, ty }; return; }
    const found = hit(p);
    if (found && (tool === "select" || found.handle >= 0)) {
      selected = found.row.id; snapshot(); gesture = { type: found.handle >= 0 ? "resize" : "move", p, box: [...found.row.bbox], row: found.row, handle: found.handle };
      renderList(); paint(); return;
    }
    if (tool === "select") { selected = found?.row.id ?? null; renderList(); paint(); return; }
    if (rows.length >= 256) return setStatus("A maximum of 256 regions is supported.", true);
    gesture = { type: "draw", p }; draft = [p.x, p.y, p.x, p.y];
  });
  listen(canvas, "pointermove", (event) => {
    if (!gesture) return;
    const p = point(event), g = gesture;
    if (g.type === "pan") { tx = g.tx + p.sx - g.p.sx; ty = g.ty + p.sy - g.p.sy; }
    else if (g.type === "draw") draft = [Math.min(g.p.x, p.x), Math.min(g.p.y, p.y), Math.max(g.p.x, p.x), Math.max(g.p.y, p.y)];
    else if (g.type === "move") {
      const dx = Math.max(-g.box[0], Math.min(image.width - g.box[2], p.x - g.p.x));
      const dy = Math.max(-g.box[1], Math.min(image.height - g.box[3], p.y - g.p.y));
      g.row.bbox = [g.box[0] + dx, g.box[1] + dy, g.box[2] + dx, g.box[3] + dy];
    } else {
      const box = [...g.box];
      if ([0, 6, 7].includes(g.handle)) box[0] = Math.min(p.x, box[2] - 1);
      if ([2, 3, 4].includes(g.handle)) box[2] = Math.max(p.x, box[0] + 1);
      if ([0, 1, 2].includes(g.handle)) box[1] = Math.min(p.y, box[3] - 1);
      if ([4, 5, 6].includes(g.handle)) box[3] = Math.max(p.y, box[1] + 1);
      if (validBox(box)) g.row.bbox = box;
    } paint();
  });
  const endGesture = (event) => {
    if (!gesture) return;
    const type = gesture.type;
    if (type === "draw" && draft && (draft[2] - draft[0]) * scale >= 4 && (draft[3] - draft[1]) * scale >= 4) {
      snapshot(); const row = { id: crypto.randomUUID(), bbox: draft.map((v) => Math.round(v * 10) / 10), text: "", enabled: true };
      if (validBox(row.bbox)) { rows.push(row); selected = row.id; markDirty(); renderList(true); }
    } else if (type === "move" || type === "resize") { markDirty(); renderList(); }
    draft = null; gesture = null; if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId); paint();
  };
  listen(canvas, "pointerup", endGesture);
  listen(canvas, "pointercancel", () => { if (gesture?.box && gesture.row) gesture.row.bbox = gesture.box; gesture = null; draft = null; paint(); });
  listen(dialog, "keydown", (event) => {
    event.stopPropagation();
    const input = ["INPUT", "TEXTAREA", "SELECT"].includes(event.target.tagName);
    if (event.isComposing) return;
    if (event.key === "Escape") { event.preventDefault(); close(); return; }
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") { event.preventDefault(); save(); return; }
    if (event.key === "Tab") {
      const focusable = [...dialog.querySelectorAll("button:not(:disabled), input:not([hidden]), canvas[tabindex], summary")].filter((el) => el.getClientRects().length);
      const first = focusable[0], last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
    }
    if (input) return;
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z") { event.preventDefault(); history(event.shiftKey ? 1 : -1); }
    else if (event.code === "Space") { event.preventDefault(); space = true; canvas.style.cursor = "grab"; }
    else if (event.key.toLowerCase() === "b") setTool("draw");
    else if (event.key.toLowerCase() === "v") setTool("select");
    else if ((event.key === "Delete" || event.key === "Backspace") && selected) { event.preventDefault(); remove(selected); }
  });
  listen(window, "keyup", (event) => { if (event.code === "Space") { space = false; setTool(tool); } });
  listen(window, "blur", () => { space = false; });
  const observer = new ResizeObserver(() => paint()); observer.observe(stage);
  if (!compatible && saved.regions?.length) setStatus("Different image detected. Existing regions were not applied.");
  updateHistory(); renderList(); setTool("draw"); requestAnimationFrame(() => { fit(); canvas.focus(); });
}

function setup(node) {
  if (node.ptSummary) return;
  node.ptSource = node.properties?.pixtext_source ?? null;
  node.ptDetections = node.properties?.pixtext_detections ?? null;
  const data = widget(node, "region_data");
  if (data) {
    data.type = "hidden"; data.computeSize = () => [0, -4]; data.draw = () => {};
    if (data.element) data.element.style.display = "none";
  }
  const summary = element("div", "pt-summary");
  const line = element("div", "pt-summaryline"), count = element("strong"), detail = element("small");
  line.append(count, element("span", "", "▢")); summary.append(line, detail, button("Open Region Editor", () => openEditor(node)));
  node.ptSummary = { count, detail };
  const dom = node.addDOMWidget("pixtext_editor", "div", summary, { serialize: false, hideOnZoom: false });
  dom.computeSize = () => [320, 112];
  node.setSize([Math.max(360, node.size[0]), Math.max(245, node.size[1])]);
  updateSummary(node);
  const imageWidget = widget(node, "image"), oldCallback = imageWidget.callback;
  imageWidget.callback = function (...args) { oldCallback?.apply(this, args); updateSummary(node); };
  const oldExecuted = node.onExecuted;
  node.onExecuted = function (message) {
    oldExecuted?.call(this, message);
    if (message.source_image?.[0]) {
      this.ptSource = { image: message.source_image[0], id: message.source_id[0] };
      if (connected(this)) {
        this.properties ??= {};
        this.properties.pixtext_source = this.ptSource;
      }
    }
    if (message.ocr_suggestions?.[0]) {
      this.ptDetections = message.ocr_suggestions[0];
      this.properties ??= {};
      this.properties.pixtext_detections = this.ptDetections;
    }
    updateSummary(this);
    if (message.region_message?.[0]) this.ptSummary.detail.textContent = message.region_message[0];
  };
  const oldConfigure = node.onConfigure;
  node.onConfigure = function (...args) {
    oldConfigure?.apply(this, args);
    this.ptSource = this.properties?.pixtext_source ?? this.ptSource;
    this.ptDetections = this.properties?.pixtext_detections ?? this.ptDetections;
    updateSummary(this);
  };
  const oldRemoved = node.onRemoved;
  node.onRemoved = function (...args) { closeActiveEditor?.(); oldRemoved?.apply(this, args); };
}

app.registerExtension({
  name: "PixText.RegionEditor",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "PixTextRegionEditor") return;
    const previous = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function (...args) { const result = previous?.apply(this, args); setup(this); return result; };
  },
});
